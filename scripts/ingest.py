"""Pipeline ingestion: data/raw/ -> parse -> chunk -> embed -> vectorstore.

Cách chạy:
    python -m scripts.ingest
    python -m scripts.ingest --topic hien_phap   # chỉ 1 chủ đề
    python -m scripts.ingest --reset             # xoá store cũ trước khi nạp
    python -m scripts.ingest --skip-existing     # bỏ qua file đã embed (resume)

Tối ưu GPU: chunks được GOM TOÀN CỤC vào buffer (mặc định 1024, override bằng
env EMBED_BUFFER_CHUNKS) rồi mới embed một lần — file nhỏ không còn làm GPU
chạy batch lẻ. Embedder tự bật FP16 + batch size phù hợp khi có CUDA.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator

# UTF-8 fix for Windows console
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.parsing import load_document
from src.chunking import chunk_document
from src.embedding import Embedder
from src.vectorstore import VectorStore
from src.schemas import DocumentMetadata
from src.parent_store import ParentStore

_HEADER_SEP = re.compile(r"^[─\-]{20,}", re.MULTILINE)


def _parse_header(text: str) -> dict:
    """
    Trích metadata từ header 9 dòng đầu của file txt vbpl.vn.
    Header format:
        NGUON: ...
        SO_HIEU: ...
        TEN: ...
        LOAI: ...
        CO_QUAN: ...
        NGAY_BAN_HANH: ...
        HIEU_LUC: ...
        URL: ...
        CHU_DE: ...
        ─────────...
    """
    meta: dict[str, str] = {}
    for line in text.splitlines()[:20]:
        if ": " in line:
            key, _, val = line.partition(": ")
            meta[key.strip()] = val.strip()
    return meta


def iter_raw_files(
    raw_dir: Path,
    topic_filter: str | None = None,
) -> Iterator[tuple[Path, DocumentMetadata]]:
    """Yield (path, DocumentMetadata) cho mọi .txt file trong raw_dir.

    Xử lý cả:
    - File gốc tại data/raw/*.txt  (bo_luat_hinh_su.txt, ...)
    - Subdirectory: all_laws/, nghi_dinh/, an_le/, nghi_quyet/, ...
    """
    # ── 1. File root (data/raw/*.txt) ─────────────────────────────────────────
    if not topic_filter:
        for txt_path in sorted(raw_dir.glob("*.txt")):
            if txt_path.name.endswith(".gitkeep"):
                continue
            try:
                raw = txt_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                print(f"  [!] Không đọc được {txt_path.name}: {exc}")
                continue
            hdr  = _parse_header(raw)
            meta = _build_meta(hdr, txt_path, "root")
            yield txt_path, meta

    # ── 2. Subdirectory ───────────────────────────────────────────────────────
    for topic_dir in sorted(raw_dir.iterdir()):
        if not topic_dir.is_dir():
            continue
        if topic_filter and topic_dir.name != topic_filter:
            continue
        # Bỏ qua các thư mục không chứa .txt (vectorstore, processed...)
        if topic_dir.name in ("vectorstore", "processed", "__pycache__"):
            continue

        # rglob để xử lý cả cấu trúc phẳng (all_laws/*.txt)
        # lẫn cấu trúc lồng nhau (hf_laws/chi_thi/*.txt, hf_laws/luat/*.txt)
        for txt_path in sorted(topic_dir.rglob("*.txt")):
            if txt_path.name.endswith(".gitkeep"):
                continue
            try:
                raw = txt_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                print(f"  [!] Không đọc được {txt_path.name}: {exc}")
                continue
            hdr  = _parse_header(raw)
            meta = _build_meta(hdr, txt_path, topic_dir.name)
            yield txt_path, meta


def _build_meta(hdr: dict, path: Path, folder: str) -> DocumentMetadata:
    """Tạo DocumentMetadata từ header dict + path info."""
    return DocumentMetadata(
        source      = hdr.get("URL") or str(path),
        doc_type    = hdr.get("LOAI") or None,
        doc_number  = hdr.get("SO_HIEU") or None,
        title       = hdr.get("TEN") or path.stem,
        issued_date = hdr.get("NGAY_BAN_HANH") or None,
        status      = hdr.get("HIEU_LUC") or None,
        linh_vuc    = hdr.get("LINH_VUC") or None,
        co_quan     = hdr.get("CO_QUAN") or None,
        folder      = folder,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Nạp data/raw/ vào vectorstore")
    p.add_argument("--topic", default=None, help="Chỉ nạp 1 chủ đề (tên thư mục)")
    p.add_argument("--reset", action="store_true", help="Xoá collection cũ trước khi nạp")
    p.add_argument("--skip-existing", action="store_true",
                   help="Bỏ qua file đã có trong store (resume sau khi ingest bị dừng)")
    p.add_argument("--no-parent-child", action="store_true",
                   help="Tắt parent-child chunking (dùng khi chỉ muốn reindex nhanh)")
    args = p.parse_args()

    if args.reset and args.skip_existing:
        print("⚠ --reset và --skip-existing không dùng cùng nhau. Bỏ --skip-existing.")
        args.skip_existing = False

    embedder = Embedder(config.EMBEDDING_MODEL)
    store    = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)

    # Parent-Child store
    use_pc = config.USE_PARENT_CHILD and not args.no_parent_child
    parent_store: ParentStore | None = None
    if use_pc:
        parent_store = ParentStore(config.PARENT_STORE_PATH)
        print(f"Parent-Child ON — store: {config.PARENT_STORE_PATH} ({parent_store.count()} entries)")
        if args.reset:
            parent_store.reset()
    else:
        print("Parent-Child OFF")

    if args.reset:
        print("Reset vector collection...")
        store.reset()

    # Load 1 lần set sources đã có để skip
    existing_sources: set[str] = set()
    if args.skip_existing:
        print("Đang đọc danh sách doc đã embed trong store...")
        for c in store.iter_all_chunks():
            existing_sources.add(c.metadata.source)
        print(f"  → {len(existing_sources)} doc đã có, sẽ bỏ qua.")

    # ── Global batching ───────────────────────────────────────────────────────
    # Gom chunks từ nhiều file vào buffer rồi embed 1 lần → GPU luôn chạy
    # batch đầy thay vì batch lẻ theo từng file nhỏ.
    EMBED_BUFFER = int(os.getenv("EMBED_BUFFER_CHUNKS", "1024"))
    STORE_SLICE  = 4096  # Chroma giới hạn ~5461 items/lần add

    LOG_EVERY = 200  # in tiến độ mỗi N docs (tránh ngập log với 89k files)

    total_docs    = 0
    total_chunks  = 0
    skipped       = 0
    buffer: list  = []
    t_start       = time.time()

    def _flush() -> None:
        """Embed + lưu toàn bộ buffer hiện tại."""
        nonlocal total_chunks
        if not buffer:
            return
        embeddings = embedder.encode([c.text for c in buffer])
        for i in range(0, len(buffer), STORE_SLICE):
            store.add(buffer[i:i + STORE_SLICE], embeddings[i:i + STORE_SLICE])
        total_chunks += len(buffer)
        buffer.clear()

    for path, meta in iter_raw_files(config.RAW_DIR, topic_filter=args.topic):
        if args.skip_existing and meta.source in existing_sources:
            skipped += 1
            continue
        try:
            doc    = load_document(path, meta)
            chunks = chunk_document(doc, parent_store=parent_store)
            if not chunks:
                print(f"  [!] Không chunk được: {path.name}")
                continue
            buffer.extend(chunks)
            total_docs += 1

            if len(buffer) >= EMBED_BUFFER:
                _flush()

            if total_docs % LOG_EVERY == 0:
                elapsed = time.time() - t_start
                done = total_chunks + len(buffer)
                rate = done / elapsed if elapsed > 0 else 0
                print(
                    f"  [{total_docs:,} docs] {done:,} chunks | "
                    f"{rate:.0f} chunks/s | {elapsed/60:.1f} phút"
                )
        except Exception as exc:
            print(f"  [!] Lỗi {path.name}: {exc}")

    _flush()  # phần còn lại trong buffer

    elapsed = time.time() - t_start
    print(f"\nXong sau {elapsed/3600:.2f} giờ. Mới nạp: {total_docs:,} tài liệu, {total_chunks:,} chunks.")
    if total_chunks and elapsed > 0:
        print(f"Throughput trung bình: {total_chunks/elapsed:.0f} chunks/s")
    if skipped:
        print(f"Đã bỏ qua: {skipped:,} tài liệu (--skip-existing).")
    print(f"Store total: {store.count():,} chunks")
    if parent_store:
        print(f"Parent store total: {parent_store.count():,} entries")


if __name__ == "__main__":
    main()

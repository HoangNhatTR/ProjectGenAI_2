"""Pipeline tự động cập nhật corpus: crawl → parse → chunk → embed → index.

Chỉ xử lý văn bản MỚI kể từ lần chạy trước — không đụng đến VB đã có.

Cách chạy:
    python -m scripts.auto_update                    # chạy thường
    python -m scripts.auto_update --since 7          # crawl 7 ngày gần nhất
    python -m scripts.auto_update --skip-crawl       # chỉ ingest, không crawl
    python -m scripts.auto_update --rebuild-bm25     # rebuild BM25 sau khi ingest

Lên lịch (Windows Task Scheduler):
    Chạy mỗi ngày lúc 3:00 sáng:
    Action: python -m scripts.auto_update
    Start in: <thư mục project>

Lên lịch (Linux cron):
    0 3 * * * cd /path/to/project && python -m scripts.auto_update >> logs/update.log 2>&1
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# UTF-8 fix cho Windows console
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.chunking import chunk_document
from src.embedding import Embedder
from src.parsing import load_document
from src.schemas import DocumentMetadata
from src.vectorstore import VectorStore


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_header(text: str) -> dict:
    """Trích metadata từ header file .txt (format vbpl.vn)."""
    meta: dict[str, str] = {}
    for line in text.splitlines()[:20]:
        if ": " in line:
            key, _, val = line.partition(": ")
            meta[key.strip()] = val.strip()
    return meta


def _get_existing_sources(store: VectorStore) -> set[str]:
    """Lấy tập hợp URL đã có trong vectorstore (để skip khi ingest)."""
    print("  Đọc danh sách VB đã embed... ", end="", flush=True)
    sources = {c.metadata.source for c in store.iter_all_chunks()}
    print(f"{len(sources)} VB.")
    return sources


# ── Step 1: Crawl ──────────────────────────────────────────────────────────────

def step_crawl(since_days: Optional[int], dry_run: bool) -> list[Path]:
    """Crawl vbpl.vn và trả về danh sách file mới đã tải."""
    from scripts.crawler import CrawlState, VBPLCrawler

    root       = Path(__file__).resolve().parent.parent
    state_file = root / "data" / "crawl_state.json"
    out_dir    = root / "data" / "raw" / "crawled"

    state   = CrawlState(state_file)
    crawler = VBPLCrawler()

    since = (
        datetime.now() - timedelta(days=since_days)
        if since_days
        else state.last_crawled(VBPLCrawler.SOURCE)
    )

    print(f"\n{'='*60}")
    print(f"BƯỚC 1: CRAWL vbpl.vn (từ {since.strftime('%d/%m/%Y')})")
    print(f"{'='*60}")

    docs = crawler.list_new(since)
    print(f"  → Tìm được {len(docs)} VB mới.")

    if dry_run:
        for d in docs:
            print(f"    [DRY] {d.doc_number} — {d.title[:60]}")
        return []

    new_files: list[Path] = []
    for i, info in enumerate(docs, 1):
        print(f"  [{i:3d}/{len(docs)}] {info.doc_number or info.item_id[:20]:<25}", end=" ", flush=True)
        text = crawler.fetch_text(info)
        if not text:
            print("SKIP (lỗi tải)")
            continue
        if state.is_new_or_changed(info.url, text):
            path = crawler.save(info, text, out_dir)
            print(f"→ lưu {path.name}")
            new_files.append(path)
            state.add_stat("total_downloaded")
        else:
            print("→ không đổi")

    state.set_crawled_now(VBPLCrawler.SOURCE)
    state.save()
    print(f"\n  → Tải mới: {len(new_files)} file vào {out_dir}")
    return new_files


# ── Step 2: Incremental ingest ────────────────────────────────────────────────

def step_ingest(new_files: list[Path], store: VectorStore, embedder: Embedder) -> int:
    """Parse → chunk → embed → upsert chỉ các file mới."""
    print(f"\n{'='*60}")
    print(f"BƯỚC 2: INGEST {len(new_files)} file mới vào vectorstore")
    print(f"{'='*60}")

    if not new_files:
        print("  Không có file mới — bỏ qua.")
        return 0

    total_chunks = 0
    ok_docs      = 0

    for i, path in enumerate(new_files, 1):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            hdr = _parse_header(raw)
            meta = DocumentMetadata(
                source      = hdr.get("URL") or str(path),
                doc_type    = hdr.get("LOAI"),
                doc_number  = hdr.get("SO_HIEU"),
                title       = hdr.get("TEN") or path.stem,
                issued_date = hdr.get("NGAY_BAN_HANH"),
                status      = hdr.get("HIEU_LUC"),
            )
            doc    = load_document(path, meta)
            chunks = chunk_document(doc)
            if not chunks:
                print(f"  [{i}] {path.name}: không chunk được")
                continue

            embeddings = embedder.encode_chunks(chunks)
            store.add(chunks, embeddings)     # upsert — an toàn nếu đã có
            total_chunks += len(chunks)
            ok_docs      += 1
            print(f"  [{i:3d}/{len(new_files)}] {path.name:<40} → {len(chunks):3d} chunks")

        except Exception as exc:
            print(f"  [{i}] [LỖI] {path.name}: {exc}")

    print(f"\n  → Ingest xong: {ok_docs}/{len(new_files)} VB, {total_chunks} chunks mới.")
    print(f"  → Vectorstore tổng: {store.count()} chunks.")
    return total_chunks


# ── Step 3: Rebuild BM25 ──────────────────────────────────────────────────────

def step_rebuild_bm25(store: VectorStore) -> None:
    """Rebuild BM25 index từ toàn bộ vectorstore hiện tại."""
    from src.bm25_index import BM25Index

    print(f"\n{'='*60}")
    print("BƯỚC 3: REBUILD BM25 INDEX")
    print(f"{'='*60}")

    bm25_path = config.DATA_DIR / "bm25" / "index.json"
    bm25      = BM25Index(bm25_path)

    print("  Đọc toàn bộ chunks từ vectorstore...", flush=True)
    all_chunks = list(store.iter_all_chunks())
    print(f"  → {len(all_chunks)} chunks.")

    print("  Build BM25...", flush=True)
    bm25.build(all_chunks)
    bm25.save()
    print(f"  → BM25 lưu tại {bm25_path} ({bm25.count()} docs).")


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tự động cập nhật corpus pháp luật từ vbpl.vn"
    )
    p.add_argument("--since", type=int, default=None,
                   help="Crawl N ngày gần nhất (override crawl state)")
    p.add_argument("--skip-crawl", action="store_true",
                   help="Bỏ qua bước crawl, chỉ ingest file trong data/raw/crawled/")
    p.add_argument("--rebuild-bm25", action="store_true",
                   help="Rebuild BM25 sau ingest (mặc định: chỉ rebuild nếu có file mới)")
    p.add_argument("--dry-run", action="store_true",
                   help="Xem danh sách VB mới, không thực sự tải/ingest")
    return p.parse_args()


def main() -> None:
    args = parse_args = _parse_args()
    start = datetime.now()

    print("=" * 60)
    print("  AUTO UPDATE — CẬP NHẬT CORPUS PHÁP LUẬT")
    print(f"  {start.strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 60)

    # ── Khởi tạo ──────────────────────────────────────────────────────────────
    embedder = Embedder(config.EMBEDDING_MODEL)
    store    = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)

    # ── Bước 1: Crawl ──────────────────────────────────────────────────────────
    if args.skip_crawl:
        # Dùng file đã có trong crawled/ mà chưa ingest
        crawled_dir = config.DATA_DIR / "raw" / "crawled"
        existing    = _get_existing_sources(store)
        new_files   = [
            p for p in crawled_dir.glob("*.txt")
            if not any(p.name in s for s in existing)
        ] if crawled_dir.exists() else []
        print(f"\nBỏ qua crawl. Tìm thấy {len(new_files)} file chưa ingest trong crawled/")
    else:
        new_files = step_crawl(args.since, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[DRY RUN] Kết thúc — không thực sự thay đổi corpus.")
        return

    # ── Bước 2: Ingest ─────────────────────────────────────────────────────────
    new_chunks = step_ingest(new_files, store, embedder)

    # ── Bước 3: Rebuild BM25 ──────────────────────────────────────────────────
    if new_chunks > 0 or args.rebuild_bm25:
        step_rebuild_bm25(store)
    else:
        print("\nBỎ QUA rebuild BM25 (không có chunk mới).")

    # ── Tóm tắt ───────────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"  HOÀN THÀNH — {elapsed:.0f}s")
    print(f"  Files mới  : {len(new_files)}")
    print(f"  Chunks mới : {new_chunks}")
    print(f"  Store tổng : {store.count()} chunks")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Re-ingest TOÀN VĂN NĐ 168/2024/NĐ-CP vào OpenSearch (sửa lỗi chỉ có 5/57 Điều).

Lỗi gốc: add_penalty_decrees.py dùng docAbs (tóm tắt) → chỉ 5 Điều lọt corpus.
Script này lấy TOÀN VĂN qua VBPLClient (documentContent.content), chunk theo
Điều/Khoản/Điểm + parent, embed BGE-M3 (CPU), xoá bản cũ rồi ingest đầy đủ.

    python -m scripts.reingest_nd168 --dry-run   # chỉ chunk + đếm, KHÔNG ghi
    python -m scripts.reingest_nd168             # chạy thật (embed + ghi OpenSearch)

Chạy bằng venv Chatbot: ../Chatbot/Scripts/python.exe (cần opensearchpy + torch).
OpenSearch phải đang bật.
"""
from __future__ import annotations

import argparse
import re
import sys
import time

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.chunking import chunk_document
from src.parent_store import ParentStore
from src.schemas import DocumentMetadata, RawDocument
from src.vbpl_client import VBPLClient

DOC_ID = "173920"  # id vbpl.vn của NĐ 168/2024 (xác minh bằng fetch_nd168_test)
DOC_NUMBER = "168/2024/NĐ-CP"

# Metadata khớp CHÍNH XÁC chunk 168 hiện có trong corpus (để đồng nhất, cùng
# source → cùng prefix chunk_id '168-2024-nd-cp_0090d543').
META = DocumentMetadata(
    source="https://vbpl.vn/van-ban/chi-tiet/168-2024-nd-cp",
    doc_type="Nghị định (NĐ)",
    doc_number=DOC_NUMBER,
    title="Nghị định 168/2024/NĐ-CP quy định xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ",
    issued_date="2024-12-26",
    status="Còn hiệu lực (có hiệu lực từ 01/01/2025, thay thế NĐ 100/2019/NĐ-CP)",
    linh_vuc="giao_thong",
    co_quan="Chính phủ",
    folder="nghi_dinh",
)
PARENT_PREFIX = "168-2024-nd-cp_"  # để dọn parent cũ (orphan) trước khi ghi lại


def fetch_full_text() -> str:
    client = VBPLClient()
    print(f">>> Fetch toàn văn id={DOC_ID} ...", flush=True)
    full = client.fetch_with_content(DOC_ID)
    if not full or not full.get("content"):
        # fallback: search lại nếu id đổi
        docs = client.search("xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ", max_docs=250)
        t = next((d for d in docs if "168/2024" in (d["so_hieu"] or "")), None)
        if t:
            full = client.fetch_with_content(t["id"])
    if not full or not full.get("content"):
        sys.exit("[FAIL] Không lấy được toàn văn NĐ 168/2024.")
    return full["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Chỉ chunk + đếm, không embed/ghi")
    args = ap.parse_args()

    text = fetch_full_text()
    arts = sorted(set(int(m) for m in re.findall(r"Điều\s+(\d+)\.", text)))
    print(f"   Toàn văn: {len(text):,} chars | {len(arts)} Điều (max={arts[-1]})", flush=True)

    parent_store = ParentStore(config.PARENT_STORE_PATH)
    doc = RawDocument(text=text, metadata=META)

    # ── Chunk (ghi parent vào store) ──────────────────────────────────────────
    if args.dry_run:
        # dry-run: KHÔNG ghi parent → dùng store giả
        class _NoStore:
            def add_batch(self, items): self._n = getattr(self, "_n", 0) + len(items)
        ns = _NoStore()
        chunks = chunk_document(doc, parent_store=ns)
        art_chunks = sum(1 for c in chunks if c.article)
        with_parent = sum(1 for c in chunks if c.parent_id)
        print(f"\n=== DRY-RUN ===")
        print(f"  Tổng chunk      : {len(chunks)}")
        print(f"  Có article      : {art_chunks}")
        print(f"  Có parent_id    : {with_parent}")
        print(f"  Parent entries  : {getattr(ns, '_n', 0)}")
        print(f"  Điều phân biệt  : {len(set(c.article for c in chunks if c.article))}")
        print(f"\n  Mẫu 3 chunk:")
        for c in chunks[:3]:
            print(f"    [{c.article}/{c.clause}/{c.point}] {c.text[:90].strip()!r}")
        return

    # ── Dọn parent cũ (orphan) của 168 rồi chunk lại ──────────────────────────
    import sqlite3
    with sqlite3.connect(parent_store.db_path) as conn:
        n = conn.execute("DELETE FROM parents WHERE id LIKE ?", (PARENT_PREFIX + "%",)).rowcount
        conn.commit()
    print(f"   Xoá {n} parent cũ của 168", flush=True)

    chunks = chunk_document(doc, parent_store=parent_store)
    print(f"   Chunk: {len(chunks)} | parent_store giờ {parent_store.count():,} entries", flush=True)

    # ── Embed (CPU) ───────────────────────────────────────────────────────────
    from src.embedding import Embedder
    embedder = Embedder(config.EMBEDDING_MODEL)
    print(f">>> Embed {len(chunks)} chunk (BGE-M3 CPU)...", flush=True)
    t0 = time.time()
    embeddings = embedder.encode([c.text for c in chunks])
    print(f"   Embed xong {time.time()-t0:.0f}s", flush=True)

    # ── Xoá chunk 168 cũ trong OpenSearch rồi ghi đầy đủ ──────────────────────
    from src.opensearch_store import OpenSearchVectorStore
    vstore = OpenSearchVectorStore(config.OPENSEARCH_URL, config.OPENSEARCH_INDEX)
    cli = vstore._connect()
    d = cli.delete_by_query(index=config.OPENSEARCH_INDEX, refresh=True,
                            body={"query": {"term": {"doc_number": DOC_NUMBER}}})
    print(f"   Xoá {d.get('deleted')} chunk 168 cũ trong OpenSearch", flush=True)

    vstore.add(chunks, embeddings)
    cli.indices.refresh(index=config.OPENSEARCH_INDEX)

    # ── Verify ────────────────────────────────────────────────────────────────
    r = cli.search(index=config.OPENSEARCH_INDEX, body={
        "size": 0, "query": {"term": {"doc_number": DOC_NUMBER}},
        "aggs": {"arts": {"cardinality": {"field": "article"}}},
    })
    cnt = cli.count(index=config.OPENSEARCH_INDEX, body={"query": {"term": {"doc_number": DOC_NUMBER}}})["count"]
    print(f"\n=== XONG ===")
    print(f"  NĐ 168/2024 giờ: {cnt} chunk | {r['aggregations']['arts']['value']} Điều phân biệt (trước: 55 chunk / 5 Điều)")


if __name__ == "__main__":
    main()

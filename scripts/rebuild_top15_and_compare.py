"""Dựng lại 15 luật VỚI nhãn Điều (article) rồi so sánh RAG vs Graph-RAG.

Mục đích: store cũ (legal_docs) build bằng code đời trước → chunk KHÔNG có
metadata `article` → cầu nối KG→chunk chết → Graph-RAG ≡ RAG. Script này
re-ingest 15 luật bằng pipeline hiện tại (đã gắn article), build BM25 riêng,
rồi đo lại — kỳ vọng KG bắt đầu đóng góp chunk riêng (only_gr > 0).

Chạy:
    python -m scripts.rebuild_top15_and_compare
"""
from __future__ import annotations

import argparse
import json
import sys
import time

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.bm25_index import BM25Index
from src.chunking import chunk_document
from src.embedding import Embedder
from src.kg.kg_retriever import KGRetriever
from src.parsing import load_document
from src.retriever import Retriever
from src.vectorstore import VectorStore
from scripts.ingest import iter_raw_files
from scripts.compare_rag_vs_graphrag import TEST_QUERIES, fmt_chunk

COLL = "legal_top15"
BM25_PATH = config.DATA_DIR / "bm25" / "top15_index.json"
TOP_K = 5


def build_store() -> tuple[VectorStore, BM25Index]:
    man = json.loads(
        (config.DATA_DIR / "comparison" / "top15_laws" / "manifest.json").read_text(encoding="utf-8")
    )
    top15 = {l["source_url"] for l in man["laws"]}
    print(f"15 luật cần nạp. Đang quét raw files + chunk (gắn article)...")

    embedder = Embedder(config.EMBEDDING_MODEL)
    store = VectorStore(config.VECTORSTORE_DIR, COLL)
    store.reset()  # collection sạch

    all_chunks = []
    seen_src = set()
    for path, meta in iter_raw_files(config.RAW_DIR):
        if meta.source not in top15 or meta.source in seen_src:
            continue
        seen_src.add(meta.source)
        doc = load_document(path, meta)
        chunks = chunk_document(doc)  # không parent_store — KG không cần
        all_chunks.extend(chunks)
        print(f"  + {path.name[:38]:38} {len(chunks):4} chunks "
              f"(article={sum(1 for c in chunks if c.article)})")

    # dedup chunk_id
    seen_ids, unique = set(), []
    for c in all_chunks:
        if c.chunk_id in seen_ids:
            continue
        seen_ids.add(c.chunk_id)
        unique.append(c)

    with_art = sum(1 for c in unique if c.article)
    print(f"\nTổng {len(unique):,} chunks ({len(seen_src)}/15 luật) | "
          f"có article: {with_art:,} ({100*with_art/max(len(unique),1):.0f}%)")

    print("Đang embed (CPU, có thể vài phút)...")
    t0 = time.time()
    SLICE = 256
    for i in range(0, len(unique), SLICE):
        batch = unique[i:i + SLICE]
        embs = embedder.encode([c.text for c in batch])
        store.add(batch, embs)
        print(f"  embedded {min(i+SLICE, len(unique)):,}/{len(unique):,} "
              f"({time.time()-t0:.0f}s)")

    print("Build BM25...")
    bm = BM25Index(BM25_PATH)
    bm.build(unique)
    bm.save()
    print(f"  BM25: {bm.count():,} chunks → {BM25_PATH.name}")
    return store, bm


def compare(store: VectorStore, bm: BM25Index) -> None:
    kg = KGRetriever()
    rag = Retriever(embedder=Embedder(config.EMBEDDING_MODEL), store=store, bm25=bm, kg_retriever=None)
    gr = Retriever(embedder=rag.embedder, store=store, bm25=bm, kg_retriever=kg)

    print(f"\nStore: {store.count():,} chunks | BM25: {bm.count():,} | KG: ready\n")

    rows = []
    for q in TEST_QUERIES:
        rag_c = rag.retrieve(q["query"], top_k=TOP_K, use_kg=False)
        gr_c = gr.retrieve(q["query"], top_k=TOP_K, use_kg=True)
        kg_hits = kg.retrieve(q["query"], top_k=10)
        kg_mapped = 0
        for h in kg_hits:
            if h.source_url and store.get_by_filter(
                {"$and": [{"source": {"$eq": h.source_url}}, {"article": {"$eq": h.article_label}}]}, limit=1):
                kg_mapped += 1

        rag_ids = {c.chunk.chunk_id for c in rag_c}
        gr_ids = {c.chunk.chunk_id for c in gr_c}
        shared = len(rag_ids & gr_ids)
        only_gr = gr_ids - rag_ids
        rows.append((q["id"], q["category"], shared, len(only_gr), len(kg_hits), kg_mapped))

        print("=" * 70)
        print(f"  {q['id']}: {q['category']}  |  {q['query']}")
        print(f"  chung={shared}/{TOP_K}  only_GR={len(only_gr)}  "
              f"KG_hits={len(kg_hits)}  KG_map_ra_chunk={kg_mapped}")
        if only_gr:
            print("  🟣 Chunk KG thêm vào (RAG không có):")
            for c in gr_c:
                if c.chunk.chunk_id in only_gr:
                    print(f"     {fmt_chunk(c.chunk)}  (score {c.score:.3f})")

    print("\n" + "=" * 70)
    print(" 📊 SUMMARY  (sau khi fix article)")
    print("=" * 70)
    print(f"  {'Q':<5}{'Category':<26}{'chung':>6}{'only_GR':>8}{'KG_hit':>7}{'KG_map':>7}")
    for r in rows:
        print(f"  {r[0]:<5}{r[1][:26]:<26}{r[2]:>6}{r[3]:>8}{r[4]:>7}{r[5]:>7}")
    n_gr = sum(1 for r in rows if r[3] > 0)
    print(f"\n  Câu Graph-RAG có chunk riêng (only_GR>0): {n_gr}/{len(rows)}")
    print(f"  (Trước khi fix: 0/10 — KG map_ra_chunk=0 mọi câu)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Re-ingest 15 luật (có article) + so sánh RAG vs Graph-RAG")
    p.add_argument("--compare-only", action="store_true",
                   help="Bỏ qua re-ingest, dùng collection legal_top15 + BM25 đã có (chạy local sau khi tải về)")
    args = p.parse_args()

    if args.compare_only:
        store = VectorStore(config.VECTORSTORE_DIR, COLL)
        bm = BM25Index(BM25_PATH)
        if store.count() == 0:
            sys.exit(f"Collection '{COLL}' rỗng — chưa có dữ liệu để so sánh. Bỏ --compare-only để build trước.")
        print(f"[compare-only] dùng collection '{COLL}' sẵn có.")
        compare(store, bm)
    else:
        store, bm = build_store()
        embedder = None  # noqa
        compare(store, bm)

"""So sánh RAG-only vs Graph-RAG trên 1 bộ test queries.

Đo:
  - Latency
  - Top-K chunks retrieved
  - Overlap % giữa 2 mode
  - KG branch contribution (chunks unique cho Graph-RAG)
  - Quality (heuristic: chunks có liên quan trực tiếp tới query keywords)

Chạy:
    python -m scripts.compare_rag_vs_graphrag
"""
from __future__ import annotations

import json
import sys
import time
from typing import Optional

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.bm25_index import BM25Index
from src.embedding import Embedder
from src.kg.kg_retriever import KGRetriever
from src.retriever import Retriever
from src.vectorstore import VectorStore


# Test queries — đa dạng để thấy khi nào KG giúp, khi nào không
TEST_QUERIES = [
    {
        "id": "Q1", "category": "🔴 Hình sự",
        "query": "Trộm cắp tài sản dưới 2 triệu đồng bị phạt như thế nào?",
        "expect_kg_help": True,
    },
    {
        "id": "Q2", "category": "🚦 Giao thông",
        "query": "Vượt đèn đỏ với xe máy bị phạt bao nhiêu tiền?",
        "expect_kg_help": True,
    },
    {
        "id": "Q3", "category": "💼 Lao động",
        "query": "Người lao động bị sa thải trái pháp luật có được bồi thường?",
        "expect_kg_help": True,
    },
    {
        "id": "Q4", "category": "🔬 Reasoning",
        "query": "Tội nào có khung phạt tù từ 7 năm trở lên?",
        "expect_kg_help": True,
    },
    {
        "id": "Q5", "category": "👤 Subject-based",
        "query": "Pháp nhân thương mại phạm tội bị xử lý như thế nào?",
        "expect_kg_help": True,
    },
    {
        "id": "Q6", "category": "❓ Định nghĩa (RAG đủ)",
        "query": "Quan hệ dân sự là gì?",
        "expect_kg_help": False,
    },
    {
        "id": "Q7", "category": "💰 Thuế",
        "query": "Mức thuế thu nhập cá nhân hiện nay là bao nhiêu?",
        "expect_kg_help": False,
    },
    {
        "id": "Q8", "category": "🏠 Đất đai",
        "query": "Thủ tục cấp giấy chứng nhận quyền sử dụng đất lần đầu?",
        "expect_kg_help": False,
    },
    {
        "id": "Q9", "category": "💊 Ma túy",
        "query": "Sử dụng trái phép chất ma túy bị xử lý ra sao?",
        "expect_kg_help": True,
    },
    {
        "id": "Q10", "category": "🛍️ Bảo vệ NTD",
        "query": "Người tiêu dùng có quyền khiếu nại sản phẩm kém chất lượng không?",
        "expect_kg_help": False,
    },
]


def fmt_chunk(c) -> str:
    src = c.metadata.source.split("/")[-1][:30]
    art = c.article or "-"
    return f"{src}|{art}"


def run_compare():
    print("Đang init...")
    embedder = Embedder(config.EMBEDDING_MODEL)
    vstore = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)
    bm25_path = config.DATA_DIR / "bm25" / "index.json"
    bm25 = BM25Index(bm25_path) if bm25_path.exists() else None
    kg_retriever = KGRetriever()

    # 2 retrievers — same vector+bm25, khác KG
    retr_rag = Retriever(embedder, vstore, bm25=bm25, kg_retriever=None)
    retr_graphrag = Retriever(embedder, vstore, bm25=bm25, kg_retriever=kg_retriever)

    # Load top 15 URLs for filter
    manifest = json.loads(
        (config.DATA_DIR / "comparison" / "top15_laws" / "manifest.json").read_text(encoding="utf-8")
    )
    top15_urls = [l["source_url"] for l in manifest["laws"]]

    print(f"  Vector: {vstore.count():,} chunks")
    print(f"  BM25:   {bm25.count():,} chunks")
    print(f"  KG:     ready")
    print(f"  Top 15 dataset: {len(top15_urls)} laws")
    print()

    # ── Run queries ────────────────────────────────────────────
    results = []
    for q in TEST_QUERIES:
        print("=" * 75)
        print(f"  {q['id']}: {q['category']}")
        print(f"  Q: {q['query']}")
        print(f"  Expected KG helpful: {q['expect_kg_help']}")
        print("=" * 75)

        TOP_K = 5

        # RAG-only (top 15 filter, no KG)
        t0 = time.time()
        rag_chunks = retr_rag.retrieve(
            q["query"], top_k=TOP_K, use_kg=False, allowed_sources=top15_urls,
        )
        rag_latency = (time.time() - t0) * 1000

        # Graph-RAG (top 15 + KG)
        t0 = time.time()
        gr_chunks = retr_graphrag.retrieve(
            q["query"], top_k=TOP_K, use_kg=True, allowed_sources=top15_urls,
        )
        gr_latency = (time.time() - t0) * 1000

        # KG branch standalone
        kg_hits = kg_retriever.retrieve(q["query"], top_k=10)

        # Overlap analysis
        rag_ids = {c.chunk.chunk_id for c in rag_chunks}
        gr_ids = {c.chunk.chunk_id for c in gr_chunks}
        shared = rag_ids & gr_ids
        only_rag = rag_ids - gr_ids
        only_gr = gr_ids - rag_ids

        print(f"\n  ⏱  RAG latency:        {rag_latency:>7.0f} ms")
        print(f"  ⏱  Graph-RAG latency:  {gr_latency:>7.0f} ms  (+{gr_latency-rag_latency:+.0f})")
        print(f"  🟢 Chunks chung:       {len(shared)}/{TOP_K}")
        print(f"  🔵 Chỉ RAG có:         {len(only_rag)}")
        print(f"  🟣 Chỉ Graph-RAG có:   {len(only_gr)}  ← bonus từ KG")
        print(f"  🕸 KG branch hits:     {len(kg_hits)}")

        if kg_hits:
            print(f"\n  KG path breakdown:")
            from collections import Counter
            paths = Counter(h.reason for h in kg_hits)
            for path, n in paths.items():
                print(f"    {path:<25s} {n}")

        # Show top results side-by-side
        print("\n  TOP CHUNKS:")
        print(f"  {'RAG':<55s} | {'Graph-RAG':<55s}")
        for i in range(TOP_K):
            r = fmt_chunk(rag_chunks[i].chunk) if i < len(rag_chunks) else "-"
            g = fmt_chunk(gr_chunks[i].chunk) if i < len(gr_chunks) else "-"
            r_score = f"{rag_chunks[i].score:.3f}" if i < len(rag_chunks) else "-"
            g_score = f"{gr_chunks[i].score:.3f}" if i < len(gr_chunks) else "-"
            r_marker = "🟢" if i < len(rag_chunks) and rag_chunks[i].chunk.chunk_id in shared else "🔵"
            g_marker = "🟢" if i < len(gr_chunks) and gr_chunks[i].chunk.chunk_id in shared else "🟣"
            print(f"  {r_marker} {r:<43s} ({r_score}) | {g_marker} {g:<43s} ({g_score})")

        results.append({
            "id": q["id"], "category": q["category"], "query": q["query"],
            "rag_latency_ms": rag_latency,
            "graphrag_latency_ms": gr_latency,
            "shared": len(shared), "only_rag": len(only_rag), "only_gr": len(only_gr),
            "kg_hits": len(kg_hits),
            "expected_kg_help": q["expect_kg_help"],
        })
        print()

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(" 📊 SUMMARY")
    print("=" * 75)
    avg_rag = sum(r["rag_latency_ms"] for r in results) / len(results)
    avg_gr = sum(r["graphrag_latency_ms"] for r in results) / len(results)
    print(f"\n  Avg RAG latency:        {avg_rag:.0f} ms")
    print(f"  Avg Graph-RAG latency:  {avg_gr:.0f} ms  (overhead +{avg_gr-avg_rag:.0f} ms)")

    helpful = [r for r in results if r["expected_kg_help"] and r["kg_hits"] > 0]
    print(f"\n  Queries với KG hits > 0: {sum(1 for r in results if r['kg_hits'] > 0)}/10")
    print(f"  Queries Graph-RAG có chunks unique: {sum(1 for r in results if r['only_gr'] > 0)}/10")

    print(f"\n  Per-category:")
    print(f"  {'Query':<8} {'Category':<28} {'RAG ms':>7} {'GR ms':>7} {'Shared':>6} {'KG hit':>6} {'GR bonus':>8}")
    for r in results:
        print(f"  {r['id']:<8} {r['category'][:28]:<28} {r['rag_latency_ms']:>7.0f} {r['graphrag_latency_ms']:>7.0f} {r['shared']:>6} {r['kg_hits']:>6} {r['only_gr']:>8}")


if __name__ == "__main__":
    run_compare()

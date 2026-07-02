"""Soi full top-k của 1 câu hỏi: RRF thô vs sau rerank. Để chẩn đoán ranking.

python -m scripts.inspect_query "tội trộm cắp tài sản bị xử lý hình sự thế nào?"
"""
from __future__ import annotations

import sys

from scripts.eval_retrieval import build_retriever
from src.reranker import rerank as _rerank

q = sys.argv[1] if len(sys.argv) > 1 else "tội trộm cắp tài sản bị xử lý hình sự như thế nào?"
K = 10
retriever, _ = build_retriever(use_kg=False)

raw = retriever.retrieve(q, top_k=K, use_hyde=False, use_parent_expansion=True)
print(f"\nCâu: {q}\n")
print("── RRF thô (trước rerank) ──")
for i, rc in enumerate(raw, 1):
    m = rc.chunk.metadata
    print(f"  {i:>2}. {rc.score:.4f} [{m.doc_number}] {m.doc_type} | art={rc.chunk.article} "
          f"| folder={m.folder} | {(m.title or '')[:45]}")

reranked = _rerank(q, raw, top_k=K, use_cross_encoder=True)
print("\n── Sau rerank (CrossEncoder ON) ──")
for i, rc in enumerate(reranked, 1):
    m = rc.chunk.metadata
    print(f"  {i:>2}. {rc.score:.4f} [{m.doc_number}] {m.doc_type} | art={rc.chunk.article} "
          f"| folder={m.folder} | {(m.title or '')[:45]}")

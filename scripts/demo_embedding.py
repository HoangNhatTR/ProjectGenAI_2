"""Chạy thử embedding + retrieval đơn giản (chưa cần vectorstore).

Lần đầu chạy sẽ tải model AITeamVN/Vietnamese_Embedding (~2.3GB) về cache HuggingFace.
Cách chạy:
    python -m scripts.demo_embedding
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src import config
from src.chunking import chunk_document
from src.embedding import Embedder
from src.parsing import load_document
from src.schemas import DocumentMetadata


QUERIES = [
    "Vượt đèn đỏ ô tô bị phạt bao nhiêu tiền?",
    "Người uống rượu lái xe bị phạt thế nào?",
    "Đi bộ qua đường cao tốc có bị phạt không?",
    "Trừ bao nhiêu điểm khi vượt quá tốc độ 15 km/h?",
]


def topk(query_emb, chunk_embs, chunks, k: int = 3):
    sims = np.array(chunk_embs) @ np.array(query_emb)
    idx = np.argsort(-sims)[:k]
    return [(chunks[i], float(sims[i])) for i in idx]


def main() -> None:
    print(f"Loading model: {config.EMBEDDING_MODEL}")
    print("(Lần đầu sẽ tải ~2.3GB từ HuggingFace, kiên nhẫn chờ...)")
    t0 = time.time()
    embedder = Embedder(config.EMBEDDING_MODEL)
    _ = embedder.dim  # trigger load
    print(f"  Loaded in {time.time() - t0:.1f}s — embedding dim = {embedder.dim}")
    print()

    # Load + chunk file luật giao thông
    path = Path("data/raw/luat_giao_thong_2025.txt")
    doc = load_document(path, DocumentMetadata(source=path.name))
    chunks = chunk_document(doc)
    print(f"File: {path.name} → {len(chunks)} chunks")

    t0 = time.time()
    chunk_embs = embedder.encode_chunks(chunks)
    elapsed = time.time() - t0
    print(f"Encode {len(chunks)} chunks: {elapsed:.2f}s ({elapsed / len(chunks) * 1000:.0f}ms/chunk)")
    print()

    # Sanity check: similarity 2 câu giống & 2 câu khác
    print("=== Sanity check ===")
    a = embedder.encode(["Tốc độ tối đa trong khu dân cư là bao nhiêu?"])[0]
    b = embedder.encode(["Quy định tốc độ ở khu đông dân cư"])[0]
    c = embedder.encode(["Cách nấu phở bò ngon"])[0]
    sim_ab = float(np.dot(a, b))
    sim_ac = float(np.dot(a, c))
    print(f"  sim(câu giống nghĩa)   = {sim_ab:.4f}")
    print(f"  sim(câu khác hẳn chủ đề) = {sim_ac:.4f}")
    assert sim_ab > sim_ac, "Embedding bất thường!"
    print()

    # Retrieval demo
    print("=== Retrieval (top-3 mỗi query) ===")
    for q in QUERIES:
        q_emb = embedder.encode([q])[0]
        results = topk(q_emb, chunk_embs, chunks, k=3)
        print(f"\n[Q] {q}")
        for rank, (ch, score) in enumerate(results, 1):
            tag = " | ".join(filter(None, [ch.article, ch.clause])) or "(preamble)"
            preview = ch.text.replace("\n", " ")[:130]
            print(f"  #{rank}  score={score:.3f}  [{tag}]  {preview}...")


if __name__ == "__main__":
    main()

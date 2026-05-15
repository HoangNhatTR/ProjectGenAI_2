"""Build full pipeline: parse → chunk → embed → store vào Chroma → query.

Lần đầu chạy sẽ index toàn bộ data/raw/. Lần sau chạy lại sẽ upsert
(không nhân đôi vì chunk_id deterministic).

    python -m scripts.demo_vectorstore
"""
from __future__ import annotations

import time
from pathlib import Path

from src import config
from src.chunking import chunk_document
from src.embedding import Embedder
from src.parsing import load_document
from src.schemas import DocumentMetadata
from src.vectorstore import VectorStore


QUERIES = [
    "Vượt đèn đỏ ô tô bị phạt bao nhiêu tiền?",
    "Tội cướp tài sản bị xử thế nào?",
    "Trừ điểm khi vượt tốc độ 15 km/h?",
    "Người đi bộ vào đường cao tốc bị phạt bao nhiêu?",
]


def ingest(store: VectorStore, embedder: Embedder) -> None:
    raw_dir = config.RAW_DIR
    files = [p for p in sorted(raw_dir.iterdir()) if p.is_file() and p.suffix.lower() == ".txt"]
    print(f"Tìm thấy {len(files)} file trong {raw_dir}")

    for path in files:
        meta = DocumentMetadata(source=path.name)
        doc = load_document(path, meta)
        chunks = chunk_document(doc)
        t0 = time.time()
        embs = embedder.encode_chunks(chunks)
        store.add(chunks, embs)
        print(f"  ✓ {path.name:<35} → {len(chunks):>3} chunks  ({time.time()-t0:.1f}s encode)")


def search(store: VectorStore, embedder: Embedder, query: str, top_k: int = 3) -> None:
    q_emb = embedder.encode([query])[0]
    results = store.query(q_emb, top_k=top_k)
    print(f"\n[Q] {query}")
    for rank, r in enumerate(results, 1):
        tag = " | ".join(filter(None, [r.chunk.article, r.chunk.clause])) or "(preamble)"
        preview = r.chunk.text.replace("\n", " ")[:130]
        print(f"  #{rank}  score={r.score:.3f}  [{r.chunk.metadata.source} → {tag}]")
        print(f"        {preview}...")


def main() -> None:
    print(f"Vectorstore dir : {config.VECTORSTORE_DIR}")
    print(f"Collection name : {config.COLLECTION_NAME}")
    print(f"Embedding model : {config.EMBEDDING_MODEL}")
    print()

    embedder = Embedder(config.EMBEDDING_MODEL)
    store = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)

    print(f"Số chunk hiện có trong store: {store.count()}")
    print()

    print("=== INGEST ===")
    ingest(store, embedder)
    print(f"\nSố chunk sau ingest: {store.count()}")

    print("\n=== QUERY ===")
    for q in QUERIES:
        search(store, embedder, q, top_k=3)


if __name__ == "__main__":
    main()

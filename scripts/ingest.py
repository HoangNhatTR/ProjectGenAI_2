"""Pipeline ingestion: data/raw/ -> parse -> clean -> chunk -> embed -> vectorstore.

Cách chạy:
    python -m scripts.ingest
"""
from __future__ import annotations

from pathlib import Path

from src import config
from src.parsing import load_document
from src.chunking import chunk_document
from src.embedding import Embedder
from src.vectorstore import VectorStore
from src.schemas import DocumentMetadata


def iter_raw_files(raw_dir: Path):
    """Duyệt data/raw/ và yield (path, metadata) cho từng file."""
    raise NotImplementedError


def main() -> None:
    embedder = Embedder(config.EMBEDDING_MODEL)
    store = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)

    for path, meta in iter_raw_files(config.RAW_DIR):
        doc = load_document(path, meta)
        chunks = chunk_document(doc)
        embeddings = embedder.encode_chunks(chunks)
        store.add(chunks, embeddings)

    print(f"Done. Total chunks in store: {store.count()}")


if __name__ == "__main__":
    main()

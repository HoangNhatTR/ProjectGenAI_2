"""(Re)build BM25 index từ chunks đã có trong Chroma.

Chạy SAU khi đã ingest xong vào vectorstore:
    python -m scripts.build_bm25

Tạo file ở `data/bm25/index.json`. Khi app.py khởi động sẽ tự load nếu tìm thấy.
"""
from __future__ import annotations

import sys

from src import config
from src.bm25_index import BM25Index
from src.vectorstore import VectorStore


def main() -> None:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    store = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)
    n = store.count()
    print(f"Vectorstore có {n} chunks.")
    if n == 0:
        print("⚠ Vectorstore trống. Chạy ingest (vd: python -m scripts.demo_vectorstore) trước.")
        sys.exit(1)

    print("Đang đọc chunks từ Chroma...")
    chunks = list(store.iter_all_chunks())
    print(f"✓ Đã đọc {len(chunks)} chunks.")

    index = BM25Index(config.DATA_DIR / "bm25" / "index.json")
    index.build(chunks)
    index.save()
    print(f"✓ Đã build & lưu BM25 index: {index.path}")
    print(f"  ({index.count()} chunks indexed)")


if __name__ == "__main__":
    main()

"""Chạy thử chunking trên các file trong data/raw/.

Cách chạy:
    python -m scripts.demo_chunking
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from src import config
from src.chunking import chunk_document
from src.parsing import load_document
from src.schemas import DocumentMetadata


def show_stats(chunks):
    sizes = [len(c.text) for c in chunks]
    print(f"Tổng số chunk : {len(chunks)}")
    print(f"Kích thước    : min={min(sizes)}  max={max(sizes)}  avg={sum(sizes)//len(sizes)}")
    n_with_art = sum(1 for c in chunks if c.article)
    n_with_cls = sum(1 for c in chunks if c.clause)
    print(f"Có 'article'  : {n_with_art}/{len(chunks)}")
    print(f"Có 'clause'   : {n_with_cls}/{len(chunks)}")
    article_counts = Counter(c.article for c in chunks if c.article)
    print(f"Số Điều unique: {len(article_counts)}")
    multi = [(a, n) for a, n in article_counts.items() if n > 1]
    if multi:
        print(f"Điều bị chia nhiều chunk (do dài): {multi}")


def show_samples(chunks, k: int = 3):
    print(f"\n--- {min(k, len(chunks))} chunk đầu ---")
    for c in chunks[:k]:
        meta = []
        if c.article:
            meta.append(c.article)
        if c.clause:
            meta.append(c.clause)
        tag = " | ".join(meta) if meta else "(preamble)"
        print(f"\n[{c.chunk_id}]  {tag}  ({len(c.text)} chars)")
        preview = c.text[:300] + ("..." if len(c.text) > 300 else "")
        print(preview)

    print(f"\n--- chunk cuối ---")
    last = chunks[-1]
    meta = " | ".join(filter(None, [last.article, last.clause])) or "(preamble)"
    print(f"\n[{last.chunk_id}]  {meta}  ({len(last.text)} chars)")
    print(last.text[:400] + ("..." if len(last.text) > 400 else ""))


def main() -> None:
    raw_dir = config.RAW_DIR
    files = [p for p in sorted(raw_dir.iterdir()) if p.is_file() and p.suffix.lower() in {".txt", ".html", ".pdf"}]
    if not files:
        print(f"[!] Không có file hợp lệ trong {raw_dir}")
        return

    for path in files:
        print("=" * 70)
        print(f"FILE: {path.name}")
        print("=" * 70)

        meta = DocumentMetadata(source=path.name)
        doc = load_document(path, meta)
        chunks = chunk_document(doc)

        print(f"Cấu hình     : CHUNK_SIZE={config.CHUNK_SIZE}  CHUNK_OVERLAP={config.CHUNK_OVERLAP}")
        print(f"Văn bản gốc  : {len(doc.text)} chars")
        show_stats(chunks)
        show_samples(chunks, k=3)
        print()


if __name__ == "__main__":
    main()

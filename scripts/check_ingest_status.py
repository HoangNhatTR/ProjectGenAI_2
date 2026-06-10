"""Kiểm tra ingest status: file nào đã embed, file nào còn thiếu.

Chạy:
    python -m scripts.check_ingest_status
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.vectorstore import VectorStore


def main() -> None:
    vs = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)
    all_chunks = list(vs.iter_all_chunks())
    embedded_sources = set(c.metadata.source for c in all_chunks)

    raw_files = list(config.RAW_DIR.rglob("*.txt"))
    print(f"Raw .txt files:        {len(raw_files)}")
    print(f"Unique docs embedded:  {len(embedded_sources)}")
    print(f"Total chunks in store: {len(all_chunks)}")

    print("\nMẫu source trong store:")
    for s in list(embedded_sources)[:3]:
        print(f"  - {s[:100]}")

    # Source trong store là URL vbpl.vn. Đọc URL từ header mỗi file raw để compare.
    def url_from_header(path: Path) -> str | None:
        try:
            with path.open(encoding="utf-8", errors="replace") as f:
                for _ in range(20):
                    line = f.readline()
                    if not line:
                        break
                    if line.startswith("URL: "):
                        return line[5:].strip()
        except Exception:
            return None
        return None

    missing = []
    for f in raw_files:
        url = url_from_header(f)
        # Khớp theo URL hoặc theo path str (fallback cho file không có header)
        if url and url in embedded_sources:
            continue
        if str(f) in embedded_sources:
            continue
        missing.append(f)

    print(f"\nFile có thể CHƯA embed: {len(missing)}")

    # Phân bố theo topic
    topic_total = Counter(f.parent.name for f in raw_files)
    topic_missing = Counter(f.parent.name for f in missing)
    print("\nPhân bố theo topic:")
    for topic, n_total in topic_total.most_common():
        n_done = n_total - topic_missing[topic]
        pct = n_done * 100 // n_total
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"  {topic:20s} [{bar}] {n_done:4d}/{n_total:4d}  ({pct}%)")

    if missing:
        print(f"\n10 file đầu bị thiếu (tổng {len(missing)}):")
        for f in missing[:10]:
            print(f"  - {f.parent.name}/{f.name}")


if __name__ == "__main__":
    main()

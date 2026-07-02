"""BƯỚC 1 (LOCAL): fetch + chunk các VB cần re-ingest → JSONL để embed trên Colab.

KHÔNG embed ở đây (đó là việc của Colab GPU). Chỉ lấy toàn văn vbpl + chunk theo
Điều (chunker đã fix) → xuất:
  data/colab_export/to_embed.jsonl  : 1 chunk / dòng (chunk_id, text, metadata)
  data/colab_export/parents.jsonl   : (parent_id, text) cho parent_store
  data/colab_export/manifest.json   : danh sách doc_number + thống kê

    python -m scripts.export_to_embed                 # scope 2019+ (149), bỏ VB đã xong
    python -m scripts.export_to_embed --min-year 2012 # cả 2012-2018 (toàn bộ 503)
    python -m scripts.export_to_embed --limit 3       # test nhỏ

Sau đó: upload to_embed.jsonl lên Colab, chạy COLAB_EMBED.md → embedded.parquet,
tải về rồi: python -m scripts.merge_embedded
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from opensearchpy import OpenSearch

from src import config
from src.chunking import chunk_document
from src.schemas import DocumentMetadata, RawDocument
from src.vbpl_client import VBPLClient
# tái dùng helper từ batch_reingest để nhất quán
from scripts.batch_reingest import (
    KEEP_OLD, SKIP, build_meta, first_chunk_source, resolve_vbpl_id, scan_flagged,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "colab_export"
PROGRESS = ROOT / "data" / "batch_reingest_progress.json"

_META_COLS = ("source", "doc_type", "doc_number", "title", "issued_date",
              "effective_date", "status", "linh_vuc", "co_quan", "folder")


class _ParentCollector:
    """Bắt parent (parent_id, text) thay vì ghi SQLite — để xuất ra JSONL."""
    def __init__(self):
        self.items: list[tuple[str, str]] = []

    def add_batch(self, items):
        self.items.extend(items)


def _chunk_row(c) -> dict:
    m = c.metadata
    row = {"chunk_id": c.chunk_id, "text": c.text,
           "article": c.article, "clause": c.clause, "point": c.point,
           "parent_id": c.parent_id}
    for f in _META_COLS:
        row[f] = getattr(m, f)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-year", type=int, default=2019)
    ap.add_argument("--max-year", type=int, default=2026)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if PROGRESS.exists():
        done = set(json.loads(PROGRESS.read_text(encoding="utf-8")).get("done", {}))

    print("Quét danh sách...", flush=True)
    flagged = [dn for dn in scan_flagged() if dn not in SKIP and dn not in done]
    todo = []
    for dn in flagged:
        s = first_chunk_source(dn)
        if not s:
            continue
        yr = (s.get("issued_date") or "")[:4]
        y = int(yr) if yr.isdigit() else 0
        if (args.min_year <= y <= args.max_year) or dn in KEEP_OLD:
            todo.append((dn, s))
    todo.sort(key=lambda x: -(int((x[1].get("issued_date") or "0")[:4]) if (x[1].get("issued_date") or "")[:4].isdigit() else 0))
    if args.limit:
        todo = todo[:args.limit]
    print(f"Cần export: {len(todo)} VB (bỏ qua {len(done)} đã xong)\n", flush=True)

    client = VBPLClient()
    f_chunks = (OUT / "to_embed.jsonl").open("w", encoding="utf-8")
    parents: list[tuple[str, str]] = []
    manifest = {"doc_numbers": [], "skipped": {}, "n_chunks": 0}

    for i, (dn, s) in enumerate(todo, 1):
        try:
            vid = resolve_vbpl_id(client, s["source"], dn)
            if not vid:
                manifest["skipped"][dn] = "no_id"; continue
            full = client.fetch_with_content(vid)
            if not full or len((full.get("content") or "")) < 500:
                manifest["skipped"][dn] = "no_content"; continue
            if dn.split("/")[0] not in (full.get("so_hieu") or ""):
                manifest["skipped"][dn] = f"mismatch:{full.get('so_hieu')}"; continue

            doc = RawDocument(text=full["content"], metadata=build_meta(s))
            coll = _ParentCollector()
            chunks = chunk_document(doc, parent_store=coll)
            arts = len({c.article for c in chunks if c.article})
            for c in chunks:
                f_chunks.write(json.dumps(_chunk_row(c), ensure_ascii=False) + "\n")
            parents.extend(coll.items)
            manifest["doc_numbers"].append(dn)
            manifest["n_chunks"] += len(chunks)
            print(f"[{i}/{len(todo)}] {dn:<18} {len(chunks):>4} chunk / {arts:>3} Điều | {full['title'][:36]}", flush=True)
        except Exception as exc:
            manifest["skipped"][dn] = f"error:{exc}"
            print(f"[{i}/{len(todo)}] {dn:<18} ERROR {exc}", flush=True)

    f_chunks.close()
    with (OUT / "parents.jsonl").open("w", encoding="utf-8") as fp:
        for pid, txt in parents:
            fp.write(json.dumps({"parent_id": pid, "text": txt}, ensure_ascii=False) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== XUẤT XONG ===")
    print(f"  VB: {len(manifest['doc_numbers'])} | chunk: {manifest['n_chunks']:,} | parent: {len(parents):,} | skip: {len(manifest['skipped'])}")
    print(f"  → {OUT/'to_embed.jsonl'}")
    print(f"  → {OUT/'parents.jsonl'}")
    print(f"  → {OUT/'manifest.json'}")
    print("\nBước tiếp: upload to_embed.jsonl lên Colab, chạy COLAB_EMBED.md → embedded.parquet")


if __name__ == "__main__":
    main()

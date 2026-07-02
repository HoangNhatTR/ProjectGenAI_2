"""BƯỚC 3 (LOCAL): ghép embedded.parquet (từ Colab) vào OpenSearch + parent_store.

Đọc data/colab_export/embedded.parquet (chunk + vector BGE-M3) + parents.jsonl +
manifest.json → XOÁ chunk cũ (unstructured) của từng doc_number → bulk add chunk
mới (có vector) → ghi parent_store.db. Stream theo lô (nhẹ RAM).

    python -m scripts.merge_embedded
    python -m scripts.merge_embedded --dry-run   # chỉ kiểm tra file, không ghi

⚠ GHI INDEX — cần duyệt. Vector phải là BGE-M3 normalize (cùng model) mới ghép đúng.
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

import pyarrow.parquet as pq
from opensearchpy import OpenSearch, helpers

from src import config
from src.parent_store import ParentStore

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "data" / "colab_export"
PARQUET = EXP / "embedded.parquet"

_OPT = ("article", "clause", "point", "parent_id", "doc_type", "doc_number",
        "title", "issued_date", "effective_date", "status", "folder",
        "co_quan", "linh_vuc")


def row_to_source(row: dict) -> dict:
    doc = {"text": row["text"], "source": row["source"], "embedding": row["embedding"]}
    for f in _OPT:
        v = row.get(f)
        if v:
            doc[f] = v
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=2000)
    args = ap.parse_args()

    if not PARQUET.exists():
        sys.exit(f"[FAIL] Không thấy {PARQUET}. Hãy chạy Colab embed + tải về trước.")
    manifest = json.loads((EXP / "manifest.json").read_text(encoding="utf-8"))
    doc_numbers = manifest["doc_numbers"]

    pf = pq.ParquetFile(PARQUET)
    n_rows = pf.metadata.num_rows
    cols = pf.schema_arrow.names
    print(f"Parquet: {n_rows:,} chunk | cột: {cols}")
    assert "embedding" in cols, "Thiếu cột 'embedding' — parquet chưa được Colab embed?"
    # kiểm tra chiều vector
    sample = next(pf.iter_batches(batch_size=1)).to_pylist()[0]
    dim = len(sample["embedding"])
    print(f"Chiều vector: {dim} (BGE-M3 phải = 1024)")
    assert dim == 1024, f"Vector {dim} chiều ≠ 1024 — sai model!"
    print(f"Doc cần merge: {len(doc_numbers)} | parents: {(EXP/'parents.jsonl').exists()}")

    if args.dry_run:
        print("\n[DRY-RUN] File hợp lệ. Bỏ --dry-run để ghi thật.")
        return

    cli = OpenSearch(hosts=[config.OPENSEARCH_URL], timeout=300)
    IDX = config.OPENSEARCH_INDEX

    # 1) Xoá chunk cũ (unstructured) của các doc sẽ merge
    print("Xoá chunk cũ...", flush=True)
    for i in range(0, len(doc_numbers), 100):
        batch = doc_numbers[i:i+100]
        r = cli.delete_by_query(index=IDX, refresh=False, conflicts="proceed",
                                body={"query": {"terms": {"doc_number": batch}}})
        print(f"  ...xoá {r.get('deleted')} (lô {i//100+1})", flush=True)

    # 2) Bulk add chunk mới (stream theo lô)
    print("Bulk add chunk mới...", flush=True)
    added = 0
    for rb in pf.iter_batches(batch_size=args.batch):
        rows = rb.to_pylist()
        actions = ({"_op_type": "index", "_index": IDX, "_id": r["chunk_id"],
                    "_source": row_to_source(r)} for r in rows)
        helpers.bulk(cli, actions, chunk_size=500, request_timeout=180)
        added += len(rows)
        print(f"  ...added {added:,}/{n_rows:,}", flush=True)

    # 3) Parent store
    pj = EXP / "parents.jsonl"
    if pj.exists():
        ps = ParentStore(config.PARENT_STORE_PATH)
        items = []
        with pj.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                items.append((d["parent_id"], d["text"]))
        ps.add_batch(items)
        print(f"Ghi {len(items):,} parent vào parent_store.db")

    cli.indices.refresh(index=IDX)
    # 4) Verify
    print("\n=== VERIFY (vài doc) ===")
    for dn in doc_numbers[:5]:
        c = cli.count(index=IDX, body={"query": {"term": {"doc_number": dn}}})["count"]
        a = cli.search(index=IDX, body={"size": 0, "query": {"term": {"doc_number": dn}},
                                        "aggs": {"a": {"cardinality": {"field": "article"}}}})
        print(f"  {dn:<18} {c} chunk / {a['aggregations']['a']['value']} Điều")
    print(f"\nXong. Merge {len(doc_numbers)} VB / {added:,} chunk. → chạy audit_coverage để xác nhận.")


if __name__ == "__main__":
    main()

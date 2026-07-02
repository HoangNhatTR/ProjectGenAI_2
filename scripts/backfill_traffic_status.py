"""Backfill hiệu lực THẬT (từ vbpl) cho VB xử phạt giao thông cũ đã bị thay thế.

KHÔNG bịa: chỉ set đúng status vbpl xác nhận (xem fetch_traffic_status.py).
reranker._temporal_factor sẽ tự demote (toàn bộ→0.50, một phần→0.90).
Reversible: chạy với --revert để xoá status (về (none)).

    python -m scripts.backfill_traffic_status            # áp dụng
    python -m scripts.backfill_traffic_status --revert   # hoàn tác
"""
from __future__ import annotations

import argparse
import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from opensearchpy import OpenSearch
from src import config

# Status THẬT từ vbpl (fetch_traffic_status.py 2026-06-15). Chỉ các VB xử phạt
# giao thông đường bộ đã bị NĐ 168/2024 thay thế — KHÔNG đụng VBHN (số hiệu trùng,
# không xác minh được) và KHÔNG đụng văn bản còn hiệu lực.
TRUE_STATUS = {
    "152/2005/NĐ-CP": "Hết hiệu lực toàn bộ",
    "34/2010/NĐ-CP":  "Hết hiệu lực toàn bộ",
    "100/2019/NĐ-CP": "Hết hiệu lực một phần (lĩnh vực đường bộ — thay bởi NĐ 168/2024/NĐ-CP)",
    "123/2021/NĐ-CP": "Hết hiệu lực một phần (lĩnh vực đường bộ — thay bởi NĐ 168/2024/NĐ-CP)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    cli = OpenSearch(hosts=[config.OPENSEARCH_URL], timeout=120)
    IDX = config.OPENSEARCH_INDEX

    for dn, status in TRUE_STATUS.items():
        if args.revert:
            script = {"source": "ctx._source.remove('status')"}
            label = "xoá status"
        else:
            script = {"source": "ctx._source.status = params.s", "params": {"s": status}}
            label = f"set '{status[:40]}'"
        r = cli.update_by_query(
            index=IDX, refresh=True, wait_for_completion=True, conflicts="proceed",
            body={"query": {"term": {"doc_number": dn}}, "script": script},
        )
        print(f"  {dn:<16} {label} → updated {r.get('updated')} chunk")

    cli.indices.refresh(index=IDX)
    print("Xong." if not args.revert else "Đã revert.")


if __name__ == "__main__":
    main()

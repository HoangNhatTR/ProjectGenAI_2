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
    # Đợt 2 (2026-07-08, fetch_traffic_status xác nhận từng VB trên vbpl):
    # cả chuỗi NĐ phạt giao thông 1995-2016 + đường thủy cũ — đo thấy chiếm
    # 17/20 slot RRF thô cho query đèn tín hiệu, đè NĐ 168 hiện hành.
    "49-CP":          "Hết hiệu lực toàn bộ",
    "78/1998/NĐ-CP":  "Hết hiệu lực toàn bộ",
    "39/2001/NĐ-CP":  "Hết hiệu lực toàn bộ",
    "15/2003/NĐ-CP":  "Hết hiệu lực toàn bộ",
    "146/2007/NĐ-CP": "Hết hiệu lực toàn bộ",
    "71/2012/NĐ-CP":  "Hết hiệu lực toàn bộ",
    "171/2013/NĐ-CP": "Hết hiệu lực toàn bộ",
    "107/2014/NĐ-CP": "Hết hiệu lực toàn bộ",
    "46/2016/NĐ-CP":  "Hết hiệu lực toàn bộ",
    "60/2011/NĐ-CP":  "Hết hiệu lực toàn bộ",
    "93/2013/NĐ-CP":  "Hết hiệu lực toàn bộ",
    "132/2015/NĐ-CP": "Hết hiệu lực toàn bộ",
    # Đợt 3 (2026-07-08): thế hệ cũ hơn trồi lên sau khi demote đợt 1-2
    "09/2005/NĐ-CP":  "Hết hiệu lực toàn bộ",
    "44/2006/NĐ-CP":  "Hết hiệu lực toàn bộ",
}

# VBHN xử phạt giao thông: số hiệu VBHN TRÙNG giữa nhiều văn bản khác nhau
# (đo 2026-07-08: doc_number 03/VBHN-BGTVT có 6.983 chunk nhưng chỉ 2.045 là
# bản xử phạt GT) → PHẢI lọc thêm title. Status SUY RA từ hiệu lực vbpl của
# các NĐ ruột (đã xác nhận ở TRUE_STATUS): 19/VBHN-2014 hợp nhất 171/2013 +
# 107/2014 (cả hai TOÀN BỘ) → toàn bộ; 03/VBHN-2022 hợp nhất 100/2019 +
# 123/2021 (một phần đường bộ) → một phần.
VBHN_TITLE_PHRASE = "xử phạt vi phạm hành chính trong lĩnh vực giao thông"
VBHN_STATUS = {
    "19/VBHN-BGTVT": "Hết hiệu lực toàn bộ (VBHN của 171/2013 + 107/2014 — đều đã hết hiệu lực)",
    "03/VBHN-BGTVT": "Hết hiệu lực một phần (lĩnh vực đường bộ — thay bởi NĐ 168/2024/NĐ-CP)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    cli = OpenSearch(hosts=[config.OPENSEARCH_URL], timeout=120)
    IDX = config.OPENSEARCH_INDEX

    def _apply(dn: str, status: str, query: dict) -> None:
        if args.revert:
            script = {"source": "ctx._source.remove('status')"}
            label = "xoá status"
        else:
            script = {"source": "ctx._source.status = params.s", "params": {"s": status}}
            label = f"set '{status[:40]}'"
        r = cli.update_by_query(
            index=IDX, refresh=True, wait_for_completion=True, conflicts="proceed",
            body={"query": query, "script": script},
        )
        print(f"  {dn:<16} {label} → updated {r.get('updated')} chunk")

    for dn, status in TRUE_STATUS.items():
        _apply(dn, status, {"term": {"doc_number": dn}})

    # VBHN: bắt buộc lọc cả title (số hiệu trùng giữa nhiều VB khác nhau)
    for dn, status in VBHN_STATUS.items():
        _apply(dn, status, {"bool": {
            "filter": [{"term": {"doc_number": dn}}],
            "must": [{"match_phrase": {"title": VBHN_TITLE_PHRASE}}],
        }})

    cli.indices.refresh(index=IDX)
    print("Xong." if not args.revert else "Đã revert.")


if __name__ == "__main__":
    main()

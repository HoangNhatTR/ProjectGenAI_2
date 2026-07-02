"""Danh sách scoped re-ingest: văn bản HIỆN HÀNH bị mất cấu trúc Điều.

Lọc từ scan: chunk≥50 & Điều<3 & ban hành 2019+ (proxy 'còn hiệu lực'), + vài
luật cũ-nhưng-còn-hiệu-lực then chốt. Sắp theo năm giảm dần. In số chunk + ước
tính thời gian re-embed CPU (NĐ 168: ~0.67s/chunk).

Read-only. python -m scripts.scoped_reingest_list
"""
from __future__ import annotations

import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from opensearchpy import OpenSearch
from src import config

cli = OpenSearch(hosts=[config.OPENSEARCH_URL], timeout=300)
IDX = config.OPENSEARCH_INDEX
LAW_TYPES = ["Bộ luật", "Bộ luật (BL)", "Luật", "Luật (Lu)", "Pháp lệnh",
             "Nghị định", "Nghị định (NĐ)"]
SEC_PER_CHUNK = 0.67  # đo từ NĐ 168 (1344 chunk / 899s)

# Luật cũ (<2019) nhưng CÒN HIỆU LỰC, tra cứu nhiều → nên fix dù năm cũ.
KEEP_OLD = {"52/2014/QH13", "50/2014/QH13", "15/2012/QH13", "65/2014/QH13",
            "67/2014/QH13", "43/2013/QH13", "45/2013/QH13", "40/2013/QH13"}
# Bỏ qua: bản dịch tiếng Anh (xử lý riêng), không phải VB nội dung VN
SKIP = {"144/2021/NĐ-CP"}


def scan_flagged():
    flagged = []
    after = None
    while True:
        comp = {"size": 500, "sources": [{"dn": {"terms": {"field": "doc_number"}}}]}
        if after:
            comp["after"] = after
        res = cli.search(index=IDX, body={
            "size": 0, "query": {"terms": {"doc_type": LAW_TYPES}},
            "aggs": {"g": {"composite": comp,
                           "aggregations": {"arts": {"cardinality": {"field": "article"}}}}},
        })
        g = res["aggregations"]["g"]
        for b in g["buckets"]:
            if b["doc_count"] >= 50 and b["arts"]["value"] < 3:
                flagged.append((b["key"]["dn"], b["doc_count"]))
        after = g.get("after_key")
        if not after or not g["buckets"]:
            return flagged


def meta(dn):
    r = cli.search(index=IDX, body={"size": 1, "_source": ["title", "issued_date", "doc_type"],
                                    "query": {"term": {"doc_number": dn}}})
    h = r["hits"]["hits"]
    s = h[0]["_source"] if h else {}
    return s.get("title", ""), (s.get("issued_date", "") or "")[:4], s.get("doc_type", "")


print("Đang quét...", flush=True)
flagged = scan_flagged()
rows = []
for dn, cnt in flagged:
    if dn in SKIP:
        continue
    title, yr, dt = meta(dn)
    y = int(yr) if yr.isdigit() else 0
    if y >= 2019 or dn in KEEP_OLD:
        rows.append((dn, cnt, y, dt, title))

rows.sort(key=lambda r: (-r[2], -r[1]))  # năm desc, chunk desc

print(f"\n=== DANH SÁCH SCOPED RE-INGEST ({len(rows)} văn bản hiện hành) ===\n")
print(f"{'#':>3} {'doc_number':<16}{'năm':>5}{'chunk':>7}  văn bản")
print("-" * 92)
total = 0
for i, (dn, cnt, y, dt, title) in enumerate(rows, 1):
    total += cnt
    print(f"{i:>3} {dn:<16}{y:>5}{cnt:>7}  {title[:50]}")

mins = total * SEC_PER_CHUNK / 60
print("-" * 92)
print(f"TỔNG: {len(rows)} văn bản | {total:,} chunk | re-embed CPU ≈ {mins:.0f} phút (~{mins/60:.1f} giờ)")
print(f"\n(Gợi ý: có thể chia nhỏ — vd chỉ luật 2024-2025 trước. Đếm theo năm:)")
from collections import Counter
yc = Counter(r[2] for r in rows)
cc = Counter()
for dn, cnt, y, dt, title in rows:
    cc[y] += cnt
for y in sorted(yc, reverse=True):
    print(f"   {y}: {yc[y]:>2} VB, {cc[y]:>6,} chunk (~{cc[y]*SEC_PER_CHUNK/60:.0f} phút)")

"""Quét rộng: văn bản Luật/Bộ luật/Nghị định/Pháp lệnh có nhiều chunk nhưng
0 Điều phân biệt (dạng A — mất cấu trúc do crawl làm phẳng dòng).

Read-only. python -m scripts.scan_unstructured
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
MIN_CHUNKS = 50          # bỏ qua văn bản quá nhỏ
LOW_ART = 3              # 'gần như mất cấu trúc' nếu < ngưỡng này

print(f"Quét doc_type={LAW_TYPES} | cờ khi chunk≥{MIN_CHUNKS} & Điều<{LOW_ART}\n", flush=True)

flagged = []          # (doc_number, chunk, arts)
after = None
pages = 0
total_docs = 0
while True:
    comp = {"size": 500, "sources": [{"dn": {"terms": {"field": "doc_number"}}}]}
    if after:
        comp["after"] = after
    body = {
        "size": 0,
        "query": {"terms": {"doc_type": LAW_TYPES}},
        "aggs": {"by_dn": {"composite": comp,
                           "aggregations": {"arts": {"cardinality": {"field": "article"}}}}},
    }
    res = cli.search(index=IDX, body=body)
    agg = res["aggregations"]["by_dn"]
    for b in agg["buckets"]:
        total_docs += 1
        cnt = b["doc_count"]
        arts = b["arts"]["value"]
        if cnt >= MIN_CHUNKS and arts < LOW_ART:
            flagged.append((b["key"]["dn"], cnt, arts))
    after = agg.get("after_key")
    pages += 1
    if not after or not agg["buckets"]:
        break
    if pages % 10 == 0:
        print(f"  ...đã quét {total_docs} văn bản, cờ {len(flagged)}", flush=True)

print(f"\nĐã quét {total_docs} văn bản (Luật/NĐ/Pháp lệnh). Cờ: {len(flagged)}\n")

# Lấy title cho các văn bản bị cờ (sắp theo chunk giảm dần — nặng nhất trước)
flagged.sort(key=lambda x: -x[1])
print(f"{'doc_number':<20}{'chunk':>7}{'Điều':>6}  văn bản")
print("-" * 90)
for dn, cnt, arts in flagged[:60]:
    r = cli.search(index=IDX, body={"size": 1, "_source": ["title", "issued_date"],
                                    "query": {"term": {"doc_number": dn}}})
    h = r["hits"]["hits"]
    title = h[0]["_source"].get("title", "")[:52] if h else ""
    yr = h[0]["_source"].get("issued_date", "")[:4] if h else ""
    print(f"{dn:<20}{cnt:>7}{arts:>6}  [{yr}] {title}")

if len(flagged) > 60:
    print(f"\n... và {len(flagged)-60} văn bản nữa.")

# Thống kê nhanh tổng chunk bị ảnh hưởng
tot_chunk = sum(c for _, c, _ in flagged)
print(f"\nTổng chunk trong các văn bản bị cờ: {tot_chunk:,}")

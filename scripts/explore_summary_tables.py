"""Khám phá corpus OpenSearch để thiết kế detection 'bảng tóm tắt mức phạt' (#3).

Chạy: python -m scripts.explore_summary_tables
Read-only — KHÔNG ghi gì vào index.
"""
from __future__ import annotations

import re
from collections import Counter

from opensearchpy import OpenSearch

from src import config

cli = OpenSearch(hosts=[config.OPENSEARCH_URL], timeout=60)
IDX = config.OPENSEARCH_INDEX

_MONEY_RE = re.compile(r"\d[\d.,]*\s*(?:triệu|nghìn|tỷ|đồng)\b", re.IGNORECASE)
_ART_RE = re.compile(r"Điều\s+\d+", re.IGNORECASE)


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# 1) doc_type / folder distribution
section("1) doc_type & folder (top buckets)")
for field in ("doc_type", "folder"):
    res = cli.search(index=IDX, body={
        "size": 0,
        "aggs": {"a": {"terms": {"field": field, "size": 15}}},
    })
    print(f"\n-- {field} --")
    for b in res["aggregations"]["a"]["buckets"]:
        print(f"  {b['key']:<30} {b['doc_count']:>10,}")

# 2) Title chứa các từ khóa 'bảng tổng hợp / tra cứu mức phạt'
section("2) Docs có title kiểu tổng hợp/tra cứu (đếm + mẫu title)")
for kw in ["tổng hợp mức phạt", "tra cứu mức phạt", "bảng tổng hợp", "toàn bộ mức phạt", "tổng hợp"]:
    res = cli.search(index=IDX, body={
        "size": 3,
        "_source": ["title", "source", "doc_type"],
        "query": {"match_phrase": {"title": kw}},
    })
    total = res["hits"]["total"]["value"]
    print(f"\n[{kw}] ~{total} chunks")
    for h in res["hits"]["hits"]:
        s = h["_source"]
        print(f"    title={s.get('title','')[:90]!r}")
        print(f"      doc_type={s.get('doc_type')} source={s.get('source','')[:60]}")

# 3) Mẫu chunk có mật độ mức tiền cao (structural table) — lấy mẫu match nhiều keyword
section("3) Mẫu chunk text-match 'mức phạt' — đo money_mentions & distinct_articles")
res = cli.search(index=IDX, body={
    "size": 40,
    "_source": ["title", "doc_type", "text", "source"],
    "query": {"match": {"text": "mức phạt vượt đèn đỏ nồng độ cồn"}},
})
title_ctr = Counter()
hi_money = 0
for h in res["hits"]["hits"]:
    s = h["_source"]
    txt = s.get("text", "")
    money = len(_MONEY_RE.findall(txt))
    arts = len({m.lower() for m in _ART_RE.findall(txt)})
    tbl = txt.count("|") + txt.count("\t")
    if money >= 5 or tbl >= 6:
        hi_money += 1
        title_ctr[(s.get("title", "")[:70], s.get("doc_type"))] += 1
print(f"\n{hi_money}/40 chunk có money≥5 hoặc table-marks≥6. Title hay gặp:")
for (t, dt), n in title_ctr.most_common(15):
    print(f"  x{n}  [{dt}] {t!r}")

print("\nDONE explore.")

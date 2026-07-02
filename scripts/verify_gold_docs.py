"""Xác minh gold cho eval retrieval: (1) coverage số hiệu gốc, (2) VBHN là bản
hợp nhất của bộ luật nào. Read-only. python -m scripts.verify_gold_docs
"""
from __future__ import annotations

from opensearchpy import OpenSearch
from src import config

cli = OpenSearch(hosts=[config.OPENSEARCH_URL], timeout=60)
IDX = config.OPENSEARCH_INDEX


def count_doc(sub: str) -> int:
    r = cli.count(index=IDX, body={"query": {"wildcard": {"doc_number": f"*{sub}*"}}})
    return int(r["count"])


def title_of(docnum: str) -> str:
    r = cli.search(index=IDX, body={
        "size": 1, "_source": ["title", "doc_type", "doc_number"],
        "query": {"term": {"doc_number": docnum}},
    })
    h = r["hits"]["hits"]
    if not h:
        # thử wildcard
        r = cli.search(index=IDX, body={
            "size": 1, "_source": ["title", "doc_type", "doc_number"],
            "query": {"wildcard": {"doc_number": f"*{docnum}*"}},
        })
        h = r["hits"]["hits"]
    if not h:
        return "(không thấy)"
    s = h[0]["_source"]
    return f"[{s.get('doc_number')}] {s.get('title','')[:85]}"


print("=== Coverage số hiệu GỐC trong gold ===")
for d in ["168/2024", "100/2015", "91/2015", "45/2019", "31/2024", "52/2014",
          "59/2020", "12/2017"]:
    print(f"  {d:<12} → {count_doc(d):>8,} chunk")

print("\n=== Danh tính các VBHN/doc xuất hiện ở top1 (MISS) ===")
for d in ["11/VBHN-VPQH", "05/VBHN-VPQH", "18/VBHN-VPQH", "21/VBHN-VPQH",
          "121/VBHN-VPQH", "133/VBHN-VPQH", "45-LCT/HĐNN8", "17/2022/UBTVQH15",
          "576/QĐ-NHNN", "155/2020/NĐ-CP", "172/2013/NĐ-CP"]:
    print(f"  {title_of(d)}")

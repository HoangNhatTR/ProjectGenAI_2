"""Điều tra case 'vượt đèn đỏ' — tìm bảng tóm tắt vs điều khoản gốc thật.

Read-only. Chạy: python -m scripts.explore_redlight
"""
from __future__ import annotations

import re

from opensearchpy import OpenSearch

from src import config

cli = OpenSearch(hosts=[config.OPENSEARCH_URL], timeout=60)
IDX = config.OPENSEARCH_INDEX
_MONEY_RE = re.compile(r"\d[\d.,]*\s*(?:triệu|nghìn|tỷ|đồng)\b", re.IGNORECASE)
_ART_RE = re.compile(r"Điều\s+\d+", re.IGNORECASE)

res = cli.search(index=IDX, body={
    "size": 12,
    "_source": ["title", "doc_type", "doc_number", "article", "clause", "text", "source"],
    "query": {
        "bool": {
            "must": [{"match": {"text": {"query": "vượt đèn đỏ vượt đèn tín hiệu giao thông"}}}],
            "should": [{"match_phrase": {"text": {"query": "đèn tín hiệu", "boost": 2}}}],
        }
    },
})

print(f"Tổng hit: {res['hits']['total']['value']}\n")
for i, h in enumerate(res["hits"]["hits"], 1):
    s = h["_source"]
    txt = s.get("text", "")
    money = len(_MONEY_RE.findall(txt))
    arts = sorted({m for m in _ART_RE.findall(txt)})
    tbl = txt.count("|") + txt.count("\t")
    nl = txt.count("\n")
    print(f"[{i}] score={h['_score']:.2f} doc={s.get('doc_number')} "
          f"art={s.get('article')} clause={s.get('clause')}")
    print(f"     title={s.get('title','')[:80]!r}")
    print(f"     money={money} distinct_arts={len(arts)} table_marks={tbl} newlines={nl} len={len(txt)}")
    snippet = re.sub(r"\s+", " ", txt)[:240]
    print(f"     text: {snippet!r}\n")

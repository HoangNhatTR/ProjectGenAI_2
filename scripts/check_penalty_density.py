"""Kiểm tra: heuristic _is_summary_table có hạ nhầm điều khoản xử phạt gốc không?

Lấy chunk NĐ 168/2024 + parent text (nếu có parent_store), đo money/table-marks
rồi chạy đúng hàm reranker._is_summary_table để xem có bị gắn cờ oan.

Read-only. Chạy: python -m scripts.check_penalty_density
"""
from __future__ import annotations

import re
from collections import defaultdict

from opensearchpy import OpenSearch

from src import config
from src import reranker
from src.schemas import Chunk, DocumentMetadata

cli = OpenSearch(hosts=[config.OPENSEARCH_URL], timeout=60)
IDX = config.OPENSEARCH_INDEX
_MONEY_RE = re.compile(r"\d[\d.,]*\s*(?:triệu|nghìn|tỷ|đồng)\b", re.IGNORECASE)

# Parent store (nếu có) để dựng lại full Điều text — đúng cái reranker thấy
ps = None
try:
    if config.PARENT_STORE_PATH.exists():
        from src.parent_store import ParentStore
        ps = ParentStore(config.PARENT_STORE_PATH)
        print(f"ParentStore: {ps.count():,} entries\n")
except Exception as e:
    print(f"(không mở được parent_store: {e})\n")

# Lấy chunks của NĐ 168/2024 (nghị định xử phạt giao thông GỐC)
res = cli.search(index=IDX, body={
    "size": 200,
    "_source": ["article", "clause", "parent_id", "text", "title", "doc_number", "doc_type"],
    "query": {"term": {"doc_number": "168/2024/NĐ-CP"}},
})
hits = res["hits"]["hits"]
print(f"Lấy {len(hits)} chunk NĐ 168/2024/NĐ-CP\n")

# Gom theo parent_id để dựng full Điều
by_parent = defaultdict(list)
for h in hits:
    s = h["_source"]
    by_parent[s.get("parent_id")].append(s)

flagged_parents = 0
checked = 0
print("Mỗi Điều (parent): money trong full text + có bị _is_summary_table gắn cờ?")
for pid, group in list(by_parent.items())[:25]:
    if not pid:
        continue
    parent_text = ps.get(pid) if ps else None
    full = parent_text or " ".join(g.get("text", "") for g in group)
    money = len(_MONEY_RE.findall(full))
    s0 = group[0]
    ch = Chunk(
        chunk_id="x", text=full,
        article=s0.get("article"), clause=s0.get("clause"),
        metadata=DocumentMetadata(source="", title=s0.get("title"), doc_type=s0.get("doc_type")),
    )
    flagged = reranker._is_summary_table(ch)
    checked += 1
    if flagged:
        flagged_parents += 1
    mark = "⚠ FLAGGED(oan?)" if flagged else "ok"
    print(f"  {s0.get('article'):<12} money={money:<3} len={len(full):<6} → {mark}")

print(f"\n==> {flagged_parents}/{checked} Điều của NĐ 168 (nghị định xử phạt GỐC) bị "
      f"_is_summary_table gắn cờ. Nếu >0 ⇒ heuristic hạ nhầm điều khoản gốc.")

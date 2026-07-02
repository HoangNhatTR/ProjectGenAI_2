"""Test: lấy TOÀN VĂN NĐ 168/2024 từ vbpl.vn để xác nhận trước khi ingest.

python -m scripts.fetch_nd168_test
"""
from __future__ import annotations

import re
import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src.vbpl_client import VBPLClient

client = VBPLClient()
print(">>> Search 'xử phạt vi phạm hành chính giao thông đường bộ' (quét sâu) ...")
docs = client.search("xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ",
                     max_docs=250)
print(f"   → {len(docs)} văn bản. Lọc các NĐ giao thông gần đây:")
for d in docs:
    if any(y in (d["so_hieu"] or "") for y in ("/2024", "/2025", "/2026")) and "NĐ" in (d["so_hieu"] or ""):
        print(f"   id={d['id']:<40} | {d['so_hieu']} | {d['title'][:55]}")

target = next((d for d in docs if "168/2024" in (d["so_hieu"] or "")), None)

if not target:
    print("\n[FAIL] Không lấy được id của NĐ 168/2024.")
    sys.exit(1)

print(f"\n>>> Fetch full text id={target['id']} ...")
full = client.fetch_with_content(target["id"])
if not full or not full.get("content"):
    print("[FAIL] Không lấy được content.")
    sys.exit(1)

content = full["content"]
arts = sorted(set(int(m) for m in re.findall(r"Điều\s+(\d+)\.", content)))
print(f"\n=== KẾT QUẢ ===")
print(f"so_hieu: {full['so_hieu']}")
print(f"title  : {full['title'][:80]}")
print(f"len content: {len(content):,} chars")
print(f"Số 'Điều N.' phân biệt: {len(arts)}  (min={arts[0] if arts else '-'}, max={arts[-1] if arts else '-'})")
print(f"Các Điều: {arts}")
print(f"\n--- 600 ký tự đầu ---\n{content[:600]}")

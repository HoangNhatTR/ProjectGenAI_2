"""Rà hiệu lực NĐ 168/2024 + các NĐ xử phạt GT 2025-2026 (thay/sửa gì).

Read-only (chỉ gọi vbpl API). python -m scripts.check_nd168_effect
"""
from __future__ import annotations

import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src.vbpl_client import VBPLClient

client = VBPLClient()

# id đã biết từ search trước
CANDIDATES = [
    ("173920", "168/2024/NĐ-CP"),
    ("ea017d50-5a6c-11f1-a489-f564974f1db8", "133/2026/NĐ-CP"),
    ("187551", "81/2026/NĐ-CP"),
    ("185666", "336/2025/NĐ-CP"),
    ("187432", "48/2026/NĐ-CP"),
    ("172770", "121/2024/NĐ-CP"),
]

for doc_id, label in CANDIDATES:
    print("\n" + "=" * 70)
    full = client.fetch_with_content(doc_id)
    if not full:
        print(f"[{label}] KHÔNG fetch được (id={doc_id})")
        continue
    print(f"[{label}] {full.get('title','')[:90]}")
    print(f"   hiệu lực: {full.get('eff_status','?')}")
    refs = full.get("references", [])
    if refs:
        print(f"   references ({len(refs)}):")
        for r in refs[:12]:
            print(f"     - {r.get('rel_type','?'):<22} {r.get('so_hieu','')} | {r.get('title','')[:55]}")
    else:
        print("   (không có references)")

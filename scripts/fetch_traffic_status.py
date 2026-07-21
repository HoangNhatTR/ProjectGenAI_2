"""Lấy hiệu lực THẬT (vbpl) của các VB giao thông cũ — để backfill status đúng.

KHÔNG bịa status. Chỉ lấy effStatus chính thức từ search vbpl rồi in ra.
python -m scripts.fetch_traffic_status
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
TARGETS = ["100/2019/NĐ-CP", "123/2021/NĐ-CP", "152/2005/NĐ-CP", "34/2010/NĐ-CP",
           "19/VBHN-BGTVT", "03/VBHN-BGTVT", "10/VBHN-BGTVT",
           # Đợt 2 (2026-07-08): NĐ phạt GT cổ vẫn đè pool retrieval — đo thấy
           # chiếm 17/20 slot thô cho query đèn tín hiệu (49-CP xếp trên NĐ 168)
           "49-CP", "78/1998/NĐ-CP", "39/2001/NĐ-CP", "15/2003/NĐ-CP",
           "146/2007/NĐ-CP", "71/2012/NĐ-CP", "171/2013/NĐ-CP", "107/2014/NĐ-CP",
           "46/2016/NĐ-CP", "60/2011/NĐ-CP", "93/2013/NĐ-CP"]

for dn in TARGETS:
    # search theo số hiệu — kết quả search đã có effStatus (không cần fetch content)
    docs = client.search(dn, max_docs=30)
    m = next((d for d in docs if dn.split("/")[0] in (d["so_hieu"] or "")
              and d["so_hieu"] and dn.replace("/", "").lower()[:6] in d["so_hieu"].replace("/", "").lower()), None)
    if not m:
        # match lỏng hơn: chứa nguyên số hiệu
        m = next((d for d in docs if dn.lower() in (d["so_hieu"] or "").lower()), None)
    if m:
        print(f"{dn:<16} → eff='{m.get('eff_status','?')}' | {m.get('title','')[:60]}")
    else:
        print(f"{dn:<16} → KHÔNG tìm thấy trên vbpl (found {len(docs)} kết quả khác)")

"""Kiểm chứng fix chunker trên dữ liệu THẬT (không ghi gì): 36/2024 + 52/2014."""
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
from src.chunking import chunk_document, _normalize_structure
from src.schemas import DocumentMetadata, RawDocument

_LINESTART = re.compile(r"(?m)^Điều\s+\d+")
client = VBPLClient()

for doc_id, dn in [("170620", "36/2024/QH15"), ("36870", "52/2014/QH13")]:
    full = client.fetch_with_content(doc_id)
    if not full or not full.get("content"):
        print(f"{dn}: KHÔNG fetch được")
        continue
    t = full["content"]
    ls_raw = len(_LINESTART.findall(t))
    ls_norm = len(_LINESTART.findall(_normalize_structure(t)))
    doc = RawDocument(text=t, metadata=DocumentMetadata(
        source=full["url"], doc_number=dn, title=full["title"], doc_type="Luật"))
    ch = chunk_document(doc, parent_store=None)
    arts = sorted({int(x.article.split()[1]) for x in ch if x.article and x.article.split()[1].isdigit()})
    print(f"\n{dn}: {len(t):,} chars | {full['title'][:50]}")
    print(f"  Điều đầu dòng RAW (fresh fetch): {ls_raw}")
    print(f"  Điều đầu dòng SAU normalize    : {ls_norm}")
    print(f"  chunk_document (CÓ fix): {len(ch)} chunk, {len(arts)} Điều (max={arts[-1] if arts else 0})")

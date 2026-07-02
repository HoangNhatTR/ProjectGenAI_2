"""Probe nhanh các model KieAI OpenAI-compat: model nào còn gọi được (chat+latency).

Chạy: ..\Chatbot\Scripts\python.exe -m scripts.probe_kieai
"""
from __future__ import annotations

import sys
import time

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.llm_client import KieAIClient

CANDIDATES = [
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-3-5-flash-openai",
    "gemini-2.5-pro",
    "gemini-3.1-pro",
    "gemini-3-pro",
    "gpt-5-2",
]

KEY = config.KIEAI_API_KEY
BASE = config.KIEAI_BASE_URL
print(f"Key: ...{KEY[-6:]}  Base: {BASE}\n")

client = KieAIClient(api_key=KEY, base_url=BASE)
msgs = [{"role": "user", "content": "Trả lời đúng 1 từ: OK"}]

rows = []
for m in CANDIDATES:
    t0 = time.time()
    try:
        r = client.chat(model=m, messages=msgs, options={"temperature": 0.0})
        dt = time.time() - t0
        content = (r["message"]["content"] or "").strip().replace("\n", " ")[:40]
        rows.append((m, "OK", f"{dt:.1f}s", content))
        print(f"  ✅ {m:<26} {dt:4.1f}s  -> {content!r}")
    except Exception as e:
        dt = time.time() - t0
        err = str(e).replace("\n", " ")[:80]
        rows.append((m, "FAIL", f"{dt:.1f}s", err))
        print(f"  ❌ {m:<26} {dt:4.1f}s  -> {err}")

print("\n=== TỔNG KẾT ===")
ok = [r for r in rows if r[1] == "OK"]
print(f"Sống: {len(ok)}/{len(rows)} -> {[r[0] for r in ok]}")

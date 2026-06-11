"""Benchmark tốc độ các model OpenRouter cho Legal AI"""
import os, sys, time

for s in (sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        try: s.reconfigure(encoding="utf-8", errors="replace")
        except: pass

# Repo root = thư mục cha của scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv()  # đọc .env ở repo root (đã chdir ở trên)

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not API_KEY:
    sys.exit("Thiếu OPENROUTER_API_KEY trong .env")
from openai import OpenAI

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    timeout=30,
    default_headers={"HTTP-Referer": "https://legal-ai-vn", "X-Title": "Legal AI VN"},
)

Q = "xe máy không đội mũ bảo hiểm bị phạt bao nhiêu? trả lời ngắn"

TESTS = [
    # (label, model_id, messages, extra_kwargs)
    ("qwen3.5-9b /no_think system",
     "qwen/qwen3.5-9b",
     [{"role": "system", "content": "/no_think"},
      {"role": "user",   "content": Q}],
     {}),
    ("qwen3.5-9b /no_think in user msg",
     "qwen/qwen3.5-9b",
     [{"role": "user", "content": "/no_think\n" + Q}],
     {}),
    ("qwen3.5-9b reasoning exclude",
     "qwen/qwen3.5-9b",
     [{"role": "user", "content": Q}],
     {"extra_body": {"reasoning": {"exclude": True}}}),
    ("google/gemini-flash-1.5",
     "google/gemini-flash-1.5",
     [{"role": "user", "content": Q}],
     {}),
    ("mistralai/mistral-7b-instruct",
     "mistralai/mistral-7b-instruct",
     [{"role": "user", "content": Q}],
     {}),
    ("qwen/qwen2.5-7b-instruct",
     "qwen/qwen2.5-7b-instruct",
     [{"role": "user", "content": Q}],
     {}),
]

print(f"\n{'Model':<40} {'Thời gian':>10}  {'Preview'}")
print("-" * 90)

for label, model_id, msgs, extra_kwargs in TESTS:
    try:
        t0 = time.time()
        kwargs = {"model": model_id, "messages": msgs, "temperature": 0.1, **extra_kwargs}
        r = client.chat.completions.create(**kwargs)
        elapsed = time.time() - t0
        preview = (r.choices[0].message.content or "")[:55].replace("\n", " ")
        print(f"{label:<40} {elapsed:>8.1f}s  {preview}")
    except Exception as e:
        print(f"{label:<40} {'ERROR':>10}  {str(e)[:55]}")

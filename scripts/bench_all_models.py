"""Benchmark tất cả model trên 9Router — tìm model nhanh nhất cho Legal AI."""
import os, sys, time, concurrent.futures

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Repo root = thư mục cha của scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.llm_client import Router9Client

from dotenv import load_dotenv
load_dotenv()  # đọc .env ở repo root (đã chdir ở trên)

API_KEY  = os.getenv("ROUTER9_API_KEY", "")
BASE_URL = os.getenv("ROUTER9_BASE_URL", "http://localhost:20128/v1")
if not API_KEY:
    sys.exit("Thiếu ROUTER9_API_KEY trong .env")
client   = Router9Client(api_key=API_KEY, base_url=BASE_URL)

# Câu hỏi test — cần câu trả lời ngắn
MSG_SIMPLE = [{"role": "user", "content": "xe may khong doi mu bao hiem phat bao nhieu? tra loi 1 dong ngan gon"}]
MSG_JSON   = [{"role": "user", "content": '{"action":"retrieve","intent":"legal","search_query":"xe may khong doi mu","tool_name":null,"tool_query":null,"direct_response":null}'}]

MODELS = [
    # cc/ — Claude qua Claude.ai Copilot
    "cc/claude-haiku-4-5-20251001",
    "cc/claude-sonnet-4-5-20250929",
    "cc/claude-sonnet-4-6",
    "cc/claude-opus-4-6",
    "cc/claude-opus-4-7",
    # gh/ — GitHub Models (thường nhanh)
    "gh/gpt-4o-mini",
    "gh/gpt-4.1",
    "gh/gpt-5-mini",
    "gh/gpt-5.4-mini",
    "gh/claude-haiku-4.5",
    "gh/claude-sonnet-4.5",
    "gh/claude-sonnet-4.6",
    "gh/gemini-3-flash-preview",
    # kc/ — Key Credentials (có API key riêng)
    "kc/google/gemini-2.5-flash",
    "kc/deepseek/deepseek-chat",
    "kc/anthropic/claude-sonnet-4-20250514",
]

TIMEOUT = 12  # giây tối đa mỗi model

def test_model(model_id: str) -> tuple[str, float, str]:
    try:
        t0 = time.time()
        r = client.chat(model=model_id, messages=MSG_SIMPLE,
                        options={"temperature": 0.1})
        elapsed = time.time() - t0
        reply = r["message"]["content"][:55].replace("\n", " ").strip()
        return model_id, elapsed, reply
    except Exception as e:
        return model_id, -1, str(e)[:70]


print("\nBenchmarking 9Router models (timeout=12s each)...")
print("=" * 80)
print(f"{'Model':<42} {'Time':>7}  Status / Preview")
print("-" * 80)

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    futures = {ex.submit(test_model, m): m for m in MODELS}
    for fut in concurrent.futures.as_completed(futures, timeout=TIMEOUT + 5):
        try:
            model_id, elapsed, reply = fut.result(timeout=TIMEOUT)
        except Exception as e:
            model_id = futures[fut]
            elapsed, reply = -1, f"TIMEOUT/ERR: {e}"
        results.append((model_id, elapsed, reply))

# Sort by speed (ERROR cuối)
results.sort(key=lambda x: x[1] if x[1] > 0 else 9999)

for model_id, elapsed, reply in results:
    if elapsed < 0:
        status = f"  ERROR  "
    else:
        status = f"{elapsed:6.1f}s"
    marker = "✅" if 0 < elapsed < 6 else ("⚠️" if 0 < elapsed < 12 else "❌")
    print(f"{model_id:<42} {status}  {marker} {reply}")

print("=" * 80)
print("\nGợi ý:")
fast = [(m, t) for m, t, _ in results if 0 < t < 5]
if fast:
    print(f"  Router (JSON nhanh): {fast[0][0]}  ({fast[0][1]:.1f}s)")
    if len(fast) > 1:
        print(f"  Generator (chất lượng): {fast[-1][0]}  ({fast[-1][1]:.1f}s)")

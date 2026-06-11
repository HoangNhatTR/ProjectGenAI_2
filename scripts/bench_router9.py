import sys, io, time, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# Repo root = thư mục cha của scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv()  # đọc .env ở repo root (đã chdir ở trên)

from src.llm_client import Router9Client
client = Router9Client(
    api_key=os.getenv("ROUTER9_API_KEY", ""),
    base_url=os.getenv("ROUTER9_BASE_URL", "http://localhost:20128/v1"),
)

MODELS = [
    'cc/claude-haiku-4-5-20251001',
    'cc/claude-sonnet-4-5-20250929',
    'cc/claude-sonnet-4-6',
    'gh/gpt-4o-mini',
    'gh/claude-haiku-4.5',
]
MSG = [{'role':'user','content':'xe may khong doi mu bao hiem phat bao nhieu? tra loi ngan'}]

print(f"\n{'Model':<36} {'Time':>6}  Response")
print("-" * 75)
for m in MODELS:
    try:
        t0 = time.time()
        r = client.chat(model=m, messages=MSG, options={'temperature':0.1})
        elapsed = time.time() - t0
        content = r['message']['content'][:55].replace('\n',' ')
        print(f"{m:<36} {elapsed:>5.1f}s  {content}")
    except Exception as e:
        print(f"{m:<36} ERROR   {str(e)[:55]}")

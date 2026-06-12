from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", DATA_DIR / "vectorstore"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# ── Vector backend ─────────────────────────────────────────────────────────────
# chroma     : Chroma local (mặc định) + lexical FTS5/BM25
# opensearch : OpenSearch server — vector kNN + BM25 native trong 1 index
VECTOR_BACKEND   = os.getenv("VECTOR_BACKEND", "chroma")
OPENSEARCH_URL   = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "legal_chunks")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama | gemini | groq | router9 | openrouter
LLM_MODEL = os.getenv("LLM_MODEL", "mistral:latest")
# Model riêng cho Router (nhỏ/nhanh hơn, chỉ cần xuất JSON)
ROUTER_MODEL = os.getenv("ROUTER_MODEL", LLM_MODEL)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_docs")

ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
ROUTER9_API_KEY      = os.getenv("ROUTER9_API_KEY", "")
ROUTER9_BASE_URL     = os.getenv("ROUTER9_BASE_URL", "http://localhost:20128/v1")
ROUTER9_MODEL        = os.getenv("ROUTER9_MODEL", "cc/claude-haiku-4-5-20251001")
KIEAI_API_KEY        = os.getenv("KIEAI_API_KEY", "")
KIEAI_BASE_URL       = os.getenv("KIEAI_BASE_URL", "https://kieai.erweima.ai/api/v1")
OPENROUTER_API_KEY   = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL  = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "5"))

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# ── Parent-Child Chunking ──────────────────────────────────────────────────────
# True: mỗi Điều lưu vào ParentStore, child chunks có parent_id → expand khi generate
USE_PARENT_CHILD = os.getenv("USE_PARENT_CHILD", "true").lower() == "true"
PARENT_STORE_PATH = PROCESSED_DIR / "parent_store.db"

# ── HyDE (Hypothetical Document Embeddings) ───────────────────────────────────
# True: sinh 1 đoạn pháp luật giả → embed → dùng làm vector query (tăng recall)
# Dùng HYDE_MODEL (fast model) để giảm latency
USE_HYDE = os.getenv("USE_HYDE", "false").lower() == "true"
HYDE_MODEL = os.getenv("HYDE_MODEL", ROUTER_MODEL)

# URL gốc của API server (dùng để tạo download link trong response)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── API server security ───────────────────────────────────────────────────────
# Nếu set, mọi request tới /v1/* phải gửi header: Authorization: Bearer <key>
# Để trống = không bắt auth (chỉ nên dùng khi chạy localhost)
API_AUTH_KEY = os.getenv("API_AUTH_KEY", "")

# Danh sách origin được phép gọi API (phân tách bằng dấu phẩy).
# Đặt CORS_ORIGINS=* để mở hoàn toàn (khi đó credentials bị tắt theo spec CORS).
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:3001,http://localhost:8501"
    ).split(",")
    if o.strip()
]

# ── Danh sách model cho UI selector ──────────────────────────────────────────
# Mỗi entry: (model_id, label_hiển_thị)
ROUTER9_MODELS: list[tuple[str, str]] = [
    # ── Claude (Anthropic via 9Router) ──
    ("cc/claude-haiku-4-5-20251001", "Claude Haiku 4.5   — nhanh, tiết kiệm"),
    ("cc/claude-sonnet-4-6",         "Claude Sonnet 4.6  — cân bằng ★"),
    ("cc/claude-opus-4-7",           "Claude Opus 4.7    — mạnh, chậm hơn"),
    ("cc/claude-opus-4-8",           "Claude Opus 4.8    — mới nhất"),
    # ── GPT (OpenAI via GitHub/9Router) ──
    ("gh/gpt-4.1",                   "GPT-4.1            — GitHub Models"),
    ("gh/gpt-4o",                    "GPT-4o             — multimodal"),
    ("gh/gpt-4o-mini",               "GPT-4o mini        — nhanh, rẻ"),
    ("gh/o3",                        "OpenAI o3          — reasoning"),
    ("gh/o4-mini",                   "OpenAI o4-mini     — reasoning nhanh"),
]

# Model IDs chỉ cho việc tra cứu nhanh
ROUTER9_MODEL_IDS: list[str] = [m[0] for m in ROUTER9_MODELS]

# ── Kie AI models ─────────────────────────────────────────────────────────────
# Mỗi entry: (model_id, label, provider_key)
# Confirmed working: deepseek-chat → deepseek-v4-flash
# Xem thêm tại: https://kie.ai/market
KIEAI_MODELS: list[tuple[str, str]] = [
    ("deepseek-chat",    "DeepSeek Chat  (→ v4-flash) ✓"),
    ("deepseek-r1",      "DeepSeek R1    (reasoning)"),
    ("gpt-4o",           "GPT-4o"),
    ("gpt-4o-mini",      "GPT-4o mini    — nhanh"),
    ("claude-sonnet",    "Claude Sonnet"),
    ("claude-haiku",     "Claude Haiku   — nhanh"),
    ("gemini-2.0-flash", "Gemini 2.0 Flash"),
    ("gemini-2.5-pro",   "Gemini 2.5 Pro"),
]
KIEAI_MODEL_IDS: list[str] = [m[0] for m in KIEAI_MODELS]

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

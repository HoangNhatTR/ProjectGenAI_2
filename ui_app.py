"""Streamlit UI cho Legal AI Agent — Redesigned v2 (modern AI chat style).

Chạy:
    streamlit run ui_app.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from typing import Optional

import streamlit as st

# UTF-8 cho stdout (Windows console khi chạy streamlit)
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.bm25_index import BM25Index
from src.embedding import Embedder
from src.generator import Generator
from src.guardrails import apply_guardrails
from src.memory import MemoryStore
from src.planner import LegalPlanner
from src.retriever import Retriever
from src.router import SmartRouter
from src.schemas import Answer
from src.session import Session, SessionStore
from src.state import ConversationState
from src.tools import LegalToolRegistry
from src.vectorstore import VectorStore
from src.doc_chat import DOC_SOURCE_PREFIX, UploadedDocStore

try:
    from src.kg import kg_queries as kgq
    _KG_AVAILABLE = True
except Exception as _kg_exc:
    _KG_AVAILABLE = False
    _kg_import_error = str(_kg_exc)


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Legal AI Agent — Tư vấn pháp luật VN",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Cached resources ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Đang khởi tạo Legal AI Agent...")
def init_agent():
    import json
    embedder = Embedder(config.EMBEDDING_MODEL)
    vstore = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)

    bm25_path = config.DATA_DIR / "bm25" / "index.json"
    bm25: Optional[BM25Index] = BM25Index(bm25_path) if bm25_path.exists() else None

    kg_retriever = None
    if _KG_AVAILABLE:
        try:
            from src.kg.kg_retriever import KGRetriever
            kg_retriever = KGRetriever()
        except Exception:
            kg_retriever = None

    retriever = Retriever(embedder, vstore, bm25=bm25, kg_retriever=kg_retriever)

    top15_urls: list[str] = []
    manifest_path = config.DATA_DIR / "comparison" / "top10_laws" / "manifest.json"
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            top15_urls = [law["source_url"] for law in data.get("laws", []) if law.get("source_url")]
        except Exception:
            top15_urls = []

    _api_key = {
        "gemini": config.GEMINI_API_KEY,
        "groq": config.GROQ_API_KEY,
    }.get(config.LLM_PROVIDER)

    generator = Generator(
        model=config.LLM_MODEL,
        host=config.OLLAMA_HOST,
        temperature=config.LLM_TEMPERATURE,
        provider=config.LLM_PROVIDER,
        api_key=_api_key,
    )
    router = SmartRouter(
        model=config.LLM_MODEL,
        host=config.OLLAMA_HOST,
        provider=config.LLM_PROVIDER,
        api_key=_api_key,
    )

    ollama_client = generator.get_client()
    tool_registry = LegalToolRegistry(
        retriever=retriever,
        ollama_client=ollama_client,
        model=config.LLM_MODEL,
    )
    planner = LegalPlanner(
        ollama_client=ollama_client,
        model=config.LLM_MODEL,
        tool_registry=tool_registry,
    )

    return {
        "embedder": embedder,
        "vstore": vstore,
        "bm25": bm25,
        "kg_retriever": kg_retriever,
        "top15_urls": top15_urls,
        "retriever": retriever,
        "generator": generator,
        "router": router,
        "planner": planner,
        "tool_registry": tool_registry,
    }


@st.cache_resource
def get_session_store() -> SessionStore:
    return SessionStore(config.DATA_DIR)


_MEMORY_DIR = config.DATA_DIR / "memory"


@st.cache_resource
def get_memory_store_for(session_id: str) -> MemoryStore:
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return MemoryStore(_MEMORY_DIR / f"{session_id}.json")


def delete_memory_file(session_id: str) -> None:
    p = _MEMORY_DIR / f"{session_id}.json"
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


# ─── Constants ────────────────────────────────────────────────────────────────

MAX_HISTORY_MESSAGES = 10
AUTO_NAME_MAXLEN = 50
SUMMARY_MIN_MESSAGES = 12
SUMMARY_KEEP_RECENT = 6
SUMMARY_MAX_RAW = MAX_HISTORY_MESSAGES

RETRIEVAL_MODES = {
    "rag_full": {
        "label": "RAG Full",
        "icon": "📚",
        "short": "Toàn bộ corpus",
        "desc": "Vector + BM25 trên toàn bộ 609 luật (~68k chunks)",
        "use_kg": False,
        "use_top10_filter": False,
        "color": "#2563EB",
        "bg": "#EFF6FF",
        "border": "#93C5FD",
    },
    "rag_top10": {
        "label": "RAG Top 15",
        "icon": "🎯",
        "short": "Top 15 luật",
        "desc": "Vector + BM25 chỉ trên top 15 luật quan trọng (~7.5k chunks)",
        "use_kg": False,
        "use_top10_filter": True,
        "color": "#059669",
        "bg": "#ECFDF5",
        "border": "#6EE7B7",
    },
    "graph_rag": {
        "label": "Graph RAG",
        "icon": "🕸️",
        "short": "Vector + KG",
        "desc": "Vector + BM25 + Knowledge Graph (top 15 luật + KG semantic)",
        "use_kg": True,
        "use_top10_filter": True,
        "color": "#7C3AED",
        "bg": "#F5F3FF",
        "border": "#C4B5FD",
    },
}
DEFAULT_RETRIEVAL_MODE = "graph_rag"

# ── Chế độ nguồn tài liệu (per-session) ──────────────────────────────────────
DOC_SOURCE_MODES = {
    "corpus_only": {
        "label": "Chỉ corpus pháp luật",
        "icon": "⚖️",
        "color": "#2563EB",
        "desc": "Trả lời dựa trên corpus 609 luật VN",
    },
    "docs_only": {
        "label": "Chỉ tài liệu upload",
        "icon": "📎",
        "color": "#059669",
        "desc": "Trả lời dựa hoàn toàn vào file bạn đính kèm",
    },
    "combined": {
        "label": "Kết hợp cả hai",
        "icon": "🔀",
        "color": "#7C3AED",
        "desc": "Merge corpus pháp luật + tài liệu upload",
    },
}
DEFAULT_DOC_MODE = "corpus_only"


def _get_doc_mode(session_id: str) -> str:
    return st.session_state.get(f"doc_mode_{session_id}", DEFAULT_DOC_MODE)


def _set_doc_mode(session_id: str, mode: str) -> None:
    st.session_state[f"doc_mode_{session_id}"] = mode


def _get_doc_store(session_id: str, embedder) -> UploadedDocStore:
    key = f"doc_store_{session_id}"
    if key not in st.session_state:
        st.session_state[key] = UploadedDocStore(embedder)
    return st.session_state[key]


def _get_session_mode(session_id: str) -> str:
    """Lấy RAG mode của session cụ thể (mỗi phiên lưu riêng)."""
    return st.session_state.get(f"rag_mode_{session_id}", DEFAULT_RETRIEVAL_MODE)


def _set_session_mode(session_id: str, mode: str) -> None:
    """Lưu RAG mode cho session cụ thể."""
    st.session_state[f"rag_mode_{session_id}"] = mode


# ─── Session state helpers ────────────────────────────────────────────────────

def init_session_state(session_store: SessionStore) -> None:
    if "active_session_id" not in st.session_state:
        recent = session_store.list_recent(limit=1)
        if recent:
            st.session_state.active_session_id = recent[0].id
        else:
            new_sess = Session.new()
            session_store.save(new_sess)
            st.session_state.active_session_id = new_sess.id

    if "conv_state" not in st.session_state:
        st.session_state.conv_state = ConversationState()

    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None

    if "last_trace" not in st.session_state:
        st.session_state.last_trace = {}

    st.session_state.setdefault("top_k", config.TOP_K)
    st.session_state.setdefault("min_score", 0.3)
    st.session_state.setdefault("use_planner", True)
    st.session_state.setdefault("use_guardrails", True)
    st.session_state.setdefault("use_memory_extract", True)
    # RAG mode được lưu per-session qua _get_session_mode / _set_session_mode


def load_active_session(session_store: SessionStore) -> Session:
    sess = session_store.load(st.session_state.active_session_id)
    if sess is None:
        sess = Session.new()
        session_store.save(sess)
        st.session_state.active_session_id = sess.id
    return sess


def maybe_update_summary(session: Session, generator: Generator) -> bool:
    total = len(session.messages)
    if total < SUMMARY_MIN_MESSAGES:
        return False
    raw_count = total - session.summary_until
    if raw_count <= SUMMARY_MAX_RAW:
        return False
    new_until = total - SUMMARY_KEEP_RECENT
    if new_until <= session.summary_until:
        return False
    to_summarize = session.messages[session.summary_until:new_until]
    new_summary = generator.summarize_history(
        messages=to_summarize, prev_summary=session.summary,
    )
    if not new_summary or new_summary == session.summary:
        return False
    session.summary = new_summary
    session.summary_until = new_until
    return True


def _group_sessions_by_date(sessions: list) -> dict:
    """Nhóm sessions theo ngày: Hôm nay / Hôm qua / 7 ngày qua / Cũ hơn."""
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    last_week = today - timedelta(days=7)

    groups: dict[str, list] = {
        "Hôm nay": [],
        "Hôm qua": [],
        "7 ngày qua": [],
        "Cũ hơn": [],
    }
    for s in sessions:
        try:
            dt = datetime.fromisoformat(s.updated_at).date()
        except Exception:
            groups["Cũ hơn"].append(s)
            continue
        if dt == today:
            groups["Hôm nay"].append(s)
        elif dt == yesterday:
            groups["Hôm qua"].append(s)
        elif dt >= last_week:
            groups["7 ngày qua"].append(s)
        else:
            groups["Cũ hơn"].append(s)
    return groups


# ─── Doc Upload Panel ─────────────────────────────────────────────────────────

def _render_doc_upload_panel(session_id: str, embedder) -> None:
    """Panel upload tài liệu trong sidebar — giống Lobe Chat attachment."""
    doc_store = _get_doc_store(session_id, embedder)
    doc_mode = _get_doc_mode(session_id)
    files = doc_store.list_files()

    n_files = len(files)
    label = f"📎 Tài liệu ({n_files})" if n_files else "📎 Đính kèm tài liệu"

    with st.expander(label, expanded=(n_files > 0)):
        # ── Nguồn trả lời ──
        st.markdown(
            '<div style="font-size:11px;font-weight:600;color:#9CA3AF;'
            'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">'
            'Nguồn trả lời</div>',
            unsafe_allow_html=True,
        )
        for key, cfg in DOC_SOURCE_MODES.items():
            is_active = key == doc_mode
            disabled = (key != "corpus_only" and doc_store.is_empty())
            border = "#3B82F6" if is_active else "transparent"
            bg = "#1E3A5F" if is_active else "transparent"
            check = "✓ " if is_active else ""
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;'
                f'padding:7px 10px;border-radius:8px;border:1.5px solid {border};'
                f'background:{bg};margin-bottom:3px;">'
                f'<span style="font-size:15px;">{cfg["icon"]}</span>'
                f'<span style="font-size:12.5px;font-weight:{"600" if is_active else "400"};'
                f'color:#F3F4F6;">{check}{cfg["label"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if not is_active and not disabled:
                if st.button(f"Dùng {cfg['icon']}", key=f"docmode_{key}_{session_id}",
                             use_container_width=True):
                    _set_doc_mode(session_id, key)
                    st.rerun()
            elif disabled and key != "corpus_only":
                st.caption("(cần upload file trước)")

        st.markdown("<hr style='border-color:#374151;margin:8px 0;'/>", unsafe_allow_html=True)

        # ── File uploader ──
        st.markdown(
            '<div style="font-size:11px;font-weight:600;color:#9CA3AF;'
            'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">'
            'Upload file</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Chọn file",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key=f"uploader_{session_id}",
            label_visibility="collapsed",
        )

        if uploaded:
            # Chỉ xử lý file chưa có trong store
            existing_names = {f["name"] for f in files}
            new_files = [f for f in uploaded if f.name not in existing_names]
            if new_files:
                for uf in new_files:
                    with st.spinner(f"Đang xử lý {uf.name}..."):
                        try:
                            n = doc_store.add_file(uf.name, uf.getvalue())
                            st.success(f"✓ {uf.name} — {n} đoạn")
                        except Exception as e:
                            st.error(f"❌ {uf.name}: {e}")
                # Auto-switch sang combined nếu có cả corpus lẫn file
                if doc_mode == "corpus_only":
                    _set_doc_mode(session_id, "combined")
                st.rerun()

        # ── Danh sách file đã upload ──
        if files:
            st.markdown(
                '<div style="font-size:11px;font-weight:600;color:#9CA3AF;'
                'text-transform:uppercase;letter-spacing:0.5px;margin:6px 0 4px 0;">'
                'Đã tải lên</div>',
                unsafe_allow_html=True,
            )
            for f in files:
                c1, c2 = st.columns([9, 1])
                ext = f["name"].rsplit(".", 1)[-1].upper()
                icon = {"PDF": "🔴", "DOCX": "🔵", "TXT": "⬜", "MD": "📄"}.get(ext, "📄")
                c1.markdown(
                    f'<div style="font-size:12px;color:#D1D5DB;padding:3px 0;">'
                    f'{icon} <strong>{f["name"][:28]}</strong>'
                    f'<span style="color:#6B7280;font-size:11px;"> · {f["chunks"]} đoạn</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if c2.button("×", key=f"rmdoc_{f['name']}_{session_id}"):
                    doc_store.remove_file(f["name"])
                    if doc_store.is_empty():
                        _set_doc_mode(session_id, "corpus_only")
                    st.rerun()

            if st.button("🗑 Xoá tất cả file", use_container_width=True,
                         key=f"clearall_{session_id}"):
                doc_store.clear()
                _set_doc_mode(session_id, "corpus_only")
                st.rerun()
        else:
            st.caption("Kéo thả hoặc click để chọn PDF, DOCX, TXT, MD")


# ─── Global CSS ───────────────────────────────────────────────────────────────

_GLOBAL_CSS = """
<style>
/* ── Base typography ── */
.stApp {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', 'Helvetica Neue', sans-serif;
  background: #F7F8FA;
}

/* ── Hide Streamlit chrome ── */
.stDeployButton, footer, #MainMenu { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; height: 0; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: #111827 !important;
  border-right: 1px solid #1F2937;
  width: 280px !important;
}
section[data-testid="stSidebar"] * { color: #E5E7EB; }
section[data-testid="stSidebar"] hr { border-color: #374151; margin: 10px 0; }

/* ── Sidebar buttons (session items) ── */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
  background: transparent !important;
  border: none !important;
  color: #D1D5DB !important;
  text-align: left !important;
  padding: 8px 10px !important;
  border-radius: 8px !important;
  font-size: 13.5px !important;
  font-weight: 400 !important;
  transition: background 0.15s, color 0.15s !important;
  box-shadow: none !important;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
  background: #1F2937 !important;
  color: #F9FAFB !important;
}

/* New chat button — special style */
div[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type > button {
  background: #1F2937 !important;
  border: 1px solid #374151 !important;
  color: #F9FAFB !important;
  font-weight: 500 !important;
}
div[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type > button:hover {
  background: #374151 !important;
}

/* ── RAG mode cards ── */
.rag-mode-card {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 10px;
  cursor: pointer;
  margin-bottom: 6px;
  border: 1.5px solid transparent;
  transition: all 0.15s ease;
}
.rag-mode-card:hover { border-color: #4B5563; background: #1F2937; }
.rag-mode-card.active { border-color: #3B82F6 !important; background: #1E3A5F !important; }
.rag-mode-icon { font-size: 20px; }
.rag-mode-text { flex: 1; }
.rag-mode-name { font-size: 13px; font-weight: 600; color: #F3F4F6; }
.rag-mode-desc { font-size: 11px; color: #9CA3AF; margin-top: 1px; }
.rag-mode-check { font-size: 14px; }

/* ── Session group headers ── */
.session-group-header {
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: 0.6px;
  text-transform: uppercase;
  color: #6B7280;
  padding: 8px 4px 4px 4px;
  margin-top: 4px;
}

/* ── Session item ── */
.session-item-active > button {
  background: #1E3A5F !important;
  color: #93C5FD !important;
  font-weight: 500 !important;
}

/* ══════════════════════════════════════════════════════════
   CHAT MESSAGES — Lobe Chat / Open WebUI style
   ══════════════════════════════════════════════════════════ */

/* Fade-in mỗi message */
@keyframes msgFadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Container mỗi message */
[data-testid="stChatMessage"] {
  animation: msgFadeIn 0.2s ease-out;
  padding: 20px 28px !important;
  border-radius: 0 !important;
  border: none !important;
  box-shadow: none !important;
  max-width: 100%;
}

/* User message: nền xám nhạt như ChatGPT/Lobe Chat */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: #F7F8FA !important;
}

/* AI message: trắng sạch */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  background: #FFFFFF !important;
}

/* Đường kẻ ngăn cách giữa các messages */
[data-testid="stChatMessage"] + [data-testid="stChatMessage"] {
  border-top: 1px solid #F1F5F9 !important;
}

/* Avatar — hình vuông bo góc kiểu Lobe Chat */
[data-testid="stChatMessageAvatarUser"] {
  background: #2563EB !important;
  border-radius: 8px !important;
  font-size: 14px !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
  background: linear-gradient(135deg, #1D4ED8 0%, #7C3AED 100%) !important;
  border-radius: 8px !important;
  font-size: 14px !important;
}

/* Content typography */
[data-testid="stChatMessageContent"] {
  font-size: 15px !important;
  line-height: 1.72 !important;
  color: #1C1C1C !important;
}
[data-testid="stChatMessageContent"] p {
  margin: 0 0 0.75em 0;
}
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }

/* Lists */
[data-testid="stChatMessageContent"] ul,
[data-testid="stChatMessageContent"] ol {
  margin: 0.4em 0 0.8em 1.5em;
  padding: 0;
}
[data-testid="stChatMessageContent"] li { margin-bottom: 4px; }

/* Headings */
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3 {
  font-weight: 700;
  margin: 1em 0 0.4em 0;
  color: #111827;
}

/* Inline code */
[data-testid="stChatMessageContent"] code:not(pre code) {
  background: #F1F5F9;
  color: #DC2626;
  padding: 2px 6px;
  border-radius: 5px;
  font-size: 13.5px;
  font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  border: 1px solid #E2E8F0;
}

/* Code blocks — dark theme như Lobe Chat / GitHub */
[data-testid="stChatMessageContent"] pre {
  background: #0D1117 !important;
  border: 1px solid #30363D !important;
  border-radius: 10px !important;
  padding: 16px 20px !important;
  margin: 10px 0 !important;
  overflow-x: auto;
}
[data-testid="stChatMessageContent"] pre code {
  background: transparent !important;
  color: #E6EDF3 !important;
  font-size: 13.5px !important;
  font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace !important;
  line-height: 1.6 !important;
  border: none !important;
  padding: 0 !important;
}

/* Blockquote */
[data-testid="stChatMessageContent"] blockquote {
  border-left: 3px solid #CBD5E1;
  padding-left: 14px;
  color: #64748B;
  margin: 8px 0;
}

/* Table */
[data-testid="stChatMessageContent"] table {
  width: 100%;
  border-collapse: collapse;
  margin: 10px 0;
  font-size: 14px;
}
[data-testid="stChatMessageContent"] th {
  background: #F8FAFC;
  border: 1px solid #E2E8F0;
  padding: 8px 12px;
  font-weight: 600;
  text-align: left;
}
[data-testid="stChatMessageContent"] td {
  border: 1px solid #E2E8F0;
  padding: 8px 12px;
}
[data-testid="stChatMessageContent"] tr:nth-child(even) td {
  background: #F8FAFC;
}

/* ── Chat input — Lobe Chat style ── */
div[data-testid="stChatInput"] {
  background: #FFFFFF !important;
  border-radius: 14px !important;
  border: 1.5px solid #E2E8F0 !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06), 0 0 0 0 transparent !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stChatInput"]:focus-within {
  border-color: #2563EB !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06), 0 0 0 3px rgba(37,99,235,0.10) !important;
}
div[data-testid="stChatInput"] textarea {
  font-size: 15px !important;
  line-height: 1.6 !important;
}

/* ── Expander ── */
details[data-testid="stExpander"] {
  border: 1px solid #E5E7EB;
  border-radius: 10px;
  background: #FAFBFD;
}

/* ── Main area background ── */
.main .block-container {
  background: #FFFFFF;
  border-radius: 0;
  padding: 1rem 0 0 0 !important;
  max-width: 820px;
  margin: 0 auto;
}

/* Chat messages wrapper */
.chat-messages-wrap {
  border: 1px solid #E5E7EB;
  border-radius: 14px;
  overflow: hidden;
  margin-bottom: 8px;
  background: #FFFFFF;
}

/* ── Tab bar — pill style nổi bật ── */
div[data-testid="stTabs"] [role="tablist"] {
  background: #1E293B;
  border-radius: 12px;
  padding: 5px 6px;
  gap: 4px;
  border-bottom: none !important;
  margin-bottom: 12px;
}
div[data-testid="stTabs"] button[data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  border-radius: 8px !important;
  color: #94A3B8 !important;
  font-size: 13.5px !important;
  font-weight: 500 !important;
  padding: 7px 18px !important;
  transition: all 0.15s ease !important;
  box-shadow: none !important;
}
div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
  background: #334155 !important;
  color: #E2E8F0 !important;
}
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
  background: #FFFFFF !important;
  color: #1D4ED8 !important;
  font-weight: 700 !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.18) !important;
}

/* ── Welcome cards ── */
.welcome-card {
  background: white;
  border: 1px solid #E5E7EB;
  border-radius: 12px;
  padding: 14px 16px;
  cursor: pointer;
  transition: all 0.15s;
  height: 100%;
}
.welcome-card:hover {
  border-color: #2563EB;
  box-shadow: 0 4px 12px rgba(37,99,235,0.08);
  transform: translateY(-1px);
}

/* ── Mode badge in header ── */
.mode-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
}

/* ── Metrics ── */
div[data-testid="stMetric"] {
  background: #FAFBFD;
  border: 1px solid #E5E7EB;
  border-radius: 10px;
  padding: 12px 14px;
}

/* ── Scrollbar ── */
section[data-testid="stSidebar"]::-webkit-scrollbar { width: 4px; }
section[data-testid="stSidebar"]::-webkit-scrollbar-track { background: transparent; }
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb { background: #374151; border-radius: 2px; }
</style>
"""


def _inject_css() -> None:
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def _render_rag_mode_selector(session_id: str) -> None:
    """Render RAG mode selector riêng cho từng session."""
    current = _get_session_mode(session_id)

    st.markdown(
        '<div style="font-size:10.5px;font-weight:600;letter-spacing:0.6px;'
        'text-transform:uppercase;color:#6B7280;padding:4px 2px 8px 2px;">🔍 Chế độ truy vấn</div>',
        unsafe_allow_html=True,
    )

    for key, cfg in RETRIEVAL_MODES.items():
        is_active = key == current
        # Card HTML: active = viền xanh + nền đậm hơn
        border_color = "#3B82F6" if is_active else "transparent"
        bg_color = "#1E3A5F" if is_active else "transparent"
        check_html = (
            '<span style="font-size:12px;color:#60A5FA;font-weight:700;">✓</span>'
            if is_active else ""
        )
        card_html = f"""
        <div style="
          display:flex;align-items:center;gap:10px;
          padding:9px 12px;border-radius:10px;margin-bottom:4px;
          border:1.5px solid {border_color};background:{bg_color};
          transition:all 0.15s;
        ">
          <span style="font-size:19px;">{cfg['icon']}</span>
          <div style="flex:1;">
            <div style="font-size:13px;font-weight:600;color:#F3F4F6;">{cfg['label']}</div>
            <div style="font-size:11px;color:#9CA3AF;margin-top:1px;">{cfg['short']}</div>
          </div>
          {check_html}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        if not is_active:
            if st.button(
                f"Chọn",
                key=f"mode_btn_{key}_{session_id}",
                use_container_width=True,
            ):
                _set_session_mode(session_id, key)
                st.rerun()
        else:
            # Spacer nhỏ khi đang active (không cần nút)
            st.markdown(
                '<div style="height:4px;"></div>',
                unsafe_allow_html=True,
            )

    # Mô tả đầy đủ mode đang dùng
    current_cfg = RETRIEVAL_MODES[current]
    st.markdown(
        f'<div style="font-size:11px;color:#6B7280;padding:4px 4px 2px 4px;">'
        f'{current_cfg["desc"]}</div>',
        unsafe_allow_html=True,
    )


def render_sidebar(agent: dict, session_store: SessionStore, memory_store: MemoryStore) -> None:
    with st.sidebar:
        # ── Brand header ──
        st.markdown(
            """
            <div style="
                padding:16px 12px 14px 12px;
                margin-bottom:4px;
            ">
              <div style="display:flex;align-items:center;gap:10px;">
                <div style="
                  width:36px;height:36px;border-radius:10px;
                  background:linear-gradient(135deg,#1D4ED8,#7C3AED);
                  display:flex;align-items:center;justify-content:center;
                  font-size:18px;flex-shrink:0;
                ">⚖️</div>
                <div>
                  <div style="font-size:15px;font-weight:700;color:#F9FAFB;letter-spacing:-0.2px;">
                    Legal AI Agent
                  </div>
                  <div style="font-size:11px;color:#6B7280;margin-top:1px;">
                    Tư vấn pháp luật VN
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── New chat button ──
        if st.button("✏️  Cuộc trò chuyện mới", use_container_width=True):
            new_sess = Session.new()
            session_store.save(new_sess)
            st.session_state.active_session_id = new_sess.id
            st.session_state.conv_state = ConversationState()
            st.session_state.last_answer = None
            st.session_state.last_trace = {}
            st.rerun()

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── RAG Mode Selector — per-session ──
        active_sid_for_mode = st.session_state.active_session_id
        _render_rag_mode_selector(active_sid_for_mode)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Document Upload Panel ──
        _render_doc_upload_panel(active_sid_for_mode, agent["embedder"])

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Session history (grouped by date) ──
        st.markdown(
            '<div style="font-size:10.5px;font-weight:600;letter-spacing:0.6px;'
            'text-transform:uppercase;color:#6B7280;padding:4px 2px 6px 2px;">💬 Lịch sử</div>',
            unsafe_allow_html=True,
        )

        recent = session_store.list_recent(limit=50)
        active_id = st.session_state.active_session_id
        groups = _group_sessions_by_date(recent)

        for group_name, sessions in groups.items():
            if not sessions:
                continue
            st.markdown(
                f'<div class="session-group-header">{group_name}</div>',
                unsafe_allow_html=True,
            )
            for s in sessions:
                is_active = s.id == active_id
                label = s.name[:32] + ("…" if len(s.name) > 32 else "")
                n_turns = len(s.messages) // 2

                col_main, col_del = st.columns([11, 1])
                with col_main:
                    # Thêm class active cho session đang chọn
                    if is_active:
                        st.markdown(
                            '<div class="session-item-active">',
                            unsafe_allow_html=True,
                        )
                    btn_label = f"{'▶ ' if is_active else ''}{label}"
                    if st.button(
                        btn_label,
                        key=f"sess_{s.id}",
                        use_container_width=True,
                        help=f"{n_turns} lượt · {s.updated_at[:10]}",
                    ):
                        if not is_active:
                            st.session_state.active_session_id = s.id
                            st.session_state.conv_state = ConversationState()
                            st.session_state.last_answer = None
                            st.session_state.last_trace = {}
                            st.rerun()
                    if is_active:
                        st.markdown("</div>", unsafe_allow_html=True)

                with col_del:
                    if len(recent) > 1 and st.button(
                        "×", key=f"del_{s.id}",
                        help="Xoá phiên",
                    ):
                        session_store.delete(s.id)
                        delete_memory_file(s.id)
                        get_memory_store_for.clear()
                        if is_active:
                            st.session_state.active_session_id = next(
                                x.id for x in recent if x.id != s.id
                            )
                            st.session_state.conv_state = ConversationState()
                            st.session_state.last_answer = None
                        st.rerun()

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Advanced Settings (collapsed) ──
        with st.expander("⚙️ Cài đặt nâng cao", expanded=False):
            st.markdown(
                '<div style="font-size:12px;font-weight:600;color:#374151;margin-bottom:6px;">'
                '🧠 Pipeline</div>',
                unsafe_allow_html=True,
            )
            st.session_state.use_planner = st.toggle(
                "Planner (multi-step)", value=st.session_state.use_planner,
                help="Lập kế hoạch nhiều bước cho câu hỏi phức tạp",
            )
            st.session_state.use_guardrails = st.toggle(
                "Guardrails (disclaimer)", value=st.session_state.use_guardrails,
                help="Tự động thêm disclaimer pháp lý",
            )
            st.session_state.use_memory_extract = st.toggle(
                "Auto-extract memory", value=st.session_state.use_memory_extract,
                help="Tự động trích xuất thông tin người dùng vào memory",
            )

            st.markdown(
                '<div style="font-size:12px;font-weight:600;color:#374151;margin:10px 0 6px 0;">'
                '📊 Tham số retrieval</div>',
                unsafe_allow_html=True,
            )
            st.session_state.top_k = st.slider("Top-K chunks", 1, 15, value=st.session_state.top_k)
            st.session_state.min_score = st.slider(
                "Min score", 0.0, 1.0, value=float(st.session_state.min_score), step=0.05,
            )

        # ── Memory per-session ──
        facts = memory_store.all()
        with st.expander(f"🧠 Memory ({len(facts)} facts)", expanded=False):
            st.caption("Chỉ áp dụng cho phiên hiện tại")
            if facts:
                for m in facts:
                    c1, c2 = st.columns([9, 1])
                    c1.markdown(f"<small>{m.content[:80]}</small>", unsafe_allow_html=True)
                    if c2.button("×", key=f"forget_{m.id}"):
                        memory_store.remove(m.id)
                        st.rerun()
            else:
                st.caption("(chưa có facts)")
            new_fact = st.text_input(
                "Thêm fact", placeholder="VD: tôi là sinh viên...",
                label_visibility="collapsed", key="new_fact_input",
            )
            if st.button("Lưu fact", use_container_width=True) and new_fact.strip():
                memory_store.add(new_fact.strip(), source_session=active_id)
                st.rerun()

        # ── System info ──
        with st.expander("ℹ️ Hệ thống", expanded=False):
            st.markdown(
                f"""<small>
                **LLM**: `{config.LLM_MODEL}`<br>
                **Embedding**: `{config.EMBEDDING_MODEL.split('/')[-1]}`<br>
                **Chunks**: {agent['vstore'].count():,}<br>
                **BM25**: {agent['bm25'].count() if agent['bm25'] else '—'}<br>
                **KG**: {'✓ Online' if agent.get('kg_retriever') else '✗ Offline'}<br>
                **Top dataset**: {len(agent.get('top15_urls', []))} luật
                </small>""",
                unsafe_allow_html=True,
            )

        # ── Xoá hội thoại ──
        st.markdown("<hr/>", unsafe_allow_html=True)
        if st.button("🗑️  Xoá hội thoại hiện tại", use_container_width=True):
            sess = load_active_session(session_store)
            sess.messages.clear()
            sess.summary = ""
            sess.summary_until = 0
            session_store.save(sess)
            st.session_state.conv_state = ConversationState()
            st.session_state.last_answer = None
            st.session_state.last_trace = {}
            st.rerun()


# ─── Sample questions ─────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    {"icon": "🚦", "category": "Giao thông",
     "q": "Vượt đèn đỏ với xe máy bị phạt bao nhiêu tiền?"},
    {"icon": "⚖️", "category": "Hình sự",
     "q": "Trộm cắp tài sản dưới 2 triệu đồng có bị truy tố hình sự không?"},
    {"icon": "💼", "category": "Lao động",
     "q": "Người lao động bị sa thải trái pháp luật có được bồi thường không?"},
    {"icon": "🏠", "category": "Đất đai",
     "q": "Thủ tục cấp Giấy chứng nhận quyền sử dụng đất lần đầu gồm những bước nào?"},
    {"icon": "💰", "category": "Thuế",
     "q": "Mức thuế thu nhập cá nhân hiện nay là bao nhiêu?"},
    {"icon": "🏢", "category": "Doanh nghiệp",
     "q": "Vốn điều lệ tối thiểu để thành lập công ty TNHH là bao nhiêu?"},
]


# ─── Welcome screen ───────────────────────────────────────────────────────────

def _render_welcome(agent: dict) -> None:
    """Hero + stats + sample question grid."""
    mode_key = _get_session_mode(st.session_state.active_session_id)
    mode_cfg = RETRIEVAL_MODES[mode_key]

    # Hero banner
    st.markdown(
        f"""
        <div style="
            padding:32px 28px;
            background:linear-gradient(135deg,#0F172A 0%,#1E3A5F 50%,#1E1B4B 100%);
            border-radius:16px;margin-bottom:20px;color:white;
            box-shadow:0 8px 32px rgba(0,0,0,0.12);
            position:relative;overflow:hidden;
        ">
          <div style="
            position:absolute;right:-10px;top:-10px;
            font-size:120px;opacity:0.06;line-height:1;
          ">⚖️</div>
          <h2 style="margin:0 0 6px 0;color:white;font-weight:700;font-size:22px;letter-spacing:-0.3px;">
            ⚖️ Legal AI Agent
          </h2>
          <p style="margin:0 0 14px 0;opacity:0.85;font-size:14px;line-height:1.6;max-width:600px;">
            Trợ lý tư vấn pháp luật Việt Nam · Trả lời kèm trích dẫn văn bản pháp luật chính thức
          </p>
          <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
            <span style="
              background:rgba(255,255,255,0.12);backdrop-filter:blur(8px);
              padding:4px 12px;border-radius:20px;font-size:12px;font-weight:500;
              border:1px solid rgba(255,255,255,0.15);
            ">🇻🇳 Pháp luật Việt Nam</span>
            <span style="
              background:rgba(255,255,255,0.12);backdrop-filter:blur(8px);
              padding:4px 12px;border-radius:20px;font-size:12px;font-weight:500;
              border:1px solid rgba(255,255,255,0.15);
            ">🔍 Hybrid Retrieval</span>
            <span style="
              background:{mode_cfg['color']}33;backdrop-filter:blur(8px);
              padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;
              border:1px solid {mode_cfg['color']}55;color:white;
            ">{mode_cfg['icon']} {mode_cfg['label']}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stats row
    n_chunks = agent["vstore"].count()
    n_bm25 = agent["bm25"].count() if agent.get("bm25") else 0
    n_laws = len(agent.get("top15_urls", []))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📚 Chunks pháp lý", f"{n_chunks:,}")
    with c2:
        st.metric("🔎 BM25 index", f"{n_bm25:,}")
    with c3:
        st.metric("🕸️ Knowledge Graph", "Online" if _KG_AVAILABLE else "Offline")
    with c4:
        st.metric("📑 Top dataset", f"{n_laws} luật")

    st.markdown("<br>", unsafe_allow_html=True)

    # Sample questions
    st.markdown(
        """
        <div style="margin:0 0 12px 0;">
          <div style="font-size:15px;font-weight:600;color:#111827;">💡 Bắt đầu với câu hỏi mẫu</div>
          <div style="font-size:13px;color:#6B7280;margin-top:2px;">
            Chọn một câu hỏi bên dưới để thử ngay
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for row_start in (0, 3):
        cols = st.columns(3)
        for col, item in zip(cols, SAMPLE_QUESTIONS[row_start:row_start + 3]):
            with col:
                if st.button(
                    f"{item['icon']} **{item['category']}**\n\n{item['q']}",
                    key=f"sample_{row_start}_{item['category']}",
                    use_container_width=True,
                ):
                    st.session_state.pending_sample_q = item["q"]
                    st.rerun()


# ─── Chat message rendering — Lobe Chat / Open WebUI style ──────────────────

def _render_messages(session: Session) -> None:
    """Render messages dùng st.chat_message (full markdown) kiểu Lobe Chat."""
    last_answer = st.session_state.last_answer
    last_trace = st.session_state.last_trace

    for i, msg in enumerate(session.messages):
        role = msg["role"]
        content = msg["content"]
        is_last_assistant = (
            role == "assistant"
            and i == len(session.messages) - 1
            and last_answer is not None
        )

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="⚖️"):
                st.markdown(content)
                if is_last_assistant:
                    render_answer_extras(last_answer, last_trace)


def render_answer_extras(answer: Answer, trace: dict) -> None:
    """Citations + pipeline trace dưới mỗi câu trả lời."""
    if answer.citations:
        n_doc = sum(1 for c in answer.citations if c.source.startswith(DOC_SOURCE_PREFIX))
        n_corpus = len(answer.citations) - n_doc
        label_parts = []
        if n_corpus:
            label_parts.append(f"⚖️ {n_corpus} pháp luật")
        if n_doc:
            label_parts.append(f"📎 {n_doc} tài liệu bạn")
        expander_label = "📚 Nguồn: " + " · ".join(label_parts) if label_parts else f"📚 Nguồn ({len(answer.citations)})"

        with st.expander(expander_label, expanded=False):
            for i, cit in enumerate(answer.citations, 1):
                is_doc = cit.source.startswith(DOC_SOURCE_PREFIX)
                tag_parts = [p for p in [cit.article, cit.clause, cit.point] if p]
                tag = " · ".join(tag_parts) or ("tài liệu bạn" if is_doc else "preamble")

                if is_doc:
                    # Nguồn từ file upload → viền xanh lá
                    source_clean = cit.source.replace(DOC_SOURCE_PREFIX, "").strip()
                    border_color = "#059669"
                    bg = "#F0FDF4"
                    num_bg = "#059669"
                    tag_bg = "#DCFCE7"
                    tag_color = "#059669"
                    tag_border = "#BBF7D0"
                else:
                    # Nguồn từ corpus pháp luật → viền xanh dương
                    source_clean = cit.source.replace(".txt", "").replace(".pdf", "")
                    source_clean = source_clean.split("/")[-1] if "/" in source_clean else source_clean
                    border_color = "#2563EB"
                    bg = "#F8FAFC"
                    num_bg = "#2563EB"
                    tag_bg = "#EFF6FF"
                    tag_color = "#1D4ED8"
                    tag_border = "#BFDBFE"

                snippet = cit.snippet.replace("\n", " ")
                snippet_short = snippet[:300] + ("…" if len(snippet) > 300 else "")
                prefix_icon = "📎 " if is_doc else ""

                st.markdown(
                    f"""
                    <div style="
                        padding:12px 14px;
                        background:{bg};
                        border:1px solid #E5E7EB;border-left:4px solid {border_color};
                        border-radius:10px;margin-bottom:8px;
                        box-shadow:0 1px 3px rgba(0,0,0,0.04);
                    ">
                      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                        <div style="
                          background:{num_bg};color:white;width:22px;height:22px;
                          border-radius:50%;display:flex;align-items:center;justify-content:center;
                          font-size:11px;font-weight:700;flex-shrink:0;
                        ">{i}</div>
                        <div style="font-size:13px;color:{border_color};font-weight:600;flex:1;">
                          {prefix_icon}{source_clean}
                        </div>
                        <span style="
                          background:{tag_bg};color:{tag_color};border:1px solid {tag_border};
                          padding:2px 8px;border-radius:10px;font-size:11px;font-weight:500;
                        ">{tag}</span>
                      </div>
                      <div style="font-size:13px;color:#4B5563;line-height:1.55;margin-left:30px;">
                        {snippet_short}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if trace:
        with st.expander("🔍 Chi tiết pipeline", expanded=False):
            mode_label = trace.get("retrieval_mode", "—")
            st.markdown(
                f'<span style="background:#F3F4F6;padding:3px 8px;border-radius:6px;'
                f'font-size:12px;">Mode: **{mode_label}**</span>',
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Intent", trace.get("intent", "—"))
            c2.metric("Action", trace.get("action", "—"))
            c3.metric("Contexts", trace.get("n_contexts", 0))

            if trace.get("search_query"):
                st.markdown(f"**🔎 Search query:** `{trace['search_query']}`")
            if trace.get("tool_name"):
                st.markdown(f"**🛠️ Tool:** `{trace['tool_name']}`")
            if trace.get("plan_steps"):
                st.markdown("**📋 Plan steps:**")
                for s in trace["plan_steps"]:
                    st.markdown(f"- {s}")
            if trace.get("new_memory"):
                st.success(f"💡 Memory mới: {trace['new_memory']}")


# ─── Main chat area ───────────────────────────────────────────────────────────

def render_chat(agent: dict, session_store: SessionStore, memory_store: MemoryStore) -> None:
    session = load_active_session(session_store)
    is_empty = len(session.messages) == 0

    active_id = st.session_state.active_session_id
    mode_key = _get_session_mode(active_id)
    mode_cfg = RETRIEVAL_MODES[mode_key]
    doc_mode_key = _get_doc_mode(active_id)
    doc_mode_cfg = DOC_SOURCE_MODES[doc_mode_key]
    doc_store = _get_doc_store(active_id, agent["embedder"])

    # ── Compact header ──
    doc_badge = ""
    if doc_mode_key != "corpus_only":
        doc_badge = (
            f'<span style="background:#ECFDF5;color:#059669;border:1px solid #6EE7B7;'
            f'padding:3px 9px;border-radius:12px;font-size:11px;font-weight:600;margin-left:6px;">'
            f'{doc_mode_cfg["icon"]} {doc_mode_cfg["label"]}</span>'
        )
    st.markdown(
        f"""
        <div style="
            display:flex;justify-content:space-between;align-items:center;
            padding:10px 14px;background:#FAFBFD;border:1px solid #E5E7EB;
            border-radius:12px;margin-bottom:12px;
        ">
          <div>
            <span style="font-size:15px;font-weight:600;color:#111827;">{session.name}</span>
            <span style="font-size:11px;color:#9CA3AF;margin-left:10px;">
              {len(session.messages)//2} lượt
            </span>
            {doc_badge}
          </div>
          <span style="
            background:{mode_cfg['bg']};color:{mode_cfg['color']};
            border:1px solid {mode_cfg['border']};
            padding:4px 12px;border-radius:16px;font-size:12px;font-weight:600;
          ">{mode_cfg['icon']} {mode_cfg['label']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Messages hoặc welcome screen ──
    if is_empty:
        _render_welcome(agent)
    else:
        st.markdown('<div class="chat-messages-wrap">', unsafe_allow_html=True)
        _render_messages(session)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Chat input ──
    pending_q = st.session_state.pop("pending_sample_q", None)
    user_input = st.chat_input("Hỏi về pháp luật Việt Nam...")
    if pending_q and not user_input:
        user_input = pending_q

    if user_input:
        process_question(user_input, agent, session, session_store, memory_store)
        st.rerun()


# ─── Pipeline helpers ────────────────────────────────────────────────────────

def _build_contexts(
    query: str,
    top_k: int,
    min_score,
    use_kg: bool,
    allowed_sources,
    retriever,
    doc_store: UploadedDocStore,
    doc_mode: str,
    agent: dict,
) -> list:
    """Xây dựng contexts theo doc_mode: corpus / docs / combined."""
    if doc_mode == "docs_only":
        if doc_store.is_empty():
            return []
        contexts = doc_store.retrieve(query, top_k=top_k)
        return contexts

    elif doc_mode == "combined" and not doc_store.is_empty():
        corpus_k = max(top_k - 3, 3)
        doc_k = min(top_k, 4)
        corpus_ctx = retriever.retrieve(
            query, top_k=corpus_k, min_score=min_score,
            use_kg=use_kg, allowed_sources=allowed_sources,
        )
        doc_ctx = doc_store.retrieve(query, top_k=doc_k)
        return (corpus_ctx + doc_ctx)[:top_k]

    else:  # corpus_only (hoặc combined khi doc_store rỗng)
        return retriever.retrieve(
            query, top_k=top_k, min_score=min_score,
            use_kg=use_kg, allowed_sources=allowed_sources,
        )


# ─── Pipeline ────────────────────────────────────────────────────────────────

def process_question(
    user_input: str,
    agent: dict,
    session: Session,
    session_store: SessionStore,
    memory_store: MemoryStore,
) -> None:
    conv_state: ConversationState = st.session_state.conv_state
    retriever = agent["retriever"]
    generator = agent["generator"]
    router = agent["router"]
    planner = agent["planner"]
    tool_registry = agent["tool_registry"]

    top_k = st.session_state.top_k
    min_score = st.session_state.min_score if st.session_state.min_score > 0 else None

    mode_key = _get_session_mode(session.id)
    mode_cfg = RETRIEVAL_MODES[mode_key]
    use_kg = mode_cfg["use_kg"]
    allowed_sources = agent.get("top15_urls", []) if mode_cfg["use_top10_filter"] else None

    # ── Doc source mode ──────────────────────────────────────────────────────
    doc_mode = _get_doc_mode(session.id)
    doc_store = _get_doc_store(session.id, agent["embedder"])

    trace: dict = {
        "retrieval_mode": mode_cfg["label"],
        "doc_mode": DOC_SOURCE_MODES[doc_mode]["label"],
    }

    with st.spinner("Đang xử lý..."):
        conv_state.update_from_question(user_input)

        raw_messages = session.messages[session.summary_until:]
        llm_history = raw_messages[-MAX_HISTORY_MESSAGES:]
        memory_text = memory_store.format_for_prompt()
        summary_text = session.summary

        # 1) Router
        decision = router.route(
            user_input,
            history=llm_history,
            memory_text=memory_text,
            summary_text=summary_text,
            state=conv_state,
        )
        trace["intent"] = decision.intent
        trace["action"] = decision.action
        trace["search_query"] = decision.search_query

        # Flow A: answer_direct
        if decision.action == "answer_direct":
            answer = Answer(
                question=user_input,
                answer=decision.direct_response or "",
                citations=[],
            )
            trace["n_contexts"] = 0

        # Flow B: use_tool
        elif decision.action == "use_tool" and decision.tool_name:
            tool_name = decision.tool_name
            tool_query = decision.tool_query or user_input
            trace["tool_name"] = tool_name

            if tool_name == "calculate_fine":
                tool_result = tool_registry.calculate_fine(description=tool_query)

            elif tool_name == "draft_document":
                if "|" in tool_query:
                    dt, det = tool_query.split("|", 1)
                else:
                    dt, det = "văn bản pháp lý", tool_query
                tool_result = tool_registry.draft_document(
                    doc_type=dt.strip(), details=det.strip(),
                )

            elif tool_name == "compare_regulations":
                if "|" in tool_query:
                    topic_a, topic_b = tool_query.split("|", 1)
                else:
                    parts = tool_query.split(" và " if " và " in tool_query else " vs ")
                    topic_a = parts[0].strip()
                    topic_b = parts[1].strip() if len(parts) > 1 else user_input
                tool_result = tool_registry.compare_regulations(
                    topic_a=topic_a.strip(), topic_b=topic_b.strip(),
                )

            elif tool_name == "validate_document":
                # Ưu tiên lấy nội dung file đã upload nếu có
                doc_text = tool_query
                doc_fname = ""
                if not doc_store.is_empty():
                    # Lấy toàn bộ text từ doc_store dưới dạng string để validate
                    all_chunks = doc_store.retrieve(user_input, top_k=20)
                    if all_chunks:
                        doc_text = "\n\n".join(c.chunk.text for c in all_chunks)
                        doc_fname = all_chunks[0].chunk.metadata.source
                tool_result = tool_registry.validate_document(
                    document_text=doc_text, filename=doc_fname,
                )

            elif tool_name == "knowledge_graph_lookup":
                tool_result = tool_registry.knowledge_graph_lookup(query=tool_query)

            else:
                tool_result = tool_registry.execute(tool_name, query=tool_query)

            # validate & compare: kết quả đã đầy đủ, không cần generate thêm
            if tool_name in ("validate_document", "compare_regulations") and tool_result.success:
                answer = Answer(
                    question=user_input,
                    answer=tool_result.result,
                    citations=[],
                )
                trace["n_contexts"] = 0
            else:
                search_q = decision.search_query or user_input
                contexts = _build_contexts(
                    search_q, top_k, min_score, use_kg, allowed_sources,
                    retriever, doc_store, doc_mode, agent,
                )
                trace["n_contexts"] = len(contexts)

                answer = generator.generate(
                    user_input, contexts,
                    history=llm_history,
                    memory_text=memory_text,
                    summary_text=summary_text,
                    tool_results=[tool_result],
                    state_context=conv_state.to_context_string(),
                )

        # Flow C: retrieve (RAG + optional Planner)
        else:
            search_query = decision.search_query or user_input
            tool_results: list = []

            if st.session_state.use_planner and doc_mode != "docs_only":
                plan = planner.create_plan(user_input, state=conv_state)
                if plan.is_complex():
                    trace["plan_steps"] = [
                        f"{s.tool_name}: {s.tool_input}" for s in plan.tool_steps()
                    ]
                    tool_results = planner.execute_plan(plan)
                    search_query = plan.retrieve_query()

            contexts = _build_contexts(
                search_query, top_k, min_score, use_kg, allowed_sources,
                retriever, doc_store, doc_mode, agent,
            )
            trace["n_contexts"] = len(contexts)
            trace["search_query"] = search_query

            no_corpus_fallback = (
                "Không tìm thấy nội dung phù hợp trong tài liệu bạn upload. "
                "Hãy kiểm tra lại nội dung file hoặc thử câu hỏi khác."
                if doc_mode == "docs_only" else
                "Không tìm thấy căn cứ pháp lý trong cơ sở dữ liệu. "
                "Vui lòng thử hỏi theo cách khác hoặc tham khảo ý kiến luật sư."
            )

            if not contexts and not tool_results:
                answer = Answer(
                    question=user_input,
                    answer=no_corpus_fallback,
                    citations=[],
                )
            else:
                answer = generator.generate(
                    user_input, contexts,
                    history=llm_history,
                    memory_text=memory_text,
                    summary_text=summary_text,
                    tool_results=tool_results if tool_results else None,
                    state_context=conv_state.to_context_string(),
                )
                if st.session_state.use_guardrails and doc_mode != "docs_only":
                    answer = apply_guardrails(answer, contexts)

        # Post-processing
        retrieved_sources = [c.source for c in answer.citations]
        conv_state.update_from_answer(answer.answer, retrieved_sources)

        session.append("user", user_input)
        session.append("assistant", answer.answer)
        if session.name == "Phiên mới" and len(session.messages) == 2:
            session.name = user_input[:AUTO_NAME_MAXLEN]
        session_store.save(session)

        if maybe_update_summary(session, generator):
            session_store.save(session)

        if st.session_state.use_memory_extract and decision.action != "answer_direct":
            new_facts = generator.extract_memory_facts(user_input, answer.answer)
            added = []
            for f in new_facts:
                fact = memory_store.add(f, source_session=session.id)
                if fact is not None:
                    added.append(fact.content)
            if added:
                trace["new_memory"] = "; ".join(added)[:150]

    st.session_state.last_answer = answer
    st.session_state.last_trace = trace


# ─── KG Explorer ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _cached_kg_stats() -> dict:
    return kgq.get_stats()


def _render_kg_stats() -> None:
    try:
        stats = _cached_kg_stats()
    except Exception as exc:
        st.error(f"❌ Không kết nối được Neo4j: {exc}")
        st.caption("Kiểm tra NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD trong .env")
        return

    nodes = stats.get("nodes", {})
    rels = stats.get("relations", {})

    st.markdown("##### 📊 Stats KG hiện tại")
    cols = st.columns(max(len(nodes), 1))
    for col, (label, n) in zip(cols, nodes.items()):
        col.metric(label, f"{n:,}")

    if rels:
        with st.expander(f"🔗 {sum(rels.values()):,} edges", expanded=False):
            for rtype, n in rels.items():
                st.markdown(f"- `-[:{rtype}]->` × **{n:,}**")


def _render_query_templates() -> None:
    st.markdown("##### 🎯 Câu hỏi mẫu")
    template_labels = {f"{k}: {v.name}": k for k, v in kgq.PREDEFINED_QUERIES.items()}
    chosen_label = st.selectbox("Chọn câu hỏi", list(template_labels.keys()), label_visibility="collapsed")
    chosen_key = template_labels[chosen_label]
    template = kgq.PREDEFINED_QUERIES[chosen_key]
    st.caption(template.description)

    params: dict = {}
    if template.needs_params():
        cols = st.columns(len(template.params_schema))
        for col, (pname, phelp) in zip(cols, template.params_schema.items()):
            params[pname] = col.text_input(pname, key=f"param_{chosen_key}_{pname}", help=phelp)

    if st.button("Chạy query", type="primary", key=f"run_{chosen_key}"):
        try:
            with st.spinner("Đang query Neo4j..."):
                rows = kgq.execute_cypher(template.cypher, params)
            if rows:
                st.success(f"Trả về {len(rows)} dòng")
                st.dataframe(rows, use_container_width=True, height=400)
            else:
                st.info("Không có kết quả")
        except Exception as exc:
            st.error(f"❌ Lỗi: {exc}")

    with st.expander("📋 Xem Cypher", expanded=False):
        st.code(template.cypher.strip(), language="cypher")


def _render_custom_cypher() -> None:
    with st.expander("⚙️ Custom Cypher (advanced)", expanded=False):
        st.caption("Read-only sandbox")
        default_q = "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC"
        cypher = st.text_area("Cypher", value=default_q, height=120, key="custom_cypher")
        if st.button("Execute", key="run_custom"):
            try:
                with st.spinner("Executing..."):
                    rows = kgq.execute_cypher(cypher)
                st.success(f"Trả về {len(rows)} dòng")
                st.dataframe(rows, use_container_width=True)
            except Exception as exc:
                st.error(f"❌ {exc}")


def _render_graph_visualization() -> None:
    with st.expander("🕸 Visualization graph (pyvis)", expanded=False):
        col1, col2, col3 = st.columns([2, 1, 1])
        center_label = col1.selectbox("Loại node center", ["Law", "Article", "Offense", "Subject"], index=2)
        center_val = col1.text_input("Tên / số hiệu / id", placeholder="vd: 'trộm cắp tài sản'")
        depth = col2.slider("Độ sâu", 1, 3, value=2)
        limit = col3.slider("Max edges", 10, 200, value=50)

        if center_val and st.button("Vẽ graph", key="draw_graph"):
            try:
                from pyvis.network import Network
            except ImportError:
                st.warning("Cài `pyvis` để xem visualization: `pip install pyvis`")
                return

            prop_field = "name" if center_label in ("Offense", "Subject") else (
                "doc_number" if center_label == "Law" else "id"
            )
            try:
                edges = kgq.get_subgraph_around(
                    center_label, prop_field, center_val, depth=depth, limit=limit,
                )
            except Exception as exc:
                st.error(f"❌ Query lỗi: {exc}")
                return

            if not edges:
                st.info("Không tìm thấy node hoặc subgraph rỗng.")
                return

            net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="#333",
                          directed=True, notebook=False)
            net.barnes_hut()

            color_map = {
                "Law": "#4F81BD", "Article": "#9BBB59", "Clause": "#C0504D",
                "Offense": "#E67E22", "Penalty": "#8E44AD", "Subject": "#16A085",
            }

            added_nodes: set[str] = set()
            for e in edges:
                src, src_lbl = e["source"], e["src_label"]
                tgt, tgt_lbl = e["target"], e["tgt_label"]
                if src not in added_nodes:
                    net.add_node(src, label=str(src)[:40], title=f"{src_lbl}: {src}",
                                 color=color_map.get(src_lbl, "#999"))
                    added_nodes.add(src)
                if tgt not in added_nodes:
                    net.add_node(tgt, label=str(tgt)[:40], title=f"{tgt_lbl}: {tgt}",
                                 color=color_map.get(tgt_lbl, "#999"))
                    added_nodes.add(tgt)
                net.add_edge(src, tgt, label=e["rel_type"], title=e["rel_type"])

            html = net.generate_html(notebook=False)
            st.components.v1.html(html, height=520)
            st.caption(f"{len(added_nodes)} nodes, {len(edges)} edges")


def render_kg_explorer() -> None:
    if not _KG_AVAILABLE:
        st.error(f"❌ KG module không load được: {_kg_import_error}")
        return

    st.markdown("### 🕸 Knowledge Graph Explorer")
    st.caption("Truy vấn trực tiếp Neo4j KG. Phase 0 (structural) cho 609 luật. Phase 1 (semantic) cho top 15 luật.")

    _render_kg_stats()
    st.divider()
    _render_query_templates()
    st.divider()
    _render_custom_cypher()
    _render_graph_visualization()


# ─── Compare tab ──────────────────────────────────────────────────────────────

def _chunk_summary(chunk, max_len: int = 220) -> tuple:
    src = chunk.metadata.source.split("/")[-1].replace(".txt", "")
    tag_parts = [p for p in [chunk.article, chunk.clause] if p]
    tag = " · ".join(tag_parts) or "(preamble)"
    text = chunk.text.replace("\n", " ").strip()
    snippet = text[:max_len] + ("…" if len(text) > max_len else "")
    return src, tag, snippet


def render_compare_tab(agent: dict) -> None:
    st.markdown("### 🆚 So sánh RAG vs Graph-RAG")
    st.caption("Cùng 1 câu hỏi, chạy retrieval 2 cấu hình rồi so sánh top-K chunks và overlap.")

    col_q, col_k = st.columns([5, 1])
    query = col_q.text_input(
        "Câu hỏi để so sánh",
        placeholder="VD: Hành vi cướp tài sản bị phạt như thế nào?",
        key="compare_query",
    )
    top_k = col_k.number_input("Top-K", min_value=1, max_value=10, value=5, key="compare_topk")

    if not st.button("🚀 So sánh", type="primary", disabled=not query.strip()):
        st.info("💡 Nhập câu hỏi rồi bấm **So sánh** để xem RAG-only vs Graph-RAG retrieve những chunks nào.")
        return

    from src.retriever import Retriever
    rag_only = Retriever(agent["embedder"], agent["vstore"], bm25=agent.get("bm25"), kg_retriever=None)
    graph_rag = agent["retriever"]

    import time
    with st.spinner("Retrieve RAG-only…"):
        t0 = time.time()
        rag_chunks = rag_only.retrieve(query, top_k=top_k)
        rag_latency = time.time() - t0

    with st.spinner("Retrieve Graph-RAG…"):
        t0 = time.time()
        gr_chunks = graph_rag.retrieve(query, top_k=top_k)
        gr_latency = time.time() - t0

    kg_hits = []
    if _KG_AVAILABLE and agent.get("retriever").kg_retriever:
        try:
            kg_hits = agent["retriever"].kg_retriever.retrieve(query, top_k=top_k * 2)
        except Exception:
            kg_hits = []

    rag_ids = {c.chunk.chunk_id for c in rag_chunks}
    gr_ids = {c.chunk.chunk_id for c in gr_chunks}
    shared = rag_ids & gr_ids
    only_rag = rag_ids - gr_ids
    only_gr = gr_ids - rag_ids

    st.markdown("##### 📊 Số liệu so sánh")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RAG latency", f"{rag_latency*1000:.0f} ms")
    m2.metric("Graph-RAG latency", f"{gr_latency*1000:.0f} ms",
              delta=f"{(gr_latency - rag_latency)*1000:+.0f} ms")
    m3.metric("Chunks chung", f"{len(shared)}/{top_k}")
    m4.metric("Chỉ RAG có", len(only_rag))
    m5.metric("Chỉ Graph-RAG có", len(only_gr))

    if kg_hits:
        with st.expander(f"🕸 KG branch: {len(kg_hits)} Article hits", expanded=False):
            for h in kg_hits[:8]:
                via_color = {"via_offense": "#E67E22", "via_article_title": "#9BBB59",
                             "via_clause_text": "#C0504D"}.get(h.reason, "#999")
                st.markdown(
                    f'<span style="background:{via_color};color:white;padding:2px 6px;'
                    f'border-radius:3px;font-size:11px;">{h.reason}</span> '
                    f'**{h.article_label}** (score {h.score:.2f}) — {h.matched_text[:100]}',
                    unsafe_allow_html=True,
                )

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🔵 RAG-only (Vector + BM25)")
        for i, r in enumerate(rag_chunks, 1):
            src, tag, snippet = _chunk_summary(r.chunk)
            badge = "🟢" if r.chunk.chunk_id in shared else "🔵"
            st.markdown(
                f"""<div style="padding:10px;background:#F0F4FA;border-left:3px solid #2563EB;
                border-radius:6px;margin-bottom:8px;">
                <div style="font-size:12px;color:#2563EB;font-weight:600;">
                  {badge} [{i}] {src} · {tag} · <span style="color:#9CA3AF;">score {r.score:.4f}</span>
                </div>
                <div style="font-size:13px;margin-top:4px;color:#374151;">{snippet}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with col_b:
        st.markdown("#### 🟣 Graph-RAG (+ KG)")
        for i, r in enumerate(gr_chunks, 1):
            src, tag, snippet = _chunk_summary(r.chunk)
            badge = "🟢" if r.chunk.chunk_id in shared else "🟣"
            st.markdown(
                f"""<div style="padding:10px;background:#F4F0FA;border-left:3px solid #7C3AED;
                border-radius:6px;margin-bottom:8px;">
                <div style="font-size:12px;color:#7C3AED;font-weight:600;">
                  {badge} [{i}] {src} · {tag} · <span style="color:#9CA3AF;">score {r.score:.4f}</span>
                </div>
                <div style="font-size:13px;margin-top:4px;color:#374151;">{snippet}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.caption("🟢 chunk chung · 🔵 chỉ RAG · 🟣 chỉ Graph-RAG (KG bonus)")


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()

    try:
        agent = init_agent()
    except Exception as exc:
        st.error(f"❌ Không khởi tạo được agent: {exc}")
        st.info(
            "Kiểm tra:\n"
            "1. Vectorstore đã ingest chưa (`python -m scripts.ingest`)\n"
            "2. Ollama server đã chạy chưa (`ollama serve`)\n"
            "3. Model LLM đã pull chưa (`ollama pull mistral`)"
        )
        st.stop()

    if agent["vstore"].count() == 0:
        st.error("⚠️ Vectorstore trống. Chạy `python -m scripts.ingest --reset` trước.")
        st.stop()

    session_store = get_session_store()
    init_session_state(session_store)

    active_sid = st.session_state.active_session_id
    memory_store = get_memory_store_for(active_sid)

    render_sidebar(agent, session_store, memory_store)

    tab_chat, tab_compare, tab_kg = st.tabs([
        "💬 Chat",
        "🆚 So sánh RAG / Graph-RAG",
        "🕸 Knowledge Graph",
    ])
    with tab_chat:
        render_chat(agent, session_store, memory_store)
    with tab_compare:
        render_compare_tab(agent)
    with tab_kg:
        render_kg_explorer()


if __name__ == "__main__":
    main()

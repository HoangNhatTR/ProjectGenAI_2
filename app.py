"""Legal AI Agent — CLI chat đa lượt với RAG + Tools + Planner + Guardrails.

Pipeline mỗi lượt (Big Update Architecture):
  user input
    → ConversationState.update_from_question()
    → SmartRouter (intent: legal/consulting/compare/calculate/draft/followup/...)
    → if answer_direct  → trả thẳng
    → if use_tool       → LegalToolRegistry.execute() → Generator.generate(tool_results)
    → if retrieve       → LegalPlanner.create_plan()
                          → complex: LegalToolRegistry.execute() (calculate/draft)
                          → Retriever.retrieve() → Reranker.rerank()
                          → Generator.generate(tool_results, contexts)
    → Guardrails.apply_guardrails()
    → ConversationState.update_from_answer()
    → Session.append() → rolling_summary → extract_memory_facts()

Cách chạy:
    python app.py
    python app.py --new
    python app.py --session "tên/id"
    python app.py --no-planner          # tắt planner (chỉ dùng RAG thuần)
    python app.py --no-guardrails       # tắt guardrails disclaimer
    python app.py --no-memory-extract   # tắt auto-extract memory
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
from pathlib import Path
from typing import Optional

# Ép UTF-8 cho stdin/stdout để in tiếng Việt OK trên Windows console
for _stream in (sys.stdout, sys.stdin, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.bm25_index import BM25Index
from src.cache import RetrievalCache
from src.embedding import Embedder
from src.generator import Generator
from src.guardrails import apply_guardrails, check_answer_quality
from src.parent_store import ParentStore
from src.reranker import rerank, preload as preload_reranker
from src.memory import MemoryStore
from src.planner import LegalPlanner
from src.retriever import Retriever
from src.router import SmartRouter
from src.schemas import Answer, Citation
from src.session import Session, SessionStore
from src.state import ConversationState
from src.tools import LegalToolRegistry
from src.vectorstore import VectorStore
from src.parsing import clean_text, parse_pdf, parse_docx, parse_txt


# ─── Help text ────────────────────────────────────────────────────────────────

HELP_TEXT = """
Lệnh:
  /quit, /exit, /q       Thoát
  /clear, /reset         Xoá lịch sử + state phiên hiện tại
  /history               Xem lịch sử phiên hiện tại
  /sources, /s           Xem chi tiết nguồn câu trả lời cuối
  /state                 Xem Conversation State hiện tại
  /new [tên]             Tạo phiên chat mới
  /sessions, /ls         Liệt kê các phiên gần đây
  /load <id|tên>         Chuyển sang phiên khác
  /rename <tên>          Đổi tên phiên hiện tại
  /delete <id|tên>       Xoá một phiên (không phải phiên hiện tại)
  /remember <fact>       Ghi nhớ thông tin về bạn (xuyên session)
  /memories, /mem        Liệt kê memory đã lưu
  /forget <id|từ khoá>   Xoá memory theo id hoặc keyword
  /summary               Xem tóm tắt rolling của phiên hiện tại
  /topk N                Đặt số chunk retrieve (mặc định {TOP_K})
  /minscore X            Đặt ngưỡng cosine (mặc định 0.3, /minscore 0 để tắt)
  /file <đường dẫn>      Tải file PDF/DOCX/TXT để kiểm tra hoặc hỏi về nội dung
  /clearfile             Xoá file đang đính kèm
  /websearch [on|off]    Bật/tắt tìm kiếm web (mặc định: bật)
  /cache [clear]         Xem thống kê cache hoặc xoá cache retrieval
  /help, /?              Hiện trợ giúp
""".strip()

# Định dạng file hỗ trợ và parser tương ứng
_FILE_PARSERS: dict[str, object] = {
    ".pdf":  parse_pdf,
    ".docx": parse_docx,
    ".txt":  parse_txt,
}

# Regex nhận diện đường dẫn file trong câu nhập tự do
# Hỗ trợ Windows (C:\...) và Unix (/home/...)
_FILE_PATH_RE = re.compile(
    r'(?:"([^"]+\.(?:pdf|docx|txt))"'   # đường dẫn trong dấu ngoặc kép
    r"|'([^']+\.(?:pdf|docx|txt))'"     # đường dẫn trong dấu ngoặc đơn
    r"|([A-Za-z]:\\[^\s,;]+\.(?:pdf|docx|txt))"  # Windows path không có ngoặc
    r"|(/(?:[^\s,;]+)/[^\s,;]+\.(?:pdf|docx|txt)))",  # Unix path không có ngoặc
    re.IGNORECASE,
)

MAX_HISTORY_MESSAGES = 10
AUTO_NAME_MAXLEN     = 50

SUMMARY_MIN_MESSAGES = 12
SUMMARY_KEEP_RECENT  = 6
SUMMARY_MAX_RAW      = MAX_HISTORY_MESSAGES


# ─── Helpers UI ───────────────────────────────────────────────────────────────

def _format_tag(article: Optional[str], clause: Optional[str]) -> str:
    return " | ".join(filter(None, [article, clause])) or "(preamble)"


def show_sources(answer: Optional[Answer]) -> None:
    if answer is None or not answer.citations:
        print("  (chưa có nguồn nào)")
        return
    for i, cit in enumerate(answer.citations, 1):
        tag = _format_tag(cit.article, cit.clause)
        # Hiển thị URL gốc đầy đủ ở /sources để tra cứu được
        print(f"  [{i}] {cit.source} → {tag}")
        snippet = cit.snippet.replace("\n", " ")
        print(f"      {snippet[:180]}{'...' if len(snippet) > 180 else ''}")


def show_history(session: Session) -> None:
    if session.summary.strip():
        print(f"  [Tóm tắt cover {session.summary_until}/{len(session.messages)} msgs]:")
        print(f"     {session.summary}")
        print()
    raw = session.messages[session.summary_until:]
    if not raw:
        print("  (không có lượt nào ngoài tóm tắt)")
        return
    for m in raw:
        prefix = "U" if m["role"] == "user" else "A"
        content = m["content"].replace("\n", " ")
        print(f"  [{prefix}] {content[:140]}{'...' if len(content) > 140 else ''}")


def show_state(state: ConversationState) -> None:
    ctx = state.to_context_string()
    if ctx:
        print(ctx)
    else:
        print("  (State trống — chưa có ngữ cảnh hội thoại)")


def show_sessions(sessions: list[Session], current_id: Optional[str] = None) -> None:
    if not sessions:
        print("  (chưa có phiên nào)")
        return
    for i, s in enumerate(sessions, 1):
        marker = " *" if s.id == current_id else "  "
        n_turns = len(s.messages) // 2
        print(f"  [{i}]{marker} {s.name}")
        print(f"        id={s.id} · {n_turns} lượt · cập nhật {s.updated_at}")


def show_memories(memory_store: MemoryStore) -> None:
    facts = memory_store.all()
    if not facts:
        print("  (chưa có memory nào)")
        return
    for m in facts:
        print(f"  [{m.id}] {m.content}")
        print(f"        ghi nhớ {m.created_at}")


def pick_session_interactively(store: SessionStore) -> Session:
    recent = store.list_recent(limit=5)
    if not recent:
        print("\n(Chưa có phiên nào, sẽ tạo phiên mới)")
        return Session.new()

    print("\nCác phiên gần đây:")
    show_sessions(recent)
    print("  [n] Tạo phiên mới")
    while True:
        try:
            choice = input("Chọn (số / n / Enter để tiếp tục phiên mới nhất): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            sys.exit(0)
        if choice == "":
            return recent[0]
        if choice == "n":
            return Session.new()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(recent):
                return recent[idx]
        print("  Lựa chọn không hợp lệ, thử lại.")


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
        messages=to_summarize,
        prev_summary=session.summary,
    )
    if not new_summary or new_summary == session.summary:
        return False
    session.summary = new_summary
    session.summary_until = new_until
    return True


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Legal AI Agent — CLI chat pháp luật Việt Nam")
    p.add_argument("--session", help="Load phiên theo id hoặc tên")
    p.add_argument("--new", action="store_true", help="Tạo phiên mới")
    p.add_argument("--no-planner", action="store_true",
                   help="Tắt planner (dùng RAG thuần, tiết kiệm 1 LLM call/lượt)")
    p.add_argument("--no-guardrails", action="store_true",
                   help="Tắt guardrails disclaimer")
    p.add_argument("--no-memory-extract", action="store_true",
                   help="Tắt auto-extract memory")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 62)
    print("  LEGAL AI AGENT — TƯ VẤN PHÁP LUẬT VIỆT NAM")
    print("=" * 62)
    print(f"  Embedding   : {config.EMBEDDING_MODEL}")
    print(f"  LLM         : {config.LLM_MODEL}")
    _router_display = config.ROUTER_MODEL if config.ROUTER_MODEL != config.LLM_MODEL else f"{config.LLM_MODEL} (chung)"
    print(f"  Router LLM  : {_router_display}")
    print(f"  Planner     : {'TẮT (--no-planner)' if args.no_planner else 'BẬT'}")
    print(f"  Guardrails  : {'TẮT (--no-guardrails)' if args.no_guardrails else 'BẬT'}")
    print(f"  Web Search  : BẬT  (gõ /websearch off để tắt)")
    print()

    # ── Khởi tạo ──────────────────────────────────────────────────────────────
    print("Đang khởi tạo...")
    embedder = Embedder(config.EMBEDDING_MODEL)
    vstore   = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)

    bm25_path = config.DATA_DIR / "bm25" / "index.json"
    bm25: Optional[BM25Index] = None
    if bm25_path.exists():
        bm25 = BM25Index(bm25_path)

    # KG retriever — optional
    kg_retriever = None
    try:
        from src.kg.kg_retriever import KGRetriever
        kg_retriever = KGRetriever()
    except Exception:
        kg_retriever = None

    _parent_store = ParentStore(config.PARENT_STORE_PATH) if config.PARENT_STORE_PATH.exists() else None

    retriever = Retriever(
        embedder, vstore, bm25=bm25, kg_retriever=kg_retriever,
        parent_store=_parent_store,
    )
    _api_key = {
        "gemini":     config.GEMINI_API_KEY,
        "groq":       config.GROQ_API_KEY,
        "router9":    config.ROUTER9_API_KEY,
        "openrouter": config.OPENROUTER_API_KEY,
    }.get(config.LLM_PROVIDER)

    _llm_host = {
        "router9":    config.ROUTER9_BASE_URL,
        "openrouter": config.OPENROUTER_BASE_URL,
    }.get(config.LLM_PROVIDER, config.OLLAMA_HOST)

    generator = Generator(
        model=config.LLM_MODEL,
        host=_llm_host,
        temperature=config.LLM_TEMPERATURE,
        provider=config.LLM_PROVIDER,
        api_key=_api_key,
    )
    _router_api_key = {
        "gemini":     config.GEMINI_API_KEY,
        "groq":       config.GROQ_API_KEY,
        "router9":    config.ROUTER9_API_KEY,
        "openrouter": config.OPENROUTER_API_KEY,
    }.get(config.LLM_PROVIDER)
    _router_host = {
        "router9":    config.ROUTER9_BASE_URL,
        "openrouter": config.OPENROUTER_BASE_URL,
    }.get(config.LLM_PROVIDER, config.OLLAMA_HOST)

    router    = SmartRouter(
        model=config.ROUTER_MODEL,
        host=_router_host,
        provider=config.LLM_PROVIDER,
        api_key=_router_api_key,
    )
    sessions  = SessionStore(config.DATA_DIR)
    memory    = MemoryStore(config.DATA_DIR / "memory.json")

    # Tools + Planner (dùng chung client với Generator)
    ollama_client = generator.get_client()

    # Wire HyDE vào retriever sau khi có LLM client
    if config.USE_HYDE:
        retriever.llm_client = ollama_client
        retriever.hyde_model  = config.HYDE_MODEL
        print(f"  HyDE        : BẬT (model={config.HYDE_MODEL})")
    else:
        print("  HyDE        : TẮT")
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

    reranker_ok = preload_reranker()
    print(f"  Reranker    : {'CrossEncoder BẬT' if reranker_ok else 'rule-based fallback'}")
    _retrieval_cache = RetrievalCache(maxsize=512, ttl=3600)

    n_chunks = vstore.count()
    print(f"  Vectorstore : {n_chunks} chunks")
    if bm25 is not None:
        print(f"  BM25 hybrid : {bm25.count()} chunks indexed")
    else:
        print("  BM25        : chưa có (vector-only). Bật: python -m scripts.build_bm25")
    print(f"  Memory      : {len(memory.all())} fact đã lưu")
    print(f"  Tools       : {', '.join(tool_registry.available_tools())}")

    if n_chunks == 0:
        print("\nVectorstore trống. Hãy chạy: python -m scripts.demo_vectorstore")
        sys.exit(1)

    # ── Session ───────────────────────────────────────────────────────────────
    if args.session:
        session = sessions.find(args.session)
        if session is None:
            print(f"Không tìm thấy phiên '{args.session}', tạo phiên mới.")
            session = Session.new(args.session)
    elif args.new:
        session = Session.new()
    else:
        session = pick_session_interactively(sessions)
    sessions.save(session)
    print(f"\nPhiên hiện tại: {session.name}  (id={session.id}, {len(session.messages)//2} lượt)")

    # ── Conversation State (per-session, in-memory) ───────────────────────────
    conv_state = ConversationState()

    print("\nGõ câu hỏi pháp lý, hoặc /help để xem các lệnh. Ctrl+C để thoát.\n")

    last_answer: Optional[Answer] = None
    top_k: int = config.TOP_K
    min_score: Optional[float] = 0.3
    web_search_enabled: bool = True

    while True:
        try:
            user_input = input(f"[{session.name}] Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break

        if not user_input:
            continue

        # ── Lệnh ──────────────────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                print("Tạm biệt!")
                break

            if cmd in ("/clear", "/reset"):
                session.messages.clear()
                session.summary = ""
                session.summary_until = 0
                conv_state.clear()
                sessions.save(session)
                last_answer = None
                print("  Đã xoá lịch sử, tóm tắt và conversation state.")
                continue

            if cmd == "/history":
                show_history(session)
                continue

            if cmd == "/state":
                show_state(conv_state)
                continue

            if cmd == "/summary":
                if session.summary.strip():
                    print(f"  Tóm tắt (cover {session.summary_until}/{len(session.messages)} msgs):")
                    print(f"\n  {session.summary}")
                else:
                    print(f"  (chưa có tóm tắt — cần >= {SUMMARY_MIN_MESSAGES} messages)")
                continue

            if cmd in ("/sources", "/s"):
                show_sources(last_answer)
                continue

            if cmd == "/new":
                session = Session.new(arg)
                conv_state.clear()
                sessions.save(session)
                last_answer = None
                print(f"  Đã tạo phiên mới: {session.name} (id={session.id})")
                continue

            if cmd in ("/sessions", "/ls"):
                show_sessions(sessions.list_recent(limit=20), current_id=session.id)
                continue

            if cmd == "/load":
                if not arg:
                    print("  Cú pháp: /load <id|tên>")
                    continue
                target = sessions.find(arg)
                if target is None:
                    print(f"  Không tìm thấy phiên '{arg}'.")
                    continue
                session = target
                conv_state.clear()
                last_answer = None
                print(f"  Đã chuyển sang: {session.name}  ({len(session.messages)//2} lượt)")
                continue

            if cmd == "/rename":
                new_name = arg.strip()
                if not new_name:
                    print("  Cú pháp: /rename <tên mới>")
                    continue
                session.name = new_name
                sessions.save(session)
                print(f"  Đã đổi tên thành: {session.name}")
                continue

            if cmd == "/delete":
                if not arg:
                    print("  Cú pháp: /delete <id|tên>")
                    continue
                target = sessions.find(arg)
                if target is None:
                    print(f"  Không tìm thấy phiên '{arg}'.")
                    continue
                if target.id == session.id:
                    print("  Không thể xoá phiên đang dùng.")
                    continue
                sessions.delete(target.id)
                print(f"  Đã xoá phiên: {target.name}")
                continue

            if cmd == "/remember":
                if not arg.strip():
                    print("  Cú pháp: /remember <thông tin về bạn>")
                    continue
                fact = memory.add(arg.strip(), source_session=session.id)
                if fact is None:
                    print("  (đã có memory tương tự, bỏ qua)")
                else:
                    print(f"  Đã ghi nhớ [{fact.id}]: {fact.content}")
                continue

            if cmd in ("/memories", "/mem"):
                show_memories(memory)
                continue

            if cmd == "/forget":
                if not arg.strip():
                    print("  Cú pháp: /forget <id hoặc từ khoá>")
                    continue
                n = memory.remove(arg.strip())
                print(f"  Đã xoá {n} memory.")
                continue

            if cmd == "/topk":
                try:
                    top_k = max(1, int(arg))
                    print(f"  top_k = {top_k}")
                except ValueError:
                    print("  Cú pháp: /topk N (vd: /topk 5)")
                continue

            if cmd == "/minscore":
                try:
                    val = float(arg)
                    min_score = val if val > 0 else None
                    print(f"  min_score = {min_score}")
                except ValueError:
                    print("  Cú pháp: /minscore X (vd: /minscore 0.3)")
                continue

            if cmd == "/websearch":
                arg_lower = arg.strip().lower()
                if arg_lower in ("on", "bật", "1", "true", ""):
                    web_search_enabled = True
                    print("  Web search: BẬT ✓  — Agent sẽ tìm thêm trên thuvienphapluat.vn / vbpl.vn khi cần.")
                elif arg_lower in ("off", "tắt", "0", "false"):
                    web_search_enabled = False
                    print("  Web search: TẮT ✗  — Agent chỉ dùng corpus nội bộ + RAG.")
                else:
                    status = "BẬT ✓" if web_search_enabled else "TẮT ✗"
                    print(f"  Web search hiện tại: {status}")
                    print("  Cú pháp: /websearch on  hoặc  /websearch off")
                continue

            if cmd == "/file":
                if not arg.strip():
                    print("  Cú pháp: /file <đường dẫn.pdf/.docx/.txt>")
                    _show_attached(conv_state)
                    continue
                fp = Path(arg.strip().strip('"').strip("'"))
                ok, text, err = _load_file(fp)
                if not ok:
                    print(f"  {err}")
                    continue
                conv_state.attached_document = {"text": text, "filename": fp.name}
                preview = text[:200].replace("\n", " ")
                print(f"  Đã tải: {fp.name}  ({len(text):,} ký tự)")
                print(f"  Preview: {preview}...")
                print("  Gợi ý: 'kiểm tra văn bản này' / 'tóm tắt file' / đặt câu hỏi về nội dung.")
                continue

            if cmd == "/clearfile":
                if conv_state.attached_document:
                    name = conv_state.attached_document["filename"]
                    conv_state.attached_document = None
                    print(f"  Đã xoá file đính kèm: {name}")
                else:
                    print("  (không có file nào đang đính kèm)")
                continue

            if cmd == "/cache":
                arg_lower = arg.strip().lower()
                if arg_lower == "clear":
                    _retrieval_cache.invalidate()
                    print("  Cache đã xoá.")
                else:
                    print(f"  Cache: {_retrieval_cache.stats_str()}")
                continue

            if cmd in ("/help", "/?"):
                print(HELP_TEXT.format(TOP_K=config.TOP_K))
                continue

            print(f"  Lệnh không nhận diện: {cmd}. Gõ /help để xem các lệnh.")
            continue

        # ══════════════════════════════════════════════════════════════════════
        # AGENT PIPELINE
        # ══════════════════════════════════════════════════════════════════════

        # 0a) Auto-detect đường dẫn file trong câu nhập tự do
        _path_match = _FILE_PATH_RE.search(user_input)
        if _path_match:
            _raw_path = next(g for g in _path_match.groups() if g)
            _fp = Path(_raw_path)
            _ok, _ftext, _ferr = _load_file(_fp)
            if _ok:
                conv_state.attached_document = {"text": _ftext, "filename": _fp.name}
                print(f"  Phát hiện file: {_fp.name} ({len(_ftext):,} ký tự) — đã đính kèm tự động.")
                # Xoá path khỏi user_input để router không bị nhiễu
                user_input = _FILE_PATH_RE.sub("", user_input).strip() or "kiểm tra văn bản vừa tải lên"
            else:
                print(f"  (Phát hiện đường dẫn nhưng không tải được: {_ferr})")

        # 0b) Cập nhật Conversation State từ câu hỏi mới
        conv_state.update_from_question(user_input)

        raw_messages = session.messages[session.summary_until:]
        llm_history  = raw_messages[-MAX_HISTORY_MESSAGES:]
        memory_text  = memory.format_for_prompt()
        summary_text = session.summary

        # answer/decision khởi tạo None — sẽ được set trong một trong các flow
        answer:   Optional[Answer] = None
        decision = None   # dùng trong post-processing

        # ── Flow D: Tiếp tục soạn thảo bị ngắt do thiếu thông tin ────────────
        if conv_state.pending_draft is not None:
            _pd = conv_state.pending_draft
            _combined = _pd["details"] + "\nThông tin bổ sung: " + user_input
            print(f"  [pending_draft | doc={_pd['doc_type']}]")
            print("  Đang ghép thông tin và soạn lại...", flush=True)
            _tr = tool_registry.draft_document(
                doc_type=_pd["doc_type"], details=_combined
            )
            if _tr.result.startswith("THIẾU_THÔNG_TIN:"):
                # Vẫn thiếu — cập nhật details tích luỹ, hỏi tiếp
                conv_state.pending_draft = {
                    "doc_type": _pd["doc_type"],
                    "details": _combined,
                }
                _q = _tr.result[len("THIẾU_THÔNG_TIN:\n"):]
                answer = Answer(question=user_input, answer=_q, citations=[])
                print(f"\nAgent: {_q}\n")
            else:
                # Đủ thông tin → soạn xong
                conv_state.pending_draft = None
                _ctxs = _retrieve_cached(_retrieval_cache, retriever, user_input, top_k, min_score)
                answer = _stream_and_collect(
                    generator,
                    question=user_input, contexts=_ctxs,
                    history=llm_history, memory_text=memory_text,
                    summary_text=summary_text,
                    tool_results=[_tr],
                    state_context=conv_state.to_context_string(),
                )
                print()
                if answer.citations:
                    _print_citations_preview(answer)
                if _tr.docx_path:
                    print(f"  📎 File DOCX: {_tr.docx_path}")

        # ── Flows A / B / C — chỉ chạy nếu Flow D chưa xử lý ─────────────────
        if answer is None:

            # 1) Router — phân loại intent + quyết định flow
            decision = router.route(
                user_input,
                history=llm_history,
                memory_text=memory_text,
                summary_text=summary_text,
                state=conv_state,
                web_search_enabled=web_search_enabled,
            )

            # ── Flow A: answer_direct (chitchat / meta / clarify) ─────────────
            if decision.action == "answer_direct":
                answer_text = decision.direct_response or ""
                print(f"  [intent={decision.intent}]")
                print(f"\nAgent: {answer_text}\n")
                answer = Answer(question=user_input, answer=answer_text, citations=[])

            # ── Flow B: use_tool trực tiếp (router biết rõ cần tool gì) ──────
            elif decision.action == "use_tool" and decision.tool_name:
                tool_name  = decision.tool_name
                tool_query = decision.tool_query or user_input
                print(f"  [intent={decision.intent} | tool={tool_name}]")
                print(f"  Đang thực thi tool: {tool_name}...", flush=True)
                _skip_generate = False

                # Safety-net: nếu web_search bị tắt nhưng router vẫn chọn → fallback RAG
                if tool_name == "web_search" and not web_search_enabled:
                    print("  [web_search TẮT → fallback RAG]")
                    search_q = decision.search_query or tool_query or user_input
                    contexts = _retrieve_cached(_retrieval_cache, retriever, search_q, top_k, min_score)
                    answer = _stream_and_collect(
                        generator,
                        question=user_input, contexts=contexts,
                        history=llm_history, memory_text=memory_text,
                        summary_text=summary_text,
                        state_context=conv_state.to_context_string(),
                    )
                    print()
                    if answer.citations:
                        _print_citations_preview(answer)
                    _skip_generate = True
                    tool_result = None  # không dùng đến

                elif tool_name == "calculate_fine":
                    tool_result = tool_registry.calculate_fine(description=tool_query)
                elif tool_name == "draft_document":
                    # Format: "<doc_type>|<details>" hoặc toàn bộ là details
                    if "|" in tool_query:
                        dt, det = tool_query.split("|", 1)
                    else:
                        dt, det = "văn bản pháp lý", tool_query
                    tool_result = tool_registry.draft_document(doc_type=dt.strip(), details=det.strip())
                    # Kiểm tra thiếu thông tin — hỏi lại user, lưu pending_draft
                    if tool_result.result.startswith("THIẾU_THÔNG_TIN:"):
                        conv_state.pending_draft = {
                            "doc_type": dt.strip(),
                            "details": det.strip(),
                        }
                        _q = tool_result.result[len("THIẾU_THÔNG_TIN:\n"):]
                        answer = Answer(question=user_input, answer=_q, citations=[])
                        print(f"\nAgent: {_q}\n")
                        _skip_generate = True
                elif tool_name == "validate_document":
                    # Ưu tiên dùng file đính kèm; fallback về tool_query (text paste)
                    if conv_state.attached_document:
                        _doc = conv_state.attached_document
                        print(f"  Kiểm tra file: {_doc['filename']} ({len(_doc['text']):,} ký tự)...",
                              flush=True)
                        tool_result = tool_registry.validate_document(
                            document_text=_doc["text"],
                            filename=_doc["filename"],
                        )
                    else:
                        tool_result = tool_registry.validate_document(
                            document_text=tool_query,
                            filename="",
                        )
                else:
                    tool_result = tool_registry.execute(tool_name, query=tool_query)

                if not _skip_generate:
                    # Retrieve thêm context cho generator nếu tool không trả đủ
                    search_q = decision.search_query or user_input
                    contexts = _retrieve_cached(_retrieval_cache, retriever, search_q, top_k, min_score)

                    answer = _stream_and_collect(
                        generator,
                        question=user_input, contexts=contexts,
                        history=llm_history, memory_text=memory_text,
                        summary_text=summary_text,
                        tool_results=[tool_result],
                        state_context=conv_state.to_context_string(),
                    )
                    print()

                    # Thông báo file DOCX nếu draft_document đã export
                    if (tool_result is not None
                            and tool_result.tool_name == "draft_document"
                            and tool_result.docx_path):
                        print(f"  📎 File DOCX: {tool_result.docx_path}")

                    if answer.citations:
                        _print_citations_preview(answer)

            # ── Flow C: retrieve (RAG + optional Planner) ─────────────────────
            else:
                search_query = decision.search_query or user_input
                intent_label = decision.intent

                # In query rewrite nếu khác câu gốc
                if search_query.strip().lower() != user_input.strip().lower():
                    print(f"  [intent={intent_label}] Tìm với: \"{search_query}\"")
                else:
                    print(f"  [intent={intent_label}]")

                tool_results: list = []
                plan = None

                # 2) Planner — chỉ gọi khi intent có thể phức tạp.
                # legal/consulting/followup luôn là simple → skip 1 LLM call.
                _SIMPLE_INTENTS = {"legal", "consulting", "followup"}
                if not args.no_planner and intent_label not in _SIMPLE_INTENTS:
                    plan = planner.create_plan(user_input, state=conv_state)
                if plan is not None and plan.is_complex():
                        print(f"  [Planner] complex → {len(plan.tool_steps())} bước tool", flush=True)
                        tool_results = planner.execute_plan(plan)
                        for tr in tool_results:
                            status = "OK" if tr.success else "FAIL"
                            print(f"    [{status}] {tr.tool_name}")
                        # Retrieve query từ plan (bước retrieve cuối)
                        search_query = plan.retrieve_query()

                # 3) Retrieve — lấy RAG context (dùng cache)
                contexts = _retrieve_cached(
                    _retrieval_cache, retriever, search_query, top_k, min_score,
                    label="đang tìm nguồn",
                )
                print(f"  ({len(contexts)} nguồn retrieved — đang sinh câu trả lời...)", flush=True)

                # 5) Hard citation check — từ chối nếu không có căn cứ pháp lý
                if not contexts and not tool_results:
                    answer = Answer(
                        question=user_input,
                        answer=(
                            "Không tìm thấy căn cứ pháp lý trong cơ sở dữ liệu cho câu hỏi này.\n"
                            "Hệ thống chỉ trả lời dựa trên văn bản pháp luật đã được nạp vào. "
                            "Vui lòng thử hỏi theo cách khác hoặc tham khảo ý kiến luật sư."
                        ),
                        citations=[],
                    )
                else:
                    # 6) Generator — sinh câu trả lời (streaming)
                    answer = _stream_and_collect(
                        generator,
                        question=user_input, contexts=contexts,
                        history=llm_history, memory_text=memory_text,
                        summary_text=summary_text,
                        tool_results=tool_results if tool_results else None,
                        state_context=conv_state.to_context_string(),
                    )

                    # 7) Guardrails
                    if not args.no_guardrails:
                        answer = apply_guardrails(answer, contexts)

                if answer is not None and not answer.answer.startswith("Không tìm thấy"):
                    print()

                if answer.citations:
                    _print_citations_preview(answer)

        # ══════════════════════════════════════════════════════════════════════
        # Post-processing
        # ══════════════════════════════════════════════════════════════════════

        # Cập nhật Conversation State từ câu trả lời
        retrieved_sources = [c.source for c in answer.citations]
        conv_state.update_from_answer(answer.answer, retrieved_sources)

        # Lưu session
        session.append("user", user_input)
        session.append("assistant", answer.answer)
        if session.name == "Phiên mới" and len(session.messages) == 2:
            session.name = user_input[:AUTO_NAME_MAXLEN]
        sessions.save(session)

        # Rolling summary
        if maybe_update_summary(session, generator):
            print(f"  [Summary] Đã cập nhật (cover {session.summary_until}/{len(session.messages)} msgs)")
            sessions.save(session)

        # Auto-extract memory — chạy background thread để không block input tiếp theo
        if not args.no_memory_extract and (decision is None or decision.action != "answer_direct"):
            _q = user_input
            _a = answer.answer
            _sid = session.id

            def _bg_extract(q: str, a: str, sid: str) -> None:
                facts = generator.extract_memory_facts(q, a)
                added = []
                for f in facts:
                    fact = memory.add(f, source_session=sid)
                    if fact is not None:
                        added.append(fact.content)
                if added:
                    preview = "; ".join(added)[:120]
                    suffix = "..." if len("; ".join(added)) > 120 else ""
                    print(f"\r  [Memory] Ghi nhớ: {preview}{suffix}\n", flush=True)

            threading.Thread(target=_bg_extract, args=(_q, _a, _sid), daemon=True).start()

        last_answer = answer


def _load_file(file_path: Path) -> tuple[bool, str, str]:
    """Đọc và parse file PDF/DOCX/TXT.

    Returns:
        (ok, text, error_msg)
    """
    suffix = file_path.suffix.lower()
    parser = _FILE_PARSERS.get(suffix)
    if parser is None:
        supported = ", ".join(_FILE_PARSERS)
        return False, "", f"Định dạng '{suffix}' chưa hỗ trợ. Hỗ trợ: {supported}"
    if not file_path.exists():
        return False, "", f"Không tìm thấy file: {file_path}"
    try:
        raw = parser(file_path)  # type: ignore[operator]
        text = clean_text(raw)
        if not text.strip():
            return False, "", "Không đọc được nội dung file (có thể bị scan/ảnh)."
        return True, text, ""
    except Exception as exc:
        return False, "", f"Lỗi đọc file: {exc}"


def _stream_and_collect(generator, **kwargs) -> "Answer":
    """Gọi stream_generate(), in từng token ra stdout, trả về Answer cuối cùng."""
    print("\nAgent: ", end="", flush=True)
    for chunk in generator.stream_generate(**kwargs):
        print(chunk, end="", flush=True)
    print()  # newline sau khi stream xong
    return getattr(generator, "_last_stream_answer", None) or generator.generate(**kwargs)


# Ngưỡng RRF score để skip CrossEncoder — chunk xuất hiện đồng thời ở BM25 lẫn vector.
# 2 * (1/61) ≈ 0.033; dùng 0.04 để chắc chắn hơn.
_SKIP_RERANK_RRF_THRESHOLD = 0.04


def _retrieve_cached(
    cache: RetrievalCache,
    retriever,
    query: str,
    top_k: int,
    min_score,
    label: str = "",
) -> list:
    """Retrieve + rerank (với cache + smart skip reranker).

    Nếu top RRF score >= _SKIP_RERANK_RRF_THRESHOLD: chunk đã rank cao ở nhiều
    branch → bỏ qua CrossEncoder (tiết kiệm 8-14s), chỉ dùng rule-based boost.
    """
    cached = cache.get(query, top_k, min_score)
    if cached is not None:
        if label:
            print(f"  ({label} — cache hit, {len(cached)} nguồn)", flush=True)
        return cached

    contexts = retriever.retrieve(
        query, top_k=top_k, min_score=min_score,
        use_hyde=config.USE_HYDE, use_parent_expansion=True,
    )
    if not contexts:
        cache.put(query, top_k, min_score, contexts)
        return contexts

    top_rrf = contexts[0].score
    # Nếu top chunk đã rank cao ở nhiều branch (RRF score cao) → skip CrossEncoder
    # để tiết kiệm 8-14s; chỉ dùng rule-based boost.
    use_ce = top_rrf < _SKIP_RERANK_RRF_THRESHOLD
    contexts = rerank(query, contexts, top_k=top_k, use_cross_encoder=use_ce)

    cache.put(query, top_k, min_score, contexts)
    return contexts


def _show_attached(conv_state: "ConversationState") -> None:
    doc = conv_state.attached_document
    if doc is None:
        print("  (không có file đính kèm)")
        return
    print(f"  File: {doc['filename']}  ({len(doc['text'])} ký tự)")
    preview = doc["text"][:300].replace("\n", " ")
    print(f"  Preview: {preview}...")


def _print_citations_preview(answer: Answer) -> None:
    """In tối đa 3 nguồn đầu sau câu trả lời."""
    preview_tags = []
    for i, cit in enumerate(answer.citations[:3], 1):
        tag = _format_tag(cit.article, cit.clause)
        # Hiển thị tên luật thân thiện hơn là URL ID
        src_display = cit.source.replace(".txt", "").replace(".pdf", "").split("/")[-1].split("\\")[-1]
        preview_tags.append(f"[{i}] {src_display} {tag}".strip())
    extra = f" • +{len(answer.citations) - 3} nữa" if len(answer.citations) > 3 else ""
    print(f"Nguồn: {' | '.join(preview_tags)}{extra}  (gõ /s để xem đầy đủ)\n")


if __name__ == "__main__":
    main()

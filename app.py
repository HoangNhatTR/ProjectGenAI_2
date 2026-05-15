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
import sys
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
from src.embedding import Embedder
from src.generator import Generator
from src.guardrails import apply_guardrails, check_answer_quality
from src.memory import MemoryStore
from src.planner import LegalPlanner
from src.reranker import rerank as rerank_chunks
from src.retriever import Retriever
from src.router import SmartRouter
from src.schemas import Answer, Citation
from src.session import Session, SessionStore
from src.state import ConversationState
from src.tools import LegalToolRegistry
from src.vectorstore import VectorStore


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
  /help, /?              Hiện trợ giúp
""".strip()

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
    print(f"  Planner     : {'TẮT (--no-planner)' if args.no_planner else 'BẬT'}")
    print(f"  Guardrails  : {'TẮT (--no-guardrails)' if args.no_guardrails else 'BẬT'}")
    print()

    # ── Khởi tạo ──────────────────────────────────────────────────────────────
    print("Đang khởi tạo...")
    embedder = Embedder(config.EMBEDDING_MODEL)
    vstore   = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)

    bm25_path = config.DATA_DIR / "bm25" / "index.json"
    bm25: Optional[BM25Index] = None
    if bm25_path.exists():
        bm25 = BM25Index(bm25_path)

    retriever = Retriever(embedder, vstore, bm25=bm25)
    _api_key = {
        "gemini": config.GEMINI_API_KEY,
        "groq":   config.GROQ_API_KEY,
    }.get(config.LLM_PROVIDER)

    generator = Generator(
        model=config.LLM_MODEL,
        host=config.OLLAMA_HOST,
        temperature=config.LLM_TEMPERATURE,
        provider=config.LLM_PROVIDER,
        api_key=_api_key,
    )
    router    = SmartRouter(
        model=config.LLM_MODEL,
        host=config.OLLAMA_HOST,
        provider=config.LLM_PROVIDER,
        api_key=_api_key,
    )
    sessions  = SessionStore(config.DATA_DIR)
    memory    = MemoryStore(config.DATA_DIR / "memory.json")

    # Tools + Planner (dùng chung client với Generator)
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

            if cmd in ("/help", "/?"):
                print(HELP_TEXT.format(TOP_K=config.TOP_K))
                continue

            print(f"  Lệnh không nhận diện: {cmd}. Gõ /help để xem các lệnh.")
            continue

        # ══════════════════════════════════════════════════════════════════════
        # AGENT PIPELINE
        # ══════════════════════════════════════════════════════════════════════

        # 0) Cập nhật Conversation State từ câu hỏi mới
        conv_state.update_from_question(user_input)

        raw_messages = session.messages[session.summary_until:]
        llm_history  = raw_messages[-MAX_HISTORY_MESSAGES:]
        memory_text  = memory.format_for_prompt()
        summary_text = session.summary

        # 1) Router — phân loại intent + quyết định flow
        decision = router.route(
            user_input,
            history=llm_history,
            memory_text=memory_text,
            summary_text=summary_text,
            state=conv_state,
        )

        # ── Flow A: answer_direct (chitchat / meta / clarify) ─────────────────
        if decision.action == "answer_direct":
            answer_text = decision.direct_response or ""
            print(f"  [intent={decision.intent}]")
            print(f"\nAgent: {answer_text}\n")
            answer = Answer(question=user_input, answer=answer_text, citations=[])

        # ── Flow B: use_tool trực tiếp (router biết rõ cần tool gì) ──────────
        elif decision.action == "use_tool" and decision.tool_name:
            tool_name  = decision.tool_name
            tool_query = decision.tool_query or user_input
            print(f"  [intent={decision.intent} | tool={tool_name}]")
            print(f"  Đang thực thi tool: {tool_name}...", flush=True)

            if tool_name == "calculate_fine":
                tool_result = tool_registry.calculate_fine(description=tool_query)
            elif tool_name == "draft_document":
                # Format: "<doc_type>|<details>" hoặc toàn bộ là details
                if "|" in tool_query:
                    dt, det = tool_query.split("|", 1)
                else:
                    dt, det = "văn bản pháp lý", tool_query
                tool_result = tool_registry.draft_document(doc_type=dt.strip(), details=det.strip())
            else:
                tool_result = tool_registry.execute(tool_name, query=tool_query)

            # Retrieve thêm context cho generator nếu tool không trả đủ
            search_q = decision.search_query or user_input
            contexts = retriever.retrieve(search_q, top_k=top_k, min_score=min_score)
            contexts = rerank_chunks(search_q, contexts)

            answer = generator.generate(
                user_input,
                contexts,
                history=llm_history,
                memory_text=memory_text,
                summary_text=summary_text,
                tool_results=[tool_result],
                state_context=conv_state.to_context_string(),
            )
            print(f"\nAgent: {answer.answer}\n")

            if answer.citations:
                _print_citations_preview(answer)

        # ── Flow C: retrieve (RAG + optional Planner) ─────────────────────────
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

            # 2) Planner — phân tích độ phức tạp
            if not args.no_planner:
                plan = planner.create_plan(user_input, state=conv_state)
                if plan.is_complex():
                    print(f"  [Planner] complex → {len(plan.tool_steps())} bước tool", flush=True)
                    tool_results = planner.execute_plan(plan)
                    for tr in tool_results:
                        status = "OK" if tr.success else "FAIL"
                        print(f"    [{status}] {tr.tool_name}")
                    # Retrieve query từ plan (bước retrieve cuối)
                    search_query = plan.retrieve_query()

            # 3) Retrieve — lấy RAG context
            contexts = retriever.retrieve(search_query, top_k=top_k, min_score=min_score)
            print(f"  ({len(contexts)} nguồn retrieved", end="")

            # 4) Reranker — xếp hạng lại theo tham chiếu điều/khoản
            contexts = rerank_chunks(search_query, contexts)
            print(f" → reranked, đang sinh câu trả lời...)", flush=True)

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
                # 6) Generator — sinh câu trả lời
                answer = generator.generate(
                    user_input,
                    contexts,
                    history=llm_history,
                    memory_text=memory_text,
                    summary_text=summary_text,
                    tool_results=tool_results if tool_results else None,
                    state_context=conv_state.to_context_string(),
                )

                # 7) Guardrails
                if not args.no_guardrails:
                    answer = apply_guardrails(answer, contexts)

            print(f"\nAgent: {answer.answer}\n")

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

        # Auto-extract memory
        if not args.no_memory_extract and decision.action != "answer_direct":
            new_facts = generator.extract_memory_facts(user_input, answer.answer)
            added = []
            for f in new_facts:
                fact = memory.add(f, source_session=session.id)
                if fact is not None:
                    added.append(fact.content)
            if added:
                preview = "; ".join(added)[:120]
                print(f"  [Memory] Ghi nhớ: {preview}{'...' if len('; '.join(added)) > 120 else ''}")

        last_answer = answer


def _print_citations_preview(answer: Answer) -> None:
    """In tối đa 3 nguồn đầu sau câu trả lời."""
    preview_tags = []
    for i, cit in enumerate(answer.citations[:3], 1):
        tag = _format_tag(cit.article, cit.clause)
        preview_tags.append(f"[{i}] {cit.source.replace('.txt','').replace('.pdf','')} {tag}")
    extra = f" • +{len(answer.citations) - 3} nữa" if len(answer.citations) > 3 else ""
    print(f"Nguồn: {' | '.join(preview_tags)}{extra}  (gõ /s để xem đầy đủ)\n")


if __name__ == "__main__":
    main()

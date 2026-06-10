"""FastAPI backend — OpenAI-compatible API cho Legal AI Agent.

Kết nối với bất kỳ frontend nào hỗ trợ OpenAI API:
  - chatbot-ui  (https://github.com/mckaywrigley/chatbot-ui)
  - open-webui
  - LibreChat
  - Curl / Python client

Chạy:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Frontend cấu hình:
    OPENAI_API_KEY  = any-string   (bất kỳ, không cần thật)
    OPENAI_API_HOST = http://localhost:8000

Models khả dụng (map sang RAG mode):
    legal-ai-graph   →  Graph-RAG (vector + BM25 + Knowledge Graph)
    legal-ai-top15   →  RAG Top 15 luật
    legal-ai-full    →  RAG Full 609 luật
    legal-ai         →  alias cho graph (mặc định)
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ── UTF-8 stdout ──────────────────────────────────────────────────────────────
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
from src.planner import LegalPlanner
from src.retriever import Retriever
from src.router import SmartRouter
from src.schemas import Answer
from src.state import ConversationState
from src.tools import LegalToolRegistry
from src.vectorstore import VectorStore

try:
    from src.kg.kg_retriever import KGRetriever
    _KG_AVAILABLE = True
except Exception:
    _KG_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_HISTORY = 10
WORD_STREAM_DELAY = 0.018   # giây giữa mỗi từ khi fake-stream

RETRIEVAL_MODES = {
    "rag_full":  {"use_kg": False, "use_top10_filter": False},
    "rag_top10": {"use_kg": False, "use_top10_filter": True},
    "graph_rag": {"use_kg": True,  "use_top10_filter": True},
}

MODEL_TO_MODE: dict[str, str] = {
    "legal-ai":        "graph_rag",
    "legal-ai-graph":  "graph_rag",
    "legal-ai-top15":  "rag_top10",
    "legal-ai-full":   "rag_full",
    # Fallback — nếu chatbot-ui gửi tên model OpenAI thật
    "gpt-4":           "graph_rag",
    "gpt-4o":          "graph_rag",
    "gpt-3.5-turbo":   "graph_rag",
    "gpt-4-turbo":     "graph_rag",
}

AVAILABLE_MODELS = [
    {"id": "legal-ai-graph",  "object": "model", "created": 1700000000,
     "owned_by": "legal-ai", "description": "Graph-RAG: vector + BM25 + Knowledge Graph"},
    {"id": "legal-ai-top15",  "object": "model", "created": 1700000000,
     "owned_by": "legal-ai", "description": "RAG Top-15: chỉ trên 15 luật trọng yếu"},
    {"id": "legal-ai-full",   "object": "model", "created": 1700000000,
     "owned_by": "legal-ai", "description": "RAG Full: toàn bộ 609 luật VN"},
    {"id": "legal-ai",        "object": "model", "created": 1700000000,
     "owned_by": "legal-ai", "description": "Alias của legal-ai-graph (mặc định)"},
]

# ── Global agent singletons (loaded 1 lần khi khởi động) ─────────────────────

_agent: dict = {}


def _load_agent() -> None:
    """Khởi tạo toàn bộ pipeline Legal AI Agent."""
    import json as _json

    print("[API] Đang khởi tạo Legal AI Agent…")

    embedder = Embedder(config.EMBEDDING_MODEL)
    vstore = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)

    bm25_path = config.DATA_DIR / "bm25" / "index.json"
    bm25 = BM25Index(bm25_path) if bm25_path.exists() else None

    kg_retriever = None
    if _KG_AVAILABLE:
        try:
            kg_retriever = KGRetriever()
        except Exception as e:
            print(f"[API] KG không khả dụng: {e}")

    retriever = Retriever(embedder, vstore, bm25=bm25, kg_retriever=kg_retriever)

    top15_urls: list[str] = []
    manifest_path = config.DATA_DIR / "comparison" / "top10_laws" / "manifest.json"
    if manifest_path.exists():
        try:
            data = _json.loads(manifest_path.read_text(encoding="utf-8"))
            top15_urls = [law["source_url"] for law in data.get("laws", []) if law.get("source_url")]
        except Exception:
            pass

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
        model=config.LLM_MODEL, host=_llm_host,
        temperature=config.LLM_TEMPERATURE,
        provider=config.LLM_PROVIDER, api_key=_api_key,
    )
    _router_model = getattr(config, "ROUTER_MODEL", config.LLM_MODEL)
    router = SmartRouter(
        model=_router_model, host=_llm_host,
        provider=config.LLM_PROVIDER, api_key=_api_key,
    )
    print(f"[API] Router model : {_router_model}")
    print(f"[API] Generator    : {config.LLM_MODEL}")

    ollama_client = generator.get_client()
    tool_registry = LegalToolRegistry(
        retriever=retriever, ollama_client=ollama_client, model=config.LLM_MODEL,
    )
    planner = LegalPlanner(
        ollama_client=ollama_client, model=config.LLM_MODEL, tool_registry=tool_registry,
    )

    _agent.update({
        "embedder": embedder,
        "vstore": vstore,
        "bm25": bm25,
        "retriever": retriever,
        "generator": generator,
        "router": router,
        "planner": planner,
        "tool_registry": tool_registry,
        "top15_urls": top15_urls,
    })

    print(f"[API] Sẵn sàng — {vstore.count():,} chunks, KG={'✓' if kg_retriever else '✗'}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_agent()
    yield
    print("[API] Tắt server.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Legal AI Agent — OpenAI-compatible API",
    version="1.0.0",
    description="Tư vấn pháp luật VN với RAG + Knowledge Graph",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models (OpenAI format) ───────────────────────────────────────────

class OAIMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "legal-ai-graph"
    messages: list[OAIMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    web_search: bool = True


class OAIDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class OAIChoice(BaseModel):
    index: int = 0
    message: Optional[OAIMessage] = None
    delta: Optional[OAIDelta] = None
    finish_reason: Optional[str] = None


class OAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "legal-ai-graph"
    choices: list[OAIChoice]
    usage: OAIUsage = Field(default_factory=OAIUsage)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    n = _agent.get("vstore", None)
    return {
        "status": "ok",
        "service": "Legal AI Agent",
        "chunks": n.count() if n else 0,
        "models": [m["id"] for m in AVAILABLE_MODELS],
    }


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": AVAILABLE_MODELS}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    for m in AVAILABLE_MODELS:
        if m["id"] == model_id:
            return m
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


_EXPORT_DIR = Path("data/exports")

@app.get("/v1/export/{filename}")
async def download_export(filename: str):
    """Tải file DOCX đã được tạo bởi draft_document."""
    # Chỉ cho phép .docx để tránh path traversal
    if not filename.endswith(".docx") or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ.")
    file_path = _EXPORT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' không tìm thấy.")
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent chưa sẵn sàng")

    # Lấy user message cuối
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="Không có user message")

    user_input = user_messages[-1].content

    # History = tất cả messages trừ cái cuối
    history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]

    # Map model → RAG mode
    rag_mode = MODEL_TO_MODE.get(request.model, "graph_rag")

    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if request.stream:
        return StreamingResponse(
            _stream_pipeline(cid, user_input, history, rag_mode, request.model,
                             web_search_enabled=request.web_search),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        answer = await asyncio.get_event_loop().run_in_executor(
            None, _run_pipeline, user_input, history, rag_mode, 5, request.web_search,
        )
        full_text = _format_answer(answer)
        return ChatCompletionResponse(
            id=cid,
            model=request.model,
            choices=[OAIChoice(
                message=OAIMessage(role="assistant", content=full_text),
                finish_reason="stop",
            )],
            usage=OAIUsage(completion_tokens=len(full_text.split())),
        )


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(
    user_input: str,
    history: list[dict],
    rag_mode: str,
    top_k: int = 5,
    web_search_enabled: bool = True,
) -> Answer:
    """Chạy toàn bộ Legal AI Agent pipeline (synchronous). Trả về Answer."""

    _t_total = time.time()

    retriever    = _agent["retriever"]
    generator    = _agent["generator"]
    router       = _agent["router"]
    planner      = _agent["planner"]
    tool_registry = _agent["tool_registry"]
    top15_urls   = _agent["top15_urls"]

    mode_cfg      = RETRIEVAL_MODES.get(rag_mode, RETRIEVAL_MODES["graph_rag"])
    use_kg        = mode_cfg["use_kg"]
    allowed_sources = top15_urls if mode_cfg["use_top10_filter"] else None

    # Rebuild ConversationState từ history
    conv_state = ConversationState()
    for msg in history[-6:]:
        if msg["role"] == "user":
            conv_state.update_from_question(msg["content"])
        elif msg["role"] == "assistant":
            conv_state.update_from_answer(msg["content"], [])
    conv_state.update_from_question(user_input)

    llm_history = history[-MAX_HISTORY:]

    # ── Router ────────────────────────────────────────────────────────────────
    _t0 = time.time()
    decision = router.route(
        user_input, history=llm_history,
        memory_text="", summary_text="",
        state=conv_state,
        web_search_enabled=web_search_enabled,
    )
    _t_router = time.time() - _t0
    print(f"  [TIMER] Router   : {_t_router:.2f}s  intent={decision.intent} action={decision.action}")

    # ── Flow A: trả lời trực tiếp ─────────────────────────────────────────────
    if decision.action == "answer_direct":
        print(f"  [TIMER] TOTAL    : {time.time()-_t_total:.2f}s  (answer_direct)")
        return Answer(
            question=user_input,
            answer=decision.direct_response or "",
            citations=[],
        )

    # ── Flow B: dùng tool ─────────────────────────────────────────────────────
    if decision.action == "use_tool" and decision.tool_name:
        tool_name  = decision.tool_name
        tool_query = decision.tool_query or user_input

        if tool_name == "calculate_fine":
            tool_result = tool_registry.calculate_fine(description=tool_query)

        elif tool_name == "draft_document":
            dt, det = (tool_query.split("|", 1) if "|" in tool_query
                       else ("văn bản pháp lý", tool_query))
            tool_result = tool_registry.draft_document(
                doc_type=dt.strip(), details=det.strip(),
            )
            # Nếu export DOCX thành công → append download link vào result
            if tool_result.success and tool_result.docx_path:
                _fname = Path(tool_result.docx_path).name
                _api_base = getattr(config, "API_BASE_URL", "http://localhost:8000")
                _link = (
                    f"\n\n---\n"
                    f"📎 **[⬇️ Tải file DOCX]({_api_base}/v1/export/{_fname})**  "
                    f"*(nhấn để tải về)*"
                )
                tool_result = tool_result.__class__(
                    tool_name=tool_result.tool_name,
                    success=tool_result.success,
                    result=tool_result.result + _link,
                    sources=tool_result.sources,
                    docx_path=tool_result.docx_path,
                )

        elif tool_name == "compare_regulations":
            # tool_query format: "chủ đề A|chủ đề B"
            if "|" in tool_query:
                topic_a, topic_b = tool_query.split("|", 1)
            else:
                # Tự phân tách từ câu hỏi gốc
                parts = tool_query.split(" và " if " và " in tool_query else " vs ")
                topic_a = parts[0].strip()
                topic_b = parts[1].strip() if len(parts) > 1 else user_input
            tool_result = tool_registry.compare_regulations(
                topic_a=topic_a.strip(), topic_b=topic_b.strip(),
            )

        elif tool_name == "validate_document":
            # tool_query là nội dung văn bản hoặc mô tả (khi không có file)
            # Thử tìm nội dung file từ system message trong history
            doc_text = tool_query
            for msg in reversed(llm_history):
                if msg.get("role") == "system" and "file" in msg.get("content", "").lower():
                    doc_text = msg["content"]
                    break
            tool_result = tool_registry.validate_document(
                document_text=doc_text, filename="",
            )

        elif tool_name == "knowledge_graph_lookup":
            tool_result = tool_registry.knowledge_graph_lookup(query=tool_query)

        else:
            tool_result = tool_registry.execute(tool_name, query=tool_query)

        # Validate và compare đã tự chứa kết quả đầy đủ — trả thẳng không qua generator
        if tool_name in ("validate_document", "compare_regulations") and tool_result.success:
            return Answer(
                question=user_input,
                answer=tool_result.result,
                citations=[],
            )

        search_q = decision.search_query or user_input
        contexts = retriever.retrieve(
            search_q, top_k=top_k, use_kg=use_kg, allowed_sources=allowed_sources,
        )

        return generator.generate(
            user_input, contexts, history=llm_history,
            tool_results=[tool_result],
            state_context=conv_state.to_context_string(),
        )

    # ── Flow C: RAG retrieve ──────────────────────────────────────────────────
    search_query = decision.search_query or user_input
    tool_results: list = []

    _t0 = time.time()
    contexts = retriever.retrieve(
        search_query, top_k=top_k, use_kg=use_kg, allowed_sources=allowed_sources,
    )
    _t_retrieve = time.time() - _t0
    print(f"  [TIMER] Retrieve : {_t_retrieve:.2f}s  ({len(contexts)} chunks)")

    if not contexts and not tool_results:
        print(f"  [TIMER] TOTAL    : {time.time()-_t_total:.2f}s  (no context)")
        return Answer(
            question=user_input,
            answer=(
                "Không tìm thấy căn cứ pháp lý trong cơ sở dữ liệu. "
                "Vui lòng thử câu hỏi khác hoặc tham khảo ý kiến luật sư."
            ),
            citations=[],
        )

    _t0 = time.time()
    answer = generator.generate(
        user_input, contexts, history=llm_history,
        tool_results=tool_results if tool_results else None,
        state_context=conv_state.to_context_string(),
    )
    _t_generate = time.time() - _t0
    print(f"  [TIMER] Generate : {_t_generate:.2f}s")
    print(f"  [TIMER] TOTAL    : {time.time()-_t_total:.2f}s  ← pipeline time")
    return apply_guardrails(answer, contexts)


def clean_llm_generated_sources(text: str) -> str:
    """Loại bỏ phần nguồn tự phát sinh của LLM (nếu có) ở cuối câu trả lời."""
    # Xóa block nguồn ở cuối (ví dụ: "Nguồn pháp lý:", "Nguồn tham khảo:")
    pattern_block = r'\n+(?:📚\s*)?(?:Nguồn pháp lý|Nguồn tham khảo|Danh sách nguồn|Nguồn trích dẫn|Tài liệu tham khảo)\s*:.*$'
    text = re.sub(pattern_block, '', text, flags=re.IGNORECASE | re.DOTALL).strip()
    
    # Xóa các dòng chứa URL nguồn tự phát ở cuối câu trả lời
    lines = text.split('\n')
    while lines:
        last_line = lines[-1].strip().lower()
        if not last_line:
            lines.pop()
            continue
        # Nếu dòng cuối cùng bắt đầu bằng "nguồn:" hoặc chứa "http://" hay "https://"
        if last_line.startswith("nguồn:") or "http://" in last_line or "https://" in last_line:
            lines.pop()
        else:
            break
    return '\n'.join(lines).strip()


def _clean_snippet(text: str, max_len: Optional[int] = None) -> str:
    """Làm sạch snippet: bỏ URL, prefix 'Nguồn:', chuẩn hoá khoảng trắng."""
    text = text.replace("\n", " ").strip()
    # Xoá "Nguồn: https://..." ở đầu (nếu có)
    text = re.sub(r'^Ngu[oồ]n:\s*https?://\S+\s*[-–—]?\s*', '', text, flags=re.IGNORECASE).strip()
    # Xoá mọi URL còn sót
    text = re.sub(r'https?://\S+', '', text).strip()
    # Chuẩn hoá khoảng trắng thừa
    text = re.sub(r'\s{2,}', ' ', text)
    if max_len is not None:
        return text[:max_len]
    return text


def _format_answer(answer: Answer) -> str:
    """Gộp câu trả lời + citations thành Markdown đầy đủ."""
    text = answer.answer

    # Làm sạch các nguồn tự phát sinh từ LLM
    text = clean_llm_generated_sources(text)

    if not answer.citations:
        return text

    lines = [text, "📚 Nguồn pháp lý:"]
    for i, cit in enumerate(answer.citations, 1):
        tag_parts = [p for p in [cit.article, cit.clause] if p]
        tag = " · ".join(tag_parts) if tag_parts else ""
        src = cit.source.replace(".txt", "").replace(".pdf", "").split("/")[-1].split("\\")[-1]
        snippet = _clean_snippet(cit.snippet, max_len=None)
        label = f"{src}{' — ' + tag if tag else ''}"
        lines.append(f"[{i}] — {label}: {snippet}")

    return "\n\n".join(lines)


async def _stream_pipeline(
    cid: str,
    user_input: str,
    history: list[dict],
    rag_mode: str,
    model_name: str,
    top_k: int = 5,
    web_search_enabled: bool = True,
) -> AsyncIterator[str]:
    """Generator SSE — chạy pipeline trong thread pool rồi stream từng từ."""

    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    created = int(time.time())
    _t_request_start = time.time()
    print(f"\n[REQUEST] '{user_input[:60]}{'...' if len(user_input)>60 else ''}'")

    # Chunk mở đầu (role)
    yield _sse({
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    })

    try:
        # Chạy pipeline trong thread pool (blocking → non-blocking)
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None, _run_pipeline, user_input, history, rag_mode, top_k, web_search_enabled,
        )
        _t_pipeline_done = time.time()
        full_text = _format_answer(answer)

    except Exception as e:
        err_text = f"Lỗi xử lý: {e}"
        yield _sse({
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"content": err_text}, "finish_reason": None}],
        })
        yield _sse({
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })
        yield "data: [DONE]\n\n"
        return

    # Stream từng token (word-by-word fake streaming)
    tokens = full_text.split(" ")
    for i, token in enumerate(tokens):
        chunk_text = token + (" " if i < len(tokens) - 1 else "")
        yield _sse({
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
        })
        await asyncio.sleep(WORD_STREAM_DELAY)

    # Chunk kết thúc
    yield _sse({
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })
    yield "data: [DONE]\n\n"

    _t_stream = time.time() - _t_pipeline_done
    _t_e2e    = time.time() - _t_request_start
    print(f"  [TIMER] Stream   : {_t_stream:.2f}s  ({len(tokens)} tokens × {WORD_STREAM_DELAY}s)")
    print(f"  [TIMER] ═══ E2E  : {_t_e2e:.2f}s  (total user-visible latency)\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

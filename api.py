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
import functools
import json
import queue
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
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
from src.parent_store import ParentStore
from src.planner import LegalPlanner
from src.reranker import rerank as _rerank
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

    _parent_store = ParentStore(config.PARENT_STORE_PATH) if config.PARENT_STORE_PATH.exists() else None
    if _parent_store:
        print(f"[API] ParentStore  : {_parent_store.count():,} entries")

    retriever = Retriever(
        embedder, vstore, bm25=bm25, kg_retriever=kg_retriever,
        parent_store=_parent_store,
    )

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

    # Wire HyDE vào retriever sau khi có LLM client
    if config.USE_HYDE:
        retriever.llm_client = ollama_client
        retriever.hyde_model  = config.HYDE_MODEL
        print(f"[API] HyDE         : BẬT (model={config.HYDE_MODEL})")
    else:
        print("[API] HyDE         : TẮT (set USE_HYDE=true để bật)")

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

# Origin lấy từ env CORS_ORIGINS (mặc định: localhost:3000/3001/8501).
# Spec CORS cấm wildcard "*" đi kèm credentials → nếu mở "*" thì tắt credentials.
_cors_allow_all = "*" in config.CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _cors_allow_all else config.CORS_ORIGINS,
    allow_credentials=not _cors_allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_auth(authorization: Optional[str]) -> None:
    """Chặn request nếu API_AUTH_KEY được set mà token không khớp.

    API_AUTH_KEY trống (mặc định) = không bắt auth — chỉ nên dùng khi localhost.
    """
    if not config.API_AUTH_KEY:
        return
    token = authorization or ""
    if token.lower().startswith("bearer "):
        token = token[7:]
    if token.strip() != config.API_AUTH_KEY:
        raise HTTPException(
            status_code=401,
            detail="API key không hợp lệ. Gửi header 'Authorization: Bearer <API_AUTH_KEY>'.",
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
    # Tham số RAG / Reranker từ frontend settings
    top_k: Optional[int] = None           # số chunk retrieve (mặc định config.TOP_K)
    ce_threshold: Optional[float] = None  # ngưỡng CrossEncoder skip (mặc định 0.04)
    llm_model: Optional[str] = None       # override LLM model (router9/kieai)


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
async def list_models(authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    return {"object": "list", "data": AVAILABLE_MODELS}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    for m in AVAILABLE_MODELS:
        if m["id"] == model_id:
            return m
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


_EXPORT_DIR = Path("data/exports")

@app.get("/v1/export/{filename}")
async def download_export(filename: str, authorization: Optional[str] = Header(None)):
    """Tải file DOCX đã được tạo bởi draft_document."""
    _require_auth(authorization)
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
    _require_auth(authorization)
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

    # Trích tham số từ request (dùng default nếu không có)
    _top_k       = request.top_k        if request.top_k        is not None else config.TOP_K
    _temperature = request.temperature  # None → giữ nguyên generator default
    _top_p       = request.top_p        # None → giữ nguyên
    _ce_thresh   = request.ce_threshold if request.ce_threshold is not None else 0.04
    _llm_model   = request.llm_model    # None → không override

    if request.stream:
        return StreamingResponse(
            _stream_pipeline(
                cid, user_input, history, rag_mode, request.model,
                top_k=_top_k,
                web_search_enabled=request.web_search,
                temperature=_temperature,
                top_p=_top_p,
                ce_threshold=_ce_thresh,
                llm_model=_llm_model,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        answer = await asyncio.get_event_loop().run_in_executor(
            None,
            functools.partial(
                _run_pipeline, user_input, history, rag_mode,
                _top_k, request.web_search,
                _temperature, _top_p, _ce_thresh, _llm_model,
            ),
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

def _make_generator(
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    llm_model: Optional[str] = None,
) -> Generator:
    """Tạo Generator riêng cho request — KHÔNG mutate singleton.

    Hai request đồng thời với model/temperature khác nhau sẽ không ghi đè
    cấu hình của nhau (trước đây set trực tiếp lên generator dùng chung
    trong thread pool → race condition).
    """
    base: Generator = _agent["generator"]
    provider, api_key, host, model = base.provider, base.api_key, base.host, base.model

    if llm_model:
        model = llm_model.strip()
        if model.startswith(("cc/", "gh/")):
            provider, api_key, host = "router9", config.ROUTER9_API_KEY, config.ROUTER9_BASE_URL
        else:
            # Không có prefix 9Router → mặc định KieAI
            provider, api_key, host = "kieai", config.KIEAI_API_KEY, config.KIEAI_BASE_URL
        print(f"  [MODEL] override → {model} (provider={provider})")

    gen = Generator(
        model=model,
        host=host,
        temperature=base.temperature if temperature is None else temperature,
        num_ctx=base.num_ctx,
        top_p=base.top_p if top_p is None else top_p,
        provider=provider,
        api_key=api_key,
    )
    # Cùng provider/host/key → tái dùng client đã connect của singleton
    # (client chỉ phụ thuộc provider+host+key; model/temperature truyền per-call)
    if (provider, host, api_key) == (base.provider, base.host, base.api_key):
        gen._client = base.get_client()
    return gen


@dataclass
class _PipelinePrep:
    """Kết quả phase chuẩn bị (router → tool → retrieve) — trước khi generate.

    final_answer được set khi flow đã có câu trả lời hoàn chỉnh
    (answer_direct, tool tự chứa kết quả, hoặc không có context).
    """
    final_answer: Optional[Answer] = None
    contexts: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    state_context: str = ""
    llm_history: list = field(default_factory=list)


def _prepare_pipeline(
    user_input: str,
    history: list[dict],
    rag_mode: str,
    top_k: int = 5,
    web_search_enabled: bool = True,
    ce_threshold: float = 0.04,
) -> _PipelinePrep:
    """Phase 1 của pipeline: router → tool → retrieve → rerank (synchronous).

    Không gọi generator — phần generate tách riêng để hỗ trợ streaming thật.
    """
    retriever     = _agent["retriever"]
    router        = _agent["router"]
    tool_registry = _agent["tool_registry"]
    top15_urls    = _agent["top15_urls"]

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
    # (chitchat/meta/clarify — không phải tư vấn pháp lý nên không cần disclaimer)
    if decision.action == "answer_direct":
        return _PipelinePrep(
            final_answer=Answer(
                question=user_input,
                answer=decision.direct_response or "",
                citations=[],
            ),
            llm_history=llm_history,
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

        # Validate và compare đã tự chứa kết quả đầy đủ — trả thẳng không qua generator.
        # Vẫn áp guardrails (disclaimer pháp lý) nhưng tắt cảnh báo "thiếu căn cứ"
        # vì kết quả tool đã tự chứa căn cứ.
        if tool_name in ("validate_document", "compare_regulations") and tool_result.success:
            ans = Answer(question=user_input, answer=tool_result.result, citations=[])
            return _PipelinePrep(
                final_answer=apply_guardrails(ans, [], warn_no_evidence=False),
                llm_history=llm_history,
            )

        search_q = decision.search_query or user_input
        contexts = retriever.retrieve(
            search_q, top_k=top_k, use_kg=use_kg, allowed_sources=allowed_sources,
        )

        return _PipelinePrep(
            contexts=contexts,
            tool_results=[tool_result],
            state_context=conv_state.to_context_string(),
            llm_history=llm_history,
        )

    # ── Flow C: RAG retrieve ──────────────────────────────────────────────────
    search_query = decision.search_query or user_input

    _t0 = time.time()
    contexts = retriever.retrieve(
        search_query, top_k=top_k, use_kg=use_kg, allowed_sources=allowed_sources,
        use_hyde=config.USE_HYDE, use_parent_expansion=True,
    )
    _t_retrieve = time.time() - _t0
    print(f"  [TIMER] Retrieve : {_t_retrieve:.2f}s  ({len(contexts)} chunks)")

    # Rerank — smart skip CrossEncoder nếu RRF score đã cao
    if contexts:
        _top_rrf = contexts[0].score
        _use_ce  = _top_rrf < ce_threshold
        contexts = _rerank(search_query, contexts, top_k=top_k, use_cross_encoder=_use_ce)
        print(f"  [TIMER] Rerank   : CE={'on' if _use_ce else 'skip'} threshold={ce_threshold} ({len(contexts)} chunks)")

    if not contexts:
        ans = Answer(
            question=user_input,
            answer=(
                "Không tìm thấy căn cứ pháp lý trong cơ sở dữ liệu. "
                "Vui lòng thử câu hỏi khác hoặc tham khảo ý kiến luật sư."
            ),
            citations=[],
        )
        return _PipelinePrep(
            final_answer=apply_guardrails(ans, []),
            llm_history=llm_history,
        )

    return _PipelinePrep(
        contexts=contexts,
        state_context=conv_state.to_context_string(),
        llm_history=llm_history,
    )


def _run_pipeline(
    user_input: str,
    history: list[dict],
    rag_mode: str,
    top_k: int = 5,
    web_search_enabled: bool = True,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    ce_threshold: float = 0.04,
    llm_model: Optional[str] = None,
) -> Answer:
    """Chạy toàn bộ pipeline (synchronous, non-streaming). Trả về Answer."""
    _t_total = time.time()

    prep = _prepare_pipeline(
        user_input, history, rag_mode, top_k, web_search_enabled, ce_threshold,
    )
    if prep.final_answer is not None:
        print(f"  [TIMER] TOTAL    : {time.time()-_t_total:.2f}s  (không cần generate)")
        return prep.final_answer

    generator = _make_generator(temperature, top_p, llm_model)

    _t0 = time.time()
    answer = generator.generate(
        user_input, prep.contexts, history=prep.llm_history,
        tool_results=prep.tool_results or None,
        state_context=prep.state_context,
    )
    print(f"  [TIMER] Generate : {time.time()-_t0:.2f}s")
    print(f"  [TIMER] TOTAL    : {time.time()-_t_total:.2f}s  ← pipeline time")

    # Tool flow: kết quả tool là căn cứ → không cảnh báo "thiếu căn cứ"
    return apply_guardrails(
        answer, prep.contexts, warn_no_evidence=not prep.tool_results,
    )


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


def _format_citations(citations: list) -> str:
    """Render block '📚 Nguồn pháp lý' từ citations. Rỗng nếu không có citation."""
    if not citations:
        return ""
    lines = ["📚 Nguồn pháp lý:"]
    for i, cit in enumerate(citations, 1):
        tag_parts = [p for p in [cit.article, cit.clause] if p]
        tag = " · ".join(tag_parts) if tag_parts else ""
        src = cit.source.replace(".txt", "").replace(".pdf", "").split("/")[-1].split("\\")[-1]
        snippet = _clean_snippet(cit.snippet, max_len=None)
        label = f"{src}{' — ' + tag if tag else ''}"
        lines.append(f"[{i}] — {label}: {snippet}")
    return "\n\n".join(lines)


def _format_answer(answer: Answer) -> str:
    """Gộp câu trả lời + citations thành Markdown đầy đủ."""
    # Làm sạch các nguồn tự phát sinh từ LLM
    text = clean_llm_generated_sources(answer.answer)

    cit_block = _format_citations(answer.citations)
    return f"{text}\n\n{cit_block}" if cit_block else text


async def _stream_pipeline(
    cid: str,
    user_input: str,
    history: list[dict],
    rag_mode: str,
    model_name: str,
    top_k: int = 5,
    web_search_enabled: bool = True,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    ce_threshold: float = 0.04,
    llm_model: Optional[str] = None,
) -> AsyncIterator[str]:
    """Generator SSE — retrieve trong thread pool, sau đó stream THẬT từ LLM.

    Token được đẩy về client ngay khi LLM sinh ra (qua worker thread + queue),
    thay vì chờ pipeline xong rồi giả lập stream từng từ như trước.
    Lưu ý: vì stream trực tiếp nên không thể "dọn" block nguồn LLM tự phát sinh
    ở cuối (clean_llm_generated_sources) — prompt đã cấm LLM tự ghi nguồn.
    """

    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _content(text: str) -> str:
        return _sse({
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        })

    def _finish() -> str:
        return _sse({
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })

    created = int(time.time())
    _t_request_start = time.time()
    print(f"\n[REQUEST] '{user_input[:60]}{'...' if len(user_input)>60 else ''}'")

    # Chunk mở đầu (role)
    yield _sse({
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    })

    loop = asyncio.get_event_loop()

    # ── Phase 1: router → tool → retrieve (blocking → thread pool) ───────────
    try:
        prep = await loop.run_in_executor(
            None,
            functools.partial(
                _prepare_pipeline, user_input, history, rag_mode,
                top_k, web_search_enabled, ce_threshold,
            ),
        )
    except Exception as e:
        yield _content(f"Lỗi xử lý: {e}")
        yield _finish()
        yield "data: [DONE]\n\n"
        return

    # Flow đã có câu trả lời hoàn chỉnh (answer_direct / tool / no-context)
    if prep.final_answer is not None:
        yield _content(_format_answer(prep.final_answer))
        yield _finish()
        yield "data: [DONE]\n\n"
        print(f"  [TIMER] ═══ E2E  : {time.time()-_t_request_start:.2f}s  (không cần generate)\n")
        return

    # ── Phase 2: stream thật từ LLM ───────────────────────────────────────────
    generator = _make_generator(temperature, top_p, llm_model)
    q: queue.Queue = queue.Queue(maxsize=512)

    def _worker() -> None:
        """Chạy stream_generate (blocking) trong thread riêng, đẩy chunk vào queue."""
        try:
            for chunk in generator.stream_generate(
                user_input, prep.contexts, history=prep.llm_history,
                tool_results=prep.tool_results or None,
                state_context=prep.state_context,
            ):
                q.put(("chunk", chunk))
            q.put(("done", getattr(generator, "_last_stream_answer", None)))
        except Exception as e:
            q.put(("error", str(e)))

    threading.Thread(target=_worker, daemon=True).start()

    streamed_text = ""
    _t_first_token: Optional[float] = None
    while True:
        kind, payload = await loop.run_in_executor(None, q.get)

        if kind == "chunk":
            if _t_first_token is None:
                _t_first_token = time.time()
                print(f"  [TIMER] TTFT     : {_t_first_token-_t_request_start:.2f}s  (first token)")
            streamed_text += payload
            yield _content(payload)

        elif kind == "error":
            yield _content(f"\n\nLỗi xử lý: {payload}")
            break

        else:  # done — append guardrails + citations sau phần text đã stream
            answer: Answer = payload or Answer(
                question=user_input, answer=streamed_text, citations=[],
            )
            guarded = apply_guardrails(
                answer, prep.contexts, warn_no_evidence=not prep.tool_results,
            )
            # apply_guardrails chỉ append vào cuối → phần thêm = đoạn sau text gốc
            tail = guarded.answer[len(answer.answer):]
            cit_block = _format_citations(guarded.citations)
            if cit_block:
                tail += "\n\n" + cit_block
            if tail:
                yield _content(tail)
            break

    yield _finish()
    yield "data: [DONE]\n\n"
    print(f"  [TIMER] ═══ E2E  : {time.time()-_t_request_start:.2f}s  (total user-visible latency)\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

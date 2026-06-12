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
from src.parent_store import ParentStore
from src.planner import LegalPlanner
from src.pipeline import LegalPipeline, RETRIEVAL_MODES, provider_credentials
from src.retriever import Retriever
from src.router import SmartRouter
from src.schemas import Answer
from src.tools import LegalToolRegistry
from src.vectorstore import VectorStore

try:
    from src.kg.kg_retriever import KGRetriever
    _KG_AVAILABLE = True
except Exception:
    _KG_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

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

    if config.VECTOR_BACKEND == "opensearch":
        # OpenSearch: vector kNN + BM25 native — backend query trực tiếp
        from src.opensearch_store import OpenSearchLexicalIndex, OpenSearchVectorStore
        vstore = OpenSearchVectorStore(config.OPENSEARCH_URL, config.OPENSEARCH_INDEX)
        bm25 = OpenSearchLexicalIndex(vstore)
        print(f"[API] Backend      : OpenSearch @ {config.OPENSEARCH_URL} (vector + BM25)")
    else:
        vstore = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)
        bm25_path = config.DATA_DIR / "bm25" / "index.json"
        bm25 = BM25Index(bm25_path) if bm25_path.exists() else None
        if bm25 is None:
            # Không có index.json (hoặc quá nặng để build) → dùng FTS5 có sẵn
            # trong chroma.sqlite3 làm nhánh lexical — zero RAM, zero build
            from src.fts_index import ChromaFTSIndex
            _fts = ChromaFTSIndex(config.VECTORSTORE_DIR)
            if _fts.is_available():
                bm25 = _fts
                print("[API] Lexical      : ChromaFTS (FTS5 trong chroma.sqlite3)")
            else:
                print("[API] Lexical      : TẮT (không có BM25 index lẫn FTS)")

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
        except Exception as e:
            print(f"[API] Không đọc được manifest top15 ({manifest_path}): {e} — mode top15 sẽ không filter")

    # Map provider → (key, host) dùng chung từ pipeline — bug cũ: dict local
    # thiếu "kieai" → api_key=None + host rơi về Ollama → crash lúc startup
    _api_key, _llm_host = provider_credentials(config.LLM_PROVIDER)

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

    pipeline = LegalPipeline(
        retriever=retriever,
        generator=generator,
        router=router,
        tool_registry=tool_registry,
        top15_urls=top15_urls,
        export_link_base=config.API_BASE_URL,
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
        "pipeline": pipeline,
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


# Đường dẫn tuyệt đối theo config — không phụ thuộc CWD lúc chạy uvicorn
_EXPORT_DIR = config.DATA_DIR / "exports"

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
    """Wrapper mỏng quanh LegalPipeline.run (logic chung ở src/pipeline.py)."""
    return _agent["pipeline"].run(
        user_input, history, rag_mode,
        top_k=top_k, web_search_enabled=web_search_enabled,
        temperature=temperature, top_p=top_p,
        ce_threshold=ce_threshold, llm_model=llm_model,
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

    pipeline: LegalPipeline = _agent["pipeline"]

    # ── Phase 1: router → tool → retrieve (blocking → thread pool) ───────────
    try:
        prep = await loop.run_in_executor(
            None,
            functools.partial(
                pipeline.prepare, user_input, history, rag_mode,
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
    generator = pipeline.make_generator(temperature, top_p, llm_model)
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
            guarded = pipeline.guard(answer, prep)
            # guardrails chỉ append vào cuối → phần thêm = đoạn sau text gốc
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

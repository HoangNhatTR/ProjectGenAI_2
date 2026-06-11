"""Pipeline chung cho Legal AI Agent — dùng bởi api.py / ui_app.py / app.py.

Trước đây logic router → tool → retrieve → rerank → generate bị viết lặp ở cả
3 entry point (API server, Streamlit UI, CLI). Module này gom về một chỗ:

  - RETRIEVAL_MODES        : cấu hình các chế độ RAG (kèm metadata hiển thị UI)
  - provider_credentials   : map provider → (api_key, host) từ config
  - make_generator         : clone Generator với overrides — KHÔNG mutate gốc
                             (an toàn khi nhiều request/tab dùng chung singleton)
  - LegalPipeline          : prepare (router→tool→retrieve→rerank) + run/guard
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from . import config
from .generator import Generator
from .guardrails import apply_guardrails
from .reranker import rerank as _rerank
from .schemas import Answer
from .state import ConversationState

MAX_HISTORY = 10

# ── Retrieval modes ────────────────────────────────────────────────────────────
# use_kg / use_top10_filter là logic; các key còn lại là metadata hiển thị UI.

RETRIEVAL_MODES: dict[str, dict] = {
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


# ── Provider / Generator factory ──────────────────────────────────────────────

def provider_credentials(provider: str) -> tuple[str, str]:
    """Map provider → (api_key, host) đọc từ config tại thời điểm gọi."""
    return {
        "router9":    (config.ROUTER9_API_KEY, config.ROUTER9_BASE_URL),
        "kieai":      (config.KIEAI_API_KEY, config.KIEAI_BASE_URL),
        "openrouter": (config.OPENROUTER_API_KEY, config.OPENROUTER_BASE_URL),
        "gemini":     (config.GEMINI_API_KEY or "", config.OLLAMA_HOST),
        "groq":       (config.GROQ_API_KEY or "", config.OLLAMA_HOST),
        "ollama":     ("", config.OLLAMA_HOST),
    }.get(provider, ("", config.OLLAMA_HOST))


def resolve_model_provider(llm_model: str) -> str:
    """Đoán provider từ model id: cc/ gh/ → 9Router; còn lại → KieAI."""
    return "router9" if llm_model.startswith(("cc/", "gh/")) else "kieai"


def make_generator(
    base: Generator,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_ctx: Optional[int] = None,
    api_key: Optional[str] = None,
    host: Optional[str] = None,
) -> Generator:
    """Clone `base` thành Generator mới với overrides — KHÔNG mutate base.

    Trước đây các entry point set trực tiếp `generator.model = ...` lên
    singleton dùng chung trong thread pool / cache_resource → hai request
    đồng thời ghi đè cấu hình của nhau. Factory này tạo instance riêng,
    và tái dùng client đã connect khi provider/host/key không đổi
    (client chỉ phụ thuộc 3 giá trị đó; model/temperature truyền per-call).
    """
    if provider and (api_key is None or host is None):
        _key, _host = provider_credentials(provider)
        api_key = _key if api_key is None else api_key
        host = _host if host is None else host

    provider = provider if provider is not None else base.provider
    api_key = api_key if api_key is not None else base.api_key
    host = host if host is not None else base.host

    gen = Generator(
        model=model if model is not None else base.model,
        host=host,
        temperature=base.temperature if temperature is None else temperature,
        num_ctx=base.num_ctx if num_ctx is None else num_ctx,
        top_p=base.top_p if top_p is None else top_p,
        provider=provider,
        api_key=api_key,
    )
    if (provider, host, api_key) == (base.provider, base.host, base.api_key):
        gen._client = base.get_client()
    return gen


# ── Pipeline ──────────────────────────────────────────────────────────────────

@dataclass
class PipelinePrep:
    """Kết quả phase chuẩn bị (router → tool → retrieve) — trước khi generate.

    final_answer được set khi flow đã có câu trả lời hoàn chỉnh
    (answer_direct, tool tự chứa kết quả, hoặc không có context).
    """
    final_answer: Optional[Answer] = None
    contexts: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    state_context: str = ""
    llm_history: list = field(default_factory=list)


class LegalPipeline:
    """Pipeline Legal AI Agent: router → tool → retrieve → rerank → generate.

    Components (retriever, generator, router, tool_registry) được inject —
    pipeline không tự load model để mỗi entry point tự quản lifecycle/cache.
    """

    def __init__(
        self,
        retriever: Any,
        generator: Generator,
        router: Any,
        tool_registry: Any,
        top15_urls: Optional[list[str]] = None,
        export_link_base: Optional[str] = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.router = router
        self.tool_registry = tool_registry
        self.top15_urls = top15_urls or []
        # Nếu set (vd http://localhost:8000) → append link tải DOCX vào kết quả
        # draft_document. API server set giá trị này; CLI/UI để None.
        self.export_link_base = export_link_base

    # ── Generator per-request ─────────────────────────────────────────────────

    def make_generator(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        llm_model: Optional[str] = None,
    ) -> Generator:
        """Generator riêng cho request, provider đoán từ prefix model id."""
        provider = None
        model = None
        if llm_model:
            model = llm_model.strip()
            provider = resolve_model_provider(model)
            logger.info(f"Model override → {model} (provider={provider})")
        return make_generator(
            self.generator,
            provider=provider, model=model,
            temperature=temperature, top_p=top_p,
        )

    # ── Phase 1: prepare ──────────────────────────────────────────────────────

    def prepare(
        self,
        user_input: str,
        history: list[dict],
        rag_mode: str,
        top_k: int = 5,
        web_search_enabled: bool = True,
        ce_threshold: float = 0.04,
    ) -> PipelinePrep:
        """Router → tool → retrieve → rerank (synchronous, chưa generate)."""
        mode_cfg = RETRIEVAL_MODES.get(rag_mode, RETRIEVAL_MODES["graph_rag"])
        use_kg = mode_cfg["use_kg"]
        allowed_sources = self.top15_urls if mode_cfg["use_top10_filter"] else None

        # Rebuild ConversationState từ history
        conv_state = ConversationState()
        for msg in history[-6:]:
            if msg["role"] == "user":
                conv_state.update_from_question(msg["content"])
            elif msg["role"] == "assistant":
                conv_state.update_from_answer(msg["content"], [])
        conv_state.update_from_question(user_input)

        llm_history = history[-MAX_HISTORY:]

        # ── Router ───────────────────────────────────────────────────────────
        _t0 = time.time()
        decision = self.router.route(
            user_input, history=llm_history,
            memory_text="", summary_text="",
            state=conv_state,
            web_search_enabled=web_search_enabled,
        )
        logger.info(
            f"Router {time.time()-_t0:.2f}s — intent={decision.intent} action={decision.action}"
        )

        # ── Flow A: trả lời trực tiếp ────────────────────────────────────────
        # (chitchat/meta/clarify — không phải tư vấn pháp lý → không disclaimer)
        if decision.action == "answer_direct":
            return PipelinePrep(
                final_answer=Answer(
                    question=user_input,
                    answer=decision.direct_response or "",
                    citations=[],
                ),
                llm_history=llm_history,
            )

        # ── Flow B: dùng tool ────────────────────────────────────────────────
        if decision.action == "use_tool" and decision.tool_name:
            return self._tool_flow(
                decision, user_input, conv_state, llm_history,
                top_k, use_kg, allowed_sources,
            )

        # ── Flow C: RAG retrieve ─────────────────────────────────────────────
        search_query = decision.search_query or user_input

        _t0 = time.time()
        contexts = self.retriever.retrieve(
            search_query, top_k=top_k, use_kg=use_kg, allowed_sources=allowed_sources,
            use_hyde=config.USE_HYDE, use_parent_expansion=True,
        )
        logger.info(f"Retrieve {time.time()-_t0:.2f}s — {len(contexts)} chunks")

        # Rerank — smart skip CrossEncoder nếu RRF score đã cao
        if contexts:
            _use_ce = contexts[0].score < ce_threshold
            contexts = _rerank(
                search_query, contexts, top_k=top_k, use_cross_encoder=_use_ce,
            )
            logger.info(f"Rerank CE={'on' if _use_ce else 'skip'} ({len(contexts)} chunks)")

        if not contexts:
            ans = Answer(
                question=user_input,
                answer=(
                    "Không tìm thấy căn cứ pháp lý trong cơ sở dữ liệu. "
                    "Vui lòng thử câu hỏi khác hoặc tham khảo ý kiến luật sư."
                ),
                citations=[],
            )
            return PipelinePrep(
                final_answer=apply_guardrails(ans, []),
                llm_history=llm_history,
            )

        return PipelinePrep(
            contexts=contexts,
            state_context=conv_state.to_context_string(),
            llm_history=llm_history,
        )

    def _tool_flow(
        self,
        decision: Any,
        user_input: str,
        conv_state: ConversationState,
        llm_history: list[dict],
        top_k: int,
        use_kg: bool,
        allowed_sources: Optional[list[str]],
    ) -> PipelinePrep:
        """Flow B: thực thi tool theo router decision."""
        tool_registry = self.tool_registry
        tool_name = decision.tool_name
        tool_query = decision.tool_query or user_input

        if tool_name == "calculate_fine":
            tool_result = tool_registry.calculate_fine(description=tool_query)

        elif tool_name == "draft_document":
            dt, det = (tool_query.split("|", 1) if "|" in tool_query
                       else ("văn bản pháp lý", tool_query))
            tool_result = tool_registry.draft_document(
                doc_type=dt.strip(), details=det.strip(),
            )
            # Nếu export DOCX thành công và có API base → append download link
            if tool_result.success and tool_result.docx_path and self.export_link_base:
                _fname = Path(tool_result.docx_path).name
                _link = (
                    f"\n\n---\n"
                    f"📎 **[⬇️ Tải file DOCX]({self.export_link_base}/v1/export/{_fname})**  "
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
                parts = tool_query.split(" và " if " và " in tool_query else " vs ")
                topic_a = parts[0].strip()
                topic_b = parts[1].strip() if len(parts) > 1 else user_input
            tool_result = tool_registry.compare_regulations(
                topic_a=topic_a.strip(), topic_b=topic_b.strip(),
            )

        elif tool_name == "validate_document":
            # tool_query là nội dung văn bản hoặc mô tả (khi không có file).
            # Thử tìm nội dung file từ system message trong history.
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

        # Validate và compare đã tự chứa kết quả đầy đủ — trả thẳng không qua
        # generator. Vẫn áp guardrails (disclaimer) nhưng tắt cảnh báo
        # "thiếu căn cứ" vì kết quả tool đã tự chứa căn cứ.
        if tool_name in ("validate_document", "compare_regulations") and tool_result.success:
            ans = Answer(question=user_input, answer=tool_result.result, citations=[])
            return PipelinePrep(
                final_answer=apply_guardrails(ans, [], warn_no_evidence=False),
                llm_history=llm_history,
            )

        search_q = decision.search_query or user_input
        contexts = self.retriever.retrieve(
            search_q, top_k=top_k, use_kg=use_kg, allowed_sources=allowed_sources,
        )

        return PipelinePrep(
            contexts=contexts,
            tool_results=[tool_result],
            state_context=conv_state.to_context_string(),
            llm_history=llm_history,
        )

    # ── Phase 2: generate + guardrails ────────────────────────────────────────

    def guard(self, answer: Answer, prep: PipelinePrep) -> Answer:
        """Áp guardrails cho answer đã generate (chính sách chung mọi flow)."""
        # Tool flow: kết quả tool là căn cứ → không cảnh báo "thiếu căn cứ"
        return apply_guardrails(
            answer, prep.contexts, warn_no_evidence=not prep.tool_results,
        )

    def run(
        self,
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

        prep = self.prepare(
            user_input, history, rag_mode, top_k, web_search_enabled, ce_threshold,
        )
        if prep.final_answer is not None:
            logger.info(f"TOTAL {time.time()-_t_total:.2f}s (không cần generate)")
            return prep.final_answer

        generator = self.make_generator(temperature, top_p, llm_model)

        _t0 = time.time()
        answer = generator.generate(
            user_input, prep.contexts, history=prep.llm_history,
            tool_results=prep.tool_results or None,
            state_context=prep.state_context,
        )
        logger.info(f"Generate {time.time()-_t0:.2f}s — TOTAL {time.time()-_t_total:.2f}s")

        return self.guard(answer, prep)

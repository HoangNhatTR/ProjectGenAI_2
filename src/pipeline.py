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
from .cache import RetrievalCache
from .confidence import compute_confidence
from .generator import Generator
from .guardrails import apply_guardrails
from .reranker import cap_per_doc, rerank as _rerank
from .retriever import _rule_expand
from .schemas import Answer
from .state import ConversationState

MAX_HISTORY = 10

# Intent chắc chắn là câu tra cứu đơn giản → khỏi tốn 1 LLM call cho planner
# (đồng bộ với app.py — trước đây chỉ CLI/Streamlit có planner, API thì không)
_SIMPLE_INTENTS = {"legal", "consulting", "followup"}

# ── Retrieval modes ────────────────────────────────────────────────────────────
# use_kg / use_top10_filter là logic; các key còn lại là metadata hiển thị UI.

RETRIEVAL_MODES: dict[str, dict] = {
    "rag_full": {
        "label": "RAG Full",
        "icon": "📚",
        "short": "Toàn bộ corpus",
        "desc": "Vector + BM25 trên toàn bộ corpus (51.673 văn bản / 4,9M chunks)",
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
        # use_top10_filter=False từ 2026-06-13: corpus đã đủ 4,9M chunks toàn bộ
        # VB (kể cả nghị định xử phạt như 168/2024) — filter top-15 luật khiến
        # câu hỏi mức phạt không bao giờ thấy nghị định, chỉ ra Bộ luật Hình sự
        "desc": "Vector + BM25 toàn corpus + Knowledge Graph semantic",
        "use_kg": True,
        "use_top10_filter": False,
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
    """Đoán provider từ model id: cc/ gh/ kc/ → 9Router; còn lại → KieAI."""
    return "router9" if llm_model.startswith(("cc/", "gh/", "kc/")) else "kieai"


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
    """Pipeline Legal AI Agent: router → tool/planner → retrieve → rerank → generate.

    Components (retriever, generator, router, tool_registry, planner) được inject —
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
        retrieval_cache: Optional[RetrievalCache] = None,
        planner: Any = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.router = router
        self.tool_registry = tool_registry
        # Planner đa bước (optional) — None = Flow C RAG thuần như cũ
        self.planner = planner
        self.top15_urls = top15_urls or []
        # Nếu set (vd http://localhost:8000) → append link tải DOCX vào kết quả
        # draft_document. API server set giá trị này; CLI/UI để None.
        self.export_link_base = export_link_base
        # Cache retrieve+rerank — tránh chạy lại embedding + BM25 + CrossEncoder
        # cho cùng (query, mode, top_k). Trước đây chỉ app.py (CLI) có cache;
        # api.py production thì không → mọi request lặp đều tốn full retrieval.
        self.retrieval_cache = retrieval_cache or RetrievalCache(maxsize=512, ttl=3600)

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

    # ── Retrieve + rerank (có cache) ──────────────────────────────────────────

    def _retrieve_rerank(
        self,
        search_query: str,
        rag_mode: str,
        top_k: int,
        use_kg: bool,
        allowed_sources: Optional[list[str]],
        ce_threshold: float,
    ) -> list:
        """Retrieve + rerank với cache (key theo rag_mode + top_k + query).

        Cùng logic cho Flow C (RAG) và Flow B (citation của tool) — gom về một
        chỗ + thêm cache. Smart skip CrossEncoder khi top RRF score đã cao.
        """
        cache = self.retrieval_cache
        cached = cache.get(search_query, top_k, None, namespace=rag_mode)
        if cached is not None:
            logger.info(f"Retrieve+rerank CACHE HIT — {len(cached)} chunks [{cache.stats_str()}]")
            return cached

        # Phễu rerank: retrieve pool RỘNG rồi CE cắt về top_k. Pool = top_k (cũ)
        # khiến CE chỉ đảo thứ tự chunk có sẵn, không cứu được chunk đúng hạng 8.
        pool = max(top_k * config.RERANK_POOL_MULT, config.RERANK_POOL_MIN)

        # Parent expansion để SAU rerank: CrossEncoder chấm child chunk ngắn
        # (~600 ký tự) thay vì full Điều đã expand (hàng nghìn ký tự) — CE trên
        # CPU scale theo độ dài input, đây là khâu tốn 1.5-5s/câu trước đây.
        # CE thấy đúng đoạn text đã match cũng cho tín hiệu relevance sạch hơn.
        _t0 = time.time()
        contexts = self.retriever.retrieve(
            search_query, top_k=pool, use_kg=use_kg, allowed_sources=allowed_sources,
            use_hyde=config.USE_HYDE, use_parent_expansion=False,
            use_multi_query=config.USE_MULTI_QUERY,
        )
        logger.info(f"Retrieve {time.time()-_t0:.2f}s — {len(contexts)} chunks (pool={pool})")

        if contexts:
            _use_ce = contexts[0].score < ce_threshold
            _t0 = time.time()
            # Rerank theo câu đã RULE-EXPAND (thuật ngữ luật) nếu có: câu thô
            # "vượt đèn đỏ" khiến reranker chấm Đ7 Khoản 1 ≈ Khoản 7 (đều thấp
            # ~0.22, K1 thắng sát nút) vì header Điều dài giống nhau + từ dân dã
            # ≠ từ luật; câu expand "không chấp hành hiệu lệnh đèn tín hiệu" cho
            # K7=1.00 (đo 2026-07-09). Retrieval đã dùng expand, rerank cũng nên.
            rerank_query = _rule_expand(search_query) or search_query
            contexts = _rerank(
                rerank_query, contexts, top_k=None, use_cross_encoder=_use_ce,
            )
            # Per-doc cap SAU rerank (trước đây TRƯỚC): chunk đúng khoản thường
            # chỉ xuất hiện ở 1 query variant (rule expansion) nên điểm RRF thô
            # thấp → cap-trước cắt mất nó trước khi rerank kịp chấm cao (bug
            # 2026-07-09: "xe máy vượt đèn đỏ" mất Đ7K7 NĐ168, generator lấy
            # nhầm Đ7K1/Đ8). Cap-sau giữ top-N chunk/VB theo điểm RERANK.
            contexts = cap_per_doc(contexts)
            # Retriever inject từ ngoài có thể không có expand_parents (test stub)
            _expand = getattr(self.retriever, "expand_parents", None)
            if _expand is not None:
                contexts = _expand(contexts)
            contexts = contexts[:top_k]
            logger.info(
                f"Rerank CE={'on' if _use_ce else 'skip'} {time.time()-_t0:.2f}s "
                f"({len(contexts)} chunks)"
            )

            # Ngưỡng "không đủ căn cứ" — chỉ khi CE chạy (score đã chuẩn hóa
            # sigmoid). MIN_EVIDENCE_SCORE=0 (mặc định) = tắt, chờ calibrate.
            if (
                contexts and _use_ce
                and config.MIN_EVIDENCE_SCORE > 0
                and contexts[0].score < config.MIN_EVIDENCE_SCORE
            ):
                logger.info(
                    f"Top score {contexts[0].score:.3f} < MIN_EVIDENCE_SCORE="
                    f"{config.MIN_EVIDENCE_SCORE} → coi như không đủ căn cứ"
                )
                contexts = []

        cache.put(search_query, top_k, None, contexts, namespace=rag_mode)
        return contexts

    # ── Phase 1: prepare ──────────────────────────────────────────────────────

    def prepare(
        self,
        user_input: str,
        history: list[dict],
        rag_mode: str,
        top_k: int = 5,
        web_search_enabled: bool = True,
        ce_threshold: float = 0.04,
        use_planner: bool = True,
        skip_router: bool = False,
        search_query: Optional[str] = None,
        no_retrieval: bool = False,
        memory_text: str = "",
        summary_text: str = "",
    ) -> PipelinePrep:
        """Router → tool → (planner) → retrieve → rerank (synchronous, chưa generate).

        skip_router=True: máy gọi máy (vd Module 2 /analyze) — đi thẳng Flow C
        RAG, không router (LLM bất định), không planner, không tool. Trước đây
        Module 2 phải "giả giọng người" + guard + retry 3 lần vì router thỉnh
        thoảng đẩy nhầm sang validate_document.

        search_query (chỉ dùng với skip_router): truy vấn RETRIEVE riêng —
        caller máy biết chính xác cần tìm gì (vd "khung hình phạt điểm c khoản 1
        Điều 250"), còn prompt generate có thể dài đầy hướng dẫn.

        no_retrieval (chỉ dùng với skip_router): caller máy biết chắc tác vụ
        KHÔNG cần tra kho luật (vd trích ngày/lập luận từ chính tài liệu đưa
        vào prompt) — bỏ hẳn bước retrieve+rerank trên 5 triệu chunk thay vì
        chỉ rút gọn search_query. Trước đây thiếu search_query khiến pipeline
        dùng NGUYÊN prompt dài (có thể hàng chục nghìn ký tự) làm câu truy vấn
        retrieve → tốn nhiều giây vô ích mỗi tài liệu (vd dossier_timeline,
        dossier_arguments) mà câu trả lời không hề dùng tới kết quả đó.

        memory_text/summary_text: long-term memory (facts về user) + rolling
        summary — CLIENT tự lưu và gửi kèm (server stateless). Trước đây 2 lớp
        này chỉ sống ở CLI/Streamlit, đường API bị hardcode chuỗi rỗng.
        """
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

        # ── Đường tắt cho machine call: RAG trực tiếp, tất định ─────────────
        if skip_router:
            if no_retrieval:
                logger.info("skip_router=True, no_retrieval=True → bỏ qua retrieve, generate thẳng")
                return PipelinePrep(
                    contexts=[],
                    state_context=conv_state.to_context_string(),
                    llm_history=llm_history,
                )
            _sq = (search_query or "").strip() or user_input
            logger.info(f"skip_router=True → Flow C trực tiếp (search='{_sq[:60]}')")
            contexts = self._retrieve_rerank(
                _sq, rag_mode, top_k, use_kg, allowed_sources, ce_threshold,
            )
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

        # ── Router ───────────────────────────────────────────────────────────
        _t0 = time.time()
        decision = self.router.route(
            user_input, history=llm_history,
            memory_text=memory_text, summary_text=summary_text,
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
                top_k, use_kg, allowed_sources, ce_threshold, rag_mode,
            )

        # ── Flow C: RAG retrieve (+ optional Planner đa bước) ────────────────
        search_query = decision.search_query or user_input
        tool_results: list = []

        # Planner: câu hỏi phức hợp (nhiều lỗi vi phạm, so sánh, tính tổng...)
        # → lập plan → execute tool steps → retrieve theo query của plan.
        # Chỉ gọi khi intent KHÔNG chắc chắn đơn giản (tiết kiệm 1 LLM call).
        if (
            use_planner
            and self.planner is not None
            and decision.intent not in _SIMPLE_INTENTS
        ):
            try:
                _t0 = time.time()
                plan = self.planner.create_plan(user_input, state=conv_state)
                if plan.is_complex():
                    tool_results = self.planner.execute_plan(plan)
                    search_query = plan.retrieve_query()
                    logger.info(
                        f"Planner {time.time()-_t0:.2f}s — complex, "
                        f"{len(tool_results)} tool steps"
                    )
            except Exception as exc:
                # Planner không được phép làm chết flow RAG
                logger.warning(f"Planner lỗi, tiếp tục RAG thuần: {exc}")
                tool_results = []

        contexts = self._retrieve_rerank(
            search_query, rag_mode, top_k, use_kg, allowed_sources, ce_threshold,
        )

        if not contexts and not tool_results:
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
            tool_results=tool_results,
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
        ce_threshold: float = 0.04,
        rag_mode: str = "graph_rag",
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

        # Retrieve cho citation của tool flow — cùng chất lượng với Flow C:
        # parent expansion (mức tiền phạt thường nằm ở câu mở đầu Khoản = parent)
        # + rerank (recency boost đè văn bản cũ xuống). Thiếu 2 bước này từng
        # khiến citation toàn Bộ luật Hình sự 2015 thay vì NĐ 168/2024.
        search_q = decision.search_query or user_input
        contexts = self._retrieve_rerank(
            search_q, rag_mode, top_k, use_kg, allowed_sources, ce_threshold,
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
        # Tool flow: kết quả tool là căn cứ → không cảnh báo "thiếu căn cứ",
        # và không kiểm chứng trích dẫn vs contexts (căn cứ nằm trong tool result)
        guarded = apply_guardrails(
            answer, prep.contexts,
            warn_no_evidence=not prep.tool_results,
            verify_cited=not prep.tool_results,
        )
        # Nhãn độ tin cậy (P2.3) — chỉ áp cho flow RAG thật (không phải tool
        # flow, căn cứ ở đó không đến từ contexts nên 4 tín hiệu vô nghĩa).
        # Tính trên answer.answer GỐC (trước khi apply_guardrails nối thêm
        # disclaimer/warning) — text đó không chứa "Điều N" nên không đổi kết
        # quả verify_citations, nhưng tính trên bản gốc cho rõ ràng.
        if not prep.tool_results:
            confidence = compute_confidence(answer.answer, answer.citations, prep.contexts)
            guarded = guarded.model_copy(update={"confidence": confidence})
        return guarded

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
        use_planner: bool = True,
        skip_router: bool = False,
        search_query: Optional[str] = None,
        no_retrieval: bool = False,
        memory_text: str = "",
        summary_text: str = "",
    ) -> Answer:
        """Chạy toàn bộ pipeline (synchronous, non-streaming). Trả về Answer."""
        _t_total = time.time()

        prep = self.prepare(
            user_input, history, rag_mode, top_k, web_search_enabled, ce_threshold,
            use_planner=use_planner, skip_router=skip_router, search_query=search_query,
            no_retrieval=no_retrieval,
            memory_text=memory_text, summary_text=summary_text,
        )
        if prep.final_answer is not None:
            logger.info(f"TOTAL {time.time()-_t_total:.2f}s (không cần generate)")
            return prep.final_answer

        generator = self.make_generator(temperature, top_p, llm_model)

        _t0 = time.time()
        answer = generator.generate(
            user_input, prep.contexts, history=prep.llm_history,
            memory_text=memory_text, summary_text=summary_text,
            tool_results=prep.tool_results or None,
            state_context=prep.state_context,
        )
        logger.info(f"Generate {time.time()-_t0:.2f}s — TOTAL {time.time()-_t_total:.2f}s")

        return self.guard(answer, prep)

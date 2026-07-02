"""Tests planner trong LegalPipeline — Flow C phải có multi-step planning.

Trước đây planner chỉ nối vào app.py (CLI) + ui_app.py (Streamlit); api.py
tạo planner nhưng KHÔNG truyền vào pipeline → frontend Next.js không bao giờ
được planning. Các test này chốt hành vi sau khi planner vào pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.pipeline import LegalPipeline
from src.planner import Plan, PlanStep
from src.schemas import Chunk, DocumentMetadata, RetrievedChunk


# ── Stubs ──────────────────────────────────────────────────────────────────────

@dataclass
class _Decision:
    action: str = "retrieve"
    intent: str = "calculate"
    tool_name: Optional[str] = None
    tool_query: Optional[str] = None
    search_query: Optional[str] = None
    direct_response: Optional[str] = None


class _StubRouter:
    def __init__(self, decision: _Decision):
        self.decision = decision

    def route(self, *a, **kw):
        return self.decision


class _StubRetriever:
    def __init__(self, chunks: Optional[list[RetrievedChunk]] = None):
        self.chunks = chunks if chunks is not None else []
        self.queries: list[str] = []

    def retrieve(self, query, **kw):
        self.queries.append(query)
        return list(self.chunks)


@dataclass
class _StubToolResult:
    tool_name: str = "calculate_fine"
    success: bool = True
    result: str = "Tổng phạt 5 triệu"
    sources: list = field(default_factory=list)
    docx_path: Optional[str] = None


class _StubPlanner:
    def __init__(self, plan: Plan, results: Optional[list] = None):
        self.plan = plan
        self.results = results or [_StubToolResult()]
        self.create_calls = 0

    def create_plan(self, question, state=None):
        self.create_calls += 1
        return self.plan

    def execute_plan(self, plan):
        return self.results


class _StubGenerator:
    """Đủ attribute cho LegalPipeline.__init__ (không generate trong prepare)."""
    model = "stub"
    host = ""
    temperature = 0.0
    num_ctx = 2048
    top_p = 1.0
    provider = "ollama"
    api_key = ""

    def get_client(self):
        return None


def _chunk(cid: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=cid, text=f"Điều 8. Nội dung {cid}", article="Điều 8",
            metadata=DocumentMetadata(source="https://vbpl.vn/x"),
        ),
        score=0.9,  # > ce_threshold → skip CrossEncoder trong rerank
    )


def _complex_plan(question: str = "q") -> Plan:
    return Plan(
        complexity="complex", reason="2 lỗi vi phạm", question=question,
        steps=[
            PlanStep(step=1, description="tính phạt", tool="calculate_fine",
                     query="vượt đèn đỏ xe máy"),
            PlanStep(step=2, description="tra thêm", tool="retrieve",
                     query="mức phạt vượt đèn đỏ không gương xe máy"),
        ],
    )


def _pipeline(planner=None, retriever=None, decision=None) -> LegalPipeline:
    return LegalPipeline(
        retriever=retriever if retriever is not None else _StubRetriever([_chunk()]),
        generator=_StubGenerator(),
        router=_StubRouter(decision or _Decision()),
        tool_registry=None,
        planner=planner,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_plan_complex_chay_tool_va_doi_search_query():
    retriever = _StubRetriever([_chunk()])
    planner = _StubPlanner(_complex_plan())
    p = _pipeline(planner=planner, retriever=retriever)

    prep = p.prepare("vượt đèn đỏ + không gương phạt bao nhiêu?", [], "graph_rag")

    assert planner.create_calls == 1
    assert len(prep.tool_results) == 1
    assert prep.final_answer is None  # vẫn cần generate
    # Retrieve phải dùng query của bước retrieve trong plan
    assert retriever.queries == ["mức phạt vượt đèn đỏ không gương xe máy"]


def test_intent_don_gian_khong_goi_planner():
    planner = _StubPlanner(_complex_plan())
    p = _pipeline(planner=planner, decision=_Decision(intent="legal"))

    prep = p.prepare("mức phạt vượt đèn đỏ?", [], "graph_rag")

    assert planner.create_calls == 0
    assert prep.tool_results == []


def test_use_planner_false_tat_planner():
    planner = _StubPlanner(_complex_plan())
    p = _pipeline(planner=planner)

    p.prepare("câu hỏi phức tạp", [], "graph_rag", use_planner=False)

    assert planner.create_calls == 0


def test_khong_co_planner_flow_nhu_cu():
    p = _pipeline(planner=None)
    prep = p.prepare("câu hỏi", [], "graph_rag")
    assert prep.tool_results == []
    assert prep.contexts  # vẫn retrieve bình thường


def test_planner_loi_khong_chet_flow():
    class _BrokenPlanner:
        def create_plan(self, *a, **kw):
            raise RuntimeError("LLM down")

    p = _pipeline(planner=_BrokenPlanner())
    prep = p.prepare("câu hỏi phức tạp", [], "graph_rag")
    assert prep.final_answer is None
    assert prep.contexts  # RAG thuần vẫn chạy


def test_tool_results_co_nhung_contexts_rong_van_generate():
    """Có kết quả tool mà retrieve rỗng → KHÔNG trả 'không tìm thấy căn cứ'."""
    retriever = _StubRetriever([])  # không tìm được gì
    planner = _StubPlanner(_complex_plan())
    p = _pipeline(planner=planner, retriever=retriever)

    prep = p.prepare("tính tổng phạt 2 lỗi", [], "graph_rag")

    assert prep.final_answer is None  # đi tiếp generate từ tool results
    assert len(prep.tool_results) == 1


def test_plan_simple_khong_execute_tool():
    simple = Plan(
        complexity="simple", reason="tra cứu", question="q",
        steps=[PlanStep(step=1, description="retrieve", tool="retrieve", query="q")],
    )
    planner = _StubPlanner(simple)
    retriever = _StubRetriever([_chunk()])
    p = _pipeline(planner=planner, retriever=retriever)

    prep = p.prepare("câu hỏi thường", [], "graph_rag")

    assert prep.tool_results == []
    # Plan simple → giữ search query của router/câu gốc, không lấy từ plan
    assert retriever.queries == ["câu hỏi thường"]

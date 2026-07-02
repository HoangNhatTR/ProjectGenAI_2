"""Tests skip_router — đường tắt máy-gọi-máy (Module 2 /analyze) trong pipeline."""
from __future__ import annotations

from src.pipeline import LegalPipeline
from src.schemas import Chunk, DocumentMetadata, RetrievedChunk


class _RouterMustNotBeCalled:
    def route(self, *a, **kw):
        raise AssertionError("Router KHÔNG được gọi khi skip_router=True")


class _PlannerMustNotBeCalled:
    def create_plan(self, *a, **kw):
        raise AssertionError("Planner KHÔNG được gọi khi skip_router=True")


class _StubRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries: list[str] = []

    def retrieve(self, query, **kw):
        self.queries.append(query)
        return list(self.chunks)


class _StubGenerator:
    model = "stub"
    host = ""
    temperature = 0.0
    num_ctx = 2048
    top_p = 1.0
    provider = "ollama"
    api_key = ""

    def get_client(self):
        return None


def _rc(cid: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=cid, text=f"Điều 250. {cid}", article="Điều 250",
                    metadata=DocumentMetadata(source="https://vbpl.vn/x")),
        score=0.9,
    )


def _pipeline(retriever) -> LegalPipeline:
    return LegalPipeline(
        retriever=retriever, generator=_StubGenerator(),
        router=_RouterMustNotBeCalled(), tool_registry=None,
        planner=_PlannerMustNotBeCalled(),
    )


def test_skip_router_khong_goi_router_va_planner():
    retriever = _StubRetriever([_rc()])
    p = _pipeline(retriever)

    prep = p.prepare("prompt dài đầy hướng dẫn 1) 2) 3)", [], "graph_rag",
                     skip_router=True)

    assert prep.final_answer is None
    assert prep.contexts
    # Không có search_query → retrieve bằng chính user_input
    assert retriever.queries == ["prompt dài đầy hướng dẫn 1) 2) 3)"]


def test_skip_router_dung_search_query_rieng():
    """Prompt generate dài; retrieve phải dùng truy vấn đích danh caller đưa."""
    retriever = _StubRetriever([_rc()])
    p = _pipeline(retriever)

    p.prepare(
        "Một người bị Tòa án xử phạt... (prompt dài)", [], "graph_rag",
        skip_router=True,
        search_query="khung hình phạt điểm c khoản 1 Điều 250 Bộ luật Hình sự",
    )

    assert retriever.queries == [
        "khung hình phạt điểm c khoản 1 Điều 250 Bộ luật Hình sự"
    ]


def test_skip_router_khong_context_tra_khong_du_can_cu():
    retriever = _StubRetriever([])
    p = _pipeline(retriever)

    prep = p.prepare("câu hỏi", [], "graph_rag", skip_router=True)

    assert prep.final_answer is not None
    assert "Không tìm thấy căn cứ" in prep.final_answer.answer


def test_khong_skip_thi_van_di_duong_router():
    """skip_router=False (mặc định) → router được gọi như cũ."""
    import pytest

    retriever = _StubRetriever([_rc()])
    p = _pipeline(retriever)

    with pytest.raises(AssertionError, match="Router KHÔNG được gọi"):
        p.prepare("câu hỏi", [], "graph_rag")

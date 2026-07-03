"""Tests memory_text/summary_text xuyên pipeline — vá lỗ hổng đường API.

Trước đây pipeline hardcode memory_text=""/summary_text="" khi gọi router và
không truyền gì cho generator → long-term memory + rolling summary chỉ sống
ở CLI/Streamlit, frontend Next.js không bao giờ có.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.pipeline import LegalPipeline
from src.schemas import Answer, Chunk, DocumentMetadata, RetrievedChunk


@dataclass
class _Decision:
    action: str = "retrieve"
    intent: str = "legal"
    tool_name: Optional[str] = None
    tool_query: Optional[str] = None
    search_query: Optional[str] = None
    direct_response: Optional[str] = None


class _CaptureRouter:
    def __init__(self):
        self.kwargs: dict = {}

    def route(self, *a, **kw):
        self.kwargs = kw
        return _Decision()


class _StubRetriever:
    def retrieve(self, query, **kw):
        return [RetrievedChunk(
            chunk=Chunk(chunk_id="c1", text="Điều 1. nội dung",
                        metadata=DocumentMetadata(source="https://vbpl.vn/x")),
            score=0.9,
        )]


class _CaptureGenerator:
    """Đóng cả 2 vai: base generator của pipeline + generator per-request."""
    model = "stub"
    host = ""
    temperature = 0.0
    num_ctx = 2048
    top_p = 1.0
    provider = "ollama"
    api_key = ""

    def __init__(self):
        self.generate_kwargs: dict = {}

    def get_client(self):
        return None

    def generate(self, question, contexts, **kw):
        self.generate_kwargs = kw
        return Answer(question=question, answer="ok Điều 1", citations=[])


def _pipeline(router, gen) -> LegalPipeline:
    p = LegalPipeline(
        retriever=_StubRetriever(), generator=gen,
        router=router, tool_registry=None,
    )
    # per-request generator = chính capture gen (né make_generator tạo Generator thật)
    p.make_generator = lambda *a, **kw: gen  # type: ignore[method-assign]
    return p


def test_router_nhan_memory_va_summary():
    router = _CaptureRouter()
    p = _pipeline(router, _CaptureGenerator())

    p.prepare("câu hỏi", [], "graph_rag",
              memory_text="- Là tài xế xe máy", summary_text="Đang hỏi về phạt nguội")

    assert router.kwargs["memory_text"] == "- Là tài xế xe máy"
    assert router.kwargs["summary_text"] == "Đang hỏi về phạt nguội"


def test_mac_dinh_van_rong_nhu_cu():
    router = _CaptureRouter()
    p = _pipeline(router, _CaptureGenerator())

    p.prepare("câu hỏi", [], "graph_rag")

    assert router.kwargs["memory_text"] == ""
    assert router.kwargs["summary_text"] == ""


def test_generator_nhan_memory_va_summary_qua_run():
    gen = _CaptureGenerator()
    p = _pipeline(_CaptureRouter(), gen)

    p.run("câu hỏi", [], "graph_rag",
          memory_text="- Chủ doanh nghiệp vận tải", summary_text="Tóm tắt cũ")

    assert gen.generate_kwargs["memory_text"] == "- Chủ doanh nghiệp vận tải"
    assert gen.generate_kwargs["summary_text"] == "Tóm tắt cũ"

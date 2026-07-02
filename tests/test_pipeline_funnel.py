"""Tests phễu rerank + per-doc cap + ngưỡng không-đủ-căn-cứ trong pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import src.pipeline as pipeline_mod
from src.pipeline import LegalPipeline
from src.schemas import Chunk, DocumentMetadata, RetrievedChunk


@dataclass
class _Decision:
    action: str = "retrieve"
    intent: str = "legal"
    tool_name: Optional[str] = None
    tool_query: Optional[str] = None
    search_query: Optional[str] = None
    direct_response: Optional[str] = None


class _StubRouter:
    def route(self, *a, **kw):
        return _Decision()


class _StubRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.requested_top_k: list[int] = []

    def retrieve(self, query, top_k, **kw):
        self.requested_top_k.append(top_k)
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


def _rc(cid: str, source: str = "https://vbpl.vn/a", score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=cid, text=f"Điều 1. {cid}",
                    metadata=DocumentMetadata(source=source)),
        score=score,
    )


def _pipeline(retriever) -> LegalPipeline:
    return LegalPipeline(
        retriever=retriever, generator=_StubGenerator(),
        router=_StubRouter(), tool_registry=None,
    )


def test_phau_rerank_retrieve_pool_rong_hon_top_k():
    """top_k=5 → retrieve phải xin pool max(5*4, 20) = 20 ứng viên."""
    retriever = _StubRetriever([_rc(f"c{i}") for i in range(3)])
    p = _pipeline(retriever)

    p.prepare("mức phạt vượt đèn đỏ", [], "graph_rag", top_k=5)

    assert retriever.requested_top_k == [20]


def test_ket_qua_cuoi_van_cat_ve_top_k():
    retriever = _StubRetriever(
        [_rc(f"c{i}", source=f"https://vbpl.vn/{i}") for i in range(20)]
    )
    p = _pipeline(retriever)

    prep = p.prepare("mức phạt vượt đèn đỏ", [], "graph_rag", top_k=5)

    assert len(prep.contexts) == 5


def test_per_doc_cap_da_dang_hoa_pool():
    """6 chunk cùng 1 VB + 1 VB khác → cap 3/VB nên VB kia phải lọt top-5."""
    chunks = [_rc(f"a{i}", source="https://vbpl.vn/to") for i in range(6)]
    chunks.append(_rc("b0", source="https://vbpl.vn/nd168"))
    retriever = _StubRetriever(chunks)
    p = _pipeline(retriever)

    prep = p.prepare("câu hỏi", [], "graph_rag", top_k=5)

    sources = [c.chunk.metadata.source for c in prep.contexts]
    assert "https://vbpl.vn/nd168" in sources
    assert sources.count("https://vbpl.vn/to") <= 3


def test_min_evidence_score_tra_khong_du_can_cu(monkeypatch):
    """CE chạy + top score dưới ngưỡng → coi như không tìm thấy căn cứ."""
    # score 0.01 < ce_threshold 0.04 → _use_ce=True; patch _rerank để không load CE thật
    retriever = _StubRetriever([_rc("c1", score=0.01)])
    monkeypatch.setattr(pipeline_mod, "_rerank", lambda q, ctxs, top_k, use_cross_encoder: ctxs)
    monkeypatch.setattr(pipeline_mod.config, "MIN_EVIDENCE_SCORE", 0.3)
    p = _pipeline(retriever)

    prep = p.prepare("câu hỏi ngoài phạm vi corpus", [], "graph_rag", top_k=5)

    assert prep.final_answer is not None
    assert "Không tìm thấy căn cứ" in prep.final_answer.answer


def test_min_evidence_score_mac_dinh_tat(monkeypatch):
    """MIN_EVIDENCE_SCORE=0 (mặc định) → không loại contexts dù score thấp."""
    retriever = _StubRetriever([_rc("c1", score=0.01)])
    monkeypatch.setattr(pipeline_mod, "_rerank", lambda q, ctxs, top_k, use_cross_encoder: ctxs)
    monkeypatch.setattr(pipeline_mod.config, "MIN_EVIDENCE_SCORE", 0.0)
    p = _pipeline(retriever)

    prep = p.prepare("câu hỏi", [], "graph_rag", top_k=5)

    assert prep.final_answer is None
    assert prep.contexts

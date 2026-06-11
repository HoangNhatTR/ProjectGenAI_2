"""Tests cho temporal factor trong reranker: ưu tiên VB hiện hành."""
from __future__ import annotations

import pytest

from src import reranker
from src.reranker import _temporal_factor, rerank
from src.schemas import Chunk, DocumentMetadata, RetrievedChunk


def _rc(score: float = 0.02, *, issued: str = None, status: str = None,
        folder: str = None, cid: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=cid,
            text="Nội dung quy định chung về xử phạt vi phạm.",
            metadata=DocumentMetadata(
                source=f"https://x/{cid}", issued_date=issued,
                status=status, folder=folder,
            ),
        ),
        score=score,
    )


@pytest.fixture(autouse=True)
def _recency_on(monkeypatch):
    monkeypatch.setattr(reranker, "RECENCY_ENABLED", True)


def test_van_ban_moi_factor_cao_hon_van_ban_cu():
    new = _temporal_factor(_rc(issued="2024-08-23").chunk)
    old = _temporal_factor(_rc(issued="1995-09-26").chunk)
    assert new > old
    assert old >= reranker.RECENCY_FLOOR * 0.999  # không bao giờ dưới sàn


def test_het_hieu_luc_toan_bo_bi_phat_nang():
    expired = _temporal_factor(_rc(status="Hết hiệu lực toàn bộ").chunk)
    partial = _temporal_factor(_rc(status="Hết hiệu lực một phần").chunk)
    valid   = _temporal_factor(_rc(status="Còn hiệu lực").chunk)
    assert expired < partial < valid == 1.0


def test_vbhn_duoc_uu_tien_nhe():
    vbhn = _temporal_factor(_rc(folder="van_ban_hop_nhat").chunk)
    thuong = _temporal_factor(_rc(folder="nghi_dinh").chunk)
    assert vbhn > thuong


def test_rerank_fallback_vb_moi_thang_vb_cu_cung_diem(monkeypatch):
    """2 chunk cùng RRF score, không CE → VB 2024 phải xếp trên VB 2005."""
    old = _rc(0.02, issued="2005-03-01", cid="old")
    new = _rc(0.02, issued="2024-01-01", cid="new")
    out = rerank("mức phạt vi phạm", [old, new], use_cross_encoder=False)
    assert out[0].chunk.chunk_id == "new"


def test_rerank_het_hieu_luc_xep_duoi(monkeypatch):
    expired = _rc(0.02, issued="2019-01-01", status="Hết hiệu lực toàn bộ", cid="exp")
    valid   = _rc(0.02, issued="2019-01-01", status="Còn hiệu lực", cid="ok")
    out = rerank("quy định", [expired, valid], use_cross_encoder=False)
    assert out[0].chunk.chunk_id == "ok"


def test_tat_recency_thi_khong_doi_thu_tu(monkeypatch):
    monkeypatch.setattr(reranker, "RECENCY_ENABLED", False)
    assert _temporal_factor(_rc(issued="1995-01-01", status="Hết hiệu lực toàn bộ").chunk) == 1.0

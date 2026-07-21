"""Tests demote tầng retrieval: VB hết hiệu lực toàn bộ + đường ngang sai miền
phải chìm trong nhánh vector/BM25 TRƯỚC RRF (fix pool-crowding 2026-07-08)."""
from __future__ import annotations

from src.retriever import RETRIEVAL_EXPIRED_FACTOR, Retriever, _stale_factor
from src.schemas import Chunk, DocumentMetadata, RetrievedChunk


def _chunk(cid: str, status: str | None = None, text: str = "nội dung") -> Chunk:
    return Chunk(
        chunk_id=cid, text=text,
        metadata=DocumentMetadata(source=f"https://vbpl.vn/{cid}", status=status),
    )


def test_stale_factor_het_hieu_luc_toan_bo():
    c = _chunk("old", status="Hết hiệu lực toàn bộ")
    assert _stale_factor("vượt đèn đỏ phạt bao nhiêu", c) == RETRIEVAL_EXPIRED_FACTOR


def test_stale_factor_mot_phan_khong_demote_o_retrieval():
    """'Một phần' chỉ demote ở rerank (×0.9) — retrieval giữ nguyên vì phần
    còn hiệu lực (vd đường sắt của 100/2019) vẫn là đáp án hợp lệ."""
    c = _chunk("part", status="Hết hiệu lực một phần (lĩnh vực đường bộ)")
    assert _stale_factor("vượt đèn đỏ phạt bao nhiêu", c) == 1.0


def test_stale_factor_duong_ngang_sai_mien():
    rail = _chunk("rail", text="[X — Điều 47. Xử phạt quy tắc giao thông tại đường ngang, cầu chung] 5. Phạt...")
    assert _stale_factor("xe máy vượt đèn đỏ phạt bao nhiêu", rail) < 1.0
    # Query CÓ nhắc đường ngang → giữ nguyên
    assert _stale_factor("vượt đèn đỏ tại đường ngang phạt bao nhiêu", rail) == 1.0


class _StubStoreStatus:
    """Vector store trả VB cổ (điểm cao) + VB hiện hành (điểm thấp hơn)."""

    def query(self, embedding, top_k, where=None):
        return [
            RetrievedChunk(chunk=_chunk("old-decree", status="Hết hiệu lực toàn bộ"), score=0.90),
            RetrievedChunk(chunk=_chunk("current-168"), score=0.60),
        ]


class _StubEmb:
    def encode(self, texts):
        return [[0.0]] * len(texts)


def test_retrieve_vector_demote_dao_thu_hang():
    """VB hết hiệu lực 0.90×0.5=0.45 phải xếp SAU VB hiện hành 0.60."""
    r = Retriever(embedder=_StubEmb(), store=_StubStoreStatus())
    out = r.retrieve("vượt đèn đỏ xe máy phạt bao nhiêu tiền", top_k=2, use_kg=False)
    assert out[0].chunk.chunk_id == "current-168"
    assert out[1].chunk.chunk_id == "old-decree"


def test_stale_factor_duong_thuy_sai_mien():
    from src.retriever import WATERWAY_MISMATCH_FACTOR
    c = _chunk("dt")
    c.metadata.title = "Nghị định 139/2021/NĐ-CP xử phạt vi phạm hành chính trong lĩnh vực giao thông đường thủy nội địa"
    assert _stale_factor("xe máy vượt đèn đỏ phạt bao nhiêu", c) == WATERWAY_MISMATCH_FACTOR
    # Query về sông nước → giữ nguyên
    assert _stale_factor("thuyền vượt đèn tín hiệu đường thủy phạt bao nhiêu", c) == 1.0

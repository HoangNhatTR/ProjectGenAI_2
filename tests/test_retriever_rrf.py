"""Unit tests cho RRF fusion — thuật toán xếp hạng cốt lõi của Retriever."""
from __future__ import annotations

import src.retriever as retriever_mod
from src.retriever import Retriever
from src.schemas import Chunk, DocumentMetadata, RetrievedChunk


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=f"text {cid}",
        metadata=DocumentMetadata(source=f"https://vbpl.vn/{cid}"),
    )


def _retriever() -> Retriever:
    # embedder/store không dùng trong _rrf_fuse
    return Retriever(embedder=None, store=None)


def test_rrf_giua_2_branch_chunk_trung_xep_cao_hon():
    """Chunk xuất hiện ở cả vector + BM25 phải xếp trên chunk chỉ có 1 branch."""
    r = _retriever()
    both = _chunk("both")        # rank 1 ở cả 2 branch
    only_vec = _chunk("only_vec")  # rank 0 vector (cao hơn) nhưng chỉ 1 branch
    only_bm = _chunk("only_bm")

    results = r._rrf_fuse(
        bm25_results=[(only_bm, 9.0), (both, 8.0)],
        vector_results=[
            RetrievedChunk(chunk=only_vec, score=0.95),
            RetrievedChunk(chunk=both, score=0.90),
        ],
        kg_chunks=[],
        top_k=3,
    )

    assert results[0].chunk.chunk_id == "both"
    assert len(results) == 3


def test_rrf_top_k_gioi_han_ket_qua():
    r = _retriever()
    vector = [RetrievedChunk(chunk=_chunk(f"c{i}"), score=1.0 - i * 0.1) for i in range(10)]
    results = r._rrf_fuse(bm25_results=[], vector_results=vector, kg_chunks=[], top_k=4)
    assert len(results) == 4
    # Giữ nguyên thứ tự vector khi chỉ có 1 branch
    assert [x.chunk.chunk_id for x in results] == ["c0", "c1", "c2", "c3"]


def test_rrf_kg_weight_mac_dinh_ngang_vector():
    """KG_WEIGHT mặc định 1.0 → KG branch ngang hàng vector, KHÔNG boost.

    (Đo thật cho thấy boost 1.5 đẩy KG noise lên top ở câu hỏi tình huống —
    xem comment tại src/retriever.py cạnh KG_WEIGHT.)
    """
    r = _retriever()
    kg_c = _chunk("kg")
    vec_c = _chunk("vec")
    results = r._rrf_fuse(
        bm25_results=[],
        vector_results=[RetrievedChunk(chunk=vec_c, score=0.99)],
        kg_chunks=[kg_c],
        top_k=2,
    )
    by_id = {x.chunk.chunk_id: x.score for x in results}
    assert abs(by_id["kg"] - by_id["vec"]) < 1e-9


def test_rrf_kg_weight_boost_khi_override(monkeypatch):
    """Cơ chế KG_WEIGHT > 1 vẫn hoạt động: chunk KG rank 0 thắng vector rank 0."""
    monkeypatch.setattr(retriever_mod, "KG_WEIGHT", 1.5)
    r = _retriever()
    kg_c = _chunk("kg")
    vec_c = _chunk("vec")
    results = r._rrf_fuse(
        bm25_results=[],
        vector_results=[RetrievedChunk(chunk=vec_c, score=0.99)],
        kg_chunks=[kg_c],
        top_k=2,
    )
    assert results[0].chunk.chunk_id == "kg"


def test_rrf_k_override_khong_mutate_instance():
    """rrf_k truyền per-call không thay đổi self.rrf_k (an toàn concurrent)."""
    r = _retriever()
    default_k = r.rrf_k
    c = _chunk("a")
    out = r._rrf_fuse(
        bm25_results=[],
        vector_results=[RetrievedChunk(chunk=c, score=1.0)],
        kg_chunks=[],
        top_k=1,
        rrf_k=5,
    )
    # score = 1/(5+0+1)
    assert abs(out[0].score - 1 / 6) < 1e-9
    assert r.rrf_k == default_k  # không bị mutate


def test_rrf_score_cong_don_dung_cong_thuc():
    """Chunk ở rank 0 cả 2 branch: score = 2 × 1/(k+1)."""
    r = _retriever()
    c = _chunk("x")
    out = r._rrf_fuse(
        bm25_results=[(c, 5.0)],
        vector_results=[RetrievedChunk(chunk=c, score=0.9)],
        kg_chunks=[],
        top_k=1,
        rrf_k=60,
    )
    assert abs(out[0].score - 2 / 61) < 1e-9

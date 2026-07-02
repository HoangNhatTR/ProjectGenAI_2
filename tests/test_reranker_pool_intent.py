"""Tests per-doc cap + fine-intent boost trong reranker."""
from __future__ import annotations

from src.reranker import (
    _fine_intent_factor,
    _is_fine_intent,
    cap_per_doc,
    rerank,
    FINE_INTENT_BOOST,
)
from src.schemas import Chunk, DocumentMetadata, RetrievedChunk


def _rc(
    cid: str,
    source: str = "https://vbpl.vn/a",
    score: float = 0.5,
    doc_type: str | None = None,
    title: str | None = None,
    text: str = "nội dung",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=cid, text=text,
            metadata=DocumentMetadata(source=source, doc_type=doc_type, title=title),
        ),
        score=score,
    )


# ── cap_per_doc ────────────────────────────────────────────────────────────────

def test_cap_gioi_han_moi_van_ban():
    chunks = [_rc(f"a{i}", source="https://vbpl.vn/to-hop-nhat") for i in range(6)]
    chunks.append(_rc("b0", source="https://vbpl.vn/nd168"))
    out = cap_per_doc(chunks, cap=3)
    assert len(out) == 4  # 3 của VB đồ sộ + 1 của VB kia
    assert [x.chunk.chunk_id for x in out] == ["a0", "a1", "a2", "b0"]


def test_cap_giu_nguyen_thu_tu_rank():
    chunks = [
        _rc("a0", source="s1"), _rc("b0", source="s2"),
        _rc("a1", source="s1"), _rc("b1", source="s2"),
    ]
    out = cap_per_doc(chunks, cap=2)
    assert [x.chunk.chunk_id for x in out] == ["a0", "b0", "a1", "b1"]


def test_cap_0_la_tat():
    chunks = [_rc(f"a{i}", source="s1") for i in range(5)]
    assert cap_per_doc(chunks, cap=0) is chunks


# ── _is_fine_intent ────────────────────────────────────────────────────────────

def test_fine_intent_cau_hanh_chinh():
    assert _is_fine_intent("vượt đèn đỏ phạt bao nhiêu?")
    assert _is_fine_intent("mức phạt không đội mũ bảo hiểm")


def test_fine_intent_khong_boost_cau_hinh_su():
    assert not _is_fine_intent("tội trộm cắp bị phạt tù bao nhiêu năm?")
    assert not _is_fine_intent("khung hình phạt tội lừa đảo là gì")


def test_fine_intent_khong_boost_cau_khong_lien_quan():
    assert not _is_fine_intent("thủ tục đăng ký kết hôn cần giấy tờ gì")


# ── _fine_intent_factor ────────────────────────────────────────────────────────

def test_factor_boost_nghi_dinh_xu_phat():
    c = _rc("x", doc_type="Nghị định",
            title="Nghị định 168/2024/NĐ-CP quy định xử phạt vi phạm hành chính")
    assert _fine_intent_factor(c.chunk) == FINE_INTENT_BOOST


def test_factor_khong_boost_luat_va_nd_thuong():
    luat = _rc("x", doc_type="Luật", title="Luật Giao thông đường bộ")
    nd_khac = _rc("y", doc_type="Nghị định", title="Nghị định về đăng kiểm")
    assert _fine_intent_factor(luat.chunk) == 1.0
    assert _fine_intent_factor(nd_khac.chunk) == 1.0


def test_factor_nhan_dien_xu_phat_trong_header_chunk():
    """Contextual header/tiêu đề Điều ở đầu text chứa 'Xử phạt' cũng được boost."""
    c = _rc("x", doc_type="Nghị định", title="Nghị định 168/2024/NĐ-CP",
            text="[168/2024/NĐ-CP — Điều 8. Xử phạt người điều khiển xe mô tô]\nPhạt tiền...")
    assert _fine_intent_factor(c.chunk) == FINE_INTENT_BOOST


# ── rerank integration (fallback path, không load CE) ─────────────────────────

def test_rerank_fine_intent_nd_xu_phat_thang_thong_tu():
    thong_tu = _rc(
        "tt", source="s1", score=0.5, doc_type="Thông tư",
        title="Thông tư quản lý đèn tín hiệu giao thông",
    )
    nd_phat = _rc(
        "nd", source="s2", score=0.5, doc_type="Nghị định",
        title="Nghị định 168/2024/NĐ-CP quy định xử phạt vi phạm hành chính",
    )
    out = rerank("vượt đèn đỏ phạt bao nhiêu?", [thong_tu, nd_phat],
                 top_k=2, use_cross_encoder=False)
    assert out[0].chunk.chunk_id == "nd"


def test_rerank_cau_hinh_su_khong_ap_boost():
    """Câu 'phạt tù' → không fine-intent → giữ nguyên thứ tự theo score gốc."""
    blhs = _rc("blhs", source="s1", score=0.6, doc_type="Bộ luật",
               title="Bộ luật Hình sự")
    nd_phat = _rc("nd", source="s2", score=0.5, doc_type="Nghị định",
                  title="Nghị định xử phạt vi phạm hành chính")
    out = rerank("tội trộm cắp bị phạt tù bao nhiêu năm?", [blhs, nd_phat],
                 top_k=2, use_cross_encoder=False)
    assert out[0].chunk.chunk_id == "blhs"

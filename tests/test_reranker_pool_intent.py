"""Tests per-doc cap + fine-intent boost + railway-mismatch demote trong reranker."""
from __future__ import annotations

from src.reranker import (
    _fine_intent_factor,
    _is_fine_intent,
    _railway_mismatch_factor,
    cap_per_doc,
    rerank,
    FINE_INTENT_BOOST,
    RAILWAY_MISMATCH_FACTOR,
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


# ── _railway_mismatch_factor ───────────────────────────────────────────────────

_RAIL_HEAD = ("[03/VBHN-BGTVT — Điều 47. Xử phạt các hành vi vi phạm quy định về "
              "quy tắc giao thông tại đường ngang, cầu chung]\n5. Phạt tiền...")
_ROAD_HEAD = ("[168/2024/NĐ-CP — Điều 6. Xử phạt người điều khiển xe ô tô vi phạm "
              "quy tắc giao thông đường bộ — Khoản 9]\n9. Phạt tiền...")


def test_railway_demote_khi_query_khong_nhac_duong_sat():
    rail = _rc("r", text=_RAIL_HEAD)
    assert _railway_mismatch_factor("ô tô vượt đèn đỏ phạt bao nhiêu", rail.chunk) \
        == RAILWAY_MISMATCH_FACTOR


def test_railway_giu_nguyen_khi_query_nhac_duong_ngang():
    rail = _rc("r", text=_RAIL_HEAD)
    assert _railway_mismatch_factor("vượt đèn đỏ tại đường ngang phạt bao nhiêu", rail.chunk) == 1.0


def test_railway_khong_oan_chunk_duong_bo():
    road = _rc("d", text=_ROAD_HEAD)
    assert _railway_mismatch_factor("ô tô vượt đèn đỏ phạt bao nhiêu", road.chunk) == 1.0


def test_rerank_duong_bo_thang_duong_ngang_khi_query_nut_giao():
    """Chunk đường ngang score gốc cao hơn vẫn phải thua chunk NĐ 168 đường bộ."""
    rail = _rc("rail", source="s1", score=0.55, doc_type="Văn bản hợp nhất",
               title="Văn bản hợp nhất 03/VBHN-BGTVT", text=_RAIL_HEAD)
    road = _rc("road", source="s2", score=0.50, doc_type="Nghị định",
               title="Nghị định 168/2024/NĐ-CP quy định xử phạt vi phạm hành chính",
               text=_ROAD_HEAD)
    out = rerank("xe máy vượt đèn đỏ tại nút giao thông phạt bao nhiêu?",
                 [rail, road], top_k=2, use_cross_encoder=False)
    assert out[0].chunk.chunk_id == "road"


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


# ── _machinery_mismatch_factor (xe máy chuyên dùng, 2026-07-09) ──────────────

def test_machinery_demote_xe_may_thuong():
    from src.reranker import _machinery_mismatch_factor, MACHINERY_MISMATCH_FACTOR
    d8 = _rc("d8", text="[168/2024/NĐ-CP — Điều 8. Xử phạt người điều khiển xe máy chuyên dùng vi phạm]")
    # Query 'xe máy' thường → demote chunk chuyên dùng
    assert _machinery_mismatch_factor("xe máy vượt đèn đỏ phạt bao nhiêu", d8.chunk) \
        == MACHINERY_MISMATCH_FACTOR


def test_machinery_giu_khi_query_chuyen_dung():
    from src.reranker import _machinery_mismatch_factor
    d8 = _rc("d8", text="[168/2024/NĐ-CP — Điều 8. Xử phạt xe máy chuyên dùng]")
    assert _machinery_mismatch_factor("xe máy chuyên dùng vượt đèn đỏ phạt bao nhiêu", d8.chunk) == 1.0


def test_machinery_khong_oan_xe_may_thuong():
    from src.reranker import _machinery_mismatch_factor
    d7 = _rc("d7", text="[168/2024/NĐ-CP — Điều 7. Xử phạt người điều khiển xe mô tô, xe gắn máy]")
    assert _machinery_mismatch_factor("xe máy vượt đèn đỏ phạt bao nhiêu", d7.chunk) == 1.0


def test_rerank_xe_may_thuong_thang_chuyen_dung():
    """Query 'xe máy' → Điều 7 (mô tô/gắn máy) phải thắng Điều 8 (chuyên dùng)
    dù CE score gần nhau (fallback rule-based path)."""
    d7 = _rc("d7", source="s1", score=0.62, doc_type="Nghị định",
             title="Nghị định 168/2024/NĐ-CP",
             text="[168/2024/NĐ-CP — Điều 7. Xử phạt người điều khiển xe mô tô, xe gắn máy] 7. Phạt tiền 4-6 triệu... không chấp hành đèn tín hiệu")
    d8 = _rc("d8", source="s2", score=0.65, doc_type="Nghị định",
             title="Nghị định 168/2024/NĐ-CP",
             text="[168/2024/NĐ-CP — Điều 8. Xử phạt người điều khiển xe máy chuyên dùng] 7. Phạt tiền 6-8 triệu... không chấp hành đèn tín hiệu")
    out = rerank("xe máy vượt đèn đỏ phạt bao nhiêu tiền?", [d8, d7],
                 top_k=2, use_cross_encoder=False)
    assert out[0].chunk.chunk_id == "d7"  # 0.65×0.7=0.455 < 0.62 → Đ7 thắng

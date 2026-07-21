"""Unit tests cho nhãn độ tin cậy (P2.3, src/confidence.py) — 4 tín hiệu tất
định, không LLM. Xem docs/tro-ly-luat-su/lo-trinh-cong-viec.md §P2.3."""
from __future__ import annotations

from src.confidence import (
    LABEL_CAO,
    LABEL_THAP,
    LABEL_TRUNG_BINH,
    compute_confidence,
)
from src.schemas import Chunk, Citation, DocumentMetadata, RetrievedChunk


def _ctx(
    text: str = "", article: str | None = None, doc_number: str | None = None,
    title: str | None = None, score: float = 0.9, status: str | None = None,
    source: str = "https://vbpl.vn/x",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"c-{article}-{doc_number}-{score}", text=text, article=article,
            metadata=DocumentMetadata(source=source, doc_number=doc_number, title=title, status=status),
        ),
        score=score,
    )


def _cit(ref: int, article: str | None = None, source: str = "https://vbpl.vn/x") -> Citation:
    return Citation(source=source, article=article, snippet="", ref=ref)


def test_no_contexts_always_thap():
    conf = compute_confidence("Bất kỳ câu trả lời nào.", [], [])
    assert conf.label == LABEL_THAP
    assert conf.reasons_vi


def test_high_confidence_strong_signals():
    ctx1 = _ctx(
        text="Điều 173. Tội trộm cắp tài sản...", article="Điều 173",
        doc_number="100/2015/QH13", title="Bộ luật Hình sự", score=0.85,
    )
    ctx2 = _ctx(
        text="Điều 173. quy định tương tự tại văn bản hợp nhất...", article="Điều 173",
        doc_number="19/VBHN-VPQH", title="Văn bản hợp nhất Bộ luật Hình sự", score=0.7,
    )
    answer_text = "Theo Điều 173 Bộ luật Hình sự [1][2], hành vi này có thể bị xử lý hình sự."
    citations = [_cit(1, "Điều 173", "https://vbpl.vn/x"), _cit(2, "Điều 173", "https://vbpl.vn/y")]
    conf = compute_confidence(answer_text, citations, [ctx1, ctx2])
    assert conf.label == LABEL_CAO
    assert conf.n_sources == 2
    assert conf.has_expired_source is False


def test_expired_source_never_cao_even_with_good_signals():
    ctx1 = _ctx(
        text="Điều 5. ...", article="Điều 5", doc_number="10/2000/QH10",
        title="Luật cũ", score=0.95, status="Hết hiệu lực toàn bộ",
    )
    ctx2 = _ctx(
        text="Điều 5. ...", article="Điều 5", doc_number="20/2010/QH12",
        title="Luật cũ 2", score=0.9, status="Hết hiệu lực toàn bộ",
    )
    answer_text = "Theo Điều 5 [1][2], quy định như sau."
    citations = [_cit(1, "Điều 5"), _cit(2, "Điều 5")]
    conf = compute_confidence(answer_text, citations, [ctx1, ctx2])
    assert conf.has_expired_source is True
    assert conf.label == LABEL_THAP
    assert any("Hết hiệu lực" in r for r in conf.reasons_vi)


def test_low_top1_score_and_single_source_gives_trung_binh_or_thap():
    ctx = _ctx(text="Điều 9. ...", article="Điều 9", doc_number="1/2020/NĐ-CP", score=0.1)
    answer_text = "Theo Điều 9 [1], có thể áp dụng quy định này."
    citations = [_cit(1, "Điều 9")]
    conf = compute_confidence(answer_text, citations, [ctx])
    assert conf.label in (LABEL_TRUNG_BINH, LABEL_THAP)
    assert conf.n_sources == 1


def test_unverified_citation_claim_lowers_pass_rate_and_flags_reason():
    # ctx KHÔNG được nhắc "Điều 250" ở đâu (kể cả trong text) — nếu không
    # verify_citations() sẽ coi nó có căn cứ thật (đúng thiết kế, xem
    # guardrails._context_evidence quét cả chunk.text, không chỉ chunk.article).
    ctx = _ctx(text="Điều 9. nội dung hoàn toàn không liên quan", article="Điều 9",
               doc_number="1/2020/NĐ-CP", score=0.8)
    # Answer bịa "Điều 250" — không có trong bất kỳ context nào
    answer_text = "Theo Điều 250 [1], mức phạt như sau."
    citations = [_cit(1, "Điều 250")]
    conf = compute_confidence(answer_text, citations, [ctx])
    assert conf.citation_pass_rate < 1.0
    assert conf.label != LABEL_CAO  # trích dẫn không đối chiếu được -> không thể "Cao"


def test_no_article_claims_in_answer_does_not_crash_and_full_pass_rate():
    ctx = _ctx(text="Nội dung chung không có Điều luật cụ thể.", score=0.6)
    conf = compute_confidence("Câu trả lời chung chung, không trích Điều nào.", [], [ctx])
    assert conf.citation_pass_rate == 1.0  # không claim gì -> không có gì "sai"


def test_citations_without_matching_ref_fall_back_to_first_context():
    """citation.ref trỏ ra ngoài phạm vi contexts (lỗi hiếm từ LLM) -> vẫn
    không crash, fallback dùng context đầu tiên."""
    ctx = _ctx(text="Điều 1. ...", article="Điều 1", score=0.5)
    citations = [_cit(99, "Điều 1")]  # ref=99 nhưng chỉ có 1 context
    conf = compute_confidence("Theo Điều 1 [99].", citations, [ctx])
    assert conf.top1_score == 0.5
    assert conf.n_sources == 1


def test_reasons_always_present_even_for_cao():
    ctx1 = _ctx(text="Điều 2. ...", article="Điều 2", doc_number="1/2020", score=0.9)
    ctx2 = _ctx(text="Điều 2. ...", article="Điều 2", doc_number="2/2021", score=0.8)
    citations = [_cit(1, "Điều 2"), _cit(2, "Điều 2")]
    conf = compute_confidence("Theo Điều 2 [1][2].", citations, [ctx1, ctx2])
    assert conf.label == LABEL_CAO
    assert len(conf.reasons_vi) >= 1  # luôn giải thích, kể cả khi tin cậy cao

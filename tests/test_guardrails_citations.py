"""Unit tests cho citation verification — chống LLM bịa căn cứ pháp lý."""
from __future__ import annotations

from src.guardrails import apply_guardrails, verify_citations
from src.schemas import Answer, Chunk, DocumentMetadata, RetrievedChunk


def _ctx(
    text: str = "",
    article: str | None = None,
    doc_number: str | None = None,
    title: str | None = None,
    source: str = "https://vbpl.vn/x",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"c-{article}-{doc_number}",
            text=text,
            article=article,
            metadata=DocumentMetadata(
                source=source, doc_number=doc_number, title=title,
            ),
        ),
        score=0.9,
    )


ND168 = _ctx(
    text="Điều 8. Xử phạt người điều khiển xe mô tô... Phạt tiền từ 4.000.000 đồng",
    article="Điều 8",
    doc_number="168/2024/NĐ-CP",
    title="Nghị định 168/2024/NĐ-CP quy định xử phạt vi phạm hành chính",
)
BLHS = _ctx(
    text="Điều 173. Tội trộm cắp tài sản...",
    article="Điều 173",
    doc_number="100/2015/QH13",
    title="Bộ luật Hình sự",
)


def test_citation_dung_khong_canh_bao():
    """Điều + số hiệu VB khớp cùng 1 chunk → verified, list rỗng."""
    ans = "Theo Khoản 4 Điều 8 Nghị định 168/2024/NĐ-CP [1], mức phạt là 4-6 triệu."
    assert verify_citations(ans, [ND168, BLHS]) == []


def test_dieu_bia_bi_neu_dich_danh():
    """Điều không có trong bất kỳ chunk nào → flag."""
    ans = "Theo Điều 99 Nghị định 168/2024/NĐ-CP, bạn bị phạt."
    flagged = verify_citations(ans, [ND168, BLHS])
    assert any("Điều 99" in f for f in flagged)


def test_so_hieu_vb_bia_bi_flag():
    """Số hiệu VB không tồn tại trong contexts → flag (kể cả khi Điều khớp)."""
    ans = "Theo Điều 8 Nghị định 99/2020/NĐ-CP, mức phạt là 4-6 triệu."
    flagged = verify_citations(ans, [ND168, BLHS])
    assert any("99/2020" in f for f in flagged)


def test_cap_dieu_vb_phai_cung_chunk():
    """Điều 173 có thật (BLHS) + NĐ 168 có thật, nhưng 'Điều 173 NĐ 168' là ghép bịa."""
    ans = "Theo Điều 173 Nghị định 168/2024/NĐ-CP, hành vi này bị xử phạt."
    flagged = verify_citations(ans, [ND168, BLHS])
    assert any("Điều 173" in f and "168/2024" in f for f in flagged)


def test_dieu_trong_text_chunk_duoc_tinh():
    """'Điều N' xuất hiện trong text chunk (không cần field article) → verified."""
    ctx = _ctx(
        text="Việc thu hồi đất quy định tại Điều 62 của Luật này...",
        article="Điều 61",
        doc_number="45/2013/QH13",
    )
    assert verify_citations("Theo Điều 62, nhà nước thu hồi đất...", [ctx]) == []


def test_so_hieu_ngan_khop_day_du():
    """'Nghị định 168/2024' (thiếu đuôi NĐ-CP) khớp doc_number đầy đủ."""
    ans = "Theo Điều 8 Nghị định 168/2024, mức phạt tăng mạnh."
    assert verify_citations(ans, [ND168]) == []


def test_khong_co_contexts_khong_flag():
    assert verify_citations("Theo Điều 8 Nghị định 168/2024/NĐ-CP...", []) == []


def test_answer_khong_trich_dan_khong_flag():
    assert verify_citations("Bạn nên tham khảo luật sư để được tư vấn.", [ND168]) == []


def test_apply_guardrails_gan_canh_bao():
    ans = Answer(
        question="Vượt đèn đỏ phạt bao nhiêu?",
        answer="Theo Điều 99 Nghị định 77/2077/NĐ-CP, phạt 10 triệu.",
        citations=[],
    )
    out = apply_guardrails(ans, [ND168])
    assert "KHÔNG đối chiếu được" in out.answer
    assert "Điều 99" in out.answer


def test_apply_guardrails_verify_cited_false_khong_canh_bao():
    """Tool flow: verify_cited=False → không kiểm chứng (căn cứ từ tool)."""
    ans = Answer(
        question="Tính phạt giúp tôi",
        answer="Theo Điều 99 Nghị định 77/2077/NĐ-CP, phạt 10 triệu.",
        citations=[],
    )
    out = apply_guardrails(ans, [ND168], verify_cited=False)
    assert "KHÔNG đối chiếu được" not in out.answer


def test_apply_guardrails_citation_dung_khong_canh_bao():
    ans = Answer(
        question="Vượt đèn đỏ phạt bao nhiêu?",
        answer="Theo Khoản 4 Điều 8 Nghị định 168/2024/NĐ-CP [1], phạt 4-6 triệu.",
        citations=[],
    )
    out = apply_guardrails(ans, [ND168])
    assert "KHÔNG đối chiếu được" not in out.answer

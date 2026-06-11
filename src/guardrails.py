"""Legal AI Guardrails — an toàn và kiểm soát chất lượng câu trả lời.

Theo Big Update.txt (mục 9: Guardrails pháp lý):
  - Không khẳng định thay luật sư
  - Không bịa căn cứ
  - Phân biệt "thông tin pháp luật" và "tư vấn pháp lý cá nhân"
  - Cảnh báo khi thiếu dữ kiện
  - Hỏi bổ sung khi case phụ thuộc thông tin quan trọng

Áp dụng SAU khi generator tạo xong câu trả lời.
"""
from __future__ import annotations

from .schemas import Answer, RetrievedChunk


# ─── Disclaimers ──────────────────────────────────────────────────────────────

_DISCLAIMER_GENERAL = (
    "\n\n---\n"
    "*Lưu ý: Thông tin trên chỉ mang tính tham khảo và không thay thế "
    "tư vấn pháp lý chính thức từ luật sư có chuyên môn.*"
)

_DISCLAIMER_PERSONAL = (
    "\n\n---\n"
    "*Đây là thông tin pháp luật chung. Kết quả thực tế phụ thuộc vào "
    "hoàn cảnh cụ thể của từng trường hợp — nên tham khảo luật sư trực tiếp.*"
)

_WARNING_NO_EVIDENCE = (
    "\n\n⚠ *Cơ sở dữ liệu chưa có văn bản pháp luật đủ căn cứ cho câu hỏi này. "
    "Vui lòng kiểm tra từ nguồn chính thức (vbpl.vn, thuvienphapluat.vn) "
    "hoặc hỏi luật sư.*"
)

_WARNING_COMPLEX_CASE = (
    "\n\n⚠ *Tình huống này có thể phụ thuộc vào nhiều yếu tố pháp lý phức tạp. "
    "Khuyến nghị tham khảo luật sư để được tư vấn chính xác nhất.*"
)


# ─── Detection helpers ────────────────────────────────────────────────────────

_NO_EVIDENCE_PHRASES = [
    "không tìm thấy căn cứ",
    "không có dữ liệu",
    "tôi chưa tìm thấy",
    "không có thông tin",
    "chưa có căn cứ",
    "không đủ căn cứ",
]

_PERSONAL_KEYWORDS = [
    "tôi bị", "của tôi", "trường hợp tôi", "tôi muốn", "mình bị",
    "em bị", "anh bị", "chị bị", "con tôi", "nhà tôi", "xe tôi",
]

_COMPLEX_KEYWORDS = [
    "khiếu nại", "khởi kiện", "tố cáo", "truy tố", "bắt giam",
    "phạt tù", "mức án", "xét xử", "kháng cáo", "thi hành án",
]


def _is_personal_consulting(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _PERSONAL_KEYWORDS)


def _is_complex_legal(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _COMPLEX_KEYWORDS)


def _answer_lacks_evidence(answer_text: str) -> bool:
    text = answer_text.lower()
    return any(phrase in text for phrase in _NO_EVIDENCE_PHRASES)


# ─── Main function ────────────────────────────────────────────────────────────

def apply_guardrails(
    answer: Answer,
    contexts: list[RetrievedChunk],
    add_disclaimer: bool = True,
    warn_no_evidence: bool = True,
) -> Answer:
    """Áp dụng guardrails lên câu trả lời đã generate.

    Logic:
      1. Nếu không có context retrieved → thêm warning thiếu căn cứ
         (tắt bằng warn_no_evidence=False khi câu trả lời đến từ tool
         tự chứa căn cứ — validate_document, compare_regulations...)
      2. Nếu câu hỏi là tư vấn cá nhân → disclaimer personal
      3. Nếu câu hỏi phức tạp (khởi kiện, tù...) → warning complex
      4. Luôn thêm disclaimer chung (nếu add_disclaimer=True)

    Returns:
        Answer mới với text đã bổ sung các cảnh báo phù hợp.
    """
    text = answer.answer

    # 1. Không có context và câu trả lời không tự nhận thiếu căn cứ
    if warn_no_evidence and not contexts and not _answer_lacks_evidence(text):
        text += _WARNING_NO_EVIDENCE

    # 2. Tư vấn tình huống cá nhân
    if _is_personal_consulting(answer.question):
        text += _DISCLAIMER_PERSONAL
    elif add_disclaimer:
        text += _DISCLAIMER_GENERAL

    # 3. Vụ việc phức tạp (hình sự, khởi kiện...)
    if _is_complex_legal(answer.question):
        text += _WARNING_COMPLEX_CASE

    return Answer(
        question=answer.question,
        answer=text,
        citations=answer.citations,
    )


def check_answer_quality(answer: Answer, contexts: list[RetrievedChunk]) -> dict:
    """Trả về báo cáo chất lượng ngắn để log/debug. Không thay đổi answer."""
    return {
        "has_citations":      bool(answer.citations),
        "has_contexts":       bool(contexts),
        "lacks_evidence":     _answer_lacks_evidence(answer.answer),
        "is_personal":        _is_personal_consulting(answer.question),
        "is_complex_legal":   _is_complex_legal(answer.question),
        "answer_length":      len(answer.answer),
    }

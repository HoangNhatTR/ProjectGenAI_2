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

import re

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

_WARNING_UNVERIFIED_CITATIONS = (
    "\n\n⚠ *Trích dẫn sau KHÔNG đối chiếu được với các nguồn hệ thống đã truy "
    "xuất: {items}. Có khả năng mô hình tự suy diễn — vui lòng kiểm tra lại "
    "trên vbpl.vn trước khi sử dụng.*"
)


# ─── Citation verification ────────────────────────────────────────────────────
# Chống LLM bịa căn cứ: "Điều N" và số hiệu văn bản nhắc trong answer phải xuất
# hiện trong contexts đã retrieve. Chỉ kiểm mức Điều + văn bản — Khoản/Điểm nằm
# trong text parent không parse chắc chắn được, kiểm sẽ cảnh báo oan.
#
# Thiên về PRECISION của cảnh báo (thà bỏ sót còn hơn báo oan làm nhờn):
#   - "Điều N" xuất hiện trong text chunk (kể cả dạng dẫn chiếu) → tính là có căn cứ
#   - cặp (Điều N, số hiệu VB) chỉ verify khi cả hai cùng nằm trong 1 chunk

_ANSWER_ARTICLE_RE = re.compile(r"[Đđ]iều\s+(\d+)")
_DOCNUM_FULL_RE = re.compile(r"\d+/\d{4}/[A-ZĐÂĂÊÔƠƯ][A-ZĐÂĂÊÔƠƯ\d\-\.]*")
# "Nghị định 168/2024" (thiếu đuôi NĐ-CP) — bắt số ngắn khi đứng sau loại VB
_DOCNUM_SHORT_RE = re.compile(
    r"(?:Nghị\s+định|Thông\s+tư|Bộ\s+luật|Luật|Nghị\s+quyết|Quyết\s+định|Pháp\s+lệnh)"
    r"\s+(?:số\s+)?(\d+/\d{4}(?:/[A-ZĐÂĂÊÔƠƯ][A-ZĐÂĂÊÔƠƯ\d\-\.]*)?)",
    re.IGNORECASE,
)
_CITE_WINDOW = 90  # ký tự quanh "Điều N" để tìm số hiệu VB đi kèm trong answer


def _doc_matches(claim: str, ctx_docs: set[str]) -> bool:
    """'168/2024' khớp '168/2024/NĐ-CP' (claim ngắn = prefix của doc đầy đủ)."""
    claim = claim.strip().upper()
    return any(
        d == claim or d.startswith(claim + "/") or claim.startswith(d + "/")
        for d in ctx_docs
    )


def _context_evidence(
    contexts: list[RetrievedChunk],
) -> tuple[set[str], set[int], list[tuple[set[str], set[int]]]]:
    """Gom bằng chứng từ contexts: (mọi số hiệu VB, mọi số Điều, theo từng chunk)."""
    per_chunk: list[tuple[set[str], set[int]]] = []
    all_docs: set[str] = set()
    all_articles: set[int] = set()
    for r in contexts:
        chunk = r.chunk
        meta = chunk.metadata
        docs: set[str] = set()
        for val in (meta.doc_number, meta.title, meta.source):
            if val:
                docs.update(_DOCNUM_FULL_RE.findall(val.upper()))
        if meta.doc_number:
            docs.add(meta.doc_number.strip().upper())
        arts: set[int] = set()
        if chunk.article:
            m = re.search(r"(\d+)", chunk.article)
            if m:
                arts.add(int(m.group(1)))
        for m in _ANSWER_ARTICLE_RE.finditer(chunk.text or ""):
            arts.add(int(m.group(1)))
        per_chunk.append((docs, arts))
        all_docs |= docs
        all_articles |= arts
    return all_docs, all_articles, per_chunk


def verify_citations(answer_text: str, contexts: list[RetrievedChunk]) -> list[str]:
    """Trả về list trích dẫn trong answer KHÔNG tìm được trong contexts.

    Rỗng = mọi trích dẫn đều có căn cứ (hoặc không có contexts để đối chiếu).
    """
    if not contexts or not answer_text:
        return []
    all_docs, all_articles, per_chunk = _context_evidence(contexts)

    unverified: list[str] = []
    seen: set = set()
    claimed_docs_in_pairs: set[str] = set()

    # 1) Các cụm "Điều N [+ số hiệu VB gần đó]"
    for m in _ANSWER_ARTICLE_RE.finditer(answer_text):
        art = int(m.group(1))
        lo = max(0, m.start() - _CITE_WINDOW)
        window = answer_text[lo : m.end() + _CITE_WINDOW]
        wdocs = set(_DOCNUM_FULL_RE.findall(window.upper()))
        wdocs.update(d.upper() for d in _DOCNUM_SHORT_RE.findall(window))
        if wdocs:
            claimed_docs_in_pairs |= wdocs
            for doc in wdocs:
                if (art, doc) in seen:
                    continue
                seen.add((art, doc))
                ok = any(
                    art in arts and _doc_matches(doc, docs)
                    for docs, arts in per_chunk
                )
                if not ok:
                    unverified.append(f"Điều {art} ({doc})")
        else:
            if (art, None) in seen:
                continue
            seen.add((art, None))
            if art not in all_articles:
                unverified.append(f"Điều {art}")

    # 2) Số hiệu VB đứng riêng (không gắn với Điều nào)
    bare_docs = set(_DOCNUM_FULL_RE.findall(answer_text.upper()))
    bare_docs.update(d.upper() for d in _DOCNUM_SHORT_RE.findall(answer_text))
    for doc in sorted(bare_docs - claimed_docs_in_pairs):
        if not _doc_matches(doc, all_docs):
            unverified.append(doc)

    return unverified


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
    verify_cited: bool = True,
) -> Answer:
    """Áp dụng guardrails lên câu trả lời đã generate.

    Logic:
      1. Nếu không có context retrieved → thêm warning thiếu căn cứ
         (tắt bằng warn_no_evidence=False khi câu trả lời đến từ tool
         tự chứa căn cứ — validate_document, compare_regulations...)
      1b. Kiểm chứng trích dẫn: Điều/số hiệu VB trong answer phải xuất hiện
         trong contexts — trích dẫn LLM tự suy diễn bị nêu đích danh
         (tắt bằng verify_cited=False khi căn cứ đến từ tool, không phải contexts)
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

    # 1b. Trích dẫn không đối chiếu được với nguồn → nêu đích danh
    if verify_cited and contexts:
        unverified = verify_citations(answer.answer, contexts)
        if unverified:
            text += _WARNING_UNVERIFIED_CITATIONS.format(
                items="; ".join(unverified[:4])
            )

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
        "has_citations":        bool(answer.citations),
        "has_contexts":         bool(contexts),
        "lacks_evidence":       _answer_lacks_evidence(answer.answer),
        "is_personal":          _is_personal_consulting(answer.question),
        "is_complex_legal":     _is_complex_legal(answer.question),
        "unverified_citations": verify_citations(answer.answer, contexts),
        "answer_length":        len(answer.answer),
    }

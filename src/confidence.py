"""P2.3 — Nhãn ĐỘ TIN CẬY (Cao/Trung bình/Thấp) cho câu trả lời, tính từ tín
hiệu retrieval TẤT ĐỊNH. KHÔNG để LLM tự chấm chính nó — bài học "router LLM
bất định" (xem docs/tro-ly-luat-su/lo-trinh-cong-viec.md §4 nguyên tắc #1).

4 tín hiệu (đúng đặc tả roadmap P2.3):
  1. Điểm rerank top-1 TRONG SỐ nguồn ĐÃ ĐƯỢC TRÍCH DẪN (không phải toàn bộ
     contexts đã retrieve — 1 câu có thể retrieve 10 chunk nhưng chỉ trích 2).
  2. Số nguồn ĐỘC LẬP (văn bản khác nhau, theo doc_number/source) trong số
     trích dẫn — 1 khẳng định lặp lại ở nhiều văn bản đáng tin hơn 1 văn bản.
  3. Tỉ lệ trích dẫn "Điều N" trong câu trả lời đối chiếu được với contexts —
     tái dùng guardrails.verify_citations() (đã có, dùng để chèn cảnh báo).
     KHÔNG gọi corpus-check thật của Module 2 (/corpus-check — HTTP, ~vài giây/
     cặp, và bản thân nó lại gọi ngược Module 1) vì quá chậm để chấm mỗi câu
     trả lời real-time.
  4. Có văn bản "Hết hiệu lực" trong nguồn đã trích hay không — có thì KHÔNG
     BAO GIỜ được nhãn "Cao" bất kể 3 tín hiệu kia tốt thế nào.

⚠ NGƯỠNG DƯỚI ĐÂY LÀ GIÁ TRỊ KHỞI ĐIỂM, chưa calibrate bằng gold Q&A đã người
có chuyên môn chấm đúng/sai thật (roadmap: nhóm "Cao" phải đúng >=90%, nhóm
"Thấp" được phép đúng <70%). Xem scripts/eval_confidence.py +
tests/eval_confidence_gold.json để đo lại và chỉnh THRESHOLD_HIGH/MED khi có
gold — đừng coi các số này là đã kiểm chứng.
"""
from __future__ import annotations

import re

from .guardrails import verify_citations
from .schemas import Citation, ConfidenceInfo, RetrievedChunk

LABEL_CAO = "cao"
LABEL_TRUNG_BINH = "trung_binh"
LABEL_THAP = "thap"

LABEL_VI = {LABEL_CAO: "Cao", LABEL_TRUNG_BINH: "Trung bình", LABEL_THAP: "Thấp"}

# Điểm rerank sau combine bị cap ở 1.0 (reranker.SCORE_CAP) — ngưỡng theo thang đó.
TOP1_SCORE_HIGH = 0.5
TOP1_SCORE_MED = 0.2

MIN_SOURCES_HIGH = 2       # >=2 văn bản độc lập cùng trích dẫn mới tính điểm "nhiều nguồn"
CITATION_PASS_HIGH = 0.9   # >=90% claim "Điều N" đối chiếu được -> điểm
CITATION_PASS_FLOOR = 0.5  # dưới ngưỡng này -> ép thẳng xuống "Thấp" bất kể tín hiệu khác

_ARTICLE_CLAIM_RE = re.compile(r"[Đđ]iều\s+(\d+)")


def _doc_key(meta) -> str:
    return (meta.doc_number or meta.source or meta.title or "").strip().lower()


def _cited_contexts(citations: list[Citation], contexts: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """contexts tại các vị trí thực sự được answer trích dẫn (citation.ref, 1-based).
    Rỗng nếu answer không trích dẫn gì cụ thể (LLM không đưa ref nào)."""
    out = []
    for c in citations:
        if c.ref and 1 <= c.ref <= len(contexts):
            out.append(contexts[c.ref - 1])
    return out


def compute_confidence(
    answer_text: str, citations: list[Citation], contexts: list[RetrievedChunk],
) -> ConfidenceInfo:
    """Tính nhãn độ tin cậy. contexts rỗng -> luôn 'Thấp' (không có gì để tin)."""
    if not contexts:
        return ConfidenceInfo(
            label=LABEL_THAP, label_vi=LABEL_VI[LABEL_THAP],
            reasons_vi=["Không truy xuất được nguồn nào cho câu hỏi này"],
        )

    cited = _cited_contexts(citations, contexts) or contexts[:1]
    top1_score = cited[0].score
    n_sources = len({_doc_key(rc.chunk.metadata) for rc in cited if _doc_key(rc.chunk.metadata)}) or 1

    unverified = verify_citations(answer_text, contexts)
    total_claims = len(set(_ARTICLE_CLAIM_RE.findall(answer_text or "")))
    pass_rate = 1.0 if total_claims == 0 else max(0.0, 1.0 - len(unverified) / total_claims)

    has_expired = any("hết hiệu lực" in (rc.chunk.metadata.status or "").lower() for rc in cited)

    reasons: list[str] = []
    points = 0

    if top1_score >= TOP1_SCORE_HIGH:
        points += 1
    elif top1_score >= TOP1_SCORE_MED:
        reasons.append(f"Điểm liên quan của nguồn hàng đầu ở mức trung bình ({top1_score:.2f})")
    else:
        reasons.append(f"Điểm liên quan của nguồn hàng đầu thấp ({top1_score:.2f})")

    if n_sources >= MIN_SOURCES_HIGH:
        points += 1
    else:
        reasons.append("Chỉ có 1 nguồn văn bản độc lập hỗ trợ câu trả lời")

    if pass_rate >= CITATION_PASS_HIGH:
        points += 1
    elif total_claims:
        reasons.append(
            f"{len(unverified)}/{total_claims} trích dẫn Điều luật chưa đối chiếu được với nguồn"
        )

    if has_expired:
        reasons.append("Có văn bản 'Hết hiệu lực' trong nguồn đã trích dẫn")
    else:
        points += 1

    if has_expired or pass_rate < CITATION_PASS_FLOOR:
        label = LABEL_THAP
    elif points >= 4:
        label = LABEL_CAO
    elif points >= 2:
        label = LABEL_TRUNG_BINH
    else:
        label = LABEL_THAP

    if not reasons:
        reasons.append("Nguồn liên quan cao, nhiều văn bản độc lập, trích dẫn khớp nguồn")

    return ConfidenceInfo(
        label=label, label_vi=LABEL_VI[label], reasons_vi=reasons,
        top1_score=round(top1_score, 3), n_sources=n_sources,
        citation_pass_rate=round(pass_rate, 3), has_expired_source=has_expired,
    )

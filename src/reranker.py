"""Legal Document Reranker.

Rule-based reranker: không cần model riêng, boost chunks dựa trên:
  1. Khớp tham chiếu Điều/Khoản/văn bản với query (boost mạnh)
  2. Mật độ từ khóa pháp lý quan trọng (boost nhẹ)
  3. Tính mới của văn bản (metadata issued_date nếu có)

Với pháp luật Việt Nam, "Điều 6 Khoản 4" phải match chính xác — đây là
điểm yếu của thuần vector search nên cần boost riêng.
"""
from __future__ import annotations

import re
from typing import Optional

from .schemas import RetrievedChunk


# ── Config ────────────────────────────────────────────────────────────────────
ARTICLE_BOOST   = 0.20   # boost khi chunk có đúng Điều/Khoản trong query
KEYWORD_BOOST   = 0.05   # boost nhỏ khi chunk có từ khóa pháp lý
MAX_BOOST       = 0.35   # tổng boost tối đa mỗi chunk
SCORE_CAP       = 1.0    # điểm tối đa sau khi boost

# Patterns để nhận diện tham chiếu điều khoản
_ARTICLE_PATTERNS = [
    r'(?:Điều|điều)\s+\d+',          # Điều 6
    r'(?:Khoản|khoản)\s+\d+',        # Khoản 2
    r'(?:Điểm|điểm)\s+[a-zA-Z]',     # Điểm a
    r'\d+/\d{4}/[A-ZĐÂĂÊÔƠƯ\-]+',   # 100/2019/NĐ-CP
    r'Nghị\s+định\s+\d+',
    r'Thông\s+tư\s+\d+',
]

# Từ khóa pháp lý có tính phân biệt cao
_LEGAL_KEYWORDS = [
    "mức phạt", "tiền phạt", "xử phạt", "tước quyền", "thu hồi",
    "tịch thu", "hình thức phạt", "phạt bổ sung", "phạt chính",
    "tái phạm", "nhiều lần", "gây hậu quả", "nguy hiểm",
]


def _extract_refs(text: str) -> set[str]:
    """Trích xuất tất cả tham chiếu điều/khoản từ text, chuẩn hóa lowercase."""
    refs: set[str] = set()
    for pattern in _ARTICLE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            refs.add(m.group(0).lower().strip())
    return refs


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: Optional[int] = None,
) -> list[RetrievedChunk]:
    """Re-rank danh sách chunk sau retrieval.

    Args:
        query:   câu hỏi gốc (đã rewrite bởi router)
        chunks:  kết quả từ Retriever (đã RRF-fused)
        top_k:   nếu chỉ định, chỉ trả top_k đầu

    Returns:
        Danh sách chunk đã sắp xếp lại, điểm có thể cao hơn bản gốc.
    """
    if not chunks:
        return chunks

    query_refs = _extract_refs(query)
    query_lower = query.lower()

    scored: list[tuple[RetrievedChunk, float]] = []
    for r in chunks:
        boost = 0.0
        chunk_text_lower = r.chunk.text.lower()
        chunk_article = (r.chunk.article or "").lower()
        chunk_clause  = (r.chunk.clause  or "").lower()

        # Boost 1: khớp tham chiếu điều/khoản
        if query_refs:
            chunk_refs = _extract_refs(r.chunk.text)
            # Thêm article/clause field vào set
            if chunk_article:
                chunk_refs.add(chunk_article)
            if chunk_clause:
                chunk_refs.add(chunk_clause)

            overlap = query_refs & chunk_refs
            if overlap:
                boost += ARTICLE_BOOST * min(len(overlap), 2)  # max 2 lần

        # Boost 2: mật độ từ khóa pháp lý xuất hiện trong cả query và chunk
        matching_kw = sum(
            1 for kw in _LEGAL_KEYWORDS
            if kw in query_lower and kw in chunk_text_lower
        )
        boost += KEYWORD_BOOST * min(matching_kw, 3)  # max 3 từ khóa

        # Cap boost
        boost = min(boost, MAX_BOOST)
        adjusted = min(SCORE_CAP, r.score + boost)
        scored.append((r, adjusted))

    # Sắp xếp theo điểm giảm dần
    scored.sort(key=lambda x: -x[1])

    result = [
        RetrievedChunk(chunk=rc.chunk, score=adj)
        for rc, adj in scored
    ]
    return result[:top_k] if top_k else result

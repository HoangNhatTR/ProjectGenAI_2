"""Legal Document Reranker — Cross-encoder + Rule-based hybrid.

Primary scorer: CrossEncoder (BAAI/bge-reranker-v2-m3 mặc định).
  - Đọc (query, chunk) cùng lúc → hiểu ngữ nghĩa sâu hơn bi-encoder.
  - Đặt RERANKER_MODEL=none trong .env để tắt và dùng rule-based thuần.

Bonus layer: rule-based legal-reference boost giữ lại logic nhận diện
"Điều 5 Khoản 2" — điểm yếu của neural model với exact legal citations.

Score cuối: sigmoid(ce_logit) + rule_boost * RULE_ALPHA (capped at 1.0)
Fallback tự động về rule-based nếu model không load được.
"""
from __future__ import annotations

import datetime
import math
import os
import re
from typing import Optional

from .schemas import RetrievedChunk


# ── Tuning knobs ──────────────────────────────────────────────────────────────
RULE_ALPHA    = 0.40   # weight của rule_boost khi cộng vào CE score
ARTICLE_BOOST = 0.20   # boost khi chunk có đúng Điều/Khoản trong query
KEYWORD_BOOST = 0.05   # boost nhỏ khi chunk có từ khóa pháp lý
MAX_BOOST     = 0.35   # tổng rule_boost tối đa mỗi chunk
SCORE_CAP     = 1.0    # điểm tối đa sau khi kết hợp

# ── Recency / hiệu lực — ưu tiên văn bản hiện hành ────────────────────────────
# Corpus chứa cả VB từ 1945 và 35% trước 2010, đa số KHÔNG có metadata hiệu lực
# → khi nội dung tương đương, văn bản mới hơn phải thắng. Tắt: RECENCY_BOOST=false
RECENCY_ENABLED      = os.getenv("RECENCY_BOOST", "true").lower() == "true"
RECENCY_DECAY_PER_YR = 0.015  # giảm 1.5%/năm tuổi văn bản
RECENCY_FLOOR        = 0.75   # sàn — VB cũ vẫn tìm được khi không có VB mới hơn
EXPIRED_FULL_FACTOR  = 0.50   # 'Hết hiệu lực toàn bộ' → phạt nặng
EXPIRED_PART_FACTOR  = 0.90   # 'Hết hiệu lực một phần' → phạt nhẹ
VBHN_FACTOR          = 1.05   # Văn bản hợp nhất (đã gộp sửa đổi) → ưu tiên nhẹ

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")

_ARTICLE_PATTERNS = [
    r'(?:Điều|điều)\s+\d+',
    r'(?:Khoản|khoản)\s+\d+',
    r'(?:Điểm|điểm)\s+[a-zA-Z]',
    r'\d+/\d{4}/[A-ZĐÂĂÊÔƠƯ\-]+',
    r'Nghị\s+định\s+\d+',
    r'Thông\s+tư\s+\d+',
]

_LEGAL_KEYWORDS = [
    "mức phạt", "tiền phạt", "xử phạt", "tước quyền", "thu hồi",
    "tịch thu", "hình thức phạt", "phạt bổ sung", "phạt chính",
    "tái phạm", "nhiều lần", "gây hậu quả", "nguy hiểm",
]


# ── Lazy-loaded cross-encoder (module-level singleton) ────────────────────────
_cross_encoder = None
_ce_available: Optional[bool] = None   # None = chưa thử load


def _get_cross_encoder():
    """Load CrossEncoder lần đầu tiên gọi; cache kết quả cho các lần sau."""
    global _cross_encoder, _ce_available

    if _ce_available is not None:
        return _cross_encoder

    from . import config as _cfg
    from loguru import logger

    model_name: str = _cfg.RERANKER_MODEL

    if model_name.lower() == "none":
        logger.info("Reranker: RERANKER_MODEL=none — dùng rule-based thuần.")
        _ce_available = False
        return None

    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Reranker: đang load '{model_name}'...")
        _cross_encoder = CrossEncoder(model_name)
        _ce_available = True
        logger.info("Reranker: CrossEncoder sẵn sàng.")
        return _cross_encoder
    except Exception as exc:
        logger.warning(
            f"Reranker: không load được '{model_name}': {exc}. "
            "Fallback về rule-based."
        )
        _ce_available = False
        return None


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


# ── Rule-based helpers ─────────────────────────────────────────────────────────

def _extract_refs(text: str) -> set[str]:
    refs: set[str] = set()
    for pattern in _ARTICLE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            refs.add(m.group(0).lower().strip())
    return refs


def _rule_boost(query_lower: str, query_refs: set[str], chunk) -> float:
    """Tính rule-based boost cho một chunk (pre-computed query refs/lower)."""
    boost = 0.0
    chunk_text_lower = chunk.text.lower()

    if query_refs:
        chunk_refs = _extract_refs(chunk.text)
        if chunk.article:
            chunk_refs.add(chunk.article.lower())
        if chunk.clause:
            chunk_refs.add(chunk.clause.lower())
        overlap = query_refs & chunk_refs
        if overlap:
            boost += ARTICLE_BOOST * min(len(overlap), 2)

    matching_kw = sum(
        1 for kw in _LEGAL_KEYWORDS
        if kw in query_lower and kw in chunk_text_lower
    )
    boost += KEYWORD_BOOST * min(matching_kw, 3)
    return min(boost, MAX_BOOST)


def _temporal_factor(chunk) -> float:
    """Hệ số nhân theo hiệu lực + tuổi văn bản + loại VBHN.

    - status 'Hết hiệu lực toàn bộ/một phần' (chỉ có ở nhóm vbpl) → phạt
    - tuổi (từ issued_date): -1.5%/năm, sàn 0.75 — VB 2024 ăn VB 2007
      khi CE score tương đương, nhưng VB cũ không bị loại hẳn
    - văn bản hợp nhất → +5% (text đã gộp mọi sửa đổi, đáng tin nhất)
    """
    if not RECENCY_ENABLED:
        return 1.0

    meta = chunk.metadata
    factor = 1.0

    status = (meta.status or "").lower()
    if "hết hiệu lực" in status:
        factor *= EXPIRED_FULL_FACTOR if "toàn bộ" in status else EXPIRED_PART_FACTOR

    if meta.issued_date:
        m = _YEAR_RE.search(meta.issued_date)
        if m:
            age = max(0, datetime.date.today().year - int(m.group(0)))
            factor *= max(RECENCY_FLOOR, 1.0 - RECENCY_DECAY_PER_YR * age)

    if (meta.folder or "") == "van_ban_hop_nhat":
        factor *= VBHN_FACTOR

    return factor


# ── Public API ─────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: Optional[int] = None,
    use_cross_encoder: bool = True,
) -> list[RetrievedChunk]:
    """Re-rank danh sách chunk sau retrieval.

    - Cross-encoder available: score = sigmoid(ce_logit) + rule_boost * RULE_ALPHA
    - Fallback (model không load được): score = rrf_score + rule_boost  (như cũ)

    Args:
        query:   câu hỏi gốc (đã rewrite bởi router)
        chunks:  kết quả từ Retriever (đã RRF-fused)
        top_k:   nếu chỉ định, chỉ trả top_k đầu

    Returns:
        Danh sách chunk đã sắp xếp lại theo relevance, score cập nhật.
    """
    if not chunks:
        return chunks

    query_lower = query.lower()
    query_refs  = _extract_refs(query)

    ce = _get_cross_encoder() if use_cross_encoder else None

    if ce is not None:
        try:
            pairs  = [[query, r.chunk.text] for r in chunks]
            logits = ce.predict(pairs, batch_size=32, show_progress_bar=False)

            scored: list[tuple[RetrievedChunk, float]] = []
            for r, logit in zip(chunks, logits):
                ce_score   = _sigmoid(logit)
                rule_b     = _rule_boost(query_lower, query_refs, r.chunk)
                temporal   = _temporal_factor(r.chunk)
                final      = min(SCORE_CAP, (ce_score + rule_b * RULE_ALPHA) * temporal)
                scored.append((r, final))

        except Exception as exc:
            from loguru import logger
            logger.warning(f"Reranker: CE predict thất bại ({exc}), fallback rule-based.")
            ce = None  # trigger fallback below

    if ce is None:
        scored = []
        for r in chunks:
            rule_b   = _rule_boost(query_lower, query_refs, r.chunk)
            temporal = _temporal_factor(r.chunk)
            adjusted = min(SCORE_CAP, (r.score + rule_b) * temporal)
            scored.append((r, adjusted))

    scored.sort(key=lambda x: -x[1])
    result = [
        RetrievedChunk(chunk=rc.chunk, score=adj)
        for rc, adj in scored
    ]
    return result[:top_k] if top_k else result


def is_cross_encoder_available() -> bool:
    """Trả True nếu cross-encoder đã load thành công (dùng cho startup info)."""
    return bool(_ce_available)


def preload() -> bool:
    """Preload model ngay lúc khởi động để tránh delay ở query đầu tiên.

    Returns True nếu cross-encoder load thành công.
    """
    return _get_cross_encoder() is not None

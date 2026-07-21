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

# ── Fine-intent boost — câu hỏi mức phạt HÀNH CHÍNH → ưu tiên NĐ xử phạt ───────
# Quan sát đã đo (eval GT): câu 'phạt bao nhiêu' hay lôi chunk THÔNG SỐ KỸ THUẬT
# (Thông tư quản lý đèn tín hiệu) lên trên chunk XỬ PHẠT của nghị định.
# Chỉ BOOST nghị định xử phạt, KHÔNG demote ai — an toàn cho nhóm khác.
# Câu hình sự (phạt tù, mức án) KHÔNG boost — đáp án nằm ở Bộ luật Hình sự.
# Tắt: FINE_INTENT_BOOST=1.0
FINE_INTENT_BOOST = float(os.getenv("FINE_INTENT_BOOST", "1.15"))
_FINE_QUERY_RE = re.compile(
    r"phạt tiền|mức phạt|xử phạt|phạt bao nhiêu|bị phạt|tiền phạt|phạt hành chính"
)
_CRIMINAL_HINT_RE = re.compile(
    r"phạt tù|đi tù|tù giam|hình sự|mức án|án phạt|khung hình phạt|truy cứu|tội"
)

# ── Railway-mismatch demote — chunk đường ngang/cầu chung cho query đường bộ ──
# Failure mode đo được (2026-07-08, inspect 'vượt đèn đỏ'): chunk Điều "quy tắc
# giao thông tại ĐƯỜNG NGANG, CẦU CHUNG" (giao đường sắt) chứa nguyên văn
# "vượt... khi đèn đỏ đã bật sáng" → CE chấm cao hơn chunk đèn tín hiệu ĐƯỜNG BỘ
# của NĐ 168 (phát biểu "không chấp hành hiệu lệnh đèn tín hiệu"). Query không
# nhắc đường sắt → demote các chunk đường ngang. Query CÓ nhắc → giữ nguyên.
# Chỉ nhìn HEADER chunk (tiêu đề Điều) — thân chunk đường bộ nhắc "đường sắt"
# tình cờ không bị oan. Tắt: RAILWAY_MISMATCH_FACTOR=1.0
RAILWAY_MISMATCH_FACTOR = float(os.getenv("RAILWAY_MISMATCH_FACTOR", "0.8"))
_RAILWAY_RE = re.compile(r"đường\s+ngang|cầu\s+chung|đường\s+sắt|tàu\s+(hỏa|lửa)", re.IGNORECASE)


def _railway_mismatch_factor(query_lower: str, chunk) -> float:
    """×RAILWAY_MISMATCH_FACTOR nếu chunk thuộc Điều đường ngang/cầu chung
    mà query không nhắc gì tới đường sắt. Còn lại 1.0."""
    if RAILWAY_MISMATCH_FACTOR >= 1.0 or _RAILWAY_RE.search(query_lower):
        return 1.0
    head = (chunk.text or "")[:160]
    return RAILWAY_MISMATCH_FACTOR if _RAILWAY_RE.search(head) else 1.0


# ── Machinery-mismatch demote — "xe máy chuyên dùng" cho query xe thường ──────
# "xe máy" (mô tô, xe gắn máy — Điều 7 NĐ168, 4-6tr) khớp lexical với "xe MÁY
# CHUYÊN DÙNG" (xe công trình/máy kéo — Điều 8 NĐ168, 6-8tr) → reranker lôi Đ8
# lên gần Đ7, generator trích nhầm mức phạt sai đối tượng (đo 2026-07-09). Query
# nói xe thường mà KHÔNG nhắc 'chuyên dùng' → demote chunk 'xe máy chuyên dùng'.
# Query CÓ 'chuyên dùng'/'máy kéo' → giữ nguyên. Tắt: MACHINERY_MISMATCH_FACTOR=1.0
MACHINERY_MISMATCH_FACTOR = float(os.getenv("MACHINERY_MISMATCH_FACTOR", "0.7"))
_MACHINERY_RE = re.compile(r"chuyên\s+dùng|máy\s+kéo|máy\s+chuyên\s+dùng", re.IGNORECASE)


def _machinery_mismatch_factor(query_lower: str, chunk) -> float:
    """×MACHINERY_MISMATCH_FACTOR nếu chunk thuộc Điều 'xe máy chuyên dùng'
    mà query không nhắc 'chuyên dùng'. Còn lại 1.0."""
    if MACHINERY_MISMATCH_FACTOR >= 1.0 or _MACHINERY_RE.search(query_lower):
        return 1.0
    head = (chunk.text or "")[:160]
    return MACHINERY_MISMATCH_FACTOR if _MACHINERY_RE.search(head) else 1.0


# ── Per-doc cap — chống 'số đông chunk' trong pool ứng viên ────────────────────
# VB đồ sộ (03/VBHN-BGTVT 6.983 chunk) lấp kín pool bằng nhiều chunk trung bình
# → chunk đúng của VB khác (NĐ 168) không còn slot cho CE thấy. Tắt: PER_DOC_CAP=0
PER_DOC_CAP = int(os.getenv("PER_DOC_CAP", "3"))

# ── CE input cap — chặn text quá dài đưa vào CrossEncoder ─────────────────────
# CE trên CPU scale theo độ dài token; chunk đã parent-expand (full Điều, có thể
# hàng chục nghìn ký tự) làm 1 lần predict tốn nhiều giây. Pipeline giờ rerank
# child chunk ngắn, cap này là lưới an toàn cho caller khác. Tắt: CE_MAX_CHARS=0
CE_MAX_CHARS = int(os.getenv("CE_MAX_CHARS", "2000"))

# ── CE tốc độ (đo 2026-07-08: Rerank 9.6-30s/câu = khâu chậm nhất pipeline) ────
# max_length: bge-reranker-v2-m3 nhận tới 8192 token; KHÔNG cap thì 1 chunk dài
# (chunk NĐ theo khoản ~2000 ký tự ≈ 1000+ token) kéo CẢ batch pad theo nó →
# attention phình 4-6×. 512 token là mức chuẩn BGE, đủ phủ child chunk ~600 ký
# tự + header Điều/Khoản ở đầu. Tắt cap: RERANKER_MAX_LENGTH=0
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
# int8 dynamic quantization (chỉ Linear layers): ~2-3× nhanh hơn trên CPU, chất
# lượng xê dịch nhẹ — BẬT phải kiểm ranking (inspect ND168 vẫn top). Mặc định tắt.
RERANKER_INT8 = os.getenv("RERANKER_INT8", "0").lower() in ("1", "true")

# ── Remote rerank API (tùy chọn) — GPU cloud thay CE local ────────────────────
# Endpoint Cohere-compatible /rerank (SiliconFlow, Jina, Pinecone...): ~1-2s/20
# cặp thay 7-41s CPU local. Bật bằng RERANK_API_URL + RERANK_API_KEY (.env);
# lỗi/timeout → tự fallback CE local (demo không chết vì mạng). Score API cùng
# thang [0,1] như sigmoid CE nên công thức kết hợp rule/temporal giữ nguyên.
RERANK_API_URL = os.getenv("RERANK_API_URL", "").strip()
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "").strip()
RERANK_API_MODEL = os.getenv("RERANK_API_MODEL", "Qwen/Qwen3-Reranker-8B")
RERANK_API_TIMEOUT = float(os.getenv("RERANK_API_TIMEOUT", "15"))
# Format API: "cohere" (SiliconFlow/Jina/Pinecone — POST {model,query,documents})
# hoặc "hf" (HF Inference text-classification — POST {inputs:[{text,text_pair}]},
# dùng được CHÍNH bge-reranker-v2-m3 → điểm tương đương CE local).
RERANK_API_KIND = os.getenv("RERANK_API_KIND", "cohere").strip().lower()


def remote_rerank_available() -> bool:
    """True nếu đã cấu hình đủ URL + key cho remote rerank."""
    return bool(RERANK_API_URL and RERANK_API_KEY)


def _remote_rerank_scores(query: str, docs: list[str]) -> Optional[list[float]]:
    """Gọi rerank API, trả list score THEO THỨ TỰ docs; None nếu lỗi (caller
    fallback CE local).

    - kind="cohere": POST {model,query,documents} → results[{index,
      relevance_score}] (có thể đã sort theo score → đặt lại theo index).
    - kind="hf": POST {inputs:[{text: query, text_pair: doc}]} → mỗi cặp một
      list [{label, score}] (score đã sigmoid — cùng thang CE local).
    """
    import requests

    try:
        headers = {
            "Authorization": f"Bearer {RERANK_API_KEY}",
            "Content-Type": "application/json",
        }
        if RERANK_API_KIND == "hf":
            resp = requests.post(
                RERANK_API_URL, headers=headers,
                json={"inputs": [{"text": query, "text_pair": d} for d in docs]},
                timeout=RERANK_API_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            # Shape THỰC TẾ (đo 2026-07-08): batch N cặp → [[{label,score}×N]]
            # (1 list ngoài, list trong có N dict THEO THỨ TỰ cặp). Phòng cả
            # shape chuẩn pipeline [[{..}] ×N] (mỗi cặp một list riêng).
            if not isinstance(data, list):
                raise ValueError(f"HF API trả {type(data)}")
            if len(data) == 1 and isinstance(data[0], list) and len(data[0]) == len(docs):
                per_pair = data[0]
            elif len(data) == len(docs):
                per_pair = [item[0] if isinstance(item, list) else item for item in data]
            else:
                raise ValueError(f"HF API shape lạ: ngoài={len(data)}, cần {len(docs)} cặp")
            return [float(p["score"]) for p in per_pair]

        # Mặc định: Cohere-compatible /rerank
        resp = requests.post(
            RERANK_API_URL, headers=headers,
            json={
                "model": RERANK_API_MODEL,
                "query": query,
                "documents": docs,
                "return_documents": False,
            },
            timeout=RERANK_API_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if len(results) != len(docs):
            raise ValueError(f"API trả {len(results)}/{len(docs)} kết quả")
        scores = [0.0] * len(docs)
        for r in results:
            scores[int(r["index"])] = float(r["relevance_score"])
        return scores
    except Exception as exc:
        from loguru import logger
        logger.warning(f"Reranker: remote API lỗi ({exc}) → fallback CE local")
        return None

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
        _kwargs = {}
        if RERANKER_MAX_LENGTH > 0:
            _kwargs["max_length"] = RERANKER_MAX_LENGTH
        _cross_encoder = CrossEncoder(model_name, **_kwargs)
        if RERANKER_INT8:
            try:
                import torch
                _cross_encoder.model = torch.ao.quantization.quantize_dynamic(
                    _cross_encoder.model, {torch.nn.Linear}, dtype=torch.qint8,
                )
                logger.info("Reranker: int8 dynamic quantization BẬT")
            except Exception as qexc:
                logger.warning(f"Reranker: quantize int8 thất bại, dùng fp32: {qexc}")
        _ce_available = True
        logger.info(f"Reranker: CrossEncoder sẵn sàng (max_length={RERANKER_MAX_LENGTH or 'model'}).")
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


def _is_fine_intent(query_lower: str) -> bool:
    """True khi câu hỏi về mức phạt HÀNH CHÍNH (không phải hình sự)."""
    return bool(
        _FINE_QUERY_RE.search(query_lower)
        and not _CRIMINAL_HINT_RE.search(query_lower)
    )


def _fine_intent_factor(chunk) -> float:
    """Boost chunk nghị định xử phạt khi query là fine-intent. Còn lại 1.0."""
    meta = chunk.metadata
    if "nghị định" not in (meta.doc_type or "").lower():
        return 1.0
    # Tiêu đề VB hoặc contextual header/tiêu đề Điều (đầu chunk) chứa "xử phạt"
    hay = ((meta.title or "") + " " + (chunk.text or "")[:160]).lower()
    return FINE_INTENT_BOOST if "xử phạt" in hay else 1.0


def cap_per_doc(
    chunks: list[RetrievedChunk], cap: int = None,  # type: ignore[assignment]
) -> list[RetrievedChunk]:
    """Giới hạn tối đa `cap` chunk/văn bản trong pool ứng viên (giữ thứ tự rank).

    Gọi TRƯỚC rerank để pool CE thấy nhiều văn bản khác nhau thay vì
    20 chunk của cùng một Thông tư cũ. cap<=0 → tắt.
    """
    if cap is None:
        cap = PER_DOC_CAP
    if cap <= 0:
        return chunks
    counts: dict[str, int] = {}
    out: list[RetrievedChunk] = []
    for r in chunks:
        src = r.chunk.metadata.source or ""
        if counts.get(src, 0) >= cap:
            continue
        counts[src] = counts.get(src, 0) + 1
        out.append(r)
    return out


# ── Public API ─────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: Optional[int] = None,
    use_cross_encoder: bool = True,
) -> list[RetrievedChunk]:
    """Re-rank danh sách chunk sau retrieval.

    Thứ tự ưu tiên scorer (2026-07-08):
    1. Remote rerank API (RERANK_API_URL — GPU cloud, ~1-2s/20 cặp)
    2. CrossEncoder local (fallback khi API lỗi/chưa cấu hình)
    3. Rule-based thuần (lưới cuối): score = rrf_score + rule_boost
    Điểm neural (1/2) kết hợp: (score + rule_boost×RULE_ALPHA) × temporal × fine × railway

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
    fine_intent = _is_fine_intent(query_lower)
    _cap = CE_MAX_CHARS if CE_MAX_CHARS > 0 else None

    def _combine(r: RetrievedChunk, base: float) -> float:
        """Điểm cuối = (neural + rule) × temporal × fine × railway × machinery."""
        rule_b   = _rule_boost(query_lower, query_refs, r.chunk)
        temporal = _temporal_factor(r.chunk)
        fine_f   = _fine_intent_factor(r.chunk) if fine_intent else 1.0
        rail_f   = _railway_mismatch_factor(query_lower, r.chunk)
        mach_f   = _machinery_mismatch_factor(query_lower, r.chunk)
        return min(SCORE_CAP, (base + rule_b * RULE_ALPHA) * temporal * fine_f * rail_f * mach_f)

    scored: Optional[list[tuple[RetrievedChunk, float]]] = None

    # ── 1. Remote rerank API (GPU cloud) — ưu tiên nếu cấu hình ──────────────
    if use_cross_encoder and remote_rerank_available():
        import time as _time
        _t0 = _time.time()
        api_scores = _remote_rerank_scores(query, [r.chunk.text[:_cap] for r in chunks])
        if api_scores is not None:
            from loguru import logger
            logger.info(
                f"Reranker: remote {RERANK_API_MODEL} "
                f"{len(chunks)} cặp {_time.time() - _t0:.2f}s"
            )
            scored = [(r, _combine(r, s)) for r, s in zip(chunks, api_scores)]

    # ── 2. CrossEncoder local ─────────────────────────────────────────────────
    ce = _get_cross_encoder() if (use_cross_encoder and scored is None) else None
    if ce is not None:
        try:
            pairs  = [[query, r.chunk.text[:_cap]] for r in chunks]
            logits = ce.predict(pairs, batch_size=32, show_progress_bar=False)
            scored = [(r, _combine(r, _sigmoid(logit))) for r, logit in zip(chunks, logits)]
        except Exception as exc:
            from loguru import logger
            logger.warning(f"Reranker: CE predict thất bại ({exc}), fallback rule-based.")

    # ── 3. Rule-based thuần (lưới cuối / use_cross_encoder=False) ────────────
    if scored is None:
        scored = []
        for r in chunks:
            rule_b   = _rule_boost(query_lower, query_refs, r.chunk)
            temporal = _temporal_factor(r.chunk)
            fine_f   = _fine_intent_factor(r.chunk) if fine_intent else 1.0
            rail_f   = _railway_mismatch_factor(query_lower, r.chunk)
            mach_f   = _machinery_mismatch_factor(query_lower, r.chunk)
            adjusted = min(SCORE_CAP, (r.score + rule_b) * temporal * fine_f * rail_f * mach_f)
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

"""Deterministic intent classifier — thay phần intent-routing của LLM router.

Vấn đề: SmartRouter dùng 1 LLM call để phân loại intent → KHÔNG tất định. Cùng
câu hỏi lúc ra `consulting` lúc ra `validate` (~50%), đặc biệt câu dày trích dẫn
formal bị đẩy nhầm sang tool `validate_document` (bộ chấm thể thức lỗi). Trước
đây phải lách bằng prompt-engineering + retry tới 3 lần.

Giải pháp: tầng phân loại TẤT ĐỊNH chạy TRƯỚC LLM:
  1. Rule layer (regex/keyword) cho các ranh giới NGUY HIỂM, bắt được bằng cấu
     trúc — quan trọng nhất: `validate` CHỈ khi thực sự có văn bản đính kèm.
  2. Embedding layer (tái dùng BGE-M3 đã load) xác nhận đây là câu hỏi pháp lý
     cần tra cứu (legal/consulting) qua cosine với prototype.

Khi classifier CHẮC CHẮN → trả quyết định ngay, KHÔNG gọi LLM (nhanh + nhất
quán + tiết kiệm quota). Khi KHÔNG chắc (followup cần rewrite, compare/draft cần
dựng tool_query có cấu trúc, hoặc câu mơ hồ) → trả low-confidence để router
fallback về LLM như cũ. Thiết kế additive: router không có classifier → y hệt cũ.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

# ── Tuning knobs (env-tunable) ────────────────────────────────────────────────
RULE_CONFIDENCE   = 0.95
# Ngưỡng cosine tối thiểu để embedding tự tin xếp câu là "cần tra cứu pháp lý".
# BGE-M3 normalize → cosine ∈ [-1,1]; câu pháp lý so với prototype thường 0.5-0.75.
RETRIEVE_MIN_COS  = float(os.getenv("INTENT_RETRIEVE_MIN_COS", "0.55"))

# Intent → (action, tool_name). Nguồn sự thật duy nhất cho mapping.
INTENT_ROUTING: dict[str, tuple[str, Optional[str]]] = {
    "legal":      ("retrieve", None),
    "consulting": ("retrieve", None),
    "followup":   ("retrieve", None),
    "compare":    ("use_tool", "compare_regulations"),
    "calculate":  ("use_tool", "calculate_fine"),
    "draft":      ("use_tool", "draft_document"),
    "validate":   ("use_tool", "validate_document"),
    "kg_query":   ("use_tool", "knowledge_graph_lookup"),
    "web_search": ("use_tool", "web_search"),
    "chitchat":   ("answer_direct", None),
    "meta":       ("answer_direct", None),
    "clarify":    ("answer_direct", None),
}

# Intent cần LLM dựng search_query/tool_query/direct_response → KHÔNG quyết định
# tất định (để router fallback LLM lo phần ngôn ngữ).
NEEDS_LLM_QUERY: frozenset[str] = frozenset(
    {"followup", "compare", "draft", "chitchat", "meta", "clarify"}
)


# ── Rule patterns ─────────────────────────────────────────────────────────────
_CHITCHAT_RE = re.compile(
    r"^\s*(xin\s+chào|chào\b|hi\b|hello|hey|cảm\s+ơn|thanks?|thank\s+you|"
    r"tạm\s+biệt|bye|ok(ay)?|được\s+rồi|tốt\s+lắm|hay\s+quá)\b",
    re.IGNORECASE,
)
_DRAFT_RE = re.compile(
    r"(soạn|viết|lập|làm\s+(giúp|hộ)?).{0,25}"
    r"(đơn|hợp\s+đồng|di\s+chúc|công\s+văn|biên\s+bản|tờ\s+trình|giấy\s+(ủy\s+quyền|xác\s+nhận))",
    re.IGNORECASE,
)
_COMPARE_RE = re.compile(
    r"so\s+sánh|khác\s+(nhau|biệt)\s+(giữa|gì)|phân\s+biệt\s+giữa|"
    r"\bgiữa\b.{1,40}\bvà\b.{1,40}(khác|giống)",
    re.IGNORECASE,
)
# calculate: TỪ 2 hành vi trở lên / cộng dồn — không phải hỏi 1 mức phạt đơn lẻ
_CALC_MULTI_RE = re.compile(
    r"(vừa\b.{1,40}\bvừa\b|đồng\s+thời|cùng\s+lúc|tổng\s+(cộng|số)\b.{0,30}phạt|"
    r"cộng\s+dồn|gộp\s+(lại\s+)?(các\s+)?(mức\s+)?phạt|nhiều\s+lỗi)",
    re.IGNORECASE,
)
_WEBSEARCH_RE = re.compile(
    r"mới\s+nhất|hiện\s+hành|còn\s+hiệu\s+lực\s+không|trên\s+(mạng|internet|online)|"
    r"vừa\s+ban\s+hành|cập\s+nhật\s+(mới|gần\s+đây)",
    re.IGNORECASE,
)
_YEAR_RECENT_RE = re.compile(r"\b(202[4-9]|20[3-9]\d)\b")
_KG_RE = re.compile(
    r"^\s*(tội\b.{0,40}(là\s+gì|là\s+tội\s+gì)|"
    r"hành\s+vi\b.{0,40}(là\s+gì|bị\s+(xử\s+lý|coi)|cấu\s+thành)|"
    r"thế\s+nào\s+là\s+tội)",
    re.IGNORECASE,
)
# Có văn bản đính kèm để validate? (cấu trúc, không phải ngữ nghĩa)
_DOC_REF_RE = re.compile(
    r"(văn\s+bản|bản\s+án|hợp\s+đồng|quyết\s+định|tài\s+liệu|đơn|nghị\s+quyết)\s+(này|sau|trên|đính\s+kèm)|"
    r"(kiểm\s+tra|phân\s+tích|rà\s+soát|tìm\s+(sai|lỗi)|chỉ\s+ra\s+(sai|lỗi))\s+"
    r".{0,20}(văn\s+bản|bản\s+án|hợp\s+đồng|quyết\s+định|tài\s+liệu)",
    re.IGNORECASE,
)
_DOC_IN_HISTORY_RE = re.compile(r"\bfile\b|đính\s+kèm|upload|tải\s+lên|\.pdf|\.docx?", re.IGNORECASE)
# Câu tư vấn tình huống cá nhân (ngôi thứ nhất) → consulting thay vì legal
_FIRST_PERSON_RE = re.compile(
    r"\b(tôi|mình|em|cháu|chúng\s+tôi)\b.{0,60}"
    r"(có\s+(bị|được|phải)|thì\s+(sao|thế\s+nào|phải\s+làm)|nên\s+làm|làm\s+(sao|gì|thế\s+nào))",
    re.IGNORECASE,
)

# Prototype câu hỏi pháp lý "cần tra cứu" — centroid cho embedding gate.
_RETRIEVE_PROTOTYPES = [
    "mức phạt vượt đèn đỏ đối với xe máy là bao nhiêu",
    "thủ tục đăng ký kết hôn cần giấy tờ gì",
    "quy định về thời hiệu khởi kiện dân sự",
    "điều kiện hưởng bảo hiểm thất nghiệp",
    "hành vi trộm cắp tài sản bị xử phạt thế nào",
    "quyền và nghĩa vụ của người lao động khi nghỉ việc",
    "mức bồi thường khi bị tai nạn giao thông",
    "thủ tục ly hôn đơn phương theo quy định pháp luật",
    "quy định về thừa kế theo di chúc và theo pháp luật",
    "xử phạt nồng độ cồn khi lái xe ô tô",
    "tôi bị sa thải trái luật thì phải làm sao",
    "trường hợp của tôi có được hưởng trợ cấp không",
]


@dataclass
class ClassifierResult:
    intent: str
    action: str
    tool_name: Optional[str]
    confidence: float
    source: str               # "rule" | "embed" | "low_conf"
    needs_llm_query: bool

    @property
    def confident(self) -> bool:
        return self.source != "low_conf"


def _routing(intent: str) -> tuple[str, Optional[str]]:
    return INTENT_ROUTING.get(intent, ("retrieve", None))


def _result(intent: str, confidence: float, source: str) -> ClassifierResult:
    action, tool = _routing(intent)
    return ClassifierResult(
        intent=intent, action=action, tool_name=tool,
        confidence=confidence, source=source,
        needs_llm_query=intent in NEEDS_LLM_QUERY,
    )


def history_has_document(history: Optional[list[dict]]) -> bool:
    """True nếu hội thoại có dấu hiệu văn bản đính kèm (system msg 'file', .pdf…)."""
    if not history:
        return False
    for msg in history:
        content = msg.get("content", "") or ""
        if msg.get("role") == "system" and _DOC_IN_HISTORY_RE.search(content):
            return True
    return False


def has_validation_target(question: Optional[str], history: Optional[list[dict]]) -> bool:
    """Có thật sự văn bản để validate không? (đính kèm HOẶC câu trỏ rõ '…này/sau').

    Dùng làm guard cấu trúc cho router: `validate` mà KHÔNG có target gần như chắc
    chắn là misfire của LLM (bug ~50% được ghi nhận) → ép về retrieve.
    """
    return history_has_document(history) or bool(_DOC_REF_RE.search(question or ""))


class IntentClassifier:
    """Phân loại intent tất định. embedder optional → không có vẫn chạy rule-only."""

    def __init__(self, embedder: Optional[Any] = None):
        self.embedder = embedder
        self._proto_vecs: Optional[list[list[float]]] = None  # lazy

    # ── Embedding prototypes (lazy) ───────────────────────────────────────────
    def _ensure_prototypes(self) -> bool:
        if self.embedder is None:
            return False
        if self._proto_vecs is None:
            try:
                self._proto_vecs = self.embedder.encode(_RETRIEVE_PROTOTYPES)
            except Exception as exc:
                logger.warning(f"IntentClassifier: encode prototype lỗi, tắt embedding layer: {exc}")
                self.embedder = None
                return False
        return True

    def _max_retrieve_cos(self, question: str) -> float:
        from .embedding import cosine_similarity
        if not self._ensure_prototypes():
            return -1.0
        try:
            qv = self.embedder.encode([question])[0]
        except Exception as exc:
            logger.warning(f"IntentClassifier: encode query lỗi: {exc}")
            return -1.0
        return max(cosine_similarity(qv, pv) for pv in self._proto_vecs)

    # ── Public ────────────────────────────────────────────────────────────────
    def classify(
        self,
        question: str,
        has_document: bool = False,
        web_search_enabled: bool = True,
    ) -> ClassifierResult:
        """Phân loại 1 câu hỏi. Rule trước (precision cao), embedding sau (gate)."""
        q = (question or "").strip()
        if not q:
            return _result("clarify", RULE_CONFIDENCE, "rule")

        # ── Rule layer (thứ tự = ưu tiên) ─────────────────────────────────────
        # chitchat: chỉ khi câu NGẮN thuần chào/cảm ơn (tránh nuốt câu pháp lý dài)
        if len(q) <= 40 and _CHITCHAT_RE.search(q):
            return _result("chitchat", RULE_CONFIDENCE, "rule")

        # validate: CHỈ khi thực sự có văn bản (đính kèm hoặc trỏ rõ "…này/sau").
        # Đây là fix lõi cho bug ~50% validate-misfire.
        if has_document or _DOC_REF_RE.search(q):
            return _result("validate", RULE_CONFIDENCE, "rule")

        if _DRAFT_RE.search(q):
            return _result("draft", RULE_CONFIDENCE, "rule")

        if _COMPARE_RE.search(q):
            return _result("compare", RULE_CONFIDENCE, "rule")

        # calculate: cần tín hiệu ≥2 hành vi / cộng dồn (1 hành vi đơn = legal)
        if _CALC_MULTI_RE.search(q):
            return _result("calculate", RULE_CONFIDENCE, "rule")

        if web_search_enabled and (
            _WEBSEARCH_RE.search(q) or (_YEAR_RECENT_RE.search(q) and "phạt" not in q.lower())
        ):
            return _result("web_search", RULE_CONFIDENCE, "rule")

        if _KG_RE.search(q):
            return _result("kg_query", RULE_CONFIDENCE, "rule")

        # ── Embedding gate: có phải câu hỏi pháp lý cần tra cứu? ───────────────
        cos = self._max_retrieve_cos(q)
        if cos >= RETRIEVE_MIN_COS:
            intent = "consulting" if _FIRST_PERSON_RE.search(q) else "legal"
            # confidence map [RETRIEVE_MIN_COS,1] → [0.6,0.95]
            conf = min(0.95, 0.6 + (cos - RETRIEVE_MIN_COS) / (1 - RETRIEVE_MIN_COS) * 0.35)
            return _result(intent, conf, "embed")

        # Không chắc → để LLM router quyết (followup/clarify/mơ hồ).
        return _result("legal", max(0.0, cos), "low_conf")

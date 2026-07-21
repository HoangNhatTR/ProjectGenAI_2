"""Phase 1: Semantic KG extraction — dùng LLM extract Offense/Penalty/Subject.

Input: 1 Article (Điều) — text đầy đủ
Output: SemanticExtraction với offenses, penalties, subjects + relations về Clause.

Provider:
- Default: Gemini Flash (1M TPM, 1500 RPD free tier) — TPM headroom rộng
- Fallback: Groq (cẩn thận với 12K TPM)

Có retry với exponential backoff cho 429 (rate limit).
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .. import config
from ..llm_client import GeminiClient, GroqClient, Router9Client, OpenRouterClient, KieAIClient


# ─── Prompt ──────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """Bạn là chuyên gia phân tích văn bản pháp luật Việt Nam.
Nhiệm vụ: Đọc 1 ĐIỀU LUẬT và trích xuất các thực thể pháp lý có trong text.

CHỈ trích xuất những gì xuất hiện RÕ RÀNG trong text. KHÔNG suy đoán, KHÔNG bịa.

═══ ĐIỀU LUẬT ═══
{article_text}
═══════════════

Trích xuất các thực thể:

1. **offenses** (Hành vi vi phạm / hành vi bị quy định):
   - name: tên ngắn gọn (VD: "vượt đèn đỏ", "trộm cắp tài sản", "cố ý gây thương tích")
   - description: mô tả ngắn từ text (tối đa 200 ký tự)

   ⚠️ KHÔNG extract những thứ sau làm Offense (chúng KHÔNG phải hành vi cụ thể):
   - Phân loại tội phạm: "tội phạm", "phạm tội", "tội phạm nghiêm trọng",
     "tội phạm ít nghiêm trọng", "tội phạm rất nghiêm trọng", "tội phạm đặc biệt nghiêm trọng"
   - Khái niệm pháp lý chung: "hành vi vi phạm", "vi phạm pháp luật", "vi phạm quy định"
   - Định nghĩa: "cấu thành tội phạm", "trách nhiệm hình sự", "miễn trách nhiệm"
   - Câu mở đầu: "phạm tội thuộc một trong các trường hợp"

   ➜ Chỉ extract nếu có ĐỘNG TỪ HÀNH ĐỘNG CỤ THỂ + ĐỐI TƯỢNG:
     ✓ "trộm cắp tài sản" (động từ trộm cắp + đối tượng tài sản)
     ✓ "vượt đèn đỏ" (động từ vượt + đối tượng đèn đỏ)
     ✓ "không có giấy phép lái xe" (mô tả tình trạng vi phạm cụ thể)
     ✗ "phạm tội" (chỉ là từ chung)
     ✗ "tội phạm ít nghiêm trọng" (phân loại, không phải hành vi)

   ➜ Nếu Điều này chỉ chứa định nghĩa/phân loại/nguyên tắc (không có hành vi cụ thể)
     → trả về offenses = [] (mảng rỗng)

2. **penalties** (Hình phạt / chế tài):
   - description: mô tả nguyên văn (VD: "phạt tiền từ 800.000đ đến 1.000.000đ")
   - type: "phat_tien" | "phat_tu" | "tu_chung_than" | "tu_hinh" | "canh_cao" | "cai_tao_khong_giam_giu" | "tuoc_quyen" | "tich_thu" | "khac"
   - amount_min: số nguyên (VND) hoặc null
   - amount_max: số nguyên (VND) hoặc null
   - duration_min: số tháng (nếu là phạt tù) hoặc null
   - duration_max: số tháng hoặc null

3. **subjects** (Chủ thể áp dụng):
   - name: tên ngắn gọn (VD: "người điều khiển xe máy", "doanh nghiệp nhà nước", "người dưới 18 tuổi")
   - type: "ca_nhan" | "to_chuc" | "phuong_tien" | "khac"

4. **relations** (Khoản nào quy định hành vi nào, hình phạt nào, chủ thể nào):
   - clause: số Khoản (chuỗi, VD: "1", "2"). Nếu không có khoản rõ ràng → "0"
   - offense_names: list tên offense (chuỗi từ phần 1)
   - penalty_indices: list index (0-based) của penalty trong phần 2
   - subject_names: list tên subject (chuỗi từ phần 3)

Nếu Điều không chứa hành vi/hình phạt cụ thể (vd: định nghĩa, nguyên tắc) → trả về tất cả arrays rỗng.

CHỈ trả JSON đúng schema, KHÔNG kèm giải thích, KHÔNG markdown:
{{
  "offenses":  [{{"name": "...", "description": "..."}}],
  "penalties": [{{"description": "...", "type": "...", "amount_min": null, "amount_max": null, "duration_min": null, "duration_max": null}}],
  "subjects":  [{{"name": "...", "type": "..."}}],
  "relations": [{{"clause": "1", "offense_names": [], "penalty_indices": [], "subject_names": []}}]
}}"""


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class SemanticExtraction:
    """Kết quả extract semantic từ 1 Article."""
    article_id: str
    offenses: list[dict] = field(default_factory=list)
    penalties: list[dict] = field(default_factory=list)
    subjects: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    raw_response: str = ""
    error: Optional[str] = None
    quota_exhausted: bool = False  # True nếu daily quota hết (caller nên switch provider)

    @property
    def is_empty(self) -> bool:
        return not (self.offenses or self.penalties or self.subjects)


# ─── Quota / error detection ──────────────────────────────────────────────────

_DAILY_QUOTA_KEYWORDS = (
    "per day",
    "rpd",
    "daily quota",
    "limit: 0",
    "free_tier_requests",       # Gemini: "generate_content_free_tier_requests, limit: 0"
    "requests per day",
    "quota for metric",
    "daily_request_limit",
    "daily limit exceeded",
)

# Hết tiền / model trả phí / quota tháng — VĨNH VIỄN (không retry được), phải SWITCH
# provider/model. 9Router trả các message này kèm HTTP 402; KieAI nhét code 402 vào body.
_CREDIT_EXHAUSTED_KEYWORDS = (
    "credits insufficient",
    "insufficient credit",
    "credits required",
    "paid model",
    "exceeded your monthly quota",
    "monthly quota",
    "insufficient_quota",
    "out of credit",
    "please top up",
    "balance is",
)

_RATE_LIMIT_KEYWORDS = (
    "429",
    "rate limit",
    "resource_exhausted",
    "resource exhausted",
    "quota",
    "too many requests",
    "ratelimit",
)

# Server-side transient errors — đáng retry (không phải lỗi code/quota)
_TRANSIENT_KEYWORDS = (
    "502", "503", "504", "500",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "internal server error",
    "timeout",
    "timed out",          # openai.APITimeoutError → "Request timed out."
    "connection error",
    "connection reset",
    "connection aborted",
    "remote disconnect",
    "read timeout",
    "empty response",   # Router9Client: choices=None flaky (vd oc/nemotron-3-ultra-free)
)


def _get_http_status(exc: Exception) -> Optional[int]:
    """Trích xuất HTTP status code từ exception attributes (ưu tiên hơn text parsing).

    Hỗ trợ:
    - groq.APIStatusError / openai.APIStatusError: có .status_code
    - groq.APIError: có .response.status_code
    - google.api_core.exceptions: có .grpc_status_code (RESOURCE_EXHAUSTED=8 → 429)
    """
    # groq, openai SDK: status_code attribute trực tiếp
    if hasattr(exc, "status_code"):
        try:
            return int(exc.status_code)
        except (TypeError, ValueError):
            pass
    # groq.APIError: response object
    if hasattr(exc, "response") and exc.response is not None:
        resp = exc.response
        if hasattr(resp, "status_code"):
            try:
                return int(resp.status_code)
            except (TypeError, ValueError):
                pass
    # google.api_core.exceptions: grpc_status_code
    # RESOURCE_EXHAUSTED = 8 trong gRPC → tương đương HTTP 429
    if hasattr(exc, "grpc_status_code"):
        try:
            code_str = str(exc.grpc_status_code)
            if "RESOURCE_EXHAUSTED" in code_str or code_str.strip() == "8":
                return 429
        except Exception:
            pass
    return None


def is_daily_quota_exhausted(error_msg: str, exc: Optional[Exception] = None) -> bool:
    """Phân biệt daily quota (cần switch provider) vs per-minute throttle (chỉ cần đợi).

    Ưu tiên: HTTP status code từ exception > text matching.
    """
    msg = error_msg.lower()
    # Transient errors (5xx, timeout) không phải quota
    if any(kw in msg for kw in _TRANSIENT_KEYWORDS):
        return False
    # Hết tiền / paid model / quota tháng → treat như quota exhausted để SWITCH ngay
    # (không retry được, sẽ lỗi mọi điều nếu không đổi provider/model).
    if any(kw in msg for kw in _CREDIT_EXHAUSTED_KEYWORDS):
        return True
    if exc is not None:
        status = _get_http_status(exc)
        # HTTP 402 Payment Required → hết credit/paid, switch provider (KHÔNG retry)
        if status == 402:
            return True
        # HTTP 429 nhưng KHÔNG có daily-quota keyword → chỉ là per-minute throttle
        if status == 429 and not any(kw in msg for kw in _DAILY_QUOTA_KEYWORDS):
            return False
    return any(kw in msg for kw in _DAILY_QUOTA_KEYWORDS)


def is_rate_limited(error_msg: str, exc: Optional[Exception] = None) -> bool:
    """Phát hiện rate limiting (cần retry với backoff).

    HTTP 429 từ exception attributes là tín hiệu chắc chắn nhất.
    """
    msg = error_msg.lower()
    # MiMo (mmf/*) trả HTTP 400 code 441 "risk_control" khi request dồn dập —
    # bản chất là rate limit đội lốt 400, phải backoff chứ không fail điều.
    if "risk_control" in msg or '"code": "441"' in msg:
        return True
    if exc is not None:
        status = _get_http_status(exc)
        if status == 429:
            return True
        # Status code rõ ràng, không phải 429 → không phải rate limit
        if status is not None and status not in (429, 500, 502, 503, 504):
            return False
    return any(kw in msg for kw in _RATE_LIMIT_KEYWORDS)


def is_transient_error(error_msg: str, exc: Optional[Exception] = None) -> bool:
    """5xx, timeout, connection drops — đáng retry."""
    if exc is not None:
        status = _get_http_status(exc)
        if status is not None and status in (500, 502, 503, 504):
            return True
    msg = error_msg.lower()
    return any(kw in msg for kw in _TRANSIENT_KEYWORDS)


def _parse_reset_after_seconds(error_msg: str) -> Optional[int]:
    """cx/Codex (9Router): 'usage limit has been reached (reset after 27m 33s)'.

    Trả về tổng số giây cần chờ tới reset, hoặc None nếu không phải lỗi dạng này.
    """
    low = error_msg.lower()
    if "usage limit" not in low or "reset after" not in low:
        return None
    m = re.search(r"reset after\s+(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?", low)
    if not m or (m.group(1) is None and m.group(2) is None):
        return None
    return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)


# ─── Extractor ───────────────────────────────────────────────────────────────

class SemanticExtractor:
    """Wrapper extract semantic entities. Default Gemini Flash, fallback Groq.

    Có retry với exponential backoff khi gặp 429 (rate limit).
    """

    def __init__(
        self,
        provider: str = "gemini",   # "gemini" | "groq"
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_text_chars: int = 8000,
        max_retries: int = 8,
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_text_chars = max_text_chars
        self.max_retries = max_retries

        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or config.GEMINI_API_KEY
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY chưa được set trong .env")
            self._client = GeminiClient(api_key=api_key)
            # gemini-2.5-flash-lite: 15 RPM, 1000 RPD, 250K TPM (free tier)
            self.model = model or "gemini-2.5-flash-lite"
        elif self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY") or config.GROQ_API_KEY
            if not api_key:
                raise RuntimeError("GROQ_API_KEY chưa được set trong .env")
            self._client = GroqClient(api_key=api_key)
            # llama-3.3-70b-versatile: 30 RPM, 6000 RPD, 12K TPM — quality cao
            self.model = model or "llama-3.3-70b-versatile"
        elif self.provider == "groq-8b":
            api_key = os.getenv("GROQ_API_KEY") or config.GROQ_API_KEY
            if not api_key:
                raise RuntimeError("GROQ_API_KEY chưa được set trong .env")
            self._client = GroqClient(api_key=api_key)
            # llama-3.1-8b-instant: 30 RPM, 14400 RPD, 20K TPM — RPD cao hơn
            self.model = model or "llama-3.1-8b-instant"
        elif self.provider == "router9":
            # 9Router local proxy — OpenAI-compatible, free Claude Sonnet via Kiro
            api_key = os.getenv("ROUTER9_API_KEY")
            base_url = os.getenv("ROUTER9_BASE_URL", "http://localhost:20128/v1")
            if not api_key:
                raise RuntimeError(
                    "ROUTER9_API_KEY chưa được set trong .env. "
                    "Vào dashboard http://localhost:20128 để generate key."
                )
            self._client = Router9Client(api_key=api_key, base_url=base_url)
            # Default: free Claude Sonnet 4.5 qua Kiro AI (quality cao)
            self.model = model or os.getenv("ROUTER9_MODEL", "kr/claude-sonnet-4.5")
        elif self.provider == "openrouter":
            # GPT qua OpenRouter (user yêu cầu thay Claude). 9Router GPT đều 402.
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY chưa được set trong .env")
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self._client = OpenRouterClient(api_key=api_key, base_url=base_url)
            # gpt-4o-mini: rẻ + hỗ trợ json_object. Đổi qua OPENROUTER_SEMANTIC_MODEL nếu muốn.
            self.model = model or os.getenv("OPENROUTER_SEMANTIC_MODEL", "openai/gpt-4o-mini")
        elif self.provider == "kieai":
            # KieAI (kieai.erweima.ai) — deepseek-chat, non-Claude, free/rẻ.
            api_key = os.getenv("KIEAI_API_KEY")
            if not api_key:
                raise RuntimeError("KIEAI_API_KEY chưa được set trong .env")
            base_url = os.getenv("KIEAI_BASE_URL", "https://kieai.erweima.ai/api/v1")
            self._client = KieAIClient(api_key=api_key, base_url=base_url)
            self.model = model or os.getenv("KIEAI_SEMANTIC_MODEL", "deepseek-chat")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _call_with_retry(self, prompt: str) -> str:
        """Gọi LLM với exponential backoff.

        Strategy:
          - Daily quota exhausted → raise sớm để caller switch provider
          - Per-minute throttle → backoff + retry
          - Server transient (5xx, timeout, conn drop) → backoff + retry
          - Lỗi khác (validation, parse...) → raise sớm
        """
        last_exc: Optional[Exception] = None
        usage_waits = 0  # số lần đã chờ "usage limit reset" (cx/Codex window ~25 phút)
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    format="json",
                    options={"temperature": self.temperature},
                )
                return response["message"]["content"].strip()
            except Exception as exc:
                last_exc = exc
                msg = str(exc)

                # Daily quota → raise sớm để caller switch provider
                if is_daily_quota_exhausted(msg, exc=exc):
                    raise

                # cx/Codex "usage limit reached (reset after Xm Ys)" → chờ tới reset
                # rồi retry (backoff thường cap 90s không đủ cho window ~25 phút).
                reset_s = _parse_reset_after_seconds(msg)
                if reset_s is not None:
                    if reset_s <= 1900 and usage_waits < 2:
                        usage_waits += 1
                        print(f"          ⏳ cx usage-limit, chờ {reset_s}s tới reset...", flush=True)
                        time.sleep(reset_s + 5)
                        continue
                    raise  # chờ quá lâu / đã chờ 2 lần → bỏ, --skip-existing resume sau

                # Rate limit hoặc transient (5xx/timeout/conn) → backoff retry
                if is_rate_limited(msg, exc=exc) or is_transient_error(msg, exc=exc):
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(min(backoff, 90))
                    continue

                # Lỗi khác (validation, parse...) → raise sớm
                raise
        raise last_exc or RuntimeError("Retry exhausted")

    def extract(self, article_id: str, article_text: str) -> SemanticExtraction:
        # Cắt nếu text quá dài (tránh blow context)
        text = article_text[: self.max_text_chars]
        prompt = _EXTRACT_PROMPT.format(article_text=text)

        try:
            raw = self._call_with_retry(prompt)
        except Exception as exc:
            err_msg = str(exc)
            # _call_with_retry đã backoff 8 lần. Nếu VẪN daily-quota/credit HOẶC rate-limit
            # (429) → switch provider/model kế trong chain thay vì fail từng điều mãi. 429
            # dai dẳng = trần rate bị siết chặt (vd Antigravity free), đổi model là đúng.
            should_switch = (
                is_daily_quota_exhausted(err_msg, exc=exc)
                or is_rate_limited(err_msg, exc=exc)
            )
            return SemanticExtraction(
                article_id=article_id,
                error=f"LLM call failed: {err_msg}",
                quota_exhausted=should_switch,
            )

        # Parse JSON (đôi khi LLM wrap markdown)
        try:
            data = _parse_json_lenient(raw)
        except Exception as exc:
            return SemanticExtraction(
                article_id=article_id, raw_response=raw,
                error=f"JSON parse failed: {exc}",
            )

        return SemanticExtraction(
            article_id=article_id,
            offenses=data.get("offenses", []) or [],
            penalties=data.get("penalties", []) or [],
            subjects=data.get("subjects", []) or [],
            relations=data.get("relations", []) or [],
            raw_response=raw,
        )


def _parse_json_lenient(text: str) -> dict:
    """Parse JSON với fallback: nếu wrap markdown thì strip."""
    text = text.strip()
    # Strip markdown code fences nếu có
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return json.loads(text)

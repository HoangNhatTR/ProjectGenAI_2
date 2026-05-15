"""Smart router: 1 LLM call quyết định flow trước khi retrieve.

Output (JSON):
  action="retrieve"       + search_query → app retrieve + generate
  action="answer_direct"  + direct_response → trả thẳng, bỏ qua RAG
  action="use_tool"       + tool_name + tool_query → gọi tool đặc biệt

THÊM MỚI (Big Update):
  - Nhận ConversationState để viết lại query có ngữ cảnh
  - 3 intent mới: compare / calculate / draft
  - action="use_tool": router chỉ định tool cần gọi
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .state import ConversationState


# ─── Prompt ───────────────────────────────────────────────────────────────────

ROUTER_PROMPT = """Bạn là router thông minh cho Legal AI Agent tư vấn pháp luật Việt Nam.
Phân loại câu hỏi và quyết định flow xử lý.

{memory_block}

{summary_block}

{state_block}

Lịch sử hội thoại gần đây:
{history}

Câu hỏi mới của người dùng: {question}

─── Intent (chọn 1) ───
- legal      : hỏi thông tin quy định / mức phạt / thủ tục cụ thể → CẦN tra cứu
- consulting : yêu cầu tư vấn tình huống pháp lý cá nhân → CẦN tra cứu
- compare    : so sánh 2 trường hợp (vd: xe máy vs ô tô, lỗi A vs lỗi B) → CẦN tra cứu
- calculate  : yêu cầu tính TỔNG tiền phạt (nhiều lỗi cộng lại) → DÙNG TOOL
- draft      : yêu cầu soạn đơn / văn bản / hợp đồng → DÙNG TOOL
- followup   : câu hỏi tiếp theo dựa trên ngữ cảnh đã có → CẦN tra cứu với query rewrite
- chitchat   : chào hỏi, cảm ơn, tạm biệt → KHÔNG tra cứu
- meta       : hỏi về chatbot → KHÔNG tra cứu
- clarify    : câu hỏi quá mơ hồ, thiếu thông tin → KHÔNG tra cứu, hỏi lại

─── Action (chọn 1) ───
- "retrieve"      : dùng RAG (legal, consulting, compare, followup)
- "use_tool"      : gọi tool đặc biệt (calculate, draft)
- "answer_direct" : trả thẳng (chitchat, meta, clarify)

─── Quy tắc ───
1. Nếu action = "retrieve" hoặc "use_tool":
   - search_query = câu hỏi viết lại HOÀN CHỈNH, ĐỘC LẬP (bổ sung ngữ cảnh từ
     state / lịch sử nếu câu cụt). VD: "thế ô tô thì sao?" với ngữ cảnh
     'vượt đèn đỏ' → "Mức phạt vượt đèn đỏ đối với ô tô là bao nhiêu?"
   - Nếu intent = "calculate": tool_name = "calculate_fine", tool_query = mô tả
     đầy đủ tình huống cần tính (bổ sung loại xe từ state nếu có)
   - Nếu intent = "draft": tool_name = "draft_document", tool_query = "<loại văn bản>|<chi tiết>"
2. Nếu action = "answer_direct":
   - direct_response = câu trả lời ngắn gọn tiếng Việt
   - Với clarify: đặt 1-2 câu hỏi cụ thể để user bổ sung
   - Với chitchat: trả lời tự nhiên, mời hỏi pháp lý
   - Với meta: giới thiệu ngắn

CHỈ trả về JSON đúng schema, KHÔNG kèm giải thích:
{{
  "action":          "retrieve" | "use_tool" | "answer_direct",
  "intent":          "legal"|"consulting"|"compare"|"calculate"|"draft"|"followup"|"chitchat"|"meta"|"clarify",
  "search_query":    chuỗi hoặc null,
  "tool_name":       "calculate_fine"|"draft_document"|null,
  "tool_query":      chuỗi hoặc null,
  "direct_response": chuỗi hoặc null
}}"""


VALID_ACTIONS = {"retrieve", "use_tool", "answer_direct"}
VALID_INTENTS = {
    "legal", "consulting", "compare", "calculate", "draft",
    "followup", "chitchat", "meta", "clarify",
}
VALID_TOOLS   = {"calculate_fine", "draft_document"}


# ─── RouterDecision ───────────────────────────────────────────────────────────

@dataclass
class RouterDecision:
    action:          str
    intent:          str
    search_query:    Optional[str] = None
    tool_name:       Optional[str] = None
    tool_query:      Optional[str] = None
    direct_response: Optional[str] = None
    raw:             Optional[str] = None   # debug

    @property
    def needs_retrieval(self) -> bool:
        return self.action == "retrieve"

    @property
    def needs_tool(self) -> bool:
        return self.action == "use_tool"


# ─── SmartRouter ──────────────────────────────────────────────────────────────

class SmartRouter:
    """LLM-based router với State awareness."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        max_history_turns: int = 4,
        provider: str = "ollama",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.host = host
        self.max_history_turns = max_history_turns
        self.provider = provider
        self.api_key = api_key
        self._client: Optional[Any] = None

    def _connect(self) -> None:
        if self._client is None:
            if self.provider == "gemini":
                from .llm_client import GeminiClient
                self._client = GeminiClient(api_key=self.api_key or "")
            elif self.provider == "groq":
                from .llm_client import GroqClient
                self._client = GroqClient(api_key=self.api_key or "")
            else:
                from ollama import Client
                self._client = Client(host=self.host)

    def route(
        self,
        question: str,
        history:      Optional[list[dict]] = None,
        memory_text:  str = "",
        summary_text: str = "",
        state:        Optional["ConversationState"] = None,
    ) -> RouterDecision:
        """Quyết định flow cho 1 lượt.

        Args:
            question:     text user vừa gửi
            history:      messages {role, content} gần đây (raw)
            memory_text:  memory đã format
            summary_text: rolling summary
            state:        ConversationState hiện tại (MỚI)
        """
        self._connect()

        # ── Format history ────────────────────────────────────────────────────
        if history:
            recent = history[-(self.max_history_turns * 2):]
            history_str = "\n".join(
                f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content']}"
                for m in recent
            ) or "(trống)"
        else:
            history_str = "(trống)"

        memory_block = memory_text or "(chưa có memory về user)"
        summary_block = (
            "Tóm tắt cuộc hội thoại đến hiện tại:\n" + summary_text.strip()
            if summary_text.strip()
            else "(Chưa có tóm tắt)"
        )

        # ── State block (MỚI) ─────────────────────────────────────────────────
        if state and state.has_context():
            state_block = state.to_context_string()
        else:
            state_block = "(Chưa có ngữ cảnh hội thoại — phiên mới)"

        prompt = ROUTER_PROMPT.format(
            memory_block=memory_block,
            summary_block=summary_block,
            state_block=state_block,
            history=history_str,
            question=question,
        )

        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0, "num_ctx": 2048},
            )
            raw = response["message"]["content"]
        except Exception:
            return self._fallback(question)

        data = _extract_json(raw)
        if not data:
            return self._fallback(question, raw=raw)

        return self._parse_decision(data, question, raw=raw)

    # ── Parse ──────────────────────────────────────────────────────────────────

    def _parse_decision(self, data: dict, question: str, raw: str = "") -> RouterDecision:
        action  = str(data.get("action",  "retrieve")).strip()
        intent  = str(data.get("intent",  "legal"   )).strip()
        search_query    = data.get("search_query")
        tool_name       = data.get("tool_name")
        tool_query      = data.get("tool_query")
        direct_response = data.get("direct_response")

        if action not in VALID_ACTIONS:
            action = "retrieve"
        if intent not in VALID_INTENTS:
            intent = "legal"

        # ── Sanity: retrieve ──────────────────────────────────────────────────
        if action == "retrieve":
            if not isinstance(search_query, str) or not search_query.strip():
                search_query = question
            return RouterDecision(
                action="retrieve", intent=intent,
                search_query=search_query.strip(),
                raw=raw,
            )

        # ── Sanity: use_tool ──────────────────────────────────────────────────
        if action == "use_tool":
            if not isinstance(tool_name, str) or tool_name not in VALID_TOOLS:
                # Fallback: nếu LLM chọn use_tool nhưng tool không hợp lệ → retrieve
                sq = search_query or question
                return RouterDecision(
                    action="retrieve", intent=intent,
                    search_query=sq if isinstance(sq, str) else question,
                    raw=raw,
                )
            if not isinstance(tool_query, str) or not tool_query.strip():
                tool_query = question
            return RouterDecision(
                action="use_tool", intent=intent,
                search_query=search_query.strip() if isinstance(search_query, str) else question,
                tool_name=tool_name,
                tool_query=tool_query.strip(),
                raw=raw,
            )

        # ── Sanity: answer_direct ─────────────────────────────────────────────
        if not isinstance(direct_response, str) or not direct_response.strip():
            # LLM chọn answer_direct nhưng không có response → retrieve
            return RouterDecision(
                action="retrieve", intent="legal",
                search_query=question, raw=raw,
            )
        return RouterDecision(
            action="answer_direct", intent=intent,
            direct_response=direct_response.strip(),
            raw=raw,
        )

    # ── Fallback ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback(question: str, raw: Optional[str] = None) -> RouterDecision:
        return RouterDecision(
            action="retrieve", intent="legal",
            search_query=question, raw=raw,
        )


# ─── Helper ───────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

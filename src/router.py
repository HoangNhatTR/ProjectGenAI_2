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

from loguru import logger

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from .utils import extract_json as _extract_json

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
- compare    : so sánh quy định giữa 2 đối tượng/luật/tình huống → DÙNG TOOL compare_regulations
- calculate  : hỏi mức phạt / xử phạt hành chính cho 1 hoặc nhiều hành vi vi phạm giao thông → DÙNG TOOL calculate_fine
               Nhận diện: "phạt bao nhiêu", "mức phạt", "bị phạt", "xử phạt", "tiền phạt", "vi phạm ... phạt"
- draft      : yêu cầu soạn đơn / văn bản / hợp đồng → DÙNG TOOL draft_document
- validate   : yêu cầu kiểm tra / phân tích văn bản đã upload (bản án, hợp đồng, quyết định...) → DÙNG TOOL validate_document
- kg_query   : hỏi cụ thể về 1 HÀNH VI / TỘI DANH → DÙNG TOOL knowledge_graph_lookup
- web_search : tìm kiếm pháp luật mới nhất trên internet — luật sửa đổi gần đây, kiểm tra
               hiệu lực, văn bản chưa có trong corpus → DÙNG TOOL web_search
- followup   : câu hỏi tiếp theo dựa trên ngữ cảnh đã có → CẦN tra cứu với query rewrite
- chitchat   : chào hỏi, cảm ơn, tạm biệt → KHÔNG tra cứu
- meta       : hỏi về chatbot → KHÔNG tra cứu
- clarify    : câu hỏi quá mơ hồ, thiếu thông tin → KHÔNG tra cứu, hỏi lại

─── Action (chọn 1) ───
- "retrieve"      : dùng RAG (legal, consulting, followup)
- "use_tool"      : gọi tool đặc biệt (compare, calculate, draft, validate, kg_query, web_search)
- "answer_direct" : trả thẳng (chitchat, meta, clarify)

─── Quy tắc ───
1. Nếu action = "retrieve" hoặc "use_tool":
   - search_query = câu hỏi viết lại HOÀN CHỈNH, ĐỘC LẬP
   - Nếu intent = "compare":
       tool_name = "compare_regulations"
       tool_query = "<đối tượng A>|<đối tượng B>"
       VD: "xe máy vượt đèn đỏ|ô tô vượt đèn đỏ" hoặc "Luật A|Luật B"
   - Nếu intent = "calculate":
       tool_name = "calculate_fine"
       tool_query = mô tả đầy đủ tình huống cần tính
   - Nếu intent = "draft":
       tool_name = "draft_document"
       tool_query = "<loại văn bản>|<toàn bộ thông tin chi tiết người dùng cung cấp>"
       VD: user nói "viết đơn ly hôn của tôi Trần Nhật Hoàng và vợ Nguyễn Thị B do bất đồng quan điểm"
           → tool_query = "đơn ly hôn|Nguyên đơn: Trần Nhật Hoàng; Bị đơn: Nguyễn Thị B; Lý do: bất đồng quan điểm hôn nhân"
       Các từ khóa nhận diện: "soạn", "viết", "lập", "làm", "giúp tôi soạn/viết/lập"
       Loại văn bản thường gặp: đơn ly hôn, đơn khiếu nại, đơn tố cáo, hợp đồng lao động,
         hợp đồng thuê nhà, di chúc, công văn, biên bản, đơn xin việc, đơn nghỉ phép
   - Nếu intent = "validate":
       tool_name = "validate_document"
       tool_query = "[nội dung văn bản cần kiểm tra hoặc mô tả yêu cầu]"
   - Nếu intent = "kg_query":
       tool_name = "knowledge_graph_lookup"
       tool_query = TÊN HÀNH VI ngắn gọn
   - Nếu intent = "web_search":
       tool_name = "web_search"
       tool_query = câu truy vấn ngắn gọn tiếng Việt, bổ sung số năm / tên văn bản nếu user đề cập
       VD: "nghị định 168 năm 2024 xử phạt giao thông" hoặc "mức phạt nồng độ cồn mới nhất 2025"
2. Nếu action = "answer_direct":
   - direct_response = câu trả lời ngắn gọn tiếng Việt
   - Với clarify: đặt 1-2 câu hỏi cụ thể để user bổ sung
   - Với chitchat: trả lời tự nhiên, mời hỏi pháp lý
   - Với meta: giới thiệu ngắn

NHẬN DIỆN "validate": khi user nói "kiểm tra", "phân tích văn bản", "tìm sai sót",
"bản án này đúng không", "hợp đồng này có vấn đề gì", "chỉ ra sai phạm", hoặc có file đính kèm.

NHẬN DIỆN "web_search": khi user dùng từ "mới nhất", "hiện hành", "còn hiệu lực không",
"tìm trên mạng/internet/online", nêu năm 2024/2025/2026, hỏi về nghị định/thông tư vừa
ban hành, hoặc nói rõ muốn kiểm tra thông tin bên ngoài hệ thống.

CHỈ trả về JSON đúng schema, KHÔNG kèm giải thích:
{{
  "action":          "retrieve" | "use_tool" | "answer_direct",
  "intent":          "legal"|"consulting"|"compare"|"calculate"|"draft"|"validate"|"kg_query"|"web_search"|"followup"|"chitchat"|"meta"|"clarify",
  "search_query":    chuỗi hoặc null,
  "tool_name":       "calculate_fine"|"draft_document"|"compare_regulations"|"validate_document"|"knowledge_graph_lookup"|"web_search"|null,
  "tool_query":      chuỗi hoặc null,
  "direct_response": chuỗi hoặc null
}}"""


VALID_ACTIONS = {"retrieve", "use_tool", "answer_direct"}
VALID_INTENTS = {
    "legal", "consulting", "compare", "calculate", "draft",
    "validate", "kg_query", "web_search", "followup", "chitchat", "meta", "clarify",
}
VALID_TOOLS = {
    "calculate_fine",
    "draft_document",
    "compare_regulations",
    "validate_document",
    "knowledge_graph_lookup",
    "web_search",
}


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
            from .llm_client import create_client
            self._client = create_client(self.provider, self.api_key or "", self.host)

    def route(
        self,
        question: str,
        history:           Optional[list[dict]] = None,
        memory_text:       str = "",
        summary_text:      str = "",
        state:             Optional["ConversationState"] = None,
        web_search_enabled: bool = True,
    ) -> RouterDecision:
        """Quyết định flow cho 1 lượt.

        Args:
            question:           text user vừa gửi
            history:            messages {role, content} gần đây (raw)
            memory_text:        memory đã format
            summary_text:       rolling summary
            state:              ConversationState hiện tại
            web_search_enabled: nếu False, loại web_search khỏi prompt
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
        if not web_search_enabled:
            prompt += (
                "\n\n⚠️ LƯU Ý BẮT BUỘC: web_search ĐÃ TẮT bởi người dùng. "
                "TUYỆT ĐỐI không dùng intent='web_search' hay tool_name='web_search'. "
                "Nếu câu hỏi thường dùng web_search, hãy dùng intent='legal' + action='retrieve' thay thế."
            )

        _is_thinking = any(x in self.model.lower() for x in ("qwen3", "qwq", "deepseek-r"))
        _opts: dict = {"temperature": 0.0, "num_ctx": 2048}
        if _is_thinking:
            _opts["think"] = False

        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options=_opts,
            )
            raw = response["message"]["content"]
        except Exception as exc:
            logger.warning(f"Router LLM lỗi, dùng fallback heuristic: {exc}")
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
            # Bổ sung intent → tool_name nếu LLM quên điền
            if intent == "compare" and tool_name not in ("compare_regulations",):
                tool_name = "compare_regulations"
            if intent == "validate" and tool_name not in ("validate_document",):
                tool_name = "validate_document"
            if intent == "web_search" and tool_name not in ("web_search",):
                tool_name = "web_search"
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
# _extract_json imported from .utils

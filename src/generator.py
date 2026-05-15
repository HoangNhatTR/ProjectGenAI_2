"""Generator: sinh câu trả lời từ LLM.

Nâng cấp (Big Update):
  - Nhận tool_results (list[ToolResult]) từ Planner/Tools
  - Nhận ConversationState để cá nhân hóa câu trả lời
  - Tool results được chèn vào context trước RAG chunks
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Optional

from .schemas import Answer, Citation, RetrievedChunk

if TYPE_CHECKING:
    from .tools import ToolResult


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """Bạn là TRỢ LÝ TƯ VẤN pháp luật Việt Nam — Legal AI Agent. Phong cách:
- Tư vấn thực tế: ngoài việc trích dẫn quy định, hãy đưa khuyến nghị, cảnh báo
  rủi ro, gợi ý bước tiếp theo cho người dùng.
- Khi câu hỏi thiếu chi tiết quan trọng (vd: loại phương tiện, mức độ vi phạm),
  CHỦ ĐỘNG hỏi lại để tư vấn chính xác hơn.
- Tận dụng thông tin đã ghi nhớ về người dùng (nếu có) để cá nhân hoá.
- Khi nêu quy định, trích dẫn rõ Điều/Khoản/Điểm và tên văn bản, kèm chỉ số [n].
- Nếu nguồn mâu thuẫn, nêu rõ.
- Nếu không đủ căn cứ trong dữ liệu, nói rõ "Không tìm thấy căn cứ trong dữ liệu"
  thay vì bịa.
{memory_block}"""


SUMMARIZE_PROMPT = """Tóm tắt ngắn gọn (khoảng 120-180 từ) các lượt trao đổi
sau đây giữa người dùng và trợ lý pháp luật. Tập trung vào:
- Chủ đề / tình huống pháp lý đang thảo luận
- Thông tin chính người dùng đã cung cấp (hoàn cảnh, vai trò, phương tiện...)
- Các quy định / kết luận trợ lý đã đề cập (kèm số điều/khoản nếu có)
- Câu hỏi đang dang dở hoặc bước tiếp theo trợ lý đề xuất

{prev_block}

Các lượt cần tóm tắt:
{messages_text}

CHỈ trả về đoạn tóm tắt thuần văn xuôi, KHÔNG markdown, KHÔNG tiêu đề."""


MEMORY_EXTRACT_PROMPT = """Cho lượt trao đổi sau đây giữa người dùng và trợ lý pháp luật:

Người dùng: {question}
Trợ lý: {answer}

Nhiệm vụ: trích xuất các fact MỚI đáng nhớ lâu dài VỀ NGƯỜI DÙNG để tư vấn tốt hơn.

Quy tắc:
- Chỉ trích thông tin về CHÍNH NGƯỜI DÙNG: nghề nghiệp, hoàn cảnh, tài sản
  liên quan (vd loại xe), mối quan tâm pháp lý, vai trò trong tình huống.
- KHÔNG trích kiến thức pháp luật chung.
- KHÔNG trích lại câu hỏi của user.
- KHÔNG đoán mò: chỉ trích cái user đã nói rõ.
- Mỗi fact NGẮN GỌN, một câu khẳng định ở ngôi thứ ba (vd: "Là tài xế xe máy").
- Nếu không có gì đáng nhớ, trả về facts = [].

CHỈ trả về JSON: {{"facts": ["fact 1", "fact 2", ...]}}"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _format_source_tag(chunk) -> str:
    parts = [chunk.metadata.source]
    if chunk.article:
        parts.append(chunk.article)
    if chunk.clause:
        parts.append(chunk.clause)
    if chunk.point:
        parts.append(chunk.point)
    return " - ".join(parts)


def build_prompt(
    question: str,
    contexts: list[RetrievedChunk],
    tool_results: Optional[list["ToolResult"]] = None,
) -> str:
    """Xây dựng user prompt cho generator.

    Cấu trúc:
      [Tool results] (nếu có)
      [RAG context chunks]
      [Câu hỏi + hướng dẫn]
    """
    parts: list[str] = []

    # ── Tool results (từ Planner) ──────────────────────────────────────────────
    if tool_results:
        tool_blocks = []
        for tr in tool_results:
            if tr.success:
                tool_blocks.append(tr.format_for_prompt())
        if tool_blocks:
            parts.append("=== KẾT QUẢ TỪ TOOLS ===\n" + "\n\n".join(tool_blocks))

    # ── RAG contexts ──────────────────────────────────────────────────────────
    if contexts:
        rag_blocks = []
        for i, r in enumerate(contexts, 1):
            tag = _format_source_tag(r.chunk)
            rag_blocks.append(f"[{i}] Nguồn: {tag}\n{r.chunk.text}")
        parts.append("=== NGỮ CẢNH VĂN BẢN PHÁP LUẬT ===\n\n" + "\n\n".join(rag_blocks))
    elif not tool_results:
        parts.append(
            "Không có ngữ cảnh nào được truy xuất. "
            "Nếu câu hỏi liên quan đến lượt trao đổi trước hoặc memory đã có, "
            "hãy dựa vào đó để trả lời. Nếu không, nói rõ là không tìm thấy căn cứ."
        )

    # ── Câu hỏi + hướng dẫn ──────────────────────────────────────────────────
    instructions: list[str] = ["Trả lời dựa trên thông tin trên, lịch sử hội thoại, và memory."]
    if contexts:
        instructions.append("Trích dẫn dạng [số nguồn] (vd: theo [1]).")
    if tool_results:
        instructions.append("Tích hợp kết quả tool vào câu trả lời một cách tự nhiên.")
    instructions.append("Đưa khuyến nghị thực tế ngoài việc nêu quy định.")
    instructions.append("Nếu thiếu chi tiết quan trọng để tư vấn, hỏi lại user.")

    question_block = (
        f"---\n\nCâu hỏi: {question}\n\nHướng dẫn:\n"
        + "\n".join(f"- {ins}" for ins in instructions)
    )
    parts.append(question_block)

    return "\n\n".join(parts)


def _build_system_prompt(
    memory_text: str,
    summary_text: str = "",
    state_context: str = "",
) -> str:
    """Gộp summary + memory + state vào system prompt."""
    extra_parts: list[str] = []
    if summary_text.strip():
        extra_parts.append("TÓM TẮT CUỘC HỘI THOẠI ĐẾN HIỆN TẠI:\n" + summary_text.strip())
    if state_context.strip():
        extra_parts.append(state_context.strip())
    if memory_text.strip():
        extra_parts.append(memory_text.strip())
    extras = ("\n\n" + "\n\n".join(extra_parts)) if extra_parts else ""
    return SYSTEM_PROMPT_TEMPLATE.format(memory_block=extras)


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


# ─── Generator ────────────────────────────────────────────────────────────────

class Generator:
    """Wrapper LLM (Ollama hoặc Gemini): generate câu trả lời + extract memory facts."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
        num_ctx: int = 8192,
        provider: str = "ollama",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.host = host
        self.temperature = temperature
        self.num_ctx = num_ctx
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

    def get_client(self) -> Any:
        """Expose LLM client cho các module khác (Tools, Planner)."""
        self._connect()
        return self._client

    def generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history:      Optional[list[dict]] = None,
        memory_text:  str = "",
        summary_text: str = "",
        tool_results: Optional[list["ToolResult"]] = None,
        state_context: str = "",
    ) -> Answer:
        """Sinh câu trả lời với ngữ cảnh + tools + lịch sử + memory + state.

        Args:
            question:      câu hỏi gốc của user
            contexts:      chunks đã retrieve (có thể rỗng nếu chỉ dùng tool)
            history:       messages raw gần đây
            memory_text:   memory đã format
            summary_text:  rolling summary
            tool_results:  kết quả từ LegalToolRegistry / Planner (MỚI)
            state_context: ngữ cảnh conversation state dạng text (MỚI)
        """
        self._connect()
        user_prompt = build_prompt(question, contexts, tool_results=tool_results)
        system_prompt = _build_system_prompt(memory_text, summary_text, state_context)

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        response = self._client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        )
        answer_text = response["message"]["content"].strip()

        # Citations chỉ từ RAG contexts (tool results không tạo Citation)
        citations = [
            Citation(
                source=r.chunk.metadata.source,
                article=r.chunk.article,
                clause=r.chunk.clause,
                point=r.chunk.point,
                snippet=r.chunk.text[:200],
            )
            for r in contexts
        ]

        return Answer(question=question, answer=answer_text, citations=citations)

    def summarize_history(
        self,
        messages: list[dict],
        prev_summary: str = "",
    ) -> str:
        """Tóm tắt messages cũ thành đoạn văn ngắn (rolling summary)."""
        if not messages:
            return prev_summary
        self._connect()

        lines: list[str] = []
        for m in messages:
            role = "Người dùng" if m.get("role") == "user" else "Trợ lý"
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content}")
        if not lines:
            return prev_summary

        prev_block = (
            "Tóm tắt trước đó (cần MỞ RỘNG với thông tin mới, không lặp lại y nguyên):\n"
            + prev_summary.strip()
            if prev_summary.strip()
            else "(Chưa có tóm tắt trước đó.)"
        )

        prompt = SUMMARIZE_PROMPT.format(
            prev_block=prev_block,
            messages_text="\n".join(lines),
        )

        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_ctx": 4096},
            )
            text = response["message"]["content"].strip()
            return text or prev_summary
        except Exception:
            return prev_summary

    def extract_memory_facts(self, question: str, answer: str) -> list[str]:
        """Extract facts mới về user từ lượt vừa trao đổi. Không bao giờ raise."""
        self._connect()
        prompt = MEMORY_EXTRACT_PROMPT.format(question=question, answer=answer)
        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0, "num_ctx": 2048},
            )
            raw = response["message"]["content"]
        except Exception:
            return []

        data = _extract_json(raw)
        if not data:
            return []
        facts = data.get("facts", [])
        if not isinstance(facts, list):
            return []
        return [
            f.strip()
            for f in facts
            if isinstance(f, str) and f.strip() and len(f.strip()) <= 300
        ]

"""Planner / Reasoning Layer cho Legal AI Agent.

Phân tích câu hỏi phức tạp → tạo plan nhiều bước → execute tools.

Ví dụ:
  "Tôi vượt đèn đỏ, không mang GPLX, bị phạt tổng bao nhiêu?"
  → Plan:
    step 1: calculate_fine("vượt đèn đỏ xe máy")
    step 2: calculate_fine("không mang GPLX xe máy")
    step 3: Tổng hợp
  → Execute → trả ToolResult list cho Generator

Câu hỏi đơn giản ("mức phạt vượt đèn đỏ là gì?") → Plan 1 bước "retrieve".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .utils import extract_json as _extract_json

if TYPE_CHECKING:
    from .state import ConversationState
    from .tools import LegalToolRegistry, ToolResult


# ─── Prompt ───────────────────────────────────────────────────────────────────

_PLAN_PROMPT = """Bạn là Legal AI Agent. Phân tích câu hỏi và lập kế hoạch thực hiện.

{state_block}

Câu hỏi của người dùng: {question}

Hãy phân tích và trả về JSON:
{{
  "complexity": "simple" hoặc "complex",
  "reason": "lý do ngắn gọn",
  "steps": [
    {{
      "step": 1,
      "description": "mô tả bước",
      "tool": "retrieve" | "legal_search" | "law_article_lookup" | "calculate_fine" | "draft_document" | "compare_regulations" | "knowledge_graph_lookup" | "validate_document" | "web_search",
      "query": "tham số cho tool (câu query hoặc mô tả tình huống)"
    }}
  ]
}}

Quy tắc:
- "simple": chỉ 1 bước "retrieve" (câu hỏi tra cứu thông thường)
- "complex": nhiều bước HOẶC dùng tool đặc biệt:
    * Câu hỏi tính tổng tiền phạt nhiều lỗi → calculate_fine (1-2 bước)
    * Câu hỏi soạn đơn / công văn → draft_document
    * Câu hỏi tra Điều/Khoản số cụ thể → law_article_lookup
    * Câu hỏi so sánh 2 trường hợp (xe máy vs ô tô...) → compare_regulations, query = "A|B"
    * Câu hỏi về hành vi vi phạm / tội danh cụ thể → knowledge_graph_lookup
    * Câu hỏi yêu cầu kiểm tra/phân tích văn bản đã cung cấp → validate_document
    * Câu hỏi cần tìm văn bản mới nhất / kiểm tra hiệu lực → web_search
- Luôn đặt bước "retrieve" cuối nếu vẫn cần tra văn bản thêm
- KHÔNG tạo quá 4 bước

CHỈ trả về JSON, KHÔNG giải thích thêm."""


# ─── Data classes ─────────────────────────────────────────────────────────────

VALID_TOOLS = {
    "retrieve",
    "legal_search",
    "law_article_lookup",
    "calculate_fine",
    "draft_document",
    "web_search",
    "compare_regulations",
    "knowledge_graph_lookup",
    "validate_document",
}


@dataclass
class PlanStep:
    step: int
    description: str
    tool: str
    query: str
    result: Optional["ToolResult"] = None


@dataclass
class Plan:
    complexity: str          # "simple" | "complex"
    reason: str
    question: str
    steps: list[PlanStep] = field(default_factory=list)

    def is_complex(self) -> bool:
        return self.complexity == "complex" and any(
            s.tool != "retrieve" for s in self.steps
        )

    def tool_steps(self) -> list[PlanStep]:
        """Chỉ trả bước cần gọi tool (loại bước 'retrieve')."""
        return [s for s in self.steps if s.tool != "retrieve"]

    def retrieve_query(self) -> str:
        """Query cho bước retrieve cuối (fallback về câu gốc)."""
        for s in reversed(self.steps):
            if s.tool == "retrieve":
                return s.query
        return self.question


# ─── Planner ──────────────────────────────────────────────────────────────────

class LegalPlanner:
    """Tạo và thực thi plan cho câu hỏi pháp luật."""

    def __init__(
        self,
        ollama_client: Any,
        model: str,
        tool_registry: "LegalToolRegistry",
    ):
        self._client = ollama_client
        self.model = model
        self.tools = tool_registry
        self._is_thinking = any(x in model.lower() for x in ("qwen3", "qwq", "deepseek-r"))

    # ── Create plan ───────────────────────────────────────────────────────────

    def create_plan(
        self,
        question: str,
        state: Optional["ConversationState"] = None,
    ) -> Plan:
        """Tạo plan cho câu hỏi dựa trên state hiện tại."""
        state_block = (
            state.to_context_string() if (state and state.has_context())
            else "(Chưa có ngữ cảnh hội thoại)"
        )
        prompt = _PLAN_PROMPT.format(
            state_block=state_block,
            question=question,
        )
        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0, "num_ctx": 2048, **({"think": False} if self._is_thinking else {})},
            )
            raw = response["message"]["content"]
            data = _extract_json(raw)
            if not data:
                return _simple_plan(question)

            complexity = str(data.get("complexity", "simple"))
            reason = str(data.get("reason", ""))
            steps_raw = data.get("steps") or []

            steps: list[PlanStep] = []
            for i, s in enumerate(steps_raw[:4]):  # max 4 bước
                tool = str(s.get("tool", "retrieve")).strip()
                if tool not in VALID_TOOLS:
                    tool = "retrieve"
                steps.append(PlanStep(
                    step=int(s.get("step", i + 1)),
                    description=str(s.get("description", "")),
                    tool=tool,
                    query=str(s.get("query", question)),
                ))

            if not steps:
                return _simple_plan(question)

            return Plan(complexity=complexity, reason=reason, question=question, steps=steps)

        except Exception:
            return _simple_plan(question)

    # ── Execute plan ──────────────────────────────────────────────────────────

    def execute_plan(self, plan: Plan) -> list["ToolResult"]:
        """Thực thi tất cả bước tool trong plan (bỏ qua bước 'retrieve')."""
        results: list["ToolResult"] = []
        for step in plan.tool_steps():
            result = self._execute_step(step)
            step.result = result
            results.append(result)
        return results

    def _execute_step(self, step: PlanStep) -> "ToolResult":
        """Thực thi 1 bước plan."""
        tool = step.tool
        query = step.query

        if tool == "legal_search":
            return self.tools.legal_search(query=query)
        elif tool == "law_article_lookup":
            return self.tools.law_article_lookup(article_ref=query)
        elif tool == "calculate_fine":
            return self.tools.calculate_fine(description=query)
        elif tool == "draft_document":
            # Tách doc_type từ description nếu có (format: "<type>|<details>")
            if "|" in query:
                doc_type, details = query.split("|", 1)
            else:
                doc_type, details = "văn bản pháp lý", query
            return self.tools.draft_document(doc_type=doc_type.strip(), details=details.strip())
        elif tool == "web_search":
            return self.tools.web_search(query=query)
        elif tool == "compare_regulations":
            if "|" in query:
                a, b = query.split("|", 1)
                return self.tools.compare_regulations(topic_a=a.strip(), topic_b=b.strip())
            return self.tools.compare_regulations(topic_a=query, topic_b="")
        elif tool == "knowledge_graph_lookup":
            return self.tools.knowledge_graph_lookup(query=query)
        elif tool == "validate_document":
            return self.tools.validate_document(document_text=query)
        else:
            from .tools import ToolResult
            return ToolResult(tool_name=tool, success=False, result=f"Bước '{tool}' không execute được.")


# ─── Helpers ──────────────────────────────────────────────────────────────────
# _extract_json imported from .utils

def _simple_plan(question: str) -> Plan:
    return Plan(
        complexity="simple",
        reason="Câu hỏi đơn giản, dùng RAG thông thường.",
        question=question,
        steps=[PlanStep(
            step=1,
            description="Retrieve văn bản pháp luật liên quan",
            tool="retrieve",
            query=question,
        )],
    )

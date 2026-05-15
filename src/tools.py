"""Tool Calling Framework cho Legal AI Agent.

Tools có sẵn:
  legal_search       → tìm văn bản pháp luật (wrapper quanh Retriever)
  law_article_lookup → tra đúng Điều/Khoản cụ thể với filter metadata
  calculate_fine     → tính tiền phạt / tổng nhiều lỗi (RAG + LLM compute)
  draft_document     → soạn đơn, công văn, hợp đồng mẫu

Cách dùng:
  registry = LegalToolRegistry(retriever, ollama_client, model)
  result   = registry.execute("calculate_fine", description="vượt đèn đỏ + không GPLX")
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .retriever import Retriever


# ═══════════════════════════════════════════════════════════════════════════════
# ToolResult
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolResult:
    tool_name: str
    success: bool
    result: str
    sources: list[str] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        status = "OK" if self.success else "THẤT BẠI"
        header = f"[Kết quả từ tool: {self.tool_name} | {status}]"
        body = self.result.strip()
        if self.sources:
            body += "\nNguồn: " + "; ".join(self.sources[:5])
        return f"{header}\n{body}"


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

class LegalToolRegistry:
    """Quản lý và thực thi các tools của Legal AI Agent."""

    def __init__(
        self,
        retriever: "Retriever",
        ollama_client: Optional[Any] = None,
        model: str = "",
    ):
        self.retriever = retriever
        self._client = ollama_client
        self.model = model

    # ── Tool 1: legal_search ─────────────────────────────────────────────────

    def legal_search(self, query: str, top_k: int = 5) -> ToolResult:
        """Tìm kiếm văn bản pháp luật liên quan đến query."""
        try:
            chunks = self.retriever.retrieve(query, top_k=top_k)
            if not chunks:
                return ToolResult(
                    tool_name="legal_search",
                    success=False,
                    result="Không tìm thấy văn bản pháp luật liên quan trong cơ sở dữ liệu.",
                )
            parts: list[str] = []
            sources: list[str] = []
            for i, r in enumerate(chunks, 1):
                src = r.chunk.metadata.source
                tags = [src]
                if r.chunk.article:
                    tags.append(r.chunk.article)
                if r.chunk.clause:
                    tags.append(r.chunk.clause)
                tag = " - ".join(tags)
                parts.append(f"[{i}] {tag}\n{r.chunk.text[:500]}")
                sources.append(tag)
            return ToolResult(
                tool_name="legal_search",
                success=True,
                result="\n\n".join(parts),
                sources=sources,
            )
        except Exception as e:
            return ToolResult(tool_name="legal_search", success=False, result=str(e))

    # ── Tool 2: law_article_lookup ────────────────────────────────────────────

    def law_article_lookup(self, article_ref: str) -> ToolResult:
        """Tra cứu Điều/Khoản cụ thể (vd: 'Điều 6 Nghị định 100/2019/NĐ-CP')."""
        try:
            # Lấy nhiều candidates hơn để filter
            chunks = self.retriever.retrieve(article_ref, top_k=10)
            ref_lower = article_ref.lower()

            # Ưu tiên chunk có article/clause field khớp với ref
            matched = [
                r for r in chunks
                if (r.chunk.article and r.chunk.article.lower() in ref_lower)
                or (r.chunk.clause and r.chunk.clause.lower() in ref_lower)
                or ref_lower in r.chunk.text.lower()
            ]
            display = (matched or chunks)[:5]

            parts: list[str] = []
            sources: list[str] = []
            for i, r in enumerate(display, 1):
                src = r.chunk.metadata.source
                tags = [src]
                if r.chunk.article:
                    tags.append(r.chunk.article)
                if r.chunk.clause:
                    tags.append(r.chunk.clause)
                tag = " - ".join(tags)
                parts.append(f"[{i}] {tag}\n{r.chunk.text[:700]}")
                sources.append(tag)

            return ToolResult(
                tool_name="law_article_lookup",
                success=True,
                result="\n\n".join(parts),
                sources=sources,
            )
        except Exception as e:
            return ToolResult(tool_name="law_article_lookup", success=False, result=str(e))

    # ── Tool 3: calculate_fine ────────────────────────────────────────────────

    def calculate_fine(self, description: str) -> ToolResult:
        """Tính tổng tiền phạt cho tình huống vi phạm (có thể nhiều lỗi).

        Flow: retrieve luật liên quan → LLM tính từng khoản → cộng tổng.
        """
        if not self._client or not self.model:
            return ToolResult(
                tool_name="calculate_fine",
                success=False,
                result="Calculator tool cần kết nối LLM (Ollama/Claude).",
            )
        try:
            chunks = self.retriever.retrieve(description, top_k=6)
            context_blocks: list[str] = []
            for i, r in enumerate(chunks, 1):
                tags = [r.chunk.metadata.source]
                if r.chunk.article:
                    tags.append(r.chunk.article)
                if r.chunk.clause:
                    tags.append(r.chunk.clause)
                tag = " - ".join(tags)
                context_blocks.append(f"[{i}] {tag}\n{r.chunk.text[:500]}")
            context = "\n\n".join(context_blocks)

            prompt = f"""Dựa trên các điều khoản pháp luật sau:

{context}

---
Tình huống cần tính mức phạt: {description}

Yêu cầu:
1. Xác định TỪng hành vi vi phạm riêng biệt trong tình huống
2. Tra mức phạt cho từng hành vi (nêu rõ Điều/Khoản/văn bản)
3. Cộng tổng tất cả mức phạt
4. Nếu có hình thức phạt bổ sung (tước GPLX, tịch thu...) nêu riêng
5. Nếu không đủ căn cứ trong dữ liệu, ghi rõ "Chưa có căn cứ"

Trả lời rõ ràng, có bảng tổng kết nếu có nhiều vi phạm."""

            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_ctx": 4096},
            )
            result_text = response["message"]["content"].strip()
            sources = list(dict.fromkeys(r.chunk.metadata.source for r in chunks))
            return ToolResult(
                tool_name="calculate_fine",
                success=True,
                result=result_text,
                sources=sources,
            )
        except Exception as e:
            return ToolResult(tool_name="calculate_fine", success=False, result=str(e))

    # ── Tool 4: draft_document ────────────────────────────────────────────────

    def draft_document(self, doc_type: str, details: str) -> ToolResult:
        """Soạn thảo văn bản pháp lý (đơn, công văn, hợp đồng...).

        Args:
            doc_type: loại văn bản, vd "đơn khiếu nại", "hợp đồng thuê nhà"
            details:  thông tin chi tiết user cung cấp
        """
        if not self._client or not self.model:
            return ToolResult(
                tool_name="draft_document",
                success=False,
                result="Draft tool cần kết nối LLM.",
            )
        try:
            prompt = f"""Soạn {doc_type} theo đúng hình thức văn bản pháp lý Việt Nam.

Thông tin:
{details}

Yêu cầu:
- Đầy đủ: tiêu đề, quốc hiệu (nếu cần), kính gửi/nơi nhận, phần nội dung, phần ký
- Dùng [TÊN], [NGÀY THÁNG NĂM], [ĐỊA CHỈ], [SỐ CMND/CCCD] làm placeholder cho thông tin chưa có
- Ngôn ngữ trang trọng, rõ ràng, đúng thể thức
- Nếu có căn cứ pháp lý nào phù hợp thì đề cập"""

            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_ctx": 4096},
            )
            result_text = response["message"]["content"].strip()
            return ToolResult(
                tool_name="draft_document",
                success=True,
                result=result_text,
            )
        except Exception as e:
            return ToolResult(tool_name="draft_document", success=False, result=str(e))

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Gọi tool theo tên. kwargs tuỳ tool."""
        dispatch = {
            "legal_search":       lambda: self.legal_search(**kwargs),
            "law_article_lookup": lambda: self.law_article_lookup(**kwargs),
            "calculate_fine":     lambda: self.calculate_fine(**kwargs),
            "draft_document":     lambda: self.draft_document(**kwargs),
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=f"Tool '{tool_name}' không tồn tại. "
                       f"Các tool hợp lệ: {', '.join(dispatch)}",
            )
        return fn()

    def available_tools(self) -> list[str]:
        return ["legal_search", "law_article_lookup", "calculate_fine", "draft_document"]

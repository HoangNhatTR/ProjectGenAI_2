"""Tool Calling Framework cho Legal AI Agent.

Tools có sẵn:
  legal_search            → tìm văn bản pháp luật (wrapper quanh Retriever)
  law_article_lookup      → tra đúng Điều/Khoản cụ thể với filter metadata
  calculate_fine          → tính tiền phạt / tổng nhiều lỗi (RAG + LLM compute)
  draft_document          → soạn đơn, công văn, hợp đồng mẫu (kèm tra cứu luật)
  knowledge_graph_lookup  → tra cứu KG (Neo4j): hành vi vi phạm → hình phạt → chủ thể
  compare_regulations     → so sánh quy định giữa 2 đối tượng / 2 luật / 2 tình huống
  validate_document       → kiểm tra tính đúng đắn của văn bản pháp lý đã upload

Cách dùng:
  registry = LegalToolRegistry(retriever, ollama_client, model)
  result   = registry.execute("calculate_fine", description="vượt đèn đỏ + không GPLX")
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

# ── Fine amount extraction ─────────────────────────────────────────────────────

# Khớp "từ 800.000 đồng đến 1.200.000 đồng" và biến thể phổ biến
_FINE_RANGE_RE = re.compile(
    r'(?:phạt tiền\s+)?[Tt]ừ\s+([\d]+(?:\.[\d]{3})*)\s*(?:đồng|VNĐ|VND)?\s+'
    r'đến\s+([\d]+(?:\.[\d]{3})*)\s*(?:đồng|VNĐ|VND)',
    re.IGNORECASE,
)


def _parse_vnd(s: str) -> int:
    """'1.000.000' → 1_000_000 (loại bỏ dấu chấm phân cách hàng nghìn VN)."""
    return int(s.replace(".", ""))


def _build_fine_table(chunks) -> str:
    """Quét chunks để trích xuất mức phạt cụ thể bằng regex → bảng có cấu trúc.

    Kết quả này sẽ được thêm vào prompt của LLM như "ground truth" số liệu,
    giảm thiểu hallucination khi LLM tự đọc số từ văn bản tự do.
    """
    rows: list[str] = []
    seen: set[tuple] = set()

    for r in chunks:
        text = r.chunk.text
        tag = " – ".join(filter(None, [
            r.chunk.metadata.source.split("/")[-1].replace(".txt", ""),
            r.chunk.article,
            r.chunk.clause,
        ]))

        for m in _FINE_RANGE_RE.finditer(text):
            try:
                lo = _parse_vnd(m.group(1))
                hi = _parse_vnd(m.group(2))
            except (ValueError, IndexError):
                continue
            # Sanity check: lo <= hi, hi không vượt 2 tỷ VND
            if lo > hi or hi > 2_000_000_000 or lo < 0:
                continue
            key = (lo, hi, tag)
            if key in seen:
                continue
            seen.add(key)
            # Ngữ cảnh 100 ký tự trước match để biết áp dụng cho hành vi gì
            ctx_start = max(0, m.start() - 100)
            ctx = text[ctx_start: m.end() + 40].replace("\n", " ").strip()
            rows.append(f"| {tag} | {lo:,} – {hi:,} đ | …{ctx}… |")

    if not rows:
        return ""
    header = "| Nguồn | Mức phạt (min – max) | Ngữ cảnh trích dẫn |\n|---|---|---|\n"
    return header + "\n".join(rows[:15])  # tối đa 15 hàng để tránh tràn token

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
    docx_path: Optional[str] = None   # đường dẫn file .docx nếu đã export

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

    def _no_think(self, opts: dict) -> dict:
        """Thêm think=False vào options nếu model là thinking model (Qwen3/QwQ/...)."""
        if any(x in self.model.lower() for x in ("qwen3", "qwq", "deepseek-r")):
            return {**opts, "think": False}
        return opts

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
            # parent expansion bắt buộc: mức tiền phạt ("Phạt tiền từ X đến Y đồng")
            # nằm ở câu mở đầu của Khoản (parent), không nằm trong chunk Điểm a/b/c
            chunks = self.retriever.retrieve(
                description, top_k=6, use_parent_expansion=True,
            )
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

            # Trích xuất mức phạt bằng regex trước khi chuyển cho LLM
            fine_table = _build_fine_table(chunks)
            table_section = (
                f"\n\n**📊 BẢNG MỨC PHẠT TRÍCH XUẤT TỰ ĐỘNG TỪ VĂN BẢN LUẬT:**\n"
                f"{fine_table}\n"
                f"*(Dùng bảng này làm căn cứ số liệu — ưu tiên hơn đọc tự do)*\n"
                if fine_table else
                "\n*(Không trích xuất được mức phạt cụ thể — đọc từ context nguyên văn)*\n"
            )

            prompt = f"""Bạn là chuyên gia tính mức xử phạt vi phạm hành chính Việt Nam.
{table_section}
**VĂN BẢN PHÁP LUẬT NGUYÊN VĂN (đọc để hiểu điều kiện áp dụng):**
{context}

---
**TÌNH HUỐNG CẦN TÍNH:** {description}

Hãy thực hiện TỪNG BƯỚC:

**Bước 1 — Xác định vi phạm:**
Liệt kê từng hành vi vi phạm riêng biệt trong tình huống.

**Bước 2 — Tra mức phạt từng hành vi:**
Với mỗi hành vi:
- Căn cứ: Điều.../Khoản.../Nghị định.../Thông tư...
- Mức phạt tiền: từ ... đến ... đồng (lấy số từ bảng trích xuất nếu có)
- Phạt bổ sung (nếu có): tước GPLX, tịch thu phương tiện...

**Bước 3 — Bảng tổng kết:**
| Hành vi | Mức phạt (min–max) | Căn cứ |
|---|---|---|
| ... | ... | ... |
| **TỔNG** | **... đến ...** | |

**Bước 4 — Lưu ý:**
- Trường hợp tái phạm, tình tiết tăng nặng
- Các biện pháp khắc phục hậu quả
- Nếu thiếu căn cứ trong dữ liệu: ghi rõ "Cần xác minh thêm"

⚠️ Chỉ dùng số liệu có trong bảng trích xuất hoặc văn bản gốc ở trên.
KHÔNG bịa số liệu. Nếu không chắc, ghi "Cần xác minh".

Trả lời bằng Markdown, súc tích, chính xác."""

            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options=self._no_think({"temperature": 0.0, "num_ctx": 4096}),
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

    # Các trường bắt buộc theo loại văn bản (required=True → hỏi lại nếu thiếu)
    _REQUIRED_FIELDS: dict[str, list[dict]] = {
        "biên bản vi phạm": [
            {"key": "thoi_gian",     "label": "⏰ Thời gian vi phạm",   "desc": "Ngày và giờ cụ thể (VD: 14:30 ngày 15/06/2025)",     "required": True},
            {"key": "dia_diem",      "label": "📍 Địa điểm vi phạm",    "desc": "Tên đường, số nhà hoặc tuyến đường cụ thể",           "required": True},
            {"key": "hanh_vi",       "label": "⚠️ Hành vi vi phạm",     "desc": "Mô tả hành vi (VD: vượt đèn đỏ, không đội MBH...)", "required": True},
            {"key": "nguoi_vi_pham", "label": "👤 Tên người vi phạm",   "desc": "Họ và tên đầy đủ người vi phạm",                     "required": True},
            {"key": "bien_so",       "label": "🔢 Biển số xe",           "desc": "Biển kiểm soát phương tiện (VD: 51G-123.45)",        "required": True},
            {"key": "loai_xe",       "label": "🚘 Loại phương tiện",     "desc": "Xe máy / ô tô / xe điện / xe đạp điện...",           "required": True},
            {"key": "can_bo",        "label": "👮 Cán bộ lập biên bản", "desc": "Họ tên, đơn vị, chức vụ người lập biên bản",         "required": False},
            {"key": "cccd_vi_pham",  "label": "🪪 CCCD người vi phạm",  "desc": "Số CCCD/CMND của người vi phạm",                     "required": False},
            {"key": "chung_kien",    "label": "👁️ Người chứng kiến",    "desc": "Họ tên người chứng kiến (nếu có)",                    "required": False},
        ],
        "đơn ly hôn": [
            {"key": "nguyen_don", "label": "👤 Tên nguyên đơn",  "desc": "Người làm đơn (họ tên đầy đủ)",             "required": True},
            {"key": "bi_don",     "label": "👥 Tên bị đơn",      "desc": "Người kia (họ tên đầy đủ)",                  "required": True},
            {"key": "ly_do",      "label": "📋 Lý do ly hôn",    "desc": "Mô tả lý do (bất đồng, ngoại tình, bạo lực...)", "required": True},
            {"key": "ngay_ket_hon", "label": "📅 Ngày kết hôn",  "desc": "Ngày đăng ký kết hôn",                       "required": False},
            {"key": "con_chung",  "label": "👶 Con chung",        "desc": "Số con, tuổi (nếu có)",                       "required": False},
            {"key": "tai_san",    "label": "🏠 Tài sản chung",   "desc": "Tài sản cần phân chia (nếu có)",              "required": False},
        ],
        "hợp đồng lao động": [
            {"key": "ten_cty",  "label": "🏢 Tên công ty",       "desc": "Tên đầy đủ của doanh nghiệp",                 "required": True},
            {"key": "ten_nld",  "label": "👤 Tên người lao động","desc": "Họ tên người lao động",                        "required": True},
            {"key": "vi_tri",   "label": "💼 Vị trí/chức danh",  "desc": "Công việc đảm nhận",                           "required": True},
            {"key": "luong",    "label": "💰 Mức lương",          "desc": "Lương cơ bản hàng tháng",                     "required": True},
            {"key": "tgian",    "label": "📅 Thời hạn HĐ",       "desc": "Xác định thời hạn / không xác định thời hạn", "required": False},
        ],
        "hợp đồng thuê nhà": [
            {"key": "ben_a",    "label": "🏠 Bên cho thuê",      "desc": "Họ tên chủ nhà",                              "required": True},
            {"key": "ben_b",    "label": "👤 Bên thuê",           "desc": "Họ tên người thuê",                            "required": True},
            {"key": "dia_chi",  "label": "📍 Địa chỉ nhà thuê",  "desc": "Địa chỉ đầy đủ căn hộ/nhà",                  "required": True},
            {"key": "gia_thue", "label": "💰 Giá thuê",           "desc": "Giá thuê hàng tháng (VNĐ)",                   "required": True},
            {"key": "tgian",    "label": "📅 Thời hạn thuê",      "desc": "Từ ngày... đến ngày...",                       "required": True},
        ],
    }

    # Thể thức chuẩn cho từng loại văn bản
    _DOC_TEMPLATES: dict[str, str] = {
        "đơn ly hôn": (
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập – Tự do – Hạnh phúc\n"
            "─────────────────────────\n\n"
            "ĐƠN XIN LY HÔN\n\n"
            "Kính gửi: Tòa án nhân dân [QUẬN/HUYỆN], [TỈNH/TP]\n\n"
            "I. THÔNG TIN NGƯỜI LÀM ĐƠN (NGUYÊN ĐƠN)\n"
            "- Họ và tên: [TÊN NGUYÊN ĐƠN]\n"
            "- Ngày sinh: [NGÀY SINH]\n"
            "- CCCD/CMND số: [SỐ CCCD], cấp ngày [NGÀY CẤP] tại [NƠI CẤP]\n"
            "- Địa chỉ thường trú: [ĐỊA CHỈ]\n\n"
            "II. THÔNG TIN NGƯỜI BỊ YÊU CẦU (BỊ ĐƠN)\n"
            "- Họ và tên: [TÊN BỊ ĐƠN]\n"
            "- Ngày sinh: [NGÀY SINH]\n"
            "- CCCD/CMND số: [SỐ CCCD]\n"
            "- Địa chỉ thường trú: [ĐỊA CHỈ]\n\n"
            "III. NỘI DUNG\n"
            "1. Về hôn nhân:\n"
            "   Chúng tôi đăng ký kết hôn ngày [NGÀY] tại UBND [NƠI ĐĂNG KÝ].\n"
            "   Giấy chứng nhận kết hôn số [SỐ].\n\n"
            "2. Lý do xin ly hôn:\n"
            "   [LÝ DO CỤ THỂ]\n\n"
            "3. Về con chung (nếu có):\n"
            "   [THÔNG TIN CON]\n"
            "   Yêu cầu quyền nuôi con: [NỘI DUNG]\n\n"
            "4. Về tài sản chung:\n"
            "   [NỘI DUNG PHÂN CHIA TÀI SẢN]\n\n"
            "IV. YÊU CẦU\n"
            "Kính đề nghị Tòa án:\n"
            "1. Giải quyết cho [TÊN NGUYÊN ĐƠN] và [TÊN BỊ ĐƠN] được ly hôn.\n"
            "2. [CÁC YÊU CẦU KHÁC]\n\n"
            "V. CAM KẾT\n"
            "Tôi xin cam đoan những thông tin trên là đúng sự thật...\n\n"
            "[ĐỊA DANH], ngày    tháng    năm\n"
            "                            NGƯỜI LÀM ĐƠN\n"
            "                            (Ký, ghi rõ họ tên)\n\n"
            "Tài liệu kèm theo:\n"
            "- Giấy chứng nhận kết hôn (bản sao)\n"
            "- CCCD hai bên (bản sao)\n"
            "- Giấy khai sinh con (nếu có)\n"
            "- Tài liệu chứng minh lý do ly hôn (nếu có)"
        ),
        "biên bản vi phạm": (
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập – Tự do – Hạnh phúc\n"
            "─────────────────────────\n\n"
            "BIÊN BẢN VI PHẠM HÀNH CHÍNH\n\n"
            "Số: ......./BB-VPHC\n\n"
            "Hôm nay, ngày ... tháng ... năm ..., vào lúc ... giờ ... phút\n"
            "Tại: ...\n\n"
            "Chúng tôi gồm:\n"
            "1. [HỌ TÊN CÁN BỘ], [CHỨC VỤ], Đơn vị: [ĐƠN VỊ]\n"
            "2. (Người chứng kiến): [HỌ TÊN], [ĐỊA CHỈ]\n\n"
            "Tiến hành lập biên bản vi phạm hành chính đối với:\n\n"
            "─── THÔNG TIN NGƯỜI VI PHẠM ───\n"
            "Họ và tên: [TÊN NGƯỜI VI PHẠM]        Giới tính: [NAM/NỮ]\n"
            "Ngày sinh: [NGÀY SINH]                 Quốc tịch: Việt Nam\n"
            "CCCD/CMND số: [SỐ CCCD]\n"
            "Địa chỉ thường trú: [ĐỊA CHỈ]\n"
            "Nghề nghiệp: [NGHỀ NGHIỆP]\n\n"
            "─── THÔNG TIN PHƯƠNG TIỆN ───\n"
            "Loại xe: [LOẠI PHƯƠNG TIỆN]            Biển số: [BIỂN SỐ]\n"
            "Màu sơn: [MÀU XE]                      Nhãn hiệu: [NHÃN HIỆU]\n"
            "Số khung: [SỐ KHUNG]                   Số máy: [SỐ MÁY]\n\n"
            "─── NỘI DUNG VI PHẠM ───\n"
            "Hành vi vi phạm:\n"
            "[MÔ TẢ HÀNH VI VI PHẠM CHI TIẾT]\n\n"
            "Căn cứ pháp lý:\n"
            "Vi phạm quy định tại [ĐIỀU/KHOẢN], [TÊN NGHỊ ĐỊNH/LUẬT]\n\n"
            "─── HÌNH THỨC XỬ PHẠT ───\n"
            "1. Phạt tiền: [SỐ TIỀN] đồng\n"
            "   (Từ ... đến ... đồng theo [ĐIỀU KHOẢN])\n"
            "2. Hình thức phạt bổ sung (nếu có):\n"
            "   □ Tước quyền sử dụng GPLX thời hạn: ...\n"
            "   □ Tịch thu tang vật, phương tiện vi phạm: ...\n"
            "3. Biện pháp khắc phục hậu quả (nếu có): ...\n\n"
            "─── TANG VẬT, PHƯƠNG TIỆN TẠM GIỮ ───\n"
            "□ Không tạm giữ\n"
            "□ Tạm giữ: [GIẤY TỜ/PHƯƠNG TIỆN] — Biên lai tạm giữ số: ...\n\n"
            "Biên bản được lập thành 02 bản có giá trị như nhau, giao cho người vi phạm\n"
            "01 bản. Người vi phạm có quyền giải trình trong vòng 02 ngày làm việc.\n\n"
            "NGƯỜI VI PHẠM          NGƯỜI CHỨNG KIẾN          NGƯỜI LẬP BIÊN BẢN\n"
            "(Ký, ghi rõ họ tên)    (Ký, ghi rõ họ tên)       (Ký, ghi rõ họ tên)\n\n"
            "Ghi chú: Người vi phạm không ký thì ghi rõ lý do."
        ),
        "đơn khiếu nại": (
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập – Tự do – Hạnh phúc\n"
            "─────────────────────────\n\n"
            "ĐƠN KHIẾU NẠI\n\n"
            "Kính gửi: [CƠ QUAN CÓ THẨM QUYỀN]\n\n"
            "Người khiếu nại: [TÊN] | CCCD: [SỐ] | Địa chỉ: [ĐỊA CHỈ]\n\n"
            "I. NỘI DUNG KHIẾU NẠI:\n[NỘI DUNG]\n\n"
            "II. CĂN CỨ PHÁP LÝ:\n[ĐIỀU LUẬT]\n\n"
            "III. YÊU CẦU:\n[YÊU CẦU CỤ THỂ]\n\n"
            "[ĐỊA DANH], ngày    tháng    năm\n"
            "Người khiếu nại\n[TÊN]"
        ),
        "đơn tố cáo": (
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập – Tự do – Hạnh phúc\n"
            "─────────────────────────\n\n"
            "ĐƠN TỐ CÁO\n\n"
            "Kính gửi: [CƠ QUAN]\n\n"
            "Người tố cáo: [TÊN] | Địa chỉ: [ĐỊA CHỈ]\n"
            "Người bị tố cáo: [TÊN] | Chức vụ: [CHỨC VỤ]\n\n"
            "I. NỘI DUNG:\n[NỘI DUNG TỐ CÁO]\n\n"
            "II. CHỨNG CỨ:\n[CHỨNG CỨ]\n\n"
            "III. YÊU CẦU:\n[YÊU CẦU]\n\n"
            "[ĐỊA DANH], ngày    tháng    năm\nNgười tố cáo\n[TÊN]"
        ),
        "hợp đồng lao động": (
            "HỢP ĐỒNG LAO ĐỘNG\nSố: .../HĐLĐ\n\n"
            "Hôm nay, ngày ... tháng ... năm ..., tại ...\n\n"
            "NGƯỜI SỬ DỤNG LAO ĐỘNG: [TÊN CÔNG TY] | MST: [MST] | Đại diện: [TÊN]\n"
            "NGƯỜI LAO ĐỘNG: [TÊN] | CCCD: [SỐ] | Địa chỉ: [ĐỊA CHỈ]\n\n"
            "Điều 1. Công việc và địa điểm làm việc\n"
            "Điều 2. Thời hạn hợp đồng\n"
            "Điều 3. Tiền lương và phụ cấp\n"
            "Điều 4. Thời giờ làm việc, nghỉ ngơi\n"
            "Điều 5. BHXH, BHYT, BHTN\n"
            "Điều 6. Điều khoản chung\n\n"
            "ĐẠI DIỆN NSDLĐ              NGƯỜI LAO ĐỘNG"
        ),
        "hợp đồng thuê nhà": (
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập – Tự do – Hạnh phúc\n"
            "─────────────────────────\n\n"
            "HỢP ĐỒNG THUÊ NHÀ Ở\n\n"
            "BÊN CHO THUÊ (Bên A): [TÊN] | CCCD: [SỐ] | Địa chỉ: [ĐỊA CHỈ]\n"
            "BÊN THUÊ (Bên B): [TÊN] | CCCD: [SỐ] | Địa chỉ: [ĐỊA CHỈ]\n\n"
            "Điều 1. Tài sản cho thuê\n"
            "Điều 2. Thời hạn thuê\n"
            "Điều 3. Giá thuê và phương thức thanh toán\n"
            "Điều 4. Quyền và nghĩa vụ Bên A\n"
            "Điều 5. Quyền và nghĩa vụ Bên B\n"
            "Điều 6. Chấm dứt hợp đồng\n"
            "Điều 7. Điều khoản chung\n\n"
            "BÊN A                       BÊN B"
        ),
        "di chúc": (
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập – Tự do – Hạnh phúc\n"
            "─────────────────────────\n\n"
            "DI CHÚC\n\n"
            "Tôi là: [TÊN] | Sinh ngày: [NGÀY SINH]\n"
            "CCCD/CMND số: [SỐ] | Địa chỉ: [ĐỊA CHỈ]\n\n"
            "Trong khi còn minh mẫn, tôi lập di chúc này để định đoạt tài sản:\n\n"
            "1. Tài sản để lại:\n[DANH SÁCH TÀI SẢN]\n\n"
            "2. Người thừa kế:\n[TÊN VÀ PHẦN THỪA KẾ]\n\n"
            "3. Điều kiện thừa kế (nếu có):\n[ĐIỀU KIỆN]\n\n"
            "4. Người thực hiện di chúc:\n[TÊN]\n\n"
            "[ĐỊA DANH], ngày    tháng    năm\n"
            "Người lập di chúc\n[TÊN] (Ký, điểm chỉ)"
        ),
        "biên bản": (
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Độc lập – Tự do – Hạnh phúc\n"
            "─────────────────────────\n\n"
            "BIÊN BẢN [NỘI DUNG]\n\n"
            "Hôm nay, ngày ... tháng ... năm ..., tại ...\n\n"
            "Thành phần:\n- Chủ trì: [TÊN]\n- Thư ký: [TÊN]\n- Tham dự: [DANH SÁCH]\n\n"
            "NỘI DUNG:\n[NỘI DUNG CHI TIẾT]\n\n"
            "Biên bản kết thúc lúc ... giờ ... phút.\n\n"
            "THƯ KÝ                      CHỦ TRÌ"
        ),
        "công văn": (
            "[CƠ QUAN BAN HÀNH]          CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
            "Số: .../CV-...              Độc lập – Tự do – Hạnh phúc\n"
            "                            [ĐỊA DANH], ngày... tháng... năm...\n\n"
            "V/v: ...\n\nKính gửi: ...\n\n[NỘI DUNG]\n\n"
            "Nơi nhận:                   THỦ TRƯỞNG ĐƠN VỊ\n"
            "- Như trên;                 [TÊN]"
        ),
    }

    # Query tra cứu luật theo loại văn bản
    _LAW_SEARCH_MAP: dict[str, list[str]] = {
        "biên bản vi phạm":  ["biên bản vi phạm hành chính thủ tục lập", "xử phạt vi phạm hành chính giao thông"],
        "đơn ly hôn":        ["ly hôn thuận tình đơn phương", "điều kiện ly hôn luật hôn nhân gia đình", "quyền nuôi con sau ly hôn"],
        "đơn khiếu nại":     ["quyền khiếu nại công dân", "thủ tục khiếu nại hành chính"],
        "đơn tố cáo":        ["quyền tố cáo công dân", "thủ tục tố cáo vi phạm pháp luật"],
        "hợp đồng lao động": ["hợp đồng lao động bộ luật lao động", "quyền nghĩa vụ người lao động"],
        "hợp đồng thuê nhà": ["hợp đồng thuê nhà ở luật nhà ở", "quyền nghĩa vụ bên thuê nhà"],
        "di chúc":           ["di chúc thừa kế bộ luật dân sự", "điều kiện hợp lệ di chúc"],
        "biên bản":          ["biên bản vi phạm hành chính thủ tục", "biên bản làm việc"],
        "công văn":          ["công văn hành chính thể thức văn bản nhà nước"],
    }

    def _validate_and_extract(
        self, doc_lower: str, details: str
    ) -> tuple[bool, str, list[str]]:
        """Phân tích thông tin người dùng cung cấp, kiểm tra trường bắt buộc.

        Returns:
            (is_complete, extracted_summary, missing_labels)
            - is_complete: True nếu đủ thông tin để soạn
            - extracted_summary: thông tin đã parse được (dùng cho LLM soạn)
            - missing_labels: danh sách trường còn thiếu (để hỏi lại user)
        """
        # Tìm required fields cho loại văn bản này
        req_fields: list[dict] = []
        for key, fields in self._REQUIRED_FIELDS.items():
            if key in doc_lower or any(w in doc_lower for w in key.split()):
                req_fields = fields
                break

        if not req_fields:
            # Không có required fields → cứ soạn (không cần validate)
            return True, details, []

        # Dùng LLM trích xuất thông tin từ câu tự nhiên
        field_list = "\n".join(
            f'- {f["key"]}: {f["label"]} — {f["desc"]} [{"BẮT BUỘC" if f["required"] else "Tuỳ chọn"}]'
            for f in req_fields
        )
        extract_prompt = f"""Từ đoạn văn sau, hãy trích xuất các thông tin bên dưới.
Nếu thông tin KHÔNG có trong đoạn văn, ghi "KHÔNG CÓ".
CHỈ trả về JSON, không giải thích.

Đoạn văn:
"{details}"

Các trường cần trích xuất:
{field_list}

Trả về JSON dạng:
{{{", ".join(f'"{f["key"]}": "giá trị hoặc KHÔNG CÓ"' for f in req_fields)}}}"""

        try:
            resp = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": extract_prompt}],
                format="json",
                options=self._no_think({"temperature": 0.0, "num_ctx": 2048}),
            )
            import json as _json
            extracted = _json.loads(resp["message"]["content"])
        except Exception:
            # Nếu extract lỗi → cứ soạn với thông tin thô
            return True, details, []

        # Kiểm tra trường bắt buộc còn thiếu
        missing_labels: list[str] = []
        for f in req_fields:
            if f["required"]:
                val = extracted.get(f["key"], "KHÔNG CÓ")
                if not val or val.upper() in ("KHÔNG CÓ", "N/A", "NONE", "NULL", ""):
                    missing_labels.append(f'{f["label"]} — {f["desc"]}')

        # Tóm tắt thông tin đã có cho LLM soạn thảo
        provided = {
            f["label"]: extracted.get(f["key"], "—")
            for f in req_fields
            if extracted.get(f["key"], "KHÔNG CÓ").upper() not in ("KHÔNG CÓ", "N/A", "")
        }
        summary = "\n".join(f"• {k}: {v}" for k, v in provided.items())
        if not summary:
            summary = details

        return len(missing_labels) == 0, summary, missing_labels

    def draft_document(self, doc_type: str, details: str) -> ToolResult:
        """Soạn thảo văn bản pháp lý hoàn chỉnh kèm điều luật dẫn chứng.

        Pipeline:
          1. Nhận diện loại văn bản → chọn template + query luật
          2. Kiểm tra thông tin bắt buộc — nếu thiếu → hỏi lại user
          3. Tra cứu corpus lấy điều luật liên quan
          4. LLM soạn văn bản hoàn chỉnh điền thông tin + dẫn điều luật

        Args:
            doc_type: loại văn bản (vd "biên bản vi phạm", "đơn ly hôn")
            details:  thông tin người dùng — có thể là câu tự nhiên
        """
        if not self._client or not self.model:
            return ToolResult(
                tool_name="draft_document",
                success=False,
                result="Draft tool cần kết nối LLM.",
            )

        doc_lower = doc_type.lower()

        # ── 0. Kiểm tra thông tin bắt buộc — hỏi lại nếu thiếu ────────────────
        is_complete, extracted_summary, missing_labels = self._validate_and_extract(
            doc_lower, details
        )
        if not is_complete:
            bullets = "\n".join(f"  • {lbl}" for lbl in missing_labels)
            ask_msg = (
                f"Để soạn **{doc_type}**, tôi cần thêm các thông tin sau:\n\n"
                f"{bullets}\n\n"
                "Bạn vui lòng cung cấp để tôi soạn chính xác nhé."
            )
            return ToolResult(
                tool_name="draft_document",
                success=False,
                result=f"THIẾU_THÔNG_TIN:\n{ask_msg}",
            )
        # Dùng thông tin đã được LLM trích xuất & chuẩn hoá
        details = extracted_summary

        # ── 1. Chọn template và search queries ─────────────────────────────────
        template = ""
        for key, tmpl in self._DOC_TEMPLATES.items():
            if key in doc_lower or any(w in doc_lower for w in key.split()):
                template = tmpl
                break

        search_queries = ["văn bản pháp lý", doc_lower]
        for key, queries in self._LAW_SEARCH_MAP.items():
            if key in doc_lower or any(w in doc_lower for w in key.split()):
                search_queries = queries
                break

        try:
            # ── 2. Tra cứu điều luật liên quan ─────────────────────────────────
            law_chunks = []
            for q in search_queries[:2]:
                chunks = self.retriever.retrieve(q, top_k=3)
                law_chunks.extend(chunks)
            # Dedup theo chunk_id
            seen: set[str] = set()
            unique_chunks = []
            for c in law_chunks:
                if c.chunk.chunk_id not in seen:
                    seen.add(c.chunk.chunk_id)
                    unique_chunks.append(c)
            law_chunks = unique_chunks[:6]

            law_context_blocks: list[str] = []
            for r in law_chunks:
                src = r.chunk.metadata.source.split("/")[-1].replace(".txt", "")
                art = f" – {r.chunk.article}" if r.chunk.article else ""
                khn = f", {r.chunk.clause}" if r.chunk.clause else ""
                law_context_blocks.append(
                    f"📌 {src}{art}{khn}\n{r.chunk.text[:400]}"
                )
            law_context = "\n\n".join(law_context_blocks) if law_context_blocks else "(Không tìm thấy điều luật cụ thể trong corpus)"

            # ── 3. Soạn văn bản hoàn chỉnh ─────────────────────────────────────
            template_section = f"\nMẫu thể thức chuẩn:\n```\n{template}\n```\n" if template else ""

            prompt = f"""Bạn là luật sư giỏi tại Việt Nam. Hãy soạn thảo {doc_type} HOÀN CHỈNH, SẴN SÀNG SỬ DỤNG.

━━━ THÔNG TIN NGƯỜI DÙNG CUNG CẤP ━━━
{details}
{template_section}
━━━ CÁC ĐIỀU LUẬT LIÊN QUAN TRA CỨU ĐƯỢC ━━━
{law_context}

━━━ YÊU CẦU SOẠN THẢO ━━━
1. **Điền đầy đủ thông tin** đã biết từ yêu cầu người dùng (tên, lý do, v.v.)
2. **Dùng [THÔNG TIN CẦN ĐIỀN]** cho các trường còn thiếu (ngày sinh, địa chỉ, CCCD...)
3. **Dẫn điều luật cụ thể** — trích đúng Điều/Khoản từ các luật tìm được ở trên
4. **Ngôn ngữ chính xác** — dùng văn phong pháp lý trang trọng, rõ ràng
5. **Đầy đủ thể thức** — quốc hiệu, tên văn bản, kính gửi, nội dung, ký tên
6. Sau văn bản, thêm mục **"⚖️ CĂN CỨ PHÁP LÝ ÁP DỤNG"** liệt kê cụ thể:
   - Tên luật / nghị định
   - Điều khoản áp dụng
   - Nội dung quy định liên quan

Soạn ngay, không giải thích thêm."""

            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options=self._no_think({"temperature": 0.05, "num_ctx": 6000}),
            )
            result_text = response["message"]["content"].strip()
            sources = list(dict.fromkeys(
                r.chunk.metadata.source.split("/")[-1].replace(".txt", "")
                for r in law_chunks
            ))

            # ── Export sang DOCX ────────────────────────────────────────────────
            docx_path: Optional[str] = None
            try:
                from pathlib import Path as _Path
                from .docx_exporter import export_draft
                _export_dir = _Path(__file__).parent.parent / "data" / "exports"
                _out = export_draft(result_text, doc_type, _export_dir)
                docx_path = str(_out)
            except Exception:
                pass   # export thất bại → vẫn trả kết quả text bình thường

            return ToolResult(
                tool_name="draft_document",
                success=True,
                result=result_text,
                sources=sources,
                docx_path=docx_path,
            )
        except Exception as e:
            return ToolResult(tool_name="draft_document", success=False, result=str(e))

    # ── Tool 5: knowledge_graph_lookup ────────────────────────────────────────

    def knowledge_graph_lookup(self, query: str) -> ToolResult:
        """Tra KG Neo4j cho thông tin có cấu trúc: hành vi → hình phạt → chủ thể.

        Tốt nhất cho câu hỏi:
          - "Trộm cắp tài sản phạt bao nhiêu?"
          - "Hành vi nào áp dụng cho pháp nhân thương mại?"
          - "Tội nào phạt tù trên 10 năm?"

        Khác với legal_search (vector search trên chunks), tool này dùng graph
        traversal để trả về thông tin đã được structured (offense + penalty + subject).
        """
        try:
            from .kg.kg_queries import search_offense_to_penalty, search_by_name
        except Exception as exc:
            return ToolResult(
                tool_name="knowledge_graph_lookup",
                success=False,
                result=f"KG module không khả dụng: {exc}",
            )

        # Strategy: thử search Offense theo full query trước, fallback từng từ
        result = search_offense_to_penalty(query)
        rows = result["results"]

        if not rows:
            # Fallback: lấy từng từ ≥ 4 ký tự, thử riêng
            for word in query.split():
                if len(word) >= 4:
                    r = search_offense_to_penalty(word)
                    if r["results"]:
                        rows = r["results"]
                        break

        if not rows:
            # Fallback cuối: search Subject hoặc Offense by name (loose)
            subj_matches = search_by_name("Subject", query, limit=5)
            offense_matches = search_by_name("Offense", query, limit=5)
            if subj_matches or offense_matches:
                parts = []
                if subj_matches:
                    parts.append("Chủ thể liên quan trong KG:")
                    for s in subj_matches:
                        parts.append(f"  • {s['name']}")
                if offense_matches:
                    parts.append("Hành vi liên quan trong KG:")
                    for o in offense_matches:
                        parts.append(f"  • {o['name']}")
                return ToolResult(
                    tool_name="knowledge_graph_lookup",
                    success=True,
                    result="\n".join(parts) +
                           "\n\n(Không tìm thấy hành vi cụ thể với hình phạt — có thể câu hỏi cần tra cứu chi tiết hơn.)",
                )

            return ToolResult(
                tool_name="knowledge_graph_lookup",
                success=False,
                result=(
                    "Không tìm thấy hành vi vi phạm trong Knowledge Graph cho "
                    f"keyword: '{query}'. Có thể chuyển sang legal_search "
                    "(vector retrieval) để tìm văn bản gốc."
                ),
            )

        # Format kết quả
        parts: list[str] = []
        sources: list[str] = []
        for i, r in enumerate(rows, 1):
            block = [f"[{i}] Hành vi: {r.get('offense', '?')}"]
            desc = r.get("description")
            if desc:
                block.append(f"    Mô tả: {desc[:200]}")

            penalties = [p for p in r.get("penalties", []) if p]
            if penalties:
                block.append("    Hình phạt:")
                for p in penalties[:8]:
                    block.append(f"      • {p}")

            subjects = [s for s in r.get("subjects", []) if s]
            if subjects:
                block.append(f"    Chủ thể áp dụng: {', '.join(subjects[:5])}")

            source_laws = [s for s in r.get("source_laws", []) if s]
            if source_laws:
                block.append(f"    Nguồn: {source_laws[0]}")
                sources.extend(source_laws)

            parts.append("\n".join(block))

        return ToolResult(
            tool_name="knowledge_graph_lookup",
            success=True,
            result="\n\n".join(parts),
            sources=list(dict.fromkeys(sources))[:5],
        )

    # ── Tool 6: compare_regulations ──────────────────────────────────────────

    def compare_regulations(self, topic_a: str, topic_b: str) -> ToolResult:
        """So sánh quy định pháp luật giữa 2 đối tượng/tình huống.

        Args:
            topic_a: đối tượng/tình huống thứ nhất
            topic_b: đối tượng/tình huống thứ hai
        """
        if not self._client or not self.model:
            return ToolResult(
                tool_name="compare_regulations",
                success=False,
                result="Compare tool cần kết nối LLM.",
            )
        try:
            # Retrieve riêng cho từng topic
            chunks_a = self.retriever.retrieve(topic_a, top_k=4)
            chunks_b = self.retriever.retrieve(topic_b, top_k=4)

            def fmt_chunks(chunks, label: str) -> str:
                if not chunks:
                    return f"[{label}] Không tìm thấy quy định liên quan."
                parts = []
                for r in chunks:
                    src = r.chunk.metadata.source.split("/")[-1]
                    art = r.chunk.article or ""
                    parts.append(f"[{src} {art}]\n{r.chunk.text[:400]}")
                return f"=== Quy định về {label} ===\n" + "\n\n".join(parts)

            ctx_a = fmt_chunks(chunks_a, topic_a)
            ctx_b = fmt_chunks(chunks_b, topic_b)

            prompt = f"""So sánh quy định pháp luật Việt Nam giữa hai đối tượng sau:

{ctx_a}

{ctx_b}

Hãy so sánh theo cấu trúc bảng Markdown:

| Tiêu chí | {topic_a} | {topic_b} |
|---|---|---|
| Mức phạt tiền | ... | ... |
| Hình thức xử phạt bổ sung | ... | ... |
| Căn cứ pháp lý | ... | ... |
| Điều kiện áp dụng | ... | ... |
| Điểm khác biệt chính | ... | ... |

Sau bảng, hãy tóm tắt ngắn gọn điểm giống và khác nhau quan trọng nhất.
Ghi rõ nếu không đủ căn cứ để so sánh một tiêu chí nào đó."""

            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options=self._no_think({"temperature": 0.0, "num_ctx": 4096}),
            )
            result_text = response["message"]["content"].strip()
            sources = list(dict.fromkeys(
                [r.chunk.metadata.source.split("/")[-1] for r in chunks_a + chunks_b]
            ))
            return ToolResult(
                tool_name="compare_regulations",
                success=True,
                result=result_text,
                sources=sources[:6],
            )
        except Exception as e:
            return ToolResult(tool_name="compare_regulations", success=False, result=str(e))

    # ── Tool 7: validate_document ─────────────────────────────────────────────

    def validate_document(self, document_text: str, filename: str = "") -> ToolResult:
        """Kiểm tra tính đúng đắn và phát hiện sai phạm của văn bản pháp lý.

        Args:
            document_text: nội dung văn bản đã extract
            filename:      tên file gốc (tùy chọn, để nhận diện loại)
        """
        try:
            from .validator import DocumentAnalyzer
            analyzer = DocumentAnalyzer(
                retriever=self.retriever,
                llm_client=self._client,
                model=self.model,
            )
            report = analyzer.analyze(document_text, filename=filename)
            return ToolResult(
                tool_name="validate_document",
                success=True,
                result=report.to_markdown(),
                sources=[c.reference for c in report.citations if c.found_in_corpus][:5],
            )
        except Exception as e:
            return ToolResult(tool_name="validate_document", success=False, result=str(e))

    # ── Tool 8: web_search ────────────────────────────────────────────────────

    def web_search(self, query: str, num_results: int = 6) -> ToolResult:
        """Tìm kiếm pháp luật trên thuvienphapluat.vn và vbpl.vn.

        Dùng khi:
        - Luật mới ban hành chưa có trong corpus (nghị định, thông tư mới nhất)
        - Kiểm tra văn bản còn hiệu lực hay đã bị sửa đổi/bãi bỏ
        - User hỏi thông tin năm gần đây (2024, 2025, 2026)
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return ToolResult(
                tool_name="web_search",
                success=False,
                result=(
                    "Chưa cài thư viện tìm kiếm.\n"
                    "Chạy lệnh: pip install duckduckgo-search>=6.0.0\n"
                    "Sau đó thử lại câu hỏi."
                ),
            )

        _SOURCES = [
            ("thuvienphapluat.vn", "Thư viện Pháp luật"),
            ("vbpl.vn",            "Cơ sở VBPL Quốc gia"),
        ]
        per_site = max(2, num_results // len(_SOURCES) + 1)

        try:
            combined: list[dict] = []
            seen_urls: set[str] = set()

            with DDGS() as ddgs:
                for domain, _ in _SOURCES:
                    site_query = f"site:{domain} {query}"
                    for r in ddgs.text(
                        site_query,
                        region="vn-vi",
                        safesearch="moderate",
                        max_results=per_site,
                    ):
                        url = r.get("href", "")
                        if url not in seen_urls:
                            seen_urls.add(url)
                            combined.append(r)

            if not combined:
                return ToolResult(
                    tool_name="web_search",
                    success=False,
                    result=(
                        f"Không tìm thấy kết quả cho '{query}' trên "
                        "thuvienphapluat.vn và vbpl.vn."
                    ),
                )

            parts:   list[str] = []
            sources: list[str] = []
            for i, r in enumerate(combined[:num_results], 1):
                title = r.get("title", "(không có tiêu đề)").strip()
                url   = r.get("href",  "").strip()
                body  = r.get("body",  "").strip()[:400]
                # Gắn nhãn nguồn
                site_label = next(
                    (label for domain, label in _SOURCES if domain in url),
                    "web",
                )
                parts.append(f"[{i}] [{site_label}] **{title}**\n{url}\n{body}")
                if url:
                    sources.append(url)

            return ToolResult(
                tool_name="web_search",
                success=True,
                result="\n\n".join(parts),
                sources=sources[:5],
            )

        except Exception as exc:
            err = str(exc)
            hint = " Thử lại sau vài giây." if "ratelimit" in err.lower() else ""
            return ToolResult(
                tool_name="web_search",
                success=False,
                result=f"Lỗi tìm kiếm: {err}.{hint}",
            )

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Gọi tool theo tên. kwargs tuỳ tool."""
        dispatch = {
            "legal_search":           lambda: self.legal_search(**kwargs),
            "law_article_lookup":     lambda: self.law_article_lookup(**kwargs),
            "calculate_fine":         lambda: self.calculate_fine(**kwargs),
            "draft_document":         lambda: self.draft_document(**kwargs),
            "knowledge_graph_lookup": lambda: self.knowledge_graph_lookup(**kwargs),
            "compare_regulations":    lambda: self._execute_compare(kwargs),
            "validate_document":      lambda: self.validate_document(
                                          document_text=kwargs.get("document_text",
                                                                    kwargs.get("query", "")),
                                          filename=kwargs.get("filename", ""),
                                      ),
            "web_search":             lambda: self.web_search(
                                          query=kwargs.get("query", kwargs.get("tool_query", "")),
                                          num_results=int(kwargs.get("num_results", 6)),
                                      ),
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

    def _execute_compare(self, kwargs: dict) -> ToolResult:
        """Helper: tách query 'A|B' thành topic_a, topic_b cho compare_regulations."""
        query = kwargs.get("query", kwargs.get("tool_query", ""))
        if "|" in query:
            parts = query.split("|", 1)
            return self.compare_regulations(topic_a=parts[0].strip(), topic_b=parts[1].strip())
        return self.compare_regulations(topic_a=query, topic_b="")

    def available_tools(self) -> list[str]:
        return [
            "legal_search",
            "law_article_lookup",
            "calculate_fine",
            "draft_document",
            "knowledge_graph_lookup",
            "compare_regulations",
            "validate_document",
            "web_search",
        ]

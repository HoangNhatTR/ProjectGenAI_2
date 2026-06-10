"""Validator — Phân tích và kiểm tra tính đúng đắn của văn bản pháp lý.

Chức năng:
  1. Nhận diện loại văn bản (bản án, hợp đồng, quyết định xử phạt, đơn...)
  2. Trích xuất các trích dẫn pháp luật (số hiệu, điều, khoản)
  3. Đối chiếu trích dẫn với corpus (RAG lookup)
  4. Kiểm tra các yếu tố bắt buộc theo loại văn bản
  5. Phát hiện mâu thuẫn, sai sót, thiếu sót
  6. Trả về báo cáo có cấu trúc
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .retriever import Retriever


# ── Patterns trích xuất trích dẫn pháp luật ──────────────────────────────────

# "100/2019/NĐ-CP", "45/2013/QH13", "01/2021/TT-BCA"
_LAW_NUMBER_RE = re.compile(
    r"\b\d{1,4}/\d{4}/(?:NĐ|QH|TT|QĐ|CT|TTLT|VBHN|NQ|PL)-\w+\b",
    re.IGNORECASE,
)
# "Điều 5", "Điều 10a", "điều 3"
_ARTICLE_RE = re.compile(r"\bĐiều\s+(\d+[a-z]?)\b", re.IGNORECASE)
# "Khoản 1", "khoản 2"
_CLAUSE_RE = re.compile(r"\bKhoản\s+(\d+)\b", re.IGNORECASE)
# "Điểm a", "điểm b"
_POINT_RE = re.compile(r"\bĐiểm\s+([a-z])\b", re.IGNORECASE)
# Bộ luật / Luật tên dài
_NAMED_LAW_RE = re.compile(
    r"(?:Bộ luật|Luật)\s+[\w\s]{3,50}(?:năm\s+\d{4})?",
    re.IGNORECASE,
)


# ── Loại văn bản + yếu tố bắt buộc ──────────────────────────────────────────

DOCUMENT_TYPES: dict[str, dict] = {
    "ban_an_hinh_su": {
        "name": "Bản án hình sự",
        "keywords": ["tòa án", "bị cáo", "viện kiểm sát", "hình phạt", "tội danh", "bản án"],
        "required": [
            "Tên Tòa án ra bản án",
            "Ngày, tháng, năm xét xử",
            "Họ tên bị cáo + thông tin nhân thân",
            "Tội danh bị truy tố và điều luật áp dụng",
            "Chứng cứ được xem xét",
            "Nhận định của Hội đồng xét xử",
            "Mức hình phạt tuyên",
            "Quyền kháng cáo",
        ],
    },
    "ban_an_dan_su": {
        "name": "Bản án dân sự",
        "keywords": ["nguyên đơn", "bị đơn", "tranh chấp", "yêu cầu khởi kiện", "dân sự"],
        "required": [
            "Tên Tòa án",
            "Nguyên đơn và bị đơn",
            "Yêu cầu của nguyên đơn",
            "Phản biện của bị đơn",
            "Chứng cứ",
            "Căn cứ pháp lý",
            "Phán quyết",
            "Nghĩa vụ thi hành án",
        ],
    },
    "quyet_dinh_xu_phat": {
        "name": "Quyết định xử phạt vi phạm hành chính",
        "keywords": ["quyết định xử phạt", "vi phạm hành chính", "phạt tiền", "người vi phạm"],
        "required": [
            "Căn cứ pháp lý ra quyết định (Luật XPVPHC)",
            "Thông tin người/tổ chức vi phạm",
            "Hành vi vi phạm cụ thể",
            "Điều khoản vi phạm (số Điều, Khoản)",
            "Mức phạt tiền",
            "Hình thức phạt bổ sung (nếu có)",
            "Biện pháp khắc phục hậu quả (nếu có)",
            "Thời hạn thi hành",
            "Quyền khiếu nại, khởi kiện",
            "Chữ ký người có thẩm quyền",
        ],
    },
    "hop_dong_lao_dong": {
        "name": "Hợp đồng lao động",
        "keywords": ["hợp đồng lao động", "người lao động", "người sử dụng lao động", "tiền lương", "thời hạn hợp đồng"],
        "required": [
            "Thông tin hai bên (họ tên, địa chỉ, CCCD/MST)",
            "Loại hợp đồng (xác định/không xác định thời hạn)",
            "Công việc phải làm",
            "Địa điểm làm việc",
            "Thời giờ làm việc, nghỉ ngơi",
            "Tiền lương và hình thức trả lương",
            "Phụ cấp, trợ cấp (nếu có)",
            "Trang thiết bị bảo hộ lao động",
            "BHXH, BHYT, BHTN",
            "Điều khoản chấm dứt hợp đồng",
        ],
    },
    "hop_dong_mua_ban": {
        "name": "Hợp đồng mua bán",
        "keywords": ["hợp đồng mua bán", "bên mua", "bên bán", "giá trị hợp đồng", "hàng hóa", "tài sản"],
        "required": [
            "Thông tin bên mua và bên bán",
            "Đối tượng hợp đồng (mô tả tài sản/hàng hóa)",
            "Số lượng, chất lượng",
            "Giá trị và phương thức thanh toán",
            "Thời hạn và địa điểm giao hàng",
            "Bảo hành (nếu có)",
            "Phạt vi phạm và bồi thường thiệt hại",
            "Điều khoản giải quyết tranh chấp",
        ],
    },
    "don_khieu_nai": {
        "name": "Đơn khiếu nại",
        "keywords": ["đơn khiếu nại", "kính gửi", "khiếu nại về", "đề nghị xem xét", "quyết định"],
        "required": [
            "Quốc hiệu, tiêu ngữ",
            "Tên đơn: ĐƠN KHIẾU NẠI",
            "Kính gửi (cơ quan có thẩm quyền)",
            "Thông tin người khiếu nại",
            "Nội dung khiếu nại (sự việc, quyết định bị khiếu nại)",
            "Căn cứ pháp lý",
            "Yêu cầu cụ thể",
            "Cam kết",
            "Ngày, chữ ký",
        ],
    },
    "don_to_cao": {
        "name": "Đơn tố cáo",
        "keywords": ["đơn tố cáo", "tố cáo hành vi", "vi phạm pháp luật", "đề nghị xử lý"],
        "required": [
            "Quốc hiệu, tiêu ngữ",
            "Tên đơn: ĐƠN TỐ CÁO",
            "Kính gửi",
            "Thông tin người tố cáo",
            "Thông tin người bị tố cáo",
            "Nội dung hành vi bị tố cáo",
            "Chứng cứ, tài liệu kèm theo",
            "Yêu cầu",
            "Ngày, chữ ký",
        ],
    },
    "cong_van": {
        "name": "Công văn hành chính",
        "keywords": ["công văn", "v/v", "kính gửi", "số:", "trân trọng"],
        "required": [
            "Tiêu đề cơ quan ban hành",
            "Số, ký hiệu công văn",
            "Ngày tháng năm",
            "Kính gửi",
            "Trích yếu nội dung (V/v ...)",
            "Nội dung chính",
            "Ký tên, đóng dấu",
        ],
    },
}


# ── Kết quả kiểm tra ─────────────────────────────────────────────────────────

@dataclass
class CitationCheck:
    reference: str          # "Điều 5 Nghị định 100/2019/NĐ-CP"
    found_in_corpus: bool
    relevant_text: str = ""
    note: str = ""


@dataclass
class ValidationReport:
    doc_type: str                              # tên loại văn bản
    doc_type_key: str                          # key nội bộ
    confidence: float                          # 0.0–1.0
    citations: list[CitationCheck] = field(default_factory=list)
    missing_elements: list[str] = field(default_factory=list)
    present_elements: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    llm_analysis: str = ""                     # phân tích tự do của LLM

    def to_markdown(self) -> str:
        lines: list[str] = []

        # Tiêu đề
        lines.append(f"## 📋 Báo cáo kiểm tra: {self.doc_type}")
        lines.append(f"*Độ tin cậy nhận diện: {self.confidence:.0%}*\n")

        # Trích dẫn pháp luật
        if self.citations:
            lines.append("### 📎 Kiểm tra trích dẫn pháp luật")
            ok_cits = [c for c in self.citations if c.found_in_corpus]
            bad_cits = [c for c in self.citations if not c.found_in_corpus]
            if ok_cits:
                lines.append("**✅ Tìm thấy trong corpus:**")
                for c in ok_cits:
                    lines.append(f"- `{c.reference}` — {c.note or 'Có trong cơ sở dữ liệu'}")
            if bad_cits:
                lines.append("**⚠️ Không tìm thấy hoặc cần xác minh:**")
                for c in bad_cits:
                    lines.append(f"- `{c.reference}` — {c.note or 'Không tìm thấy trong corpus'}")
            lines.append("")

        # Yếu tố bắt buộc
        if self.present_elements or self.missing_elements:
            lines.append("### 📝 Các yếu tố bắt buộc")
            for el in self.present_elements:
                lines.append(f"- ✅ {el}")
            for el in self.missing_elements:
                lines.append(f"- ❌ **Thiếu:** {el}")
            lines.append("")

        # Vấn đề phát hiện
        if self.issues:
            lines.append("### ⚠️ Vấn đề phát hiện")
            for issue in self.issues:
                lines.append(f"- {issue}")
            lines.append("")

        # Đề xuất
        if self.suggestions:
            lines.append("### 💡 Đề xuất cải thiện")
            for s in self.suggestions:
                lines.append(f"- {s}")
            lines.append("")

        # Phân tích của LLM
        if self.llm_analysis:
            lines.append("### 🤖 Phân tích chi tiết")
            lines.append(self.llm_analysis)

        return "\n".join(lines)


# ── DocumentAnalyzer ─────────────────────────────────────────────────────────

class DocumentAnalyzer:
    """Phân tích văn bản pháp lý đã upload và tạo báo cáo kiểm tra."""

    def __init__(self, retriever: "Retriever", llm_client: Optional[Any] = None, model: str = ""):
        self.retriever = retriever
        self._client = llm_client
        self.model = model

    # ── Public ────────────────────────────────────────────────────────────────

    def analyze(self, document_text: str, filename: str = "") -> ValidationReport:
        """Pipeline đầy đủ: phân loại → trích dẫn → kiểm tra → báo cáo."""
        text = document_text.strip()
        if not text:
            return ValidationReport(
                doc_type="Không xác định",
                doc_type_key="unknown",
                confidence=0.0,
                issues=["Văn bản trống hoặc không đọc được nội dung."],
            )

        # 1. Nhận diện loại văn bản
        doc_type_key, confidence = self._classify(text)
        doc_type_meta = DOCUMENT_TYPES.get(doc_type_key, {})
        doc_type_name = doc_type_meta.get("name", "Văn bản pháp lý")

        # 2. Trích xuất trích dẫn
        raw_citations = self._extract_citations(text)

        # 3. Kiểm tra trích dẫn với corpus
        citation_checks = [self._check_citation(ref) for ref in raw_citations[:10]]

        # 4. Kiểm tra yếu tố bắt buộc
        required = doc_type_meta.get("required", [])
        present, missing = self._check_required_elements(text, required)

        # 5. Phát hiện vấn đề sơ bộ
        issues = self._detect_basic_issues(text, doc_type_key, citation_checks)

        # 6. LLM phân tích sâu
        llm_analysis = ""
        suggestions: list[str] = []
        if self._client and self.model:
            llm_analysis, suggestions = self._llm_deep_analysis(
                text, doc_type_name, missing, citation_checks,
            )

        return ValidationReport(
            doc_type=doc_type_name,
            doc_type_key=doc_type_key,
            confidence=confidence,
            citations=citation_checks,
            missing_elements=missing,
            present_elements=present,
            issues=issues,
            suggestions=suggestions,
            llm_analysis=llm_analysis,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _classify(self, text: str) -> tuple[str, float]:
        """Nhận diện loại văn bản dựa trên keyword scoring."""
        text_lower = text.lower()
        best_key = "unknown"
        best_score = 0.0

        for key, meta in DOCUMENT_TYPES.items():
            keywords = meta.get("keywords", [])
            hits = sum(1 for kw in keywords if kw.lower() in text_lower)
            score = hits / max(len(keywords), 1)
            if score > best_score:
                best_score = score
                best_key = key

        return best_key, min(best_score * 1.5, 1.0)  # scale lên 1.0

    def _extract_citations(self, text: str) -> list[str]:
        """Trích xuất tất cả trích dẫn pháp luật từ văn bản."""
        refs: set[str] = set()

        # Số hiệu văn bản (100/2019/NĐ-CP)
        for m in _LAW_NUMBER_RE.finditer(text):
            refs.add(m.group(0).strip())

        # "Điều X Nghị định/Luật..."
        for m in _ARTICLE_RE.finditer(text):
            # Lấy ngữ cảnh xung quanh để tìm tên luật
            start = max(0, m.start() - 5)
            end   = min(len(text), m.end() + 80)
            context = text[start:end].strip()
            # Tìm số hiệu trong context
            law_m = _LAW_NUMBER_RE.search(context)
            if law_m:
                refs.add(f"Điều {m.group(1)} {law_m.group(0)}")
            else:
                refs.add(f"Điều {m.group(1)}")

        # Bộ luật / Luật tên dài
        for m in _NAMED_LAW_RE.finditer(text):
            name = m.group(0).strip()
            if len(name) > 8:  # bỏ qua quá ngắn
                refs.add(name[:80])

        return sorted(refs)[:15]  # giới hạn 15 trích dẫn

    def _check_citation(self, ref: str) -> CitationCheck:
        """Kiểm tra 1 trích dẫn bằng cách tìm trong corpus."""
        try:
            chunks = self.retriever.retrieve(ref, top_k=3)
            if not chunks:
                return CitationCheck(
                    reference=ref, found_in_corpus=False,
                    note="Không tìm thấy trong corpus — cần xác minh thủ công",
                )
            top = chunks[0]
            snippet = top.chunk.text[:120].replace("\n", " ")
            return CitationCheck(
                reference=ref,
                found_in_corpus=True,
                relevant_text=snippet,
                note=f"Tìm thấy trong {top.chunk.metadata.source.split('/')[-1]}, score={top.score:.2f}",
            )
        except Exception as e:
            return CitationCheck(
                reference=ref, found_in_corpus=False, note=f"Lỗi tra cứu: {e}",
            )

    def _check_required_elements(self, text: str, required: list[str]) -> tuple[list[str], list[str]]:
        """Kiểm tra sơ bộ xem các yếu tố bắt buộc có trong văn bản không."""
        text_lower = text.lower()
        present: list[str] = []
        missing: list[str] = []

        # Mapping yếu tố → keywords để tìm
        ELEMENT_HINTS: dict[str, list[str]] = {
            "thông tin người vi phạm": ["họ tên", "ngày sinh", "địa chỉ", "cmnd", "cccd"],
            "mức phạt tiền": ["phạt tiền", "đồng", "vnđ", "phạt:", "mức phạt"],
            "căn cứ pháp lý": ["căn cứ", "điều", "khoản", "theo quy định"],
            "ngày, chữ ký": ["ký tên", "ký,", "chữ ký", "ngày   tháng   năm", "ngày...tháng"],
            "kính gửi": ["kính gửi:", "kính gửi -"],
            "quốc hiệu, tiêu ngữ": ["cộng hòa xã hội chủ nghĩa", "độc lập - tự do"],
            "quyền kháng cáo": ["kháng cáo", "khiếu nại", "khởi kiện"],
            "thời hạn thi hành": ["trong vòng", "trong thời hạn", "ngày kể từ"],
        }

        for el in required:
            el_lower = el.lower()
            # Tìm hints cho yếu tố này
            hints = next(
                (v for k, v in ELEMENT_HINTS.items() if k in el_lower),
                [el_lower[:20]],  # dùng prefix của tên yếu tố làm hint
            )
            if any(h in text_lower for h in hints):
                present.append(el)
            else:
                missing.append(el)

        return present, missing

    def _detect_basic_issues(
        self,
        text: str,
        doc_type_key: str,
        citation_checks: list[CitationCheck],
    ) -> list[str]:
        """Phát hiện các vấn đề cơ bản."""
        issues: list[str] = []

        # Kiểm tra trích dẫn không tìm được
        bad_cits = [c for c in citation_checks if not c.found_in_corpus]
        if bad_cits:
            issues.append(
                f"⚠️ {len(bad_cits)} trích dẫn pháp luật không tìm thấy trong corpus: "
                + ", ".join(f"`{c.reference}`" for c in bad_cits[:3])
            )

        # Quá ngắn
        if len(text) < 200:
            issues.append("⚠️ Văn bản quá ngắn — có thể thiếu nội dung hoặc chỉ trích đoạn.")

        # Thiếu quốc hiệu (với văn bản nhà nước)
        state_docs = {"ban_an_hinh_su", "ban_an_dan_su", "quyet_dinh_xu_phat", "cong_van"}
        if doc_type_key in state_docs:
            if "cộng hòa xã hội chủ nghĩa" not in text.lower():
                issues.append("❌ Thiếu quốc hiệu 'Cộng hòa Xã hội Chủ nghĩa Việt Nam'.")

        # Thiếu ngày tháng rõ ràng
        if not re.search(r"\bngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}\b", text, re.I):
            if not re.search(r"\d{1,2}/\d{1,2}/\d{4}", text):
                issues.append("⚠️ Không tìm thấy ngày tháng năm cụ thể trong văn bản.")

        # Quyết định xử phạt: kiểm tra có số hiệu
        if doc_type_key == "quyet_dinh_xu_phat":
            if not re.search(r"số[:.]?\s*\d+", text, re.I):
                issues.append("⚠️ Thiếu số hiệu quyết định.")

        return issues

    def _llm_deep_analysis(
        self,
        text: str,
        doc_type_name: str,
        missing_elements: list[str],
        citation_checks: list[CitationCheck],
    ) -> tuple[str, list[str]]:
        """Phân tích sâu bằng LLM — phát hiện lỗi ngữ nghĩa, mâu thuẫn nội dung."""
        # Tóm tắt thông tin đã biết để đưa vào prompt
        missing_str = "\n".join(f"- {el}" for el in missing_elements) if missing_elements else "(không có)"
        bad_cits_str = "\n".join(
            f"- {c.reference}" for c in citation_checks if not c.found_in_corpus
        ) or "(không có)"

        # Cắt văn bản để tránh quá dài
        doc_snippet = text[:3000] + ("..." if len(text) > 3000 else "")

        prompt = f"""Bạn là chuyên gia pháp lý Việt Nam. Hãy phân tích văn bản pháp lý sau:

Loại văn bản: {doc_type_name}
Các yếu tố bắt buộc còn thiếu: {missing_str}
Trích dẫn pháp luật không tìm thấy: {bad_cits_str}

Nội dung văn bản:
---
{doc_snippet}
---

Hãy thực hiện:
1. Xác nhận loại văn bản và tính đúng đắn về thể thức
2. Chỉ ra CỤ THỂ những sai sót về nội dung (nếu có):
   - Sai về mức phạt / hình phạt (nếu đề cập)
   - Viện dẫn điều luật sai / không chính xác
   - Thiếu căn cứ pháp lý bắt buộc
   - Mâu thuẫn nội tại trong văn bản
3. Đưa ra 3-5 đề xuất cải thiện cụ thể (nếu có vấn đề)

Trả lời ngắn gọn, rõ ràng, dùng danh sách. Nếu văn bản đúng thì nêu rõ.
Format đề xuất theo dạng: ĐỀXUẤT: [nội dung]"""

        try:
            resp = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_ctx": 4096},
            )
            analysis = resp["message"]["content"].strip()

            # Tách phần đề xuất
            suggestions: list[str] = []
            lines = analysis.split("\n")
            clean_lines: list[str] = []
            for line in lines:
                if line.strip().startswith("ĐỀXUẤT:"):
                    suggestions.append(line.replace("ĐỀXUẤT:", "").strip())
                else:
                    clean_lines.append(line)

            return "\n".join(clean_lines).strip(), suggestions

        except Exception as e:
            return f"(Lỗi phân tích LLM: {e})", []

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .schemas import DocumentMetadata, RawDocument


def _render_table_md(rows: list[list]) -> str:
    """Render 1 bảng (list-of-rows) thành markdown table.

    Quy ước: dòng đầu là header. Cell có '\n' thay bằng space; '|' escape thành '\\|'.
    Pad ngắn cho đủ cột.
    """
    if not rows:
        return ""
    max_cols = max(len(r) for r in rows)
    padded = [list(r) + [""] * (max_cols - len(r)) for r in rows]

    def fmt(cell) -> str:
        s = "" if cell is None else str(cell)
        return s.replace("\n", " ").replace("|", "\\|").strip()

    header = "| " + " | ".join(fmt(c) for c in padded[0]) + " |"
    sep = "| " + " | ".join("---" for _ in padded[0]) + " |"
    body = ["| " + " | ".join(fmt(c) for c in row) + " |" for row in padded[1:]]
    return "\n".join([header, sep, *body])


def parse_pdf(path: Path) -> str:
    """Trích text từ file PDF, GIỮ NGUYÊN bảng dưới dạng markdown.

    Hai bước cho mỗi page:
      1. find_tables() → bbox; filter chars NGOÀI bbox → extract_text() (text thường).
      2. extract_tables() → render markdown → append cuối page.
    Nếu page không có bảng → fallback extract_text() như trước.
    """
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            try:
                tables = page.find_tables()
            except Exception:
                tables = []

            if not tables:
                parts.append(page.extract_text() or "")
                continue

            bboxes = [t.bbox for t in tables]

            def _not_in_table(obj, _bboxes=bboxes) -> bool:
                try:
                    cx0, ctop, cx1, cbottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
                except KeyError:
                    return True
                for x0, top, x1, bottom in _bboxes:
                    if cx0 >= x0 and ctop >= top and cx1 <= x1 and cbottom <= bottom:
                        return False
                return True

            try:
                non_table_text = page.filter(_not_in_table).extract_text() or ""
            except Exception:
                non_table_text = page.extract_text() or ""

            md_blocks: list[str] = []
            for t in tables:
                try:
                    rows = t.extract()
                except Exception:
                    rows = None
                if rows:
                    md = _render_table_md(rows)
                    if md:
                        md_blocks.append(md)

            page_text = non_table_text.rstrip()
            if md_blocks:
                page_text = page_text + ("\n\n" if page_text else "") + "\n\n".join(md_blocks)
            parts.append(page_text)
    return "\n".join(parts)


def parse_docx(path: Path) -> str:
    """Trích text + bảng từ file .docx, giữ thứ tự document.

    Dùng python-docx ở mức XML để duyệt body theo thứ tự (paragraph hoặc table).
    Bảng được render dưới dạng markdown như parse_pdf.
    """
    from docx import Document as DocxDocument
    from docx.oxml.ns import qn

    doc = DocxDocument(str(path))
    body = doc.element.body

    P_TAG = qn("w:p")
    TBL_TAG = qn("w:tbl")
    TR_TAG = qn("w:tr")
    TC_TAG = qn("w:tc")
    T_TAG = qn("w:t")

    parts: list[str] = []
    for elem in body.iterchildren():
        if elem.tag == P_TAG:
            text = "".join((t.text or "") for t in elem.iter(T_TAG)).strip()
            if text:
                parts.append(text)
        elif elem.tag == TBL_TAG:
            rows: list[list[str]] = []
            for row in elem.iter(TR_TAG):
                cells: list[str] = []
                for cell in row.iter(TC_TAG):
                    cell_text = " ".join((t.text or "") for t in cell.iter(T_TAG)).strip()
                    cells.append(cell_text)
                if cells:
                    rows.append(cells)
            if rows:
                md = _render_table_md(rows)
                if md:
                    parts.append(md)
    return "\n\n".join(parts)


def parse_html(path: Path) -> str:
    """Trích text từ file HTML bằng BeautifulSoup."""
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def parse_txt(path: Path) -> str:
    """Đọc file .txt (UTF-8, fallback bỏ qua ký tự lỗi)."""
    return path.read_text(encoding="utf-8", errors="ignore")


_PAGE_NUMBER_RE = re.compile(r"^\s*(?:trang\s*)?\d{1,4}\s*/?\s*\d{0,4}\s*$", re.IGNORECASE)
_FORM_FEED_RE = re.compile(r"\f+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAIL_WS_RE = re.compile(r"[ \t]+(?=\n)")
_INLINE_WS_RE = re.compile(r"[ \t]{2,}")
_SEPARATOR_LINE_RE = re.compile(r"^[=\-_*]{3,}\s*$")
_APPENDIX_MARKER_RE = re.compile(
    r"(?im)^\s*(?:"
    r"MỘT SỐ ĐIỂM MỚI"
    r"|ĐIỂM (?:MỚI|NỔI BẬT)"
    r"|TÓM TẮT"
    r"|GHI CHÚ"
    r"|LƯU Ý"
    r"|KẾT LUẬN"
    r"|TỔNG HỢP"
    r"|TÀI LIỆU THAM KHẢO"
    r")\b"
)


def _strip_appendix(text: str) -> str:
    """Cắt bỏ phần phụ lục/ghi chú/tổng hợp xuất hiện sau nội dung chính.

    Chỉ cắt khi marker xuất hiện *sau* một Điều — tránh cắt nhầm nội dung
    có cùng từ khoá nằm bên trong văn bản chính.
    """
    last_article = list(re.finditer(r"(?m)^Điều\s+\d+", text))
    if not last_article:
        return text
    cutoff_floor = last_article[-1].start()
    m = _APPENDIX_MARKER_RE.search(text, pos=cutoff_floor)
    if m:
        return text[: m.start()].rstrip()
    return text


def clean_text(text: str) -> str:
    """Chuẩn hoá Unicode (NFC), bỏ header/footer/số trang, separator, phụ lục, gộp khoảng trắng."""
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = _FORM_FEED_RE.sub("\n", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if _PAGE_NUMBER_RE.match(stripped):
            continue
        if _SEPARATOR_LINE_RE.match(stripped):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = _TRAIL_WS_RE.sub("", text)
    text = _INLINE_WS_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    text = _strip_appendix(text)
    return text.strip()


_PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".html": parse_html,
    ".htm": parse_html,
    ".txt": parse_txt,
}


def load_document(path: Path, metadata: DocumentMetadata) -> RawDocument:
    """Đọc + làm sạch một văn bản gốc, trả về RawDocument."""
    suffix = path.suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"Unsupported file type: {suffix} ({path.name})")

    raw = parser(path)
    cleaned = clean_text(raw)
    return RawDocument(text=cleaned, metadata=metadata)

"""Phase 0: Structural KG extraction — pure regex, KHÔNG dùng LLM.

Reuse src.chunking._iter_articles và _iter_clauses để parse Điều/Khoản.
Output:
  - law_node:        dict — node :Law
  - article_nodes:   list[dict] — nodes :Article
  - clause_nodes:    list[dict] — nodes :Clause
  - internal_cites:  list[dict] — Article -[REFERS_TO]-> Article (cùng luật)
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..chunking import _iter_articles, _iter_clauses
from ..schemas import RawDocument


# Text vbpl.vn thường flatten body thành 1 dòng dài → regex cần "Điều X" ở đầu
# dòng (multiline) sẽ miss. Pre-process: insert newline trước mỗi marker.
#
# CHÚ Ý: chỉ tách Điều khi pattern là "Điều X. <chữ-hoa>" (header chính thức),
# bỏ qua "tại Điều X BLHS" / "theo Điều X" (tham chiếu, không phải header mới).
_ARTICLE_INLINE_RE = re.compile(
    r"(?<!\n)(?<![a-zàáâãèéêìíòóôõùúýđ])"  # không follow ngay sau chữ thường tiếng Việt
    r"(Điều\s+\d+\.\s+)(?=[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝ])"  # cần "Điều X." + uppercase ngay sau
)
_CLAUSE_INLINE_RE = re.compile(r"(?<![\n\d])\s(\d{1,2})\.\s+(?=[A-ZĐ])")


def _normalize_for_kg(text: str) -> str:
    """Khôi phục cấu trúc đa dòng cho text bị flatten.

    1. Thêm \\n\\n trước header "Điều X. <Tiêu đề>"
    2. Thêm \\n trước mỗi "<số>. " (clause marker) — chỉ khi theo sau là chữ hoa
    """
    text = _ARTICLE_INLINE_RE.sub(r"\n\n\1", text)
    text = _CLAUSE_INLINE_RE.sub(r"\n\1. ", text)
    return text


# Bắt mọi tham chiếu "Điều X" trong text. Group 1 = số điều.
_REFER_RE = re.compile(r"Điều\s+(\d+)")

# Header trong file txt vbpl.vn có dạng:
#   HIEU_LUC: Còn hiệu lực
#   HIEU_LUC: Đã được sửa đổi bởi ... (số_hiệu)
#   HIEU_LUC: Hết hiệu lực, được thay thế bởi ... (số_hiệu)
# Bắt số_hiệu trong các trường hợp này để tạo edge AMENDS.
_AMEND_RE = re.compile(
    r"(?:sửa đổi|bổ sung|thay thế|huỷ bỏ|hủy bỏ)(?:\s+bởi|\s+bằng)?[^\n]*?"
    r"(\d+[A-Z]?[\-/\d]*?/[A-Z\d\-/]+)",
    re.IGNORECASE,
)

CLAUSE_TEXT_PREVIEW = 300  # ký tự đầu của Clause để lưu vào KG (full text nằm ở vectorstore)


def _law_id_from_source(source: str) -> str:
    """Sinh id ổn định cho Law từ source URL/path. Ưu tiên URL vbpl.vn (id cuối)."""
    s = source.strip().rstrip("/")
    # https://vbpl.vn/van-ban/chi-tiet/12345 → "vbpl_12345"
    m = re.search(r"vbpl\.vn/[^/]+/[^/]+/(\d+)", s)
    if m:
        return f"vbpl_{m.group(1)}"
    # fallback: dùng phần cuối path
    return s.replace("\\", "/").split("/")[-1].replace(".txt", "").replace(".", "_")


def _article_id(law_id: str, art_num: str) -> str:
    return f"{law_id}::dieu_{art_num}"


def _clause_id(article_id: str, clause_num: str) -> str:
    return f"{article_id}::khoan_{clause_num}"


def extract_structural(doc: RawDocument) -> dict[str, Any]:
    """Extract toàn bộ nodes + edges nội bộ từ 1 RawDocument."""
    meta = doc.metadata
    law_id = _law_id_from_source(meta.source)

    # ── Law node ─────────────────────────────────────────────────────────────
    law_props: dict[str, Any] = {"source": meta.source}
    for fld in ("doc_type", "doc_number", "title", "issued_date", "effective_date", "status"):
        v = getattr(meta, fld, None)
        if v:
            law_props[fld] = v

    law_node = {"id": law_id, "props": law_props}

    # ── Article + Clause + internal citations ───────────────────────────────
    article_nodes: list[dict[str, Any]] = []
    clause_nodes: list[dict[str, Any]] = []
    internal_cites: list[dict[str, str]] = []

    seen_articles: set[str] = set()  # số điều đã thấy trong luật này

    normalized = _normalize_for_kg(doc.text)
    for _chap, art_label, art_text in _iter_articles(normalized):
        if art_label is None:
            continue  # preamble — bỏ
        if art_label in seen_articles:
            continue  # duplicate (lỗi parse) — skip
        seen_articles.add(art_label)

        art_id = _article_id(law_id, art_label)
        # Lấy 1 dòng đầu sau "Điều X" làm title
        first_line = art_text.split("\n", 1)
        head = first_line[0] if first_line else ""
        title = re.sub(r"^Điều\s+\d+\.?\s*", "", head).strip()[:200] or None

        art_props: dict[str, Any] = {"number": int(art_label)}
        if title:
            art_props["title"] = title

        article_nodes.append({
            "id": art_id,
            "law_id": law_id,
            "props": art_props,
        })

        # Tách Khoản
        for clause_label, clause_text in _iter_clauses(art_text):
            if clause_label is None:
                continue
            cls_id = _clause_id(art_id, clause_label)
            clause_nodes.append({
                "id": cls_id,
                "article_id": art_id,
                "props": {
                    "number": int(clause_label),
                    "text_preview": clause_text[:CLAUSE_TEXT_PREVIEW],
                },
            })

        # Internal citations: tìm "Điều Y" khác trong text của Điều này
        for m in _REFER_RE.finditer(art_text):
            target_num = m.group(1)
            if target_num == art_label:
                continue  # self-reference, bỏ
            target_id = _article_id(law_id, target_num)
            internal_cites.append({"src": art_id, "dst": target_id})

    # ── Amendment edges (Law level) ─────────────────────────────────────────
    amend_targets: list[str] = []
    status_text = (meta.status or "").lower()
    if status_text and any(kw in status_text for kw in ("sửa đổi", "bổ sung", "thay thế", "huỷ", "hủy")):
        for m in _AMEND_RE.finditer(status_text):
            amend_targets.append(m.group(1).strip())

    return {
        "law_node": law_node,
        "article_nodes": article_nodes,
        "clause_nodes": clause_nodes,
        "internal_cites": internal_cites,
        "amend_targets_raw": amend_targets,  # số_hiệu thô, resolve sang law_id sau
    }


def dedup_citations(cites: list[dict[str, str]], valid_article_ids: set[str]) -> list[dict[str, str]]:
    """Lọc citations: (1) target phải tồn tại, (2) dedupe."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for c in cites:
        key = (c["src"], c["dst"])
        if key in seen:
            continue
        if c["dst"] not in valid_article_ids:
            continue
        seen.add(key)
        out.append(c)
    return out


def resolve_amendments(
    amend_pairs: list[tuple[str, str]],  # (src_law_id, target_doc_number)
    docnum_to_law_id: dict[str, str],
) -> list[dict[str, str]]:
    """Map số_hiệu thô → law_id; trả về edges cho add_amendment_edges()."""
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for src_law_id, doc_num in amend_pairs:
        target_id = docnum_to_law_id.get(doc_num)
        if not target_id or target_id == src_law_id:
            continue
        key = (src_law_id, target_id)
        if key in seen:
            continue
        seen.add(key)
        out.append({"src": src_law_id, "dst": target_id})
    return out

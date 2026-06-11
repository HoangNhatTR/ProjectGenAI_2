from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterator, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .schemas import Chunk, DocumentMetadata, RawDocument

# ── Regex patterns ─────────────────────────────────────────────────────────────

_BOUNDARY_RE = re.compile(
    r"(?m)^(?:"
    r"Điều\s+(?P<art>\d+)\.?|"
    r"CHƯƠNG\s+(?P<chap>[IVXLCDM]+|\d+)\b"
    r")"
)
_CLAUSE_RE = re.compile(r"(?m)^(\d+)\.\s")

# Điểm a), b), c) ... đ) tại đầu dòng
_POINT_RE = re.compile(r"(?m)^([a-zđ])\)\s")

_HAS_ARTICLE_RE = re.compile(r"(?m)^Điều\s+\d+")

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", " ", ""]

# Số ký tự tối đa cho 1 parent entry.
# Điều dài hơn ngưỡng này KHÔNG bị cắt cụt nữa (bug cũ với MAX=2000 làm mất
# ~95% nội dung các điều xử phạt lớn) — thay vào đó parent chuyển xuống cấp
# Khoản (kèm dòng tiêu đề Điều để giữ ngữ cảnh).
MAX_PARENT_CHARS = 6000

# Độ dài tối đa của tiêu đề Điều khi đưa vào contextual header
_ART_TITLE_MAX = 90


def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_DEFAULT_SEPARATORS,
        length_function=len,
    )


# ── Contextual header helpers ──────────────────────────────────────────────────

def _doc_label(meta: DocumentMetadata) -> str:
    """Nhãn ngắn định danh văn bản: ưu tiên số hiệu > tên > stem."""
    if meta.doc_number:
        return meta.doc_number
    if meta.title and not meta.title.isdigit():
        return meta.title[:60]
    return Path(meta.source).stem


def _article_title(article_text: str, max_len: int = _ART_TITLE_MAX) -> str:
    """Dòng đầu của Điều: 'Điều 8. Xử phạt người điều khiển xe...' (cắt gọn)."""
    first = article_text.strip().splitlines()[0].strip()
    return first[:max_len]


def _with_header(text: str, *parts: Optional[str]) -> str:
    """Prepend contextual header '[A — B — C]' vào text trước khi embed.

    Header giúp embedding/BM25 phân biệt được cùng một câu chữ vi phạm
    xuất hiện trong nhiều văn bản khác nhau, và giúp chunk Điểm a/b/c
    mang theo ngữ cảnh Điều/Khoản của nó.
    """
    header = " — ".join(p for p in parts if p)
    return f"[{header}]\n{text}" if header else text


# ── Structural iterators ───────────────────────────────────────────────────────

def _iter_articles(text: str) -> Iterator[tuple[Optional[str], Optional[str], str]]:
    """Yield (chapter_label, article_label, content) cho từng Điều.

    - Preamble trước Điều đầu tiên: yield (None, None, preamble_text)
    - Header CHƯƠNG: chỉ cập nhật chapter context, không emit content riêng.
    """
    matches = list(_BOUNDARY_RE.finditer(text))
    if not matches:
        cleaned = text.strip()
        if cleaned:
            yield (None, None, cleaned)
        return

    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            yield (None, None, pre)

    current_chapter: Optional[str] = None
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if m.group("chap") is not None:
            current_chapter = m.group("chap")
            continue
        article_label = m.group("art")
        content = text[m.start():end].strip()
        if content:
            yield (current_chapter, article_label, content)


def _iter_clauses(article_text: str) -> Iterator[tuple[Optional[str], str]]:
    """Tách 1 Điều → các Khoản. Phần trước Khoản 1 là tiêu đề (label=None)."""
    matches = list(_CLAUSE_RE.finditer(article_text))
    if not matches:
        yield (None, article_text)
        return

    if matches[0].start() > 0:
        head = article_text[: matches[0].start()].strip()
        if head:
            yield (None, head)

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(article_text)
        label = m.group(1)
        body = article_text[m.start():end].strip()
        if body:
            yield (label, body)


def _iter_points(clause_text: str) -> Iterator[tuple[Optional[str], str]]:
    """Tách 1 Khoản → các Điểm a), b), c)... Yield (point_label, text).

    Nếu không có Điểm, yield (None, clause_text) toàn bộ.
    """
    matches = list(_POINT_RE.finditer(clause_text))
    if not matches:
        yield (None, clause_text)
        return

    if matches[0].start() > 0:
        head = clause_text[: matches[0].start()].strip()
        if head:
            yield (None, head)

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(clause_text)
        body = clause_text[m.start():end].strip()
        if body:
            yield (m.group(1), body)


# ── Chunking strategies ────────────────────────────────────────────────────────

def _chunk_prefix(doc: RawDocument) -> str:
    """Prefix duy nhất cho chunk_id: stem + hash ngắn của source.

    Hash chống trùng id khi 2 văn bản khác nhau có cùng stem
    (vd cùng tên file trong 2 thư mục loại VB khác nhau).
    """
    stem = Path(doc.metadata.source).stem
    uid = hashlib.md5(doc.metadata.source.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{uid}"


def chunk_recursive(doc: RawDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Fallback RecursiveCharacterTextSplitter cho văn bản không có Điều/Khoản."""
    splitter = _make_splitter(chunk_size, chunk_overlap)
    prefix = _chunk_prefix(doc)
    label = _doc_label(doc.metadata)
    parts = splitter.split_text(doc.text)
    return [
        Chunk(
            chunk_id=f"{prefix}_{i:04d}",
            text=_with_header(part, label),
            metadata=doc.metadata,
        )
        for i, part in enumerate(parts)
    ]


def chunk_by_legal_structure(
    doc: RawDocument,
    chunk_size: int,
    chunk_overlap: int,
    parent_store: Optional[Any] = None,
) -> list[Chunk]:
    """Chunk theo Điều → Khoản → Điểm với parent-child + contextual header.

    Parent-child (khi parent_store được cung cấp):
      - Điều ≤ MAX_PARENT_CHARS → 1 parent entry = full text Điều
      - Điều dài hơn → parent theo TỪNG KHOẢN (kèm dòng tiêu đề Điều) —
        không còn cắt cụt mất nội dung như trước
      - Child chunks mang parent_id → Retriever expand khi generate

    Contextual header: mỗi child chunk được prepend
      '[<số hiệu VB> — <Điều X. tiêu đề> — Khoản Y]'
    để embedding/BM25 giữ được ngữ cảnh dù text gốc chỉ là 'a) ...'.
    """
    splitter = _make_splitter(chunk_size, chunk_overlap)
    prefix = _chunk_prefix(doc)
    doc_label = _doc_label(doc.metadata)
    chunks: list[Chunk] = []
    parent_batch: list[tuple[str, str]] = []

    def add(
        text: str,
        article: Optional[str] = None,
        clause: Optional[str] = None,
        point: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> None:
        text = text.strip()
        if not text:
            return
        chunks.append(
            Chunk(
                chunk_id=f"{prefix}_{len(chunks):04d}",
                text=text,
                article=article,
                clause=clause,
                point=point,
                metadata=doc.metadata,
                parent_id=parent_id,
            )
        )

    for _chapter, article_label, article_text in _iter_articles(doc.text):
        # ── Preamble / văn bản không thuộc Điều nào ──────────────────────────
        if article_label is None:
            for sub in splitter.split_text(article_text):
                add(_with_header(sub, doc_label))
            continue

        article_str = f"Điều {article_label}"
        art_title = _article_title(article_text)
        # Dòng tiêu đề đầy đủ của Điều — dùng cho parent text cấp Khoản
        art_first_line = article_text.strip().splitlines()[0].strip()

        # ── Parent cấp Điều (chỉ khi Điều đủ ngắn để không bị cắt) ───────────
        article_parent_id: Optional[str] = None
        use_clause_parents = False
        if parent_store is not None:
            if len(article_text) <= MAX_PARENT_CHARS:
                article_parent_id = f"{prefix}_p_{article_label}"
                parent_batch.append((article_parent_id, article_text))
            else:
                use_clause_parents = True  # Điều quá dài → parent theo Khoản

        # ── Điều ngắn → 1 child chunk ────────────────────────────────────────
        # (text đã bắt đầu bằng 'Điều X. ...' nên header chỉ cần số hiệu VB)
        if len(article_text) <= chunk_size:
            add(
                _with_header(article_text, doc_label),
                article=article_str,
                parent_id=article_parent_id,
            )
            continue

        # ── Điều dài → Khoản → Điểm ─────────────────────────────────────────
        for clause_label, clause_text in _iter_clauses(article_text):
            clause_str = f"Khoản {clause_label}" if clause_label else None

            # Parent cấp Khoản cho Điều quá dài — kèm tiêu đề Điều giữ ngữ cảnh
            parent_id = article_parent_id
            if use_clause_parents:
                k = clause_label or "head"
                parent_id = f"{prefix}_p_{article_label}_k_{k}"
                if clause_label:
                    parent_text = f"{art_first_line}\n{clause_text}"
                else:
                    parent_text = clause_text  # head đã chứa tiêu đề Điều
                parent_batch.append((parent_id, parent_text[:MAX_PARENT_CHARS]))

            # Head của Điều (trước Khoản 1) đã chứa 'Điều X. ...' → header gọn
            head_parts = (doc_label,) if clause_label is None else (doc_label, art_title)

            for point_label, point_text in _iter_points(clause_text):
                point_str = f"Điểm {point_label}" if point_label else None
                # Chunk Điểm cần đủ ngữ cảnh: VB + Điều + Khoản
                parts = head_parts + ((clause_str,) if point_label else ())

                if len(point_text) <= chunk_size:
                    add(
                        _with_header(point_text, *parts),
                        article=article_str,
                        clause=clause_str,
                        point=point_str,
                        parent_id=parent_id,
                    )
                else:
                    for sub in splitter.split_text(point_text):
                        add(
                            _with_header(sub, *parts),
                            article=article_str,
                            clause=clause_str,
                            point=point_str,
                            parent_id=parent_id,
                        )

    # Ghi tất cả parent chunks 1 lần duy nhất (1 transaction / document)
    if parent_store is not None and parent_batch:
        parent_store.add_batch(parent_batch)

    return chunks


def chunk_document(
    doc: RawDocument,
    parent_store: Optional[Any] = None,
) -> list[Chunk]:
    """Chọn chiến lược chunk dựa trên cấu trúc văn bản."""
    if _HAS_ARTICLE_RE.search(doc.text):
        return chunk_by_legal_structure(
            doc, config.CHUNK_SIZE, config.CHUNK_OVERLAP,
            parent_store=parent_store,
        )
    return chunk_recursive(doc, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

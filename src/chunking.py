from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .schemas import Chunk, RawDocument

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

# Số ký tự tối đa lưu cho 1 parent chunk (1 Điều)
MAX_PARENT_CHARS = 2000


def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_DEFAULT_SEPARATORS,
        length_function=len,
    )


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

def chunk_recursive(doc: RawDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Fallback RecursiveCharacterTextSplitter cho văn bản không có Điều/Khoản."""
    splitter = _make_splitter(chunk_size, chunk_overlap)
    parts = splitter.split_text(doc.text)
    src_stem = Path(doc.metadata.source).stem
    return [
        Chunk(
            chunk_id=f"{src_stem}_{i:04d}",
            text=part,
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
    """Chunk theo Điều → Khoản → Điểm với optional parent-child.

    Khi parent_store được cung cấp:
      - Mỗi Điều → 1 parent entry trong SQLite (text đầy đủ, không embed)
      - Child chunks có parent_id → Retriever expand về parent text khi generate
    Khi parent_store là None: hành vi tương tự cũ, chỉ thêm Điểm level.
    """
    splitter = _make_splitter(chunk_size, chunk_overlap)
    src_stem = Path(doc.metadata.source).stem
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
                chunk_id=f"{src_stem}_{len(chunks):04d}",
                text=text,
                article=article,
                clause=clause,
                point=point,
                metadata=doc.metadata,
                parent_id=parent_id,
            )
        )

    for _chapter, article_label, article_text in _iter_articles(doc.text):
        if article_label is None:
            for sub in splitter.split_text(article_text):
                add(sub)
            continue

        article_str = f"Điều {article_label}"

        # ── Parent chunk: lưu full text của Điều vào store ───────────────────
        parent_id: Optional[str] = None
        if parent_store is not None:
            parent_id = f"{src_stem}_p_{article_label}"
            parent_batch.append((parent_id, article_text[:MAX_PARENT_CHARS]))

        # ── Điều ngắn → 1 child chunk ────────────────────────────────────────
        if len(article_text) <= chunk_size:
            add(article_text, article=article_str, parent_id=parent_id)
            continue

        # ── Điều dài → Khoản → Điểm ─────────────────────────────────────────
        for clause_label, clause_text in _iter_clauses(article_text):
            clause_str = f"Khoản {clause_label}" if clause_label else None

            for point_label, point_text in _iter_points(clause_text):
                point_str = f"Điểm {point_label}" if point_label else None

                if len(point_text) <= chunk_size:
                    add(
                        point_text,
                        article=article_str,
                        clause=clause_str,
                        point=point_str,
                        parent_id=parent_id,
                    )
                else:
                    for sub in splitter.split_text(point_text):
                        add(
                            sub,
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

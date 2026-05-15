from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .schemas import Chunk, RawDocument


_BOUNDARY_RE = re.compile(
    r"(?m)^(?:"
    r"Điều\s+(?P<art>\d+)\.?|"
    r"CHƯƠNG\s+(?P<chap>[IVXLCDM]+|\d+)\b"
    r")"
)
_CLAUSE_RE = re.compile(r"(?m)^(\d+)\.\s")
_HAS_ARTICLE_RE = re.compile(r"(?m)^Điều\s+\d+")

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", " ", ""]


def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_DEFAULT_SEPARATORS,
        length_function=len,
    )


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
    """Tách 1 Điều thành các Khoản. Phần trước Khoản 1 là tiêu đề Điều (label=None)."""
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


def chunk_recursive(doc: RawDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Fallback dùng RecursiveCharacterTextSplitter cho văn bản không có cấu trúc Điều/Khoản."""
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
) -> list[Chunk]:
    """Chunk theo Điều / Khoản — fallback recursive nếu segment quá dài.

    Ý tưởng: regex bắt mẫu "Điều \\d+", "\\d+\\." (khoản); ưu tiên giữ nguyên 1 Điều
    trong 1 chunk, chỉ split khi vượt chunk_size; điền (article, clause) vào metadata.
    """
    splitter = _make_splitter(chunk_size, chunk_overlap)
    src_stem = Path(doc.metadata.source).stem
    chunks: list[Chunk] = []

    def add(text: str, article: Optional[str] = None, clause: Optional[str] = None) -> None:
        text = text.strip()
        if not text:
            return
        chunks.append(
            Chunk(
                chunk_id=f"{src_stem}_{len(chunks):04d}",
                text=text,
                article=article,
                clause=clause,
                metadata=doc.metadata,
            )
        )

    for _chapter, article_label, article_text in _iter_articles(doc.text):
        if article_label is None:
            for sub in splitter.split_text(article_text):
                add(sub)
            continue

        article_str = f"Điều {article_label}"

        if len(article_text) <= chunk_size:
            add(article_text, article=article_str)
            continue

        for clause_label, clause_text in _iter_clauses(article_text):
            clause_str = f"Khoản {clause_label}" if clause_label else None
            if len(clause_text) <= chunk_size:
                add(clause_text, article=article_str, clause=clause_str)
            else:
                for sub in splitter.split_text(clause_text):
                    add(sub, article=article_str, clause=clause_str)

    return chunks


def chunk_document(doc: RawDocument) -> list[Chunk]:
    """Chọn chiến lược chunk dựa trên cấu trúc văn bản."""
    if _HAS_ARTICLE_RE.search(doc.text):
        return chunk_by_legal_structure(doc, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    return chunk_recursive(doc, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

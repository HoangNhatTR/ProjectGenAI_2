"""Document Chat — parse, chunk, embed và retrieve tài liệu người dùng upload.

Mỗi session có 1 UploadedDocStore riêng lưu in-memory.
Tích hợp vào pipeline qua _to_retrieved_chunks() → RetrievedChunk đúng format.
"""
from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .schemas import Chunk, DocumentMetadata, RetrievedChunk

if TYPE_CHECKING:
    from .embedding import Embedder

# Prefix đặc biệt để phân biệt citation từ tài liệu upload
DOC_SOURCE_PREFIX = "[📎 TÀI LIỆU]"


def parse_file(filename: str, file_bytes: bytes) -> str:
    """Extract plain text từ PDF / DOCX / TXT / MD."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n\n".join(p for p in pages if p.strip())
        except ImportError:
            raise ImportError("Cài pypdf để đọc PDF: pip install pypdf")

    elif ext in (".docx",):
        try:
            from docx import Document
            doc = Document(io.BytesIO(file_bytes))
            paras = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paras)
        except ImportError:
            raise ImportError("Cài python-docx để đọc DOCX: pip install python-docx")

    elif ext in (".txt", ".md", ".markdown"):
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return file_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        return file_bytes.decode("utf-8", errors="replace")

    else:
        raise ValueError(f"Định dạng '{ext}' chưa hỗ trợ (PDF / DOCX / TXT / MD)")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Chia text thành các đoạn chunk_size ký tự, overlap ký tự chồng lặp.

    Cố gắng cắt tại dòng mới gần nhất để giữ tính hoàn chỉnh của đoạn văn.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        segment = text[start:end]

        # Cắt tại newline gần nhất nếu không phải đoạn cuối
        if end < len(text):
            nl = segment.rfind("\n")
            if nl > chunk_size // 3:
                end = start + nl
                segment = text[start:end]

        stripped = segment.strip()
        if len(stripped) > 40:
            chunks.append(stripped)

        start = end - overlap if end < len(text) else len(text)

    return chunks


class UploadedDocStore:
    """Kho tài liệu upload in-memory của 1 session.

    Hỗ trợ cosine similarity search và convert sang RetrievedChunk
    để tương thích với pipeline hiện có.
    """

    def __init__(self, embedder: "Embedder") -> None:
        self._embedder = embedder
        self._chunks: list[dict] = []   # {"chunk_id", "filename", "text", "emb"}
        self._files: dict[str, int] = {}  # filename → số chunk

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_file(self, filename: str, file_bytes: bytes) -> int:
        """Parse, chunk, embed và lưu file. Trả về số chunk được thêm."""
        text = parse_file(filename, file_bytes)
        if not text.strip():
            raise ValueError("Không extract được nội dung từ file")

        raw_chunks = chunk_text(text)
        if not raw_chunks:
            raise ValueError("File không có nội dung đủ dài")

        embeddings = self._embedder.encode(raw_chunks)

        added = 0
        for i, (t, emb) in enumerate(zip(raw_chunks, embeddings)):
            self._chunks.append({
                "chunk_id": f"doc_{uuid.uuid4().hex[:8]}_{i}",
                "filename": filename,
                "text": t,
                "emb": np.array(emb, dtype=np.float32),
            })
            added += 1

        self._files[filename] = added
        return added

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Cosine similarity search → trả về RetrievedChunk tương thích pipeline."""
        if not self._chunks:
            return []

        q_emb = np.array(self._embedder.encode([query])[0], dtype=np.float32)
        q_emb /= np.linalg.norm(q_emb) + 1e-9

        scored: list[tuple[float, dict]] = []
        for c in self._chunks:
            emb = c["emb"]
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            score = float(np.dot(q_emb, emb))
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[RetrievedChunk] = []
        for score, c in scored[:top_k]:
            if score < 0.15:
                continue
            source_label = f"{DOC_SOURCE_PREFIX} {c['filename']}"
            chunk = Chunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                article=None,
                clause=None,
                point=None,
                metadata=DocumentMetadata(
                    source=source_label,
                    doc_type="upload",
                    title=c["filename"],
                ),
            )
            results.append(RetrievedChunk(chunk=chunk, score=score))

        return results

    def list_files(self) -> list[dict]:
        return [{"name": k, "chunks": v} for k, v in self._files.items()]

    def remove_file(self, filename: str) -> None:
        self._chunks = [c for c in self._chunks if c["filename"] != filename]
        self._files.pop(filename, None)

    def clear(self) -> None:
        self._chunks.clear()
        self._files.clear()

    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    def total_chunks(self) -> int:
        return len(self._chunks)

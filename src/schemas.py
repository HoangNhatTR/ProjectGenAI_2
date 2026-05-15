from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata gắn với mỗi văn bản pháp luật gốc."""

    source: str
    doc_type: Optional[str] = None  # luật / nghị định / thông tư / bản án ...
    doc_number: Optional[str] = None  # ví dụ "45/2019/QH14"
    title: Optional[str] = None
    issued_date: Optional[str] = None  # ISO yyyy-mm-dd
    effective_date: Optional[str] = None
    status: Optional[str] = None  # còn hiệu lực / hết hiệu lực ...


class RawDocument(BaseModel):
    """Văn bản đã parse + clean nhưng chưa chunk."""

    text: str
    metadata: DocumentMetadata


class Chunk(BaseModel):
    """Đoạn nhỏ đã chunk, sẵn sàng để embed."""

    chunk_id: str
    text: str
    article: Optional[str] = None  # "Điều X"
    clause: Optional[str] = None   # "Khoản Y"
    point: Optional[str] = None    # "Điểm Z"
    metadata: DocumentMetadata


class RetrievedChunk(BaseModel):
    """Chunk trả về từ retriever kèm điểm số."""

    chunk: Chunk
    score: float


class Citation(BaseModel):
    source: str
    article: Optional[str] = None
    clause: Optional[str] = None
    point: Optional[str] = None
    snippet: str


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)

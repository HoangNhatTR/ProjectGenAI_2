from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata gắn với mỗi văn bản pháp luật gốc."""

    source: str
    doc_type: Optional[str] = None      # Luật / Nghị định / Thông tư / Án lệ ...
    doc_number: Optional[str] = None    # ví dụ "45/2019/QH14"
    title: Optional[str] = None
    issued_date: Optional[str] = None   # ISO yyyy-mm-dd
    effective_date: Optional[str] = None
    status: Optional[str] = None        # Còn hiệu lực / Hết hiệu lực ...
    linh_vuc: Optional[str] = None      # lĩnh vực pháp lý: hinh_su, dat_dai, ...
    co_quan: Optional[str] = None       # cơ quan ban hành
    folder: Optional[str] = None        # all_laws / nghi_dinh / an_le / nghi_quyet


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
    parent_id: Optional[str] = None  # ID trong ParentStore → context đầy đủ cấp Điều


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
    # Chỉ số [n] LLM dùng trong thân bài (1-based theo thứ tự contexts).
    # Block "📚 Nguồn pháp lý" phải in đúng số này — đánh số lại từ 1 làm
    # UI tra map thất bại khi LLM chỉ trích [3], [5].
    ref: Optional[int] = None
    # Tên văn bản (metadata.title) — UI hiển thị header popup trích dẫn
    title: Optional[str] = None


class ConfidenceInfo(BaseModel):
    """Nhãn độ tin cậy (P2.3) — tính từ tín hiệu retrieval TẤT ĐỊNH (xem
    src/confidence.py), KHÔNG phải LLM tự chấm. reasons_vi luôn có giá trị
    (kể cả khi label='cao' — giải thích vì sao tin cậy cao)."""

    label: str = Field(description="'cao' | 'trung_binh' | 'thap'")
    label_vi: str = Field(description="'Cao' | 'Trung bình' | 'Thấp'")
    reasons_vi: list[str] = Field(default_factory=list)
    # Tín hiệu thô — không bắt buộc hiển thị UI, dùng để debug/calibrate
    # (xem scripts/eval_confidence.py).
    top1_score: float = 0.0
    n_sources: int = 0
    citation_pass_rate: float = 0.0
    has_expired_source: bool = False


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Optional[ConfidenceInfo] = None

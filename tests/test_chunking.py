"""Unit tests cho chunking: header injection, parent-child không cắt cụt, id unique."""
from __future__ import annotations

from src.chunking import (
    MAX_PARENT_CHARS,
    chunk_by_legal_structure,
    chunk_document,
    chunk_recursive,
)
from src.schemas import DocumentMetadata, RawDocument


class FakeParentStore:
    def __init__(self):
        self.data: dict[str, str] = {}

    def add_batch(self, items):
        self.data.update(dict(items))


def _doc(text: str, source: str = "https://vbpl.vn/168_2024_ND-CP.txt") -> RawDocument:
    return RawDocument(
        text=text,
        metadata=DocumentMetadata(
            source=source,
            doc_number="168/2024/NĐ-CP",
            title="Nghị định quy định xử phạt vi phạm hành chính giao thông đường bộ",
        ),
    )


SHORT_DOC = """Điều 1. Phạm vi điều chỉnh
Nghị định này quy định về xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ.

Điều 2. Đối tượng áp dụng
1. Cá nhân, tổ chức Việt Nam.
2. Cá nhân, tổ chức nước ngoài.
"""


def _long_doc_text() -> str:
    """Điều 8 dài > MAX_PARENT_CHARS với nhiều Khoản, mỗi Khoản nhiều Điểm."""
    filler = "người điều khiển xe mô tô, xe gắn máy thực hiện hành vi vi phạm quy định " * 4
    parts = ["Điều 8. Xử phạt người điều khiển xe mô tô, xe gắn máy vi phạm quy tắc giao thông"]
    for k in range(1, 6):
        parts.append(f"{k}. Phạt tiền từ {k}00.000 đồng đến {k}50.000 đồng đối với một trong các hành vi sau:")
        for p in "abcdđe":
            parts.append(f"{p}) Hành vi {p} khoản {k}: {filler}")
    return "Điều 7. Điều ngắn trước đó.\n\n" + "\n".join(parts) + "\n"


# ── Contextual header ──────────────────────────────────────────────────────────

def test_header_chua_so_hieu_van_ban():
    chunks = chunk_by_legal_structure(_doc(SHORT_DOC), chunk_size=600, chunk_overlap=80)
    assert all(c.text.startswith("[168/2024/NĐ-CP") for c in chunks)


def test_chunk_diem_co_header_dieu_va_khoan():
    doc = _doc(_long_doc_text())
    chunks = chunk_by_legal_structure(doc, chunk_size=600, chunk_overlap=80)
    point_chunks = [c for c in chunks if c.point]
    assert point_chunks, "phải có chunk cấp Điểm"
    for c in point_chunks:
        assert "Điều 8" in c.text.splitlines()[0], "header phải chứa tiêu đề Điều"
        assert c.clause and c.clause.startswith("Khoản")
        assert ("— " + c.clause + "]") in c.text.splitlines()[0], "header phải chứa Khoản"


def test_fallback_recursive_cung_co_header():
    doc = _doc("Văn bản không có cấu trúc điều khoản nào cả. " * 30)
    chunks = chunk_recursive(doc, chunk_size=300, chunk_overlap=50)
    assert chunks
    assert all(c.text.startswith("[168/2024/NĐ-CP]") for c in chunks)


# ── Parent-child: không cắt cụt ────────────────────────────────────────────────

def test_dieu_ngan_parent_la_full_dieu():
    store = FakeParentStore()
    chunks = chunk_by_legal_structure(
        _doc(SHORT_DOC), chunk_size=600, chunk_overlap=80, parent_store=store,
    )
    parent_ids = {c.parent_id for c in chunks if c.parent_id}
    assert parent_ids, "chunks phải có parent_id"
    # Parent cấp Điều, chứa nguyên văn
    assert any(pid.endswith("_p_1") for pid in parent_ids)
    full = next(v for k, v in store.data.items() if k.endswith("_p_2"))
    assert "Cá nhân, tổ chức nước ngoài" in full


def test_dieu_dai_parent_theo_khoan_khong_mat_noi_dung():
    text = _long_doc_text()
    assert len(text) > MAX_PARENT_CHARS, "fixture phải dài hơn ngưỡng parent"

    store = FakeParentStore()
    chunks = chunk_by_legal_structure(
        _doc(text), chunk_size=600, chunk_overlap=80, parent_store=store,
    )

    # Điều 8 dài → parent theo Khoản (id dạng _p_8_k_<n>)
    clause_parents = {k: v for k, v in store.data.items() if "_p_8_k_" in k}
    assert len(clause_parents) >= 5, f"phải có parent cho từng Khoản, got {list(store.data)}"

    # BUG CŨ: Điểm cuối của Khoản cuối bị cắt mất khi MAX=2000.
    # Giờ nội dung Khoản 5 Điểm e phải nằm trong parent của Khoản 5.
    k5_parent = next(v for k, v in clause_parents.items() if k.endswith("_k_5"))
    assert "Hành vi e khoản 5" in k5_parent
    # Parent cấp Khoản phải kèm tiêu đề Điều để giữ ngữ cảnh
    assert k5_parent.splitlines()[0].startswith("Điều 8.")

    # Child chunk của Khoản 5 trỏ đúng parent Khoản 5
    k5_chunks = [c for c in chunks if c.clause == "Khoản 5"]
    assert k5_chunks and all(c.parent_id and c.parent_id.endswith("_k_5") for c in k5_chunks)

    # Điều 7 ngắn vẫn dùng parent cấp Điều
    assert any(k.endswith("_p_7") for k in store.data)


# ── chunk_id unique ────────────────────────────────────────────────────────────

def test_chunk_id_khong_trung_giua_2_van_ban_cung_stem():
    doc_a = _doc(SHORT_DOC, source="https://vbpl.vn/nghi_dinh/115-CP.txt")
    doc_b = _doc(SHORT_DOC, source="https://vbpl.vn/quyet_dinh/115-CP.txt")
    ids_a = {c.chunk_id for c in chunk_document(doc_a)}
    ids_b = {c.chunk_id for c in chunk_document(doc_b)}
    assert ids_a.isdisjoint(ids_b), "2 văn bản cùng stem không được trùng chunk_id"


def test_metadata_dieu_khoan_diem_giu_nguyen():
    chunks = chunk_by_legal_structure(_doc(_long_doc_text()), chunk_size=600, chunk_overlap=80)
    sample = next(c for c in chunks if c.point == "Điểm a" and c.clause == "Khoản 1")
    assert sample.article == "Điều 8"

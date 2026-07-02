"""Test chuẩn hoá heading Điều/Chương bị làm phẳng (fix 503 VB / 78k chunk).

Mục tiêu: text crawl-flattened ('...CHUNG Điều 1. Phạm vi...') được tách theo
Điều; trích dẫn ('tại Điều 9 Thông tư', 'Điều 5 của Luật') KHÔNG bị tách nhầm;
text đã đúng cấu trúc giữ nguyên hành vi.
"""
from __future__ import annotations

from src.chunking import _normalize_structure, chunk_document
from src.schemas import DocumentMetadata, RawDocument


def _doc(text: str) -> RawDocument:
    return RawDocument(text=text, metadata=DocumentMetadata(
        source="https://vbpl.vn/van-ban/chi-tiet/99999", doc_number="99/2024/QH15",
        title="Luật Thử nghiệm", doc_type="Luật",
    ))


def _articles(chunks) -> set[str]:
    return {c.article for c in chunks if c.article}


# ── _normalize_structure ──────────────────────────────────────────────────────

def test_normalize_chen_xuong_dong_truoc_heading_giua_dong():
    out = _normalize_structure("NHỮNG QUY ĐỊNH CHUNG Điều 1. Phạm vi điều chỉnh")
    assert "\nĐiều 1. Phạm vi" in out


def test_normalize_heading_co_tieu_de_chu_hoa_co_dau():
    out = _normalize_structure("xxx Điều 3. Áp dụng pháp luật về dân sự")
    assert "\nĐiều 3. Áp dụng" in out


def test_normalize_khong_dung_trich_dan():
    # 'Điều 9 Thông tư', 'Điều 5 của' — không có '. <Hoa>' ngay sau số → giữ nguyên
    src = "thực hiện theo quy định tại Điều 9 Thông tư này và Điều 5 của Luật khác"
    assert _normalize_structure(src) == src


def test_normalize_giu_nguyen_text_da_dung_cau_truc():
    src = "Điều 1. Phạm vi\nĐiều 2. Giải thích từ ngữ\nĐiều 3. Nguyên tắc"
    assert _normalize_structure(src) == src


def test_normalize_idempotent():
    src = "A Điều 1. Phạm vi điều chỉnh B Điều 2. Giải thích từ ngữ"
    once = _normalize_structure(src)
    assert _normalize_structure(once) == once


# ── chunk_document end-to-end ─────────────────────────────────────────────────

def test_chunk_text_lam_phang_tach_duoc_dieu():
    # Mô phỏng 36/2024 / 52/2014: Điều nằm giữa dòng, không có \n
    flat = ("Chương I NHỮNG QUY ĐỊNH CHUNG "
            "Điều 1. Phạm vi điều chỉnh Luật này quy định về trật tự an toàn giao thông. "
            "Điều 2. Giải thích từ ngữ Trong Luật này các từ ngữ được hiểu như sau. "
            "Điều 3. Nguyên tắc bảo đảm trật tự an toàn giao thông đường bộ.")
    chunks = chunk_document(_doc(flat), parent_store=None)
    arts = _articles(chunks)
    assert {"Điều 1", "Điều 2", "Điều 3"}.issubset(arts), f"got {arts}"


def test_chunk_trich_dan_khong_tao_dieu_gia():
    # Toàn bộ là trích dẫn — KHÔNG có heading thật → không có article nào
    ref = ("Việc xử phạt thực hiện theo quy định tại Điều 9 Thông tư này, "
           "Điều 5 của Luật Giao thông và khoản 2 Điều 8 Nghị định liên quan.")
    chunks = chunk_document(_doc(ref), parent_store=None)
    assert _articles(chunks) == set(), f"không nên tạo Điều giả: {_articles(chunks)}"


def test_chunk_text_da_dung_cau_truc_van_hoat_dong():
    structured = ("Điều 1. Phạm vi điều chỉnh\nLuật này quy định ...\n"
                  "Điều 2. Đối tượng áp dụng\nÁp dụng với mọi tổ chức ...")
    chunks = chunk_document(_doc(structured), parent_store=None)
    assert {"Điều 1", "Điều 2"}.issubset(_articles(chunks))

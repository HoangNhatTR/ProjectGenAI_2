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


# ── Điểm chunk gắn câu dẫn khoản (self-contained, 2026-07-08) ────────────────

def test_diem_chunk_gan_cau_dan_khoan():
    """Mỗi điểm con phải chứa MỨC PHẠT (câu dẫn khoản) + hành vi của điểm đó,
    và KHÔNG còn chunk câu-dẫn đứng riêng lẻ."""
    from src.chunking import chunk_document
    from src.schemas import DocumentMetadata, RawDocument

    text = (
        "Điều 7. Xử phạt người điều khiển xe mô tô, xe gắn máy\n"
        "7. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều "
        "khiển xe thực hiện một trong các hành vi vi phạm sau đây:\n"
        "a) Điều khiển xe lạng lách, đánh võng trên đường bộ;\n"
        "c) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;\n"
    ) + ("x" * 6000)  # ép Điều dài → đi nhánh Khoản → Điểm
    doc = RawDocument(text=text, metadata=DocumentMetadata(
        source="https://vbpl.vn/test-168", doc_number="168/2024/NĐ-CP",
        title="Nghị định 168/2024/NĐ-CP"))

    chunks = chunk_document(doc)
    diem_c = [c for c in chunks if c.point == "Điểm c"]
    assert diem_c, "phải có chunk Điểm c"
    txt_c = diem_c[0].text
    assert "đèn tín hiệu" in txt_c            # hành vi của điểm c
    assert "4.000.000" in txt_c               # mức phạt từ câu dẫn khoản

    # Không còn chunk câu-dẫn đứng riêng (point=None nhưng cùng khoản có điểm)
    lead_only = [c for c in chunks if c.clause == "Khoản 7" and c.point is None]
    assert not lead_only, "câu dẫn khoản phải gộp vào điểm, không đứng riêng"

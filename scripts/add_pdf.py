"""Thêm MỘT văn bản mới (PDF / DOCX / TXT) vào cơ sở dữ liệu đang chạy —
TỰ ĐỘNG đọc metadata từ chính file, chỉ cần đưa đường dẫn.

Backend hiện tại là OpenSearch (VECTOR_BACKEND=opensearch, index legal_chunks),
KHÔNG phải Chroma — nên `scripts.ingest` (ghi Chroma, chỉ đọc .txt) không dùng
được. Script này đi đúng đường:

    parse PDF  ->  clean  ->  chunk theo Điều/Khoản (+ parent_store.db)
              ->  embed BGE-M3  ->  ghi OpenSearch  ->  (tùy chọn) cập nhật KG Neo4j

Metadata (số hiệu, cơ quan, loại, ngày, tên) được TỰ TRÍCH từ trang đầu văn bản.
Muốn ép giá trị nào thì truyền flag tương ứng để ghi đè.

Cách chạy — NHANH NHẤT chỉ cần 1 dòng:
    ../Chatbot/Scripts/python.exe -m scripts.add_pdf --pdf "duong/dan/file.pdf"

Các biến thể:
    ... --pdf "file.pdf" --dry-run          # chỉ xem metadata + chunk, KHÔNG ghi
    ... --pdf "file.pdf" --kg               # ghi xong cập nhật luôn KG Neo4j
    ... --pdf "file.pdf" --so-hieu "15/2026/TT-BYT" --ngay 2026-03-01   # ghi đè

Idempotent: chạy lại cùng file không nhân bản (xoá bản cũ cùng số hiệu / nguồn
trước khi ghi). OpenSearch phải đang bật; dùng --kg thì cần NEO4J_* trong .env.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.parsing import load_document
from src.chunking import chunk_document
from src.parent_store import ParentStore
from src.schemas import DocumentMetadata

# Cơ quan ban hành hay gặp -> (tên hiển thị, mã dùng trong số hiệu TT-xxx)
_ISSUERS: list[tuple[str, str, str]] = [
    ("BỘ Y TẾ", "Bộ Y tế", "BYT"),
    ("BỘ CÔNG THƯƠNG", "Bộ Công Thương", "BCT"),
    ("BỘ TÀI CHÍNH", "Bộ Tài chính", "BTC"),
    ("BỘ CÔNG AN", "Bộ Công an", "BCA"),
    ("BỘ GIAO THÔNG VẬN TẢI", "Bộ Giao thông vận tải", "BGTVT"),
    ("BỘ LAO ĐỘNG", "Bộ Lao động - Thương binh và Xã hội", "BLĐTBXH"),
    ("BỘ GIÁO DỤC", "Bộ Giáo dục và Đào tạo", "BGDĐT"),
    ("BỘ TƯ PHÁP", "Bộ Tư pháp", "BTP"),
    ("BỘ KẾ HOẠCH", "Bộ Kế hoạch và Đầu tư", "BKHĐT"),
    ("NGÂN HÀNG NHÀ NƯỚC", "Ngân hàng Nhà nước", "NHNN"),
    ("BỘ NÔNG NGHIỆP", "Bộ Nông nghiệp và Phát triển nông thôn", "BNNPTNT"),
    ("BỘ XÂY DỰNG", "Bộ Xây dựng", "BXD"),
    ("BỘ NỘI VỤ", "Bộ Nội vụ", "BNV"),
    ("BỘ THÔNG TIN", "Bộ Thông tin và Truyền thông", "BTTTT"),
    ("BỘ KHOA HỌC", "Bộ Khoa học và Công nghệ", "BKHCN"),
    ("BỘ TÀI NGUYÊN", "Bộ Tài nguyên và Môi trường", "BTNMT"),
    ("BỘ VĂN HÓA", "Bộ Văn hóa, Thể thao và Du lịch", "BVHTTDL"),
    ("BỘ QUỐC PHÒNG", "Bộ Quốc phòng", "BQP"),
    ("BỘ NGOẠI GIAO", "Bộ Ngoại giao", "BNG"),
    ("THỦ TƯỚNG", "Thủ tướng Chính phủ", "TTg"),
    ("CHÍNH PHỦ", "Chính phủ", "CP"),
    ("QUỐC HỘI", "Quốc hội", "QH"),
]

# Từ khoá heading -> (loại chuẩn hoá, folder mặc định)
_TYPES: list[tuple[str, str, str]] = [
    ("THÔNG TƯ LIÊN TỊCH", "Thông tư liên tịch (TTLT)", "nghi_dinh"),
    ("THÔNG TƯ", "Thông tư (TT)", "nghi_dinh"),
    ("NGHỊ ĐỊNH", "Nghị định (NĐ)", "nghi_dinh"),
    ("NGHỊ QUYẾT", "Nghị quyết (NQ)", "nghi_quyet"),
    ("QUYẾT ĐỊNH", "Quyết định (QĐ)", "nghi_dinh"),
    ("LUẬT", "Luật (Lu)", "all_laws"),
]


def _first_page_text(path: Path) -> str:
    """Text trang đầu (raw) để dò metadata. PDF: page 0; khác: 2000 ký tự đầu."""
    if path.suffix.lower() == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                if pdf.pages:
                    return unicodedata.normalize("NFC", pdf.pages[0].extract_text() or "")
        except Exception:
            pass
    try:
        return unicodedata.normalize("NFC", path.read_text(encoding="utf-8", errors="ignore")[:2000])
    except Exception:
        return ""


def _slug(s: str) -> str:
    keep = "".join(c if c.isalnum() else "-" for c in s.lower())
    while "--" in keep:
        keep = keep.replace("--", "-")
    return keep.strip("-")


def auto_meta(path: Path) -> dict:
    """Trả về dict metadata tự trích + cờ is_draft. Trường không dò được = None."""
    head = _first_page_text(path)
    upper = head.upper()

    # ── Cơ quan ban hành (ưu tiên xuất hiện sớm ở letterhead) ─────────────────
    co_quan = code = None
    best_pos = 10 ** 9
    for key, name, mcode in _ISSUERS:
        i = upper.find(key)
        if 0 <= i < best_pos:
            best_pos, co_quan, code = i, name, mcode

    # ── Loại văn bản + folder ─────────────────────────────────────────────────
    loai = folder = None
    for key, norm, fld in _TYPES:
        if key in upper:
            loai, folder = norm, fld
            break

    # ── Số hiệu: BÁM chữ "Số:" (tránh vớ nhầm số văn bản được viện dẫn) ───────
    # "Số: 15/2026/TT-BYT" -> 15/2026/TT-BYT ; "Số: /2026/TT..." -> dự thảo
    doc_number = None
    is_draft = False
    m = re.search(r"Số\s*:\s*(\d{1,4})?\s*/\s*(20\d\d)\s*/\s*([A-ZĐ][A-ZĐ\-]*)", head)
    if m:
        num, year, suf = m.group(1), m.group(2), m.group(3).rstrip("-")
        # Watermark hay cắt cụt suffix TT-BYT -> BYT: dựng lại từ mã cơ quan
        if loai and loai.startswith("Thông tư") and code:
            suf = f"TT-{code}"
        elif loai and loai.startswith("Nghị định"):
            suf = "NĐ-CP"
        if num:
            doc_number = f"{num}/{year}/{suf}"
        else:
            is_draft = True  # có ô "Số:" nhưng bỏ trống -> dự thảo

    # ── Ngày ban hành ─────────────────────────────────────────────────────────
    issued = None
    d = re.search(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(20\d\d)", head)
    if d:
        issued = f"{d.group(3)}-{int(d.group(2)):02d}-{int(d.group(1)):02d}"

    # ── Tên: lấy từ tên file (đáng tin hơn header bị watermark) ───────────────
    title = re.sub(r"[_\-]+", " ", path.stem).strip()
    title = re.sub(r"\s{2,}", " ", title)

    return {
        "co_quan": co_quan, "loai": loai, "folder": folder,
        "doc_number": doc_number, "is_draft": is_draft,
        "issued": issued, "title": title,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Thêm 1 văn bản PDF/DOCX/TXT vào DB (tự trích metadata)")
    p.add_argument("--pdf", required=True, help="Đường dẫn file .pdf/.docx/.txt")
    # Mọi flag dưới đây TÙY CHỌN — bỏ trống thì tự trích từ file
    p.add_argument("--so-hieu", default=None, help="Ghi đè số hiệu")
    p.add_argument("--ten", default=None, help="Ghi đè tên")
    p.add_argument("--loai", default=None, help="Ghi đè loại")
    p.add_argument("--co-quan", default=None, help="Ghi đè cơ quan")
    p.add_argument("--linh-vuc", default=None, help="Lĩnh vực, vd giao_thong, y_te")
    p.add_argument("--ngay", default=None, help="Ghi đè ngày ban hành ISO")
    p.add_argument("--trang-thai", default=None, help="Tình trạng hiệu lực")
    p.add_argument("--folder", default=None, help="Ghi đè nhóm thư mục")
    p.add_argument("--source", default=None, help="Ghi đè URL/định danh nguồn")
    p.add_argument("--dry-run", action="store_true", help="Chỉ xem metadata + chunk, KHÔNG ghi")
    p.add_argument("--kg", action="store_true", help="Cập nhật luôn KG Neo4j")
    args = p.parse_args()

    path = Path(args.pdf)
    if not path.exists():
        sys.exit(f"[FAIL] Không thấy file: {path}")

    # ── Tự trích rồi cho flag ghi đè ──────────────────────────────────────────
    auto = auto_meta(path)
    so_hieu   = args.so_hieu   or auto["doc_number"]
    loai      = args.loai      or auto["loai"]
    co_quan   = args.co_quan   or auto["co_quan"]
    ngay      = args.ngay      or auto["issued"]
    ten       = args.ten       or auto["title"]
    folder    = args.folder    or auto["folder"] or "nghi_dinh"
    trang_thai = args.trang_thai or ("Dự thảo" if auto["is_draft"] else "Còn hiệu lực")

    source = args.source or (
        f"https://vbpl.vn/van-ban/chi-tiet/{_slug(so_hieu)}" if so_hieu
        else str(path.resolve())
    )

    print("── Metadata tự trích ─────────────────────────────")
    print(f"  Số hiệu   : {so_hieu or '(không có — DỰ THẢO, dùng path làm định danh)'}")
    print(f"  Tên       : {ten}")
    print(f"  Loại      : {loai}")
    print(f"  Cơ quan   : {co_quan}")
    print(f"  Ngày BH   : {ngay or '(không có)'}")
    print(f"  Tình trạng: {trang_thai}")
    print(f"  Folder    : {folder}")
    print(f"  Nguồn/id  : {source}")
    print("──────────────────────────────────────────────────")

    meta = DocumentMetadata(
        source=source, doc_type=loai, doc_number=so_hieu, title=ten,
        issued_date=ngay, status=trang_thai, linh_vuc=args.linh_vuc,
        co_quan=co_quan, folder=folder,
    )

    print(f">>> Đọc & làm sạch {path.name} ...", flush=True)
    doc = load_document(path, meta)
    print(f"   Toàn văn sau clean: {len(doc.text):,} ký tự", flush=True)
    if len(doc.text) < 100:
        sys.exit("[FAIL] Rất ít text — PDF có thể là ảnh scan (cần OCR trước).")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    if args.dry_run:
        class _NoStore:
            def add_batch(self, items):
                self._n = getattr(self, "_n", 0) + len(items)
        ns = _NoStore()
        chunks = chunk_document(doc, parent_store=ns)
        arts = sorted({c.article for c in chunks if c.article})
        print("\n=== DRY-RUN (không ghi gì) ===")
        print(f"  Tổng chunk     : {len(chunks)}")
        print(f"  Điều phân biệt : {len(arts)}")
        print(f"  Parent entries : {getattr(ns, '_n', 0)}")
        return

    parent_store = ParentStore(config.PARENT_STORE_PATH)
    chunks = chunk_document(doc, parent_store=parent_store)
    if not chunks:
        sys.exit("[FAIL] Không chunk được — kiểm tra lại text/parse.")
    print(f"   Chunk: {len(chunks)} | parent_store: {parent_store.count():,} entries", flush=True)

    # ── Embed ─────────────────────────────────────────────────────────────────
    from src.embedding import Embedder
    embedder = Embedder(config.EMBEDDING_MODEL)
    print(f">>> Embed {len(chunks)} chunk ({config.EMBEDDING_MODEL}) ...", flush=True)
    t0 = time.time()
    embeddings = embedder.encode([c.text for c in chunks])
    print(f"   Embed xong {time.time()-t0:.0f}s", flush=True)

    # ── Ghi OpenSearch (idempotent: xoá bản cũ cùng số hiệu/nguồn) ────────────
    from src.opensearch_store import OpenSearchVectorStore
    vstore = OpenSearchVectorStore(config.OPENSEARCH_URL, config.OPENSEARCH_INDEX)
    vstore.ensure_index()
    cli = vstore._connect()
    term = {"doc_number": so_hieu} if so_hieu else {"source": source}
    d = cli.delete_by_query(index=config.OPENSEARCH_INDEX, refresh=True,
                            body={"query": {"term": term}})
    if d.get("deleted"):
        print(f"   Xoá {d['deleted']} chunk cũ ({list(term)[0]}={list(term.values())[0]})", flush=True)

    vstore.add(chunks, embeddings)
    cli.indices.refresh(index=config.OPENSEARCH_INDEX)
    cnt = cli.count(index=config.OPENSEARCH_INDEX,
                    body={"query": {"term": term}})["count"]
    print(f"\n=== XONG (OpenSearch) === {cnt} chunk đã vào index legal_chunks")

    # ── (tùy chọn) KG ─────────────────────────────────────────────────────────
    if args.kg:
        try:
            _update_kg(doc)
        except Exception as exc:
            print(f"[!] Bỏ qua KG (OpenSearch đã ghi xong): {exc}")

    print(f"\nTổng chunk trong index legal_chunks: {vstore.count():,}")


def _update_kg(doc) -> None:
    from src.kg.neo4j_client import Neo4jClient
    from src.kg.structural_extractor import (
        extract_structural, dedup_citations, resolve_amendments,
    )
    print("\n>>> Cập nhật KG Neo4j ...", flush=True)
    result = extract_structural(doc)
    law_node = result["law_node"]
    article_ids = {a["id"] for a in result["article_nodes"]}

    client = Neo4jClient.from_env()
    client.create_constraints()
    client.upsert_laws([law_node])
    if result["article_nodes"]:
        client.upsert_articles(result["article_nodes"])
    if result["clause_nodes"]:
        client.upsert_clauses(result["clause_nodes"])
    good = dedup_citations(result["internal_cites"], article_ids)
    if good:
        client.add_internal_citations(good)
    if doc.metadata.doc_number and result["amend_targets_raw"]:
        pairs = [(law_node["id"], t) for t in result["amend_targets_raw"]]
        edges = resolve_amendments(pairs, {doc.metadata.doc_number: law_node["id"]})
        if edges:
            client.add_amendment_edges(edges)
    print(f"   KG: +1 Law, +{len(result['article_nodes'])} Điều, "
          f"+{len(result['clause_nodes'])} Khoản, +{len(good)} trích dẫn")
    client.close()


if __name__ == "__main__":
    main()

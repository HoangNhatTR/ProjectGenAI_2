"""Kiểm tra các luật quan trọng còn thiếu trong dataset.

Script này:
1. Quét tất cả file trong data/raw/all_laws/ để lập chỉ mục
2. So sánh với danh sách ~80 luật quan trọng nhất VN
3. Báo cáo luật nào ĐÃ CÓ và luật nào CÒN THIẾU
4. Tự động sinh lệnh crawl cho từng luật còn thiếu

Cách chạy:
    python -m scripts.check_missing_laws              # báo cáo đầy đủ
    python -m scripts.check_missing_laws --missing    # chỉ hiện luật thiếu
    python -m scripts.check_missing_laws --export     # xuất missing ra JSON
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

RAW_DIR  = Path(__file__).resolve().parent.parent / "data" / "raw"
ALL_LAWS = RAW_DIR / "all_laws"

# ─── Danh sách luật quan trọng cần kiểm tra ───────────────────────────────────
# Format: (so_hieu_pattern, ten_luat, linh_vuc, note)
IMPORTANT_LAWS: list[tuple[str, str, str, str]] = [
    # ── Hiến pháp ─────────────────────────────────────────────────────────────
    ("92/2013", "Hiến pháp 2013",                    "hien_phap",   ""),

    # ── Bộ luật nền tảng ──────────────────────────────────────────────────────
    ("91/2015",  "Bộ luật Dân sự 2015",              "dan_su",      "BLDS"),
    ("100/2015", "Bộ luật Hình sự 2015",             "hinh_su",     "BLHS"),
    ("101/2015", "Bộ luật Tố tụng Dân sự 2015",     "dan_su",      "BLTTDS"),
    ("101/2015", "Bộ luật Tố tụng Hình sự 2015",    "hinh_su",     "BLTTHS — số 101/2015/QH13"),
    ("38/2005",  "Bộ luật Lao động 2012",            "lao_dong",    "Bản cũ — xem thêm 45/2019"),
    ("45/2019",  "Bộ luật Lao động 2019",            "lao_dong",    "Hiệu lực 2021"),

    # ── Đất đai & BĐS ─────────────────────────────────────────────────────────
    ("13/2003",  "Luật Đất đai 2003",                "dat_dai",     "Hết hiệu lực"),
    ("45/2013",  "Luật Đất đai 2013",                "dat_dai",     "Hết hiệu lực 2025"),
    ("31/2024",  "Luật Đất đai 2024",                "dat_dai",     "QUAN TRỌNG NHẤT"),
    ("27/2023",  "Luật Nhà ở 2023",                  "nha_o",       "Hiệu lực 01/08/2024"),
    ("29/2023",  "Luật Kinh doanh BĐS 2023",         "bds",         "Hiệu lực 01/08/2024"),
    ("36/2009",  "Luật Quy hoạch đô thị 2009",       "do_thi",      ""),

    # ── Doanh nghiệp & Đầu tư ─────────────────────────────────────────────────
    ("59/2020",  "Luật Doanh nghiệp 2020",           "doanh_nghiep","Hiệu lực 2021"),
    ("61/2020",  "Luật Đầu tư 2020",                 "dau_tu",      "Hiệu lực 2021"),
    ("54/2019",  "Luật Đầu tư công 2019",            "dau_tu",      ""),
    ("21/2012",  "Luật Phá sản 2014",                "doanh_nghiep","Số 51/2014/QH13"),
    ("68/2014",  "Luật Doanh nghiệp 2014",           "doanh_nghiep","Hết hiệu lực"),
    ("88/2015",  "Luật Kế toán 2015",                "ke_toan",     ""),
    ("03/2003",  "Luật Kế toán 2003",                "ke_toan",     "Hết hiệu lực"),
    ("67/2014",  "Luật Kinh doanh bảo hiểm",         "bao_hiem",    ""),
    ("54/2010",  "Luật Ngân hàng Nhà nước 2010",     "ngan_hang",   ""),
    ("47/2010",  "Luật Các tổ chức tín dụng 2010",   "ngan_hang",   ""),
    ("17/2017",  "Luật Các tổ chức tín dụng sửa đổi","ngan_hang",   ""),
    ("32/2024",  "Luật Các tổ chức tín dụng 2024",   "ngan_hang",   "QUAN TRỌNG"),
    ("54/2019",  "Luật Chứng khoán 2019",            "chung_khoan", "Số 54/2019/QH14"),

    # ── Thuế ──────────────────────────────────────────────────────────────────
    ("13/2008",  "Luật Thuế GTGT 2008",              "thue",        ""),
    ("31/2013",  "Luật Thuế GTGT sửa đổi 2013",      "thue",        ""),
    ("14/2008",  "Luật Thuế TNDN 2008",              "thue",        ""),
    ("04/2014",  "Luật Thuế TNDN sửa đổi 2014",      "thue",        ""),
    ("04/2007",  "Luật Thuế TNCN 2007",              "thue",        ""),
    ("26/2012",  "Luật Thuế TNCN sửa đổi 2012",      "thue",        ""),
    ("57/2010",  "Luật Thuế tiêu thụ đặc biệt",      "thue",        ""),
    ("27/2008",  "Luật Quản lý thuế 2006",           "thue",        "Số 78/2006/QH11"),
    ("38/2019",  "Luật Quản lý thuế 2019",           "thue",        "QUAN TRỌNG"),
    ("05/2017",  "Luật Thuế xuất khẩu nhập khẩu",   "thue",        "Số 107/2016/QH13"),

    # ── Lao động & BHXH ───────────────────────────────────────────────────────
    ("58/2014",  "Luật BHXH 2014",                   "bhxh",        ""),
    ("25/2008",  "Luật BHYT 2008",                   "bhyt",        ""),
    ("46/2014",  "Luật BHYT sửa đổi 2014",           "bhyt",        ""),
    ("38/2013",  "Luật Việc làm 2013",               "lao_dong",    ""),
    ("84/2015",  "Luật ATVSLĐ 2015",                 "lao_dong",    "An toàn VSLĐ"),

    # ── Hôn nhân & gia đình ───────────────────────────────────────────────────
    ("52/2014",  "Luật Hôn nhân và Gia đình 2014",   "dan_su",      "Số 52/2014/QH13"),
    ("36/2014",  "Luật Hộ tịch 2014",                "dan_su",      ""),
    ("24/2000",  "Luật Hôn nhân & GĐ 2000",          "dan_su",      "Hết hiệu lực"),
    ("02/2016",  "Luật Nuôi con nuôi",               "dan_su",      "Số 52/2010/QH12"),

    # ── Hành chính & Tố tụng ─────────────────────────────────────────────────
    ("64/2006",  "Luật Tố tụng Hành chính",          "hanh_chinh",  "Số 93/2015/QH13"),
    ("02/2011",  "Luật Khiếu nại 2011",              "hanh_chinh",  "Số 02/2011/QH13"),
    ("03/2011",  "Luật Tố cáo 2011",                 "hanh_chinh",  "Số 03/2011/QH13"),
    ("25/2018",  "Luật Tố cáo 2018",                 "hanh_chinh",  "Số 25/2018/QH14"),
    ("15/2012",  "Luật Xử lý VPHC 2012",             "hanh_chinh",  "Số 15/2012/QH13"),
    ("67/2020",  "Luật Xử lý VPHC sửa đổi 2020",    "hanh_chinh",  "Số 67/2020/QH14"),

    # ── Giao thông ────────────────────────────────────────────────────────────
    ("23/2008",  "Luật Giao thông đường bộ 2008",    "giao_thong",  ""),
    ("36/2024",  "Luật Trật tự ATGT đường bộ 2024",  "giao_thong",  "Hiệu lực 2025"),
    ("35/2024",  "Luật Đường bộ 2024",               "giao_thong",  "Hiệu lực 2025"),

    # ── Hình sự & Thi hành án ─────────────────────────────────────────────────
    ("26/2014",  "Luật Thi hành án Hình sự",         "hinh_su",     "Số 41/2019/QH14"),
    ("26/2008",  "Luật Thi hành án Dân sự",          "dan_su",      ""),

    # ── Môi trường & Tài nguyên ───────────────────────────────────────────────
    ("55/2014",  "Luật Bảo vệ môi trường 2014",      "moi_truong",  "Hết hiệu lực"),
    ("72/2020",  "Luật Bảo vệ môi trường 2020",      "moi_truong",  "Hiệu lực 2022"),
    ("17/2012",  "Luật Tài nguyên nước 2012",        "tai_nguyen",  ""),
    ("28/2023",  "Luật Tài nguyên nước 2023",        "tai_nguyen",  "QUAN TRỌNG"),

    # ── Y tế & Giáo dục ───────────────────────────────────────────────────────
    ("40/2009",  "Luật Khám bệnh chữa bệnh 2009",    "y_te",        "Hết hiệu lực"),
    ("15/2023",  "Luật Khám bệnh chữa bệnh 2023",    "y_te",        "Hiệu lực 2024"),
    ("43/2009",  "Luật Giáo dục 2005",               "giao_duc",    "Hết hiệu lực"),
    ("43/2019",  "Luật Giáo dục 2019",               "giao_duc",    ""),
    ("34/2018",  "Luật Giáo dục Đại học sửa đổi",   "giao_duc",    ""),

    # ── Công nghệ thông tin & Sở hữu trí tuệ ─────────────────────────────────
    ("67/2006",  "Luật Công nghệ thông tin 2006",    "cntt",        ""),
    ("50/2005",  "Luật Sở hữu trí tuệ 2005",        "shtt",        ""),
    ("07/2022",  "Luật Sở hữu trí tuệ sửa đổi 2022","shtt",        ""),

    # ── Xây dựng ──────────────────────────────────────────────────────────────
    ("50/2014",  "Luật Xây dựng 2014",               "xay_dung",    ""),
    ("62/2020",  "Luật Xây dựng sửa đổi 2020",      "xay_dung",    ""),

    # ── Hải quan & Thương mại ─────────────────────────────────────────────────
    ("54/2014",  "Luật Hải quan 2014",               "hai_quan",    ""),
    ("36/2005",  "Luật Thương mại 2005",             "thuong_mai",  ""),
    ("05/2005",  "Luật Cạnh tranh 2004",             "canh_tranh",  "Hết hiệu lực"),
    ("23/2018",  "Luật Cạnh tranh 2018",             "canh_tranh",  ""),
]

# ─── Normalize ────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    """Chuẩn hóa string để so sánh: lowercase, bỏ dấu, bỏ ký tự đặc biệt."""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_so_hieu(text: str) -> str:
    """Trích số hiệu từ header SO_HIEU: ..."""
    m = re.search(r"^SO_HIEU:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def extract_ten(text: str) -> str:
    """Trích tên VB từ header TEN: ..."""
    m = re.search(r"^TEN:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def extract_hieu_luc(text: str) -> str:
    m = re.search(r"^HIEU_LUC:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""

# ─── Build index ──────────────────────────────────────────────────────────────

def build_index() -> list[dict]:
    """Đọc tất cả file all_laws và tạo index."""
    index = []
    files = list(ALL_LAWS.glob("*.txt"))
    print(f"  Đọc {len(files)} file trong all_laws/...")

    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
            header = text[:500]
            so_hieu   = extract_so_hieu(header)
            ten       = extract_ten(header)
            hieu_luc  = extract_hieu_luc(header)
            index.append({
                "file":      fp.name,
                "so_hieu":   so_hieu,
                "ten":       ten,
                "hieu_luc":  hieu_luc,
                "so_norm":   normalize(so_hieu),
                "ten_norm":  normalize(ten),
            })
        except Exception:
            pass
    return index


def find_in_index(pattern: str, ten: str, index: list[dict]) -> Optional[dict]:
    """Tìm xem luật (pattern, ten) đã có trong index chưa."""
    pat_norm = normalize(pattern)
    ten_norm = normalize(ten)

    # 1. So sánh số hiệu
    for entry in index:
        if pat_norm and pat_norm in entry["so_norm"]:
            return entry
        if pat_norm and entry["so_norm"] and pat_norm in entry["so_norm"]:
            return entry

    # 2. So sánh tên (fuzzy: kiểm tra từng từ quan trọng)
    ten_keywords = [w for w in ten_norm.split() if len(w) > 4]
    for entry in index:
        if not entry["ten_norm"]:
            continue
        matched = sum(1 for kw in ten_keywords if kw in entry["ten_norm"])
        if matched >= max(2, len(ten_keywords) - 1):
            return entry

    return None


# Thêm Optional import ở đầu
from typing import Optional

# ─── Report ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Kiểm tra luật còn thiếu trong dataset")
    p.add_argument("--missing", action="store_true", help="Chỉ hiện luật còn thiếu")
    p.add_argument("--export",  type=str, metavar="FILE",
                   help="Xuất danh sách luật thiếu ra file JSON")
    args = p.parse_args()

    print("\n" + "="*70)
    print("  KIỂM TRA LUẬT QUAN TRỌNG TRONG DATASET")
    print("="*70)

    index = build_index()

    found_list   = []
    missing_list = []

    for so_hieu, ten, linh_vuc, note in IMPORTANT_LAWS:
        entry = find_in_index(so_hieu, ten, index)
        if entry:
            found_list.append({
                "ten": ten, "so_hieu": so_hieu,
                "file": entry["file"],
                "hieu_luc": entry["hieu_luc"],
                "note": note,
            })
        else:
            missing_list.append({
                "ten": ten, "so_hieu": so_hieu,
                "linh_vuc": linh_vuc,
                "note": note,
            })

    # ── Báo cáo ───────────────────────────────────────────────────────────────

    print(f"\n  ✓ ĐÃ CÓ: {len(found_list)}/{len(IMPORTANT_LAWS)} luật quan trọng")
    print(f"  ✗ CÒN THIẾU: {len(missing_list)}/{len(IMPORTANT_LAWS)} luật quan trọng\n")

    if not args.missing:
        print("─"*70)
        print("  ĐÃ CÓ TRONG DATASET:")
        print("─"*70)
        for item in sorted(found_list, key=lambda x: x["ten"]):
            hl = item["hieu_luc"][:25] if item["hieu_luc"] else "?"
            print(f"  ✓ {item['ten']:<45} [{hl}]")
            if item["note"]:
                print(f"    → {item['note']}")

    print("\n" + "─"*70)
    print("  CÒN THIẾU — CẦN BỔ SUNG:")
    print("─"*70)

    # Nhóm theo lĩnh vực
    by_linh_vuc: dict[str, list] = {}
    for item in missing_list:
        by_linh_vuc.setdefault(item["linh_vuc"], []).append(item)

    for linh_vuc, items in sorted(by_linh_vuc.items()):
        print(f"\n  [{linh_vuc.upper()}]")
        for item in items:
            note_str = f"  ({item['note']})" if item["note"] else ""
            print(f"    ✗ {item['ten']}{note_str}")

    # ── Lệnh crawl gợi ý ──────────────────────────────────────────────────────

    if missing_list:
        print("\n" + "─"*70)
        print("  GỢI Ý: Chạy crawl_vbpl.py để bổ sung:")
        print("─"*70)
        print("  # Crawl toàn bộ HP+BL+Lu (bao gồm nhiều luật còn thiếu):")
        print("  python -m scripts.crawl_vbpl --topic all_laws --max-docs 800")
        print()
        print("  # Crawl riêng từng lĩnh vực bị thiếu:")

        seen_lv: set[str] = set()
        for item in missing_list:
            lv = item["linh_vuc"]
            if lv not in seen_lv and lv in {
                "dat_dai", "hinh_su", "dan_su", "lao_dong", "thue",
                "ngan_hang", "doanh_nghiep", "bhxh",
            }:
                seen_lv.add(lv)
                print(f"  python -m scripts.crawl_vbpl --topic {lv}")

        print()
        print("  # Crawl Nghị định & Thông tư (văn bản dưới luật):")
        print("  python -m scripts.crawl_nghi_dinh --topic all")
        print()
        print("  # Crawl Án lệ:")
        print("  python -m scripts.crawl_an_le")

    # ── Export ────────────────────────────────────────────────────────────────

    if args.export:
        export_path = Path(args.export)
        export_data = {
            "tong_quan": {
                "tong_kiem_tra": len(IMPORTANT_LAWS),
                "da_co": len(found_list),
                "con_thieu": len(missing_list),
            },
            "da_co":    found_list,
            "con_thieu": missing_list,
        }
        export_path.write_text(
            json.dumps(export_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  → Đã xuất báo cáo ra: {export_path}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()

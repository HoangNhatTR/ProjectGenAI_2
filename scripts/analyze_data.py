"""Phân tích toàn diện cấu trúc dữ liệu pháp luật.

Đếm: Bộ luật/Luật → Chương → Mục → Điều → Khoản → Điểm
Báo cáo thống kê theo từng folder, theo lĩnh vực, và tổng thể.

Cách chạy:
    python -m scripts.analyze_data
    python -m scripts.analyze_data --export report.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# ─── Regex patterns ───────────────────────────────────────────────────────────
P_CHUONG = re.compile(r"(?:CHƯƠNG|Chương)\s+(?:[IVXLCDM]+|\d+)")
P_MUC    = re.compile(r"(?:MỤC|Mục)\s+\d+")
P_DIEU   = re.compile(r"Điều\s+\d+")
# Khoản: "1. [Chữ hoa hoặc thường]" — xuất hiện sau Điều
P_KHOAN  = re.compile(r"(?<![./])\b(\d+)\.\s+[A-ZĐÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖ"
                      r"ơờớợởỡùúụủũưừứựửữỳýỵỷỹa-zđ]")
# Điểm: "a) Chữ" hoặc "b) Chữ"
P_DIEM   = re.compile(r"(?<!\w)([a-zđ])\)\s+[A-ZĐÀÁẠẢÃÂÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴa-zđ]")

# ─── Trích metadata từ header ─────────────────────────────────────────────────

def get_field(text: str, field: str) -> str:
    m = re.search(rf"^{field}:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def count_elements(text: str) -> dict:
    return {
        "chuong": len(P_CHUONG.findall(text)),
        "muc":    len(P_MUC.findall(text)),
        "dieu":   len(P_DIEU.findall(text)),
        "khoan":  len(P_KHOAN.findall(text)),
        "diem":   len(P_DIEM.findall(text)),
    }

# ─── Phân tích 1 folder ───────────────────────────────────────────────────────

def analyze_dir(folder: Path, label: str) -> dict:
    if not folder.exists():
        return {}

    files = [f for f in folder.iterdir() if f.is_file() and f.suffix == ".txt"]
    if not files:
        return {}

    totals = {"files": len(files), "size_bytes": 0,
              "chuong": 0, "muc": 0, "dieu": 0, "khoan": 0, "diem": 0}
    by_linh_vuc: dict[str, dict] = defaultdict(
        lambda: {"files": 0, "dieu": 0, "khoan": 0, "diem": 0}
    )
    loai_count: dict[str, int] = defaultdict(int)
    hieu_luc_count: dict[str, int] = defaultdict(int)
    file_details: list[dict] = []

    print(f"\n  Phân tích [{label}] — {len(files)} files...", end="", flush=True)
    for i, fp in enumerate(files):
        if i % 100 == 0 and i > 0:
            print(f" {i}...", end="", flush=True)
        try:
            txt = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        totals["size_bytes"] += fp.stat().st_size

        # Metadata
        so_hieu   = get_field(txt[:400], "SO_HIEU")
        ten       = get_field(txt[:400], "TEN")
        loai      = get_field(txt[:400], "LOAI")
        hieu_luc  = get_field(txt[:400], "HIEU_LUC")
        linh_vuc  = get_field(txt[:400], "LINH_VUC") or "unknown"

        loai_short = re.search(r"\((\w+)\)", loai)
        loai_code  = loai_short.group(1) if loai_short else loai[:10] or "?"
        loai_count[loai_code] += 1

        hl_short = hieu_luc[:30] if hieu_luc else "Không rõ"
        hieu_luc_count[hl_short] += 1

        # Đếm cấu trúc
        cnt = count_elements(txt)
        for k, v in cnt.items():
            totals[k] += v

        by_linh_vuc[linh_vuc]["files"]  += 1
        by_linh_vuc[linh_vuc]["dieu"]   += cnt["dieu"]
        by_linh_vuc[linh_vuc]["khoan"]  += cnt["khoan"]
        by_linh_vuc[linh_vuc]["diem"]   += cnt["diem"]

        file_details.append({
            "file": fp.name, "so_hieu": so_hieu, "ten": ten[:60],
            "loai": loai_code, "hieu_luc": hl_short,
            "linh_vuc": linh_vuc, **cnt
        })

    print(" xong.")
    return {
        "label":         label,
        "totals":        totals,
        "by_linh_vuc":  dict(by_linh_vuc),
        "loai_count":   dict(loai_count),
        "hieu_luc":     dict(hieu_luc_count),
        "top_files":    sorted(file_details, key=lambda x: -x["dieu"])[:20],
        "all_files":    file_details,
    }

# ─── In báo cáo ───────────────────────────────────────────────────────────────

def print_report(results: dict[str, dict]) -> None:
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("  THỐNG KÊ TOÀN BỘ DỮ LIỆU PHÁP LUẬT")
    print(SEP)

    grand = {"files": 0, "size_bytes": 0,
             "chuong": 0, "muc": 0, "dieu": 0, "khoan": 0, "diem": 0}

    for key, r in results.items():
        if not r:
            continue
        t = r["totals"]
        for k in grand:
            grand[k] += t.get(k, 0)

        size_mb = round(t["size_bytes"] / 1_048_576, 2)
        print(f"\n  ┌─ {r['label']} ({'─'*(50-len(r['label']))}")
        print(f"  │  Files       : {t['files']:,}")
        print(f"  │  Dung lượng  : {size_mb} MB")
        print(f"  │  Chương      : {t['chuong']:,}")
        print(f"  │  Mục         : {t['muc']:,}")
        print(f"  │  Điều        : {t['dieu']:,}")
        print(f"  │  Khoản ~     : {t['khoan']:,}")
        print(f"  │  Điểm  ~     : {t['diem']:,}")

        # Loại văn bản
        if r.get("loai_count"):
            print(f"  │  Loại VB     :", end="")
            for lc, cnt in sorted(r["loai_count"].items(), key=lambda x: -x[1]):
                print(f" {lc}:{cnt}", end="")
            print()

        # Hiệu lực
        if r.get("hieu_luc"):
            print(f"  │  Hiệu lực:")
            for hl, cnt in sorted(r["hieu_luc"].items(), key=lambda x: -x[1])[:5]:
                print(f"  │    {hl:<30}: {cnt}")

        # Phân bổ lĩnh vực
        if r.get("by_linh_vuc"):
            print(f"  │  Theo lĩnh vực:")
            for lv, lv_data in sorted(r["by_linh_vuc"].items(),
                                      key=lambda x: -x[1]["dieu"])[:10]:
                print(f"  │    {lv:<20}: {lv_data['files']:3} files | "
                      f"{lv_data['dieu']:,} Điều | {lv_data['khoan']:,} Khoản~")

        print(f"  └{'─'*60}")

    # ── TỔNG KẾT ──────────────────────────────────────────────────────────────
    total_mb = round(grand["size_bytes"] / 1_048_576, 2)
    print(f"\n{SEP}")
    print("  TỔNG KẾT TOÀN HỆ THỐNG")
    print(SEP)
    print(f"  Tổng số văn bản   : {grand['files']:,} files")
    print(f"  Tổng dung lượng   : {total_mb} MB")
    print(f"  Chương            : {grand['chuong']:,}")
    print(f"  Mục               : {grand['muc']:,}")
    print(f"  Điều              : {grand['dieu']:,}")
    print(f"  Khoản (ước tính)  : {grand['khoan']:,}")
    print(f"  Điểm (ước tính)   : {grand['diem']:,}")
    print(f"\n  → Tổng quy phạm pháp luật ở tầng Điều+Khoản+Điểm:")
    total_qppl = grand["dieu"] + grand["khoan"] + grand["diem"]
    print(f"    {grand['dieu']:,} + {grand['khoan']:,} + {grand['diem']:,} = {total_qppl:,} đơn vị")

    # ── Đánh giá & Khuyến nghị ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  ĐÁNH GIÁ & NHỮNG GÌ CẦN BỔ SUNG")
    print(SEP)

    print("""
  ĐÃ CÓ:
  ✓ Tầng Luật (QH ban hành): ~609 văn bản, phủ rộng hầu hết lĩnh vực
  ✓ Metadata đầy đủ: SO_HIEU, NGAY_BAN_HANH, HIEU_LUC, LINH_VUC
  ✓ Relationship map: quan hệ thay thế/sửa đổi giữa các VB
  ✓ 2 file cũ đã được thêm header chuẩn

  CÒN THIẾU (ưu tiên cao → thấp):
  ✗ [P1] Nghị định & Thông tư: chỉ có 1/~2000+ văn bản dưới luật
       → Chạy: python -m scripts.crawl_nghi_dinh --topic all
       → Ước tính sẽ có thêm ~500-1500 NĐ+TT quan trọng

  ✗ [P1] Án lệ: 0/70+ án lệ đã được HĐTP ban hành
       → Chạy: python -m scripts.crawl_an_le

  ✗ [P2] 7 Luật quan trọng còn thiếu (xem check_missing_laws.py):
       Hiến pháp 2013, Hộ tịch 2014, Thi hành án HS,
       Tài nguyên nước 2012, Thuế TNDN sửa đổi, Thuế TNCN 2007

  ✗ [P2] Nghị quyết của HĐTP/UBTVQH hướng dẫn áp dụng pháp luật
       → Loại type_code "NQ" trên vbpl.vn

  ✗ [P3] Văn bản địa phương (63 tỉnh): Quyết định UBND, Chỉ thị...
       → Rất lớn, cân nhắc theo nhu cầu cụ thể

  ✗ [P3] Điều ước quốc tế VN là thành viên (WTO, ASEAN, EVFTA...)
       → Ảnh hưởng thuế, thương mại, đầu tư nước ngoài
    """)
    print(SEP)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--export", type=str, default=None)
    args = p.parse_args()

    print("\nPhân tích dữ liệu pháp luật...", flush=True)

    results = {}

    # Root raw (bo_luat_hinh_su.txt, luat_giao_thong_2025.txt)
    root_files = [f for f in RAW_DIR.iterdir() if f.is_file() and f.suffix == ".txt"]
    if root_files:
        r = analyze_dir(RAW_DIR, "Bộ luật cũ (root)")
        results["root"] = r

    # all_laws
    results["all_laws"]  = analyze_dir(RAW_DIR / "all_laws",  "Tất cả Luật (all_laws)")
    results["nghi_dinh"] = analyze_dir(RAW_DIR / "nghi_dinh", "Nghị định")
    results["thong_tu"]  = analyze_dir(RAW_DIR / "thong_tu",  "Thông tư")
    results["an_le"]     = analyze_dir(RAW_DIR / "an_le",     "Án lệ")

    print_report(results)

    if args.export:
        out = {k: {kk: vv for kk, vv in v.items() if kk != "all_files"}
               for k, v in results.items() if v}
        Path(args.export).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  → Đã xuất báo cáo: {args.export}")


if __name__ == "__main__":
    main()

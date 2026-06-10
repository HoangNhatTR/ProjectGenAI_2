"""Bổ sung & làm giàu metadata cho toàn bộ file văn bản pháp luật.

Công việc cụ thể:
1. Thêm header chuẩn cho file cũ chưa có (bo_luat_hinh_su.txt, luat_giao_thong_2025.txt...)
2. Thêm trường VAN_BAN_THAY_THE, VAN_BAN_SUA_DOI_BO_SUNG, VAN_BAN_LIEN_QUAN
   bằng cách phân tích nội dung văn bản
3. Xây dựng file relationship_map.json — bản đồ quan hệ giữa các văn bản
4. Thêm trường LINH_VUC (lĩnh vực pháp lý) dựa trên nội dung

Cách chạy:
    python -m scripts.enrich_metadata              # xử lý tất cả
    python -m scripts.enrich_metadata --dry-run    # chỉ xem, không sửa file
    python -m scripts.enrich_metadata --dir nghi_dinh   # chỉ 1 thư mục

Kết quả:
    - Các file .txt được cập nhật header
    - data/raw/relationship_map.json  ← bản đồ quan hệ VB
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# ─── Metadata cho file cũ không có header ─────────────────────────────────────

# Hardcode metadata cho các file cũ đã biết trước
KNOWN_OLD_FILES: dict[str, dict] = {
    "bo_luat_hinh_su.txt": {
        "SO_HIEU":       "100/2015/QH13",
        "TEN":           "Bộ luật Hình sự 2015 (sửa đổi, bổ sung 2017)",
        "LOAI":          "Bộ luật (BL)",
        "CO_QUAN":       "Quốc hội",
        "NGAY_BAN_HANH": "2015-11-27",
        "HIEU_LUC":      "Còn hiệu lực",
        "URL":           "https://vbpl.vn/botuphap/Pages/vbpq-toanvan.aspx?ItemID=122630",
        "CHU_DE":        "Hình sự",
        "LINH_VUC":      "hinh_su",
    },
    "luat_giao_thong_2025.txt": {
        "SO_HIEU":       "36/2024/QH15",
        "TEN":           "Luật Trật tự, an toàn giao thông đường bộ 2024",
        "LOAI":          "Luật (Lu)",
        "CO_QUAN":       "Quốc hội",
        "NGAY_BAN_HANH": "2024-06-27",
        "HIEU_LUC":      "Còn hiệu lực (hiệu lực từ 01/01/2025)",
        "URL":           "https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=188040",
        "CHU_DE":        "Giao thông",
        "LINH_VUC":      "giao_thong",
    },
}

# ─── Từ điển lĩnh vực ─────────────────────────────────────────────────────────

LINH_VUC_KEYWORDS: dict[str, list[str]] = {
    "hinh_su":      ["hình sự", "tội phạm", "hình phạt", "tố tụng hình sự", "thi hành án hình"],
    "dan_su":       ["dân sự", "hợp đồng", "thừa kế", "tài sản", "sở hữu", "hôn nhân", "gia đình"],
    "dat_dai":      ["đất đai", "quyền sử dụng đất", "bồi thường", "thu hồi đất", "cấp giấy"],
    "doanh_nghiep": ["doanh nghiệp", "công ty", "thành lập", "đăng ký kinh doanh", "cổ phần"],
    "thue":         ["thuế", "thuế giá trị gia tăng", "thuế thu nhập", "khai thuế", "hoàn thuế"],
    "lao_dong":     ["lao động", "tiền lương", "hợp đồng lao động", "sa thải", "thất nghiệp"],
    "ngan_hang":    ["ngân hàng", "tín dụng", "cho vay", "lãi suất", "tổ chức tín dụng"],
    "giao_thong":   ["giao thông", "xe cơ giới", "đường bộ", "vi phạm giao thông", "bằng lái"],
    "hanh_chinh":   ["khiếu nại", "tố cáo", "xử phạt vi phạm hành chính", "hành chính"],
    "moi_truong":   ["môi trường", "ô nhiễm", "chất thải", "bảo vệ môi trường"],
    "bds":          ["bất động sản", "nhà ở", "chung cư", "kinh doanh bất động sản"],
    "bhxh":         ["bảo hiểm xã hội", "bảo hiểm y tế", "bảo hiểm thất nghiệp", "bhxh"],
}

# ─── Patterns quan hệ văn bản ─────────────────────────────────────────────────

# Tìm văn bản bị thay thế / sửa đổi
_THAY_THE_PATTERNS = [
    re.compile(r"thay thế\s+(?:cho\s+)?([^\.,;\n]{5,80})", re.I),
    re.compile(r"bãi bỏ\s+([^\.,;\n]{5,80})", re.I),
    re.compile(r"hết hiệu lực\s+([^\.,;\n]{5,80})", re.I),
]

_SUA_DOI_PATTERNS = [
    re.compile(r"sửa đổi[,\s]+bổ sung\s+([^\.,;\n]{5,80})", re.I),
    re.compile(r"bổ sung\s+([^\.,;\n]{5,80})", re.I),
]

_SO_HIEU_IN_TEXT = re.compile(
    r"\b(\d{1,3}/\d{4}/(?:QH|NĐ|TT|TTLT|UB|CP|BTC|BCT|BCA|BTP|BLĐTBXH|BKHĐT|BXD|BYT|BGDĐT|BTTTT|NN|CT|QĐ|NQ)\S*)",
    re.I,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize_vi(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def detect_linh_vuc(text: str) -> str:
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for lv, keywords in LINH_VUC_KEYWORDS.items():
        scores[lv] = sum(text_lower.count(kw) for kw in keywords)
    if not any(scores.values()):
        return ""
    return max(scores, key=scores.__getitem__)


def extract_relationships(text: str) -> dict[str, list[str]]:
    """Phân tích nội dung để tìm quan hệ giữa các văn bản."""
    result: dict[str, list[str]] = {
        "thay_the": [],
        "sua_doi":  [],
        "lien_quan": [],
    }

    # Tìm văn bản bị thay thế
    for pat in _THAY_THE_PATTERNS:
        for m in pat.finditer(text[:3000]):
            snippet = m.group(1).strip()
            sh_matches = _SO_HIEU_IN_TEXT.findall(snippet)
            result["thay_the"].extend(sh_matches)

    # Tìm văn bản bị sửa đổi
    for pat in _SUA_DOI_PATTERNS:
        for m in pat.finditer(text[:3000]):
            snippet = m.group(1).strip()
            sh_matches = _SO_HIEU_IN_TEXT.findall(snippet)
            result["sua_doi"].extend(sh_matches)

    # Tìm các số hiệu liên quan trong phần điều khoản chuyển tiếp (cuối văn bản)
    tail = text[-2000:]
    for sh in _SO_HIEU_IN_TEXT.findall(tail):
        if sh not in result["thay_the"] and sh not in result["sua_doi"]:
            result["lien_quan"].append(sh)

    # Dedup
    for key in result:
        result[key] = sorted(set(result[key]))

    return result


def read_header_fields(text: str) -> dict[str, str]:
    """Trích toàn bộ trường header từ file."""
    fields: dict[str, str] = {}
    for line in text.splitlines()[:25]:
        if ": " in line and not line.startswith("─"):
            key, _, val = line.partition(": ")
            key = key.strip()
            if re.match(r"^[A-Z_]+$", key):
                fields[key] = val.strip()
    return fields


def build_header(fields: dict[str, str]) -> str:
    """Xây dựng header string từ dict."""
    order = [
        "NGUON", "SO_HIEU", "TEN", "LOAI", "CO_QUAN",
        "NGAY_BAN_HANH", "HIEU_LUC", "URL", "CHU_DE", "LINH_VUC",
        "VAN_BAN_THAY_THE", "VAN_BAN_SUA_DOI", "VAN_BAN_LIEN_QUAN",
    ]
    lines = []
    for key in order:
        if key in fields and fields[key]:
            lines.append(f"{key}: {fields[key]}")
    # Thêm các field không có trong order
    for key, val in fields.items():
        if key not in order and val:
            lines.append(f"{key}: {val}")
    lines.append("─" * 60)
    return "\n".join(lines) + "\n\n"

# ─── Core processing ──────────────────────────────────────────────────────────

HEADER_SEP = re.compile(r"^[─\-]{20,}", re.MULTILINE)


def process_file(fp: Path, dry_run: bool = False) -> Optional[dict]:
    """
    Xử lý 1 file:
    - Thêm header nếu chưa có
    - Bổ sung LINH_VUC
    - Bổ sung VAN_BAN_THAY_THE, VAN_BAN_SUA_DOI, VAN_BAN_LIEN_QUAN
    Trả về dict thông tin để build relationship_map.
    """
    try:
        text = fp.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  [!] Không đọc được {fp.name}: {e}")
        return None

    changed = False

    # ── 1. Xử lý file chưa có header ──────────────────────────────────────────
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if not first_line.startswith("NGUON:"):
        known = KNOWN_OLD_FILES.get(fp.name)
        if known:
            print(f"  + Thêm header cho: {fp.name}")
            fields = {"NGUON": "vbpl.vn"}
            fields.update(known)
            content_body = text
            text = build_header(fields) + content_body
            changed = True
        else:
            # File không có header và không trong danh sách đã biết
            # Thử tự extract metadata từ nội dung
            linh_vuc = detect_linh_vuc(text[:2000])
            # Tìm số hiệu trong text
            sh_m = re.search(r"((?:Luật|Bộ luật|Nghị định|Thông tư)\s+số\s+[\d/A-ZĐ\-]+)", text[:500], re.I)
            so_hieu = sh_m.group(1) if sh_m else ""
            # Tìm tiêu đề
            title_lines = [l.strip() for l in text.splitlines()[:5] if len(l.strip()) > 10]
            ten = title_lines[0] if title_lines else fp.stem

            fields = {
                "NGUON":       "unknown",
                "SO_HIEU":     so_hieu,
                "TEN":         ten[:200],
                "LOAI":        "",
                "CO_QUAN":     "",
                "NGAY_BAN_HANH": "",
                "HIEU_LUC":    "",
                "URL":         "",
                "CHU_DE":      linh_vuc,
                "LINH_VUC":    linh_vuc,
            }
            text = build_header(fields) + text
            changed = True
            print(f"  + Thêm header tự động cho: {fp.name}")

    # ── 2. Bổ sung / cập nhật LINH_VUC ────────────────────────────────────────
    fields = read_header_fields(text)
    body_start = HEADER_SEP.search(text)
    body = text[body_start.end():].strip() if body_start else text

    if not fields.get("LINH_VUC"):
        lv = detect_linh_vuc(body[:3000])
        if lv:
            fields["LINH_VUC"] = lv
            changed = True

    # ── 3. Bổ sung quan hệ văn bản ────────────────────────────────────────────
    rels = extract_relationships(body)

    thay_the_str  = "; ".join(rels["thay_the"])  if rels["thay_the"]  else ""
    sua_doi_str   = "; ".join(rels["sua_doi"])   if rels["sua_doi"]   else ""
    lien_quan_str = "; ".join(rels["lien_quan"][:5]) if rels["lien_quan"] else ""  # tối đa 5

    if thay_the_str and not fields.get("VAN_BAN_THAY_THE"):
        fields["VAN_BAN_THAY_THE"] = thay_the_str
        changed = True

    if sua_doi_str and not fields.get("VAN_BAN_SUA_DOI"):
        fields["VAN_BAN_SUA_DOI"] = sua_doi_str
        changed = True

    if lien_quan_str and not fields.get("VAN_BAN_LIEN_QUAN"):
        fields["VAN_BAN_LIEN_QUAN"] = lien_quan_str
        changed = True

    # ── 4. Ghi file nếu có thay đổi ───────────────────────────────────────────
    if changed:
        new_header = build_header(fields)
        new_text   = new_header + body
        if not dry_run:
            fp.write_text(new_text, encoding="utf-8")

    return {
        "file":         fp.name,
        "so_hieu":      fields.get("SO_HIEU", ""),
        "ten":          fields.get("TEN", ""),
        "linh_vuc":     fields.get("LINH_VUC", ""),
        "hieu_luc":     fields.get("HIEU_LUC", ""),
        "thay_the":     rels["thay_the"],
        "sua_doi":      rels["sua_doi"],
        "lien_quan":    rels["lien_quan"],
        "changed":      changed,
    }


def build_relationship_map(records: list[dict], out_path: Path) -> None:
    """Xây dựng và lưu bản đồ quan hệ giữa các văn bản."""
    # Index theo số hiệu
    by_so_hieu: dict[str, dict] = {}
    for r in records:
        sh = r["so_hieu"]
        if sh:
            by_so_hieu[sh] = r

    # Build graph
    graph: list[dict] = []
    for r in records:
        if r["thay_the"] or r["sua_doi"] or r["lien_quan"]:
            graph.append({
                "van_ban":      r["so_hieu"] or r["file"],
                "ten":          r["ten"],
                "linh_vuc":     r["linh_vuc"],
                "thay_the":     r["thay_the"],
                "sua_doi":      r["sua_doi"],
                "lien_quan":    r["lien_quan"][:10],
            })

    out_data = {
        "tong_van_ban":    len(records),
        "co_quan_he":      len(graph),
        "relationship_graph": graph,
    }
    out_path.write_text(
        json.dumps(out_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  → Đã lưu relationship_map.json ({len(graph)} VB có quan hệ)")

# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Enrich metadata cho văn bản pháp luật")
    p.add_argument("--dry-run", action="store_true",
                   help="Chỉ phân tích, không sửa file")
    p.add_argument("--dir",    type=str, default=None,
                   help="Chỉ xử lý thư mục con cụ thể (vd: all_laws, nghi_dinh)")
    p.add_argument("--no-map", action="store_true",
                   help="Bỏ qua bước build relationship_map.json")
    args = p.parse_args()

    if args.dry_run:
        print("  [DRY RUN] Sẽ không sửa file nào.\n")

    # Xác định các thư mục cần xử lý
    if args.dir:
        dirs_to_scan = [RAW_DIR / args.dir]
    else:
        dirs_to_scan = [
            RAW_DIR,               # file root (bo_luat_hinh_su.txt, v.v.)
            RAW_DIR / "all_laws",
            RAW_DIR / "nghi_dinh",
            RAW_DIR / "thong_tu",
            RAW_DIR / "an_le",
        ]

    all_records: list[dict] = []
    total_changed = 0

    for scan_dir in dirs_to_scan:
        if not scan_dir.exists():
            continue

        # Lấy file .txt trực tiếp trong thư mục (không đệ quy)
        files = [f for f in scan_dir.iterdir() if f.is_file() and f.suffix == ".txt"]
        if not files:
            continue

        print(f"\n{'─'*60}")
        print(f"  Xử lý: {scan_dir.relative_to(RAW_DIR.parent.parent)} ({len(files)} files)")
        print(f"{'─'*60}")

        for fp in sorted(files):
            record = process_file(fp, dry_run=args.dry_run)
            if record:
                all_records.append(record)
                if record["changed"]:
                    total_changed += 1

    # ── Báo cáo ───────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  XONG. Đã xử lý {len(all_records)} file.")
    print(f"  Có thay đổi: {total_changed} file.")

    if args.dry_run:
        print("  [DRY RUN] Không có file nào bị sửa.")

    linh_vuc_count: dict[str, int] = {}
    for r in all_records:
        lv = r.get("linh_vuc", "unknown") or "unknown"
        linh_vuc_count[lv] = linh_vuc_count.get(lv, 0) + 1

    print("\n  Phân bổ theo lĩnh vực:")
    for lv, cnt in sorted(linh_vuc_count.items(), key=lambda x: -x[1]):
        print(f"    {lv:<20}: {cnt} văn bản")

    # ── Build relationship map ─────────────────────────────────────────────────

    if not args.no_map and all_records:
        map_path = RAW_DIR / "relationship_map.json"
        if not args.dry_run:
            build_relationship_map(all_records, map_path)
        else:
            rels_count = sum(
                1 for r in all_records
                if r["thay_the"] or r["sua_doi"] or r["lien_quan"]
            )
            print(f"\n  [DRY RUN] Sẽ build relationship_map.json ({rels_count} VB có quan hệ)")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()

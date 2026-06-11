"""Dọn corpus data/raw — loại văn bản địa phương giá trị thấp + khử trùng lặp.

Mục tiêu: dữ liệu sạch, tập trung văn bản quy phạm cấp trung ương:
  1. Loại NQ/QĐ/CT của HĐND & UBND tỉnh/huyện (điều hành địa phương, noise
     cho tư vấn pháp luật toàn quốc — lọt vào từ các lần crawl filter cũ)
  2. Khử trùng lặp theo (SO_HIEU + NGAY_BAN_HANH): cùng số hiệu VÀ cùng ngày
     ban hành → gần như chắc chắn cùng 1 văn bản (crawl từ 2 nguồn) → giữ bản
     tốt nhất (ưu tiên có metadata HIEU_LUC từ vbpl.vn, rồi file lớn hơn).
     LƯU Ý: không dedup theo SO_HIEU đơn thuần — số hiệu VN không duy nhất
     (VB cũ kiểu '115-CP' tái dùng nhiều năm, mỗi cơ quan đánh số riêng).

AN TOÀN: mặc định DRY-RUN (chỉ báo cáo). Chạy với --apply mới di chuyển file
sang data/raw_excluded/ (giữ nguyên cấu trúc thư mục — đảo ngược được bằng
cách move ngược lại, KHÔNG xóa gì).

Cách chạy:
    python -m scripts.clean_corpus            # dry-run, in báo cáo
    python -m scripts.clean_corpus --apply    # thực thi di chuyển
"""
from __future__ import annotations

import argparse
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
EXCLUDED_DIR = ROOT / "data" / "raw_excluded"

# Cơ quan địa phương → loại (normalized, lowercase, bỏ dấu)
LOCAL_AUTHORITY_KEYWORDS = [
    "hoi dong nhan dan",   # HĐND tỉnh/huyện
    "uy ban nhan dan",     # UBND các cấp
    "hdnd",
    "ubnd",
]

# Chỉ xét các folder có nguy cơ chứa VB địa phương
SCAN_FOLDERS = ["nghi_quyet", "quyet_dinh", "chi_thi"]


def normalize(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def parse_header(path: Path) -> dict[str, str]:
    """Đọc header key: value trong ~12 dòng đầu file."""
    try:
        head = path.open(encoding="utf-8", errors="replace").read(900)
    except Exception:
        return {}
    meta: dict[str, str] = {}
    for line in head.splitlines()[:12]:
        if ": " in line:
            k, _, v = line.partition(": ")
            meta[k.strip()] = v.strip()
    return meta


def is_local_authority(meta: dict) -> bool:
    auth = normalize(meta.get("CO_QUAN", ""))
    so_hieu = normalize(meta.get("SO_HIEU", ""))
    if any(kw in auth for kw in LOCAL_AUTHORITY_KEYWORDS):
        return True
    # SO_HIEU dạng '365/NQ-HĐND', '12/QĐ-UBND' khi CO_QUAN trống
    return "-hdnd" in so_hieu or "-ubnd" in so_hieu


def main() -> None:
    p = argparse.ArgumentParser(description="Dọn corpus: loại VB địa phương + khử trùng lặp")
    p.add_argument("--apply", action="store_true",
                   help="Thực thi di chuyển (mặc định dry-run chỉ báo cáo)")
    args = p.parse_args()

    to_exclude: list[tuple[Path, str]] = []  # (path, lý do)

    # ── 1. Văn bản địa phương trong các folder nguy cơ ────────────────────────
    print("=" * 70)
    print("  [1/2] Quét văn bản cơ quan địa phương (HĐND/UBND)...")
    local_by_folder: Counter = Counter()
    scanned = 0
    for base in (RAW_DIR, RAW_DIR / "hf_laws"):
        for folder_name in SCAN_FOLDERS:
            folder = base / folder_name
            if not folder.is_dir():
                continue
            for f in folder.rglob("*.txt"):
                scanned += 1
                if is_local_authority(parse_header(f)):
                    to_exclude.append((f, "địa phương"))
                    local_by_folder[str(folder.relative_to(RAW_DIR))] += 1
    print(f"  Đã quét {scanned:,} file:")
    for k, v in local_by_folder.most_common():
        print(f"    {k:30s}: {v:7,} VB địa phương")

    # ── 2. Trùng lặp (SO_HIEU, NGAY_BAN_HANH) ────────────────────────────────
    print("\n  [2/2] Quét trùng lặp theo (SO_HIEU + NGAY_BAN_HANH)...")
    excluded_set = {f for f, _ in to_exclude}
    by_key: dict[tuple[str, str], list[tuple[Path, dict]]] = defaultdict(list)
    n_files = 0
    for f in RAW_DIR.rglob("*.txt"):
        if f in excluded_set:
            continue
        n_files += 1
        meta = parse_header(f)
        sh = normalize(meta.get("SO_HIEU", ""))
        d  = (meta.get("NGAY_BAN_HANH") or "").strip()
        # Cả 2 trường phải có — thiếu ngày thì không dám kết luận trùng
        if sh and d:
            by_key[(sh, d)].append((f, meta))

    n_dup_groups = 0
    for (sh, d), entries in by_key.items():
        if len(entries) < 2:
            continue
        n_dup_groups += 1
        # Giữ bản tốt nhất: có HIEU_LUC (vbpl.vn) > file lớn hơn
        entries.sort(
            key=lambda e: (bool(e[1].get("HIEU_LUC", "").strip()),
                           e[0].stat().st_size),
            reverse=True,
        )
        for f, _ in entries[1:]:
            to_exclude.append((f, f"trùng ({sh[:25]}, {d})"))

    print(f"  Đã quét {n_files:,} file → {n_dup_groups:,} nhóm trùng (số hiệu + ngày)")

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    n_local = sum(1 for _, r in to_exclude if r == "địa phương")
    n_dup = len(to_exclude) - n_local
    print("\n" + "=" * 70)
    print(f"  TỔNG LOẠI: {len(to_exclude):,} file "
          f"(địa phương: {n_local:,} | trùng lặp: {n_dup:,})")
    remaining = sum(1 for _ in RAW_DIR.rglob('*.txt')) - len(to_exclude)
    print(f"  CÒN LẠI  : {remaining:,} file")
    print("=" * 70)

    if not args.apply:
        print("\nDRY-RUN — chưa di chuyển gì. Chạy lại với --apply để thực thi.")
        return

    # ── Apply: move sang data/raw_excluded (đảo ngược được) ──────────────────
    print(f"\nDi chuyển {len(to_exclude):,} file → {EXCLUDED_DIR} ...")
    moved = 0
    for f, _reason in to_exclude:
        rel = f.relative_to(RAW_DIR)
        dest = EXCLUDED_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(f), str(dest))
            moved += 1
            if moved % 5000 == 0:
                print(f"  ... {moved:,}/{len(to_exclude):,}")
        except Exception as exc:
            print(f"  [!] Lỗi move {f.name}: {exc}")
    print(f"✓ Xong — đã di chuyển {moved:,} file.")
    print("  Đảo ngược: move ngược nội dung data/raw_excluded/ về data/raw/")


if __name__ == "__main__":
    main()

"""Tải và xử lý dataset từ HuggingFace: vohuutridung/vietnamese-legal-documents.

Dataset gốc: 518,255 VB từ thuvienphapluat.vn (1924–2026), 3.6 GB content.
Script này:
  1. Download metadata + content (streaming để tiết kiệm RAM)
  2. Filter chỉ giữ loại VB có giá trị pháp lý cao (~80-120k VB)
  3. Dedup với data đã có (so sánh document_number)
  4. Convert sang .txt với header chuẩn của project
  5. Lưu vào data/raw/hf_laws/ (tách riêng để quản lý)
  6. Ghi progress manifest → resume được

Cách chạy:
    python -m scripts.load_hf_dataset                    # full (filter mặc định)
    python -m scripts.load_hf_dataset --dry-run          # đếm, không lưu
    python -m scripts.load_hf_dataset --max-docs 5000    # giới hạn số VB
    python -m scripts.load_hf_dataset --types luat nd tt # chỉ 1 số loại
    python -m scripts.load_hf_dataset --resume           # tiếp tục nếu bị dừng

Kết quả:
    data/raw/hf_laws/<loai>/<id>.txt
    data/raw/hf_manifest.json  ← id đã xử lý, resume được
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Cho phép download từ HuggingFace (override TRANSFORMERS_OFFLINE nếu có)
import os as _os
_os.environ["TRANSFORMERS_OFFLINE"] = "0"
_os.environ["HF_DATASETS_OFFLINE"]  = "0"

ROOT    = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
HF_DIR  = RAW_DIR / "hf_laws"
MANIFEST_PATH = RAW_DIR / "hf_manifest.json"

# ─── Loại VB quan trọng cần giữ ──────────────────────────────────────────────
# Key = chuỗi cần có trong legal_type (normalized, lowercase, bỏ dấu)
# Value = folder con để lưu

IMPORTANT_TYPES: dict[str, str] = {
    # Tầng Luật (Quốc hội)
    "hien phap":          "hien_phap",
    "bo luat":            "bo_luat",
    "luat":               "luat",
    "phap lenh":          "phap_lenh",
    "nghi quyet":         "nghi_quyet",
    "van ban hop nhat":   "van_ban_hop_nhat",  # văn bản hợp nhất

    # Tầng Chính phủ / Bộ
    "nghi dinh":          "nghi_dinh",
    "thong tu lien tich": "thong_tu",
    "thong tu":           "thong_tu",
    "chi thi":            "chi_thi",
}

# Quyết định: CHỈ giữ từ các cơ quan TW tầng cao nhất
# (nhiều QĐ cấp tỉnh/huyện → noise, không cần cho RAG pháp lý)
QD_KEEP_AUTHORITIES: list[str] = [
    "thu tuong chinh phu",  # Thủ tướng Chính phủ
    "chinh phu",            # Chính phủ (Chính phủ ký)
    "ngan hang nha nuoc",   # NHNN
    "hoi dong tham phan",   # HĐTP TAND Tối cao
    "tand toi cao",         # TAND Tối cao
    "vien kiem sat nhan dan toi cao",  # VKSND Tối cao
    "uy ban thuong vu quoc hoi",       # UBTVQH
    "hoi dong bau cu quoc gia",
]

# Chỉ thị: tương tự, chỉ từ cấp TW
CT_KEEP_AUTHORITIES: list[str] = [
    "thu tuong",
    "chinh phu",
    "bo truong",
    " bo ",
    "ngan hang nha nuoc",
    "tong cuc",
]

# Loại VB cần loại bỏ dứt khoát (nhiều, giá trị thấp)
EXCLUDED_TYPES: set[str] = {
    "cong van",         # Công văn
    "thong bao",        # Thông báo
    "thong cao",        # Thông cáo
    "ke hoach",         # Kế hoạch
    "bao cao",          # Báo cáo
    "to trinh",         # Tờ trình
    "bien ban",         # Biên bản
    "hop dong",         # Hợp đồng
    "van ban khac",     # Văn bản khác
    "huong dan",        # Hướng dẫn (thường là nội bộ)
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    # "đ/Đ" không NFD-decomposable → phải thay thủ công trước khi NFD
    s = s.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def get_folder(legal_type: str, authority: str = "") -> str | None:
    """Trả về folder nếu loại VB quan trọng, None nếu cần bỏ qua."""
    lt_norm  = normalize(legal_type or "")
    auth_norm = normalize(authority or "")

    # Kiểm tra loại bị loại trừ trước
    for excl in EXCLUDED_TYPES:
        if excl in lt_norm:
            return None

    # Quyết định: CHỈ TW tầng cao (Thủ tướng, CP, NHNN, TAND TC...)
    if "quyet dinh" in lt_norm:
        if any(kw in auth_norm for kw in QD_KEEP_AUTHORITIES):
            return "quyet_dinh"
        return None

    # Chỉ thị: chỉ từ Bộ trở lên
    if "chi thi" in lt_norm:
        if any(kw in auth_norm for kw in CT_KEEP_AUTHORITIES):
            return "chi_thi"
        return None

    # Các loại VB khác (Luật, NĐ, TT...): check IMPORTANT_TYPES
    for key, folder in IMPORTANT_TYPES.items():
        if key in lt_norm:
            return folder

    return None


_MULTI_NL = re.compile(r"\n{3,}")
_TRAIL_WS = re.compile(r"[ \t]+$", re.MULTILINE)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAIL_WS.sub("", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def safe_filename(s: str, max_len: int = 80) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\-.]", "_", s)
    return s[:max_len].strip("_")


def parse_date(raw: str) -> str:
    """Chuyển DD/MM/YYYY hoặc YYYY-MM-DD về YYYY-MM-DD."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    if re.match(r"\d{2}/\d{2}/\d{4}", raw):
        d, m, y = raw[:10].split("/")
        return f"{y}-{m}-{d}"
    return raw[:10]  # Giả sử đã là ISO


def build_header(row: dict, folder: str) -> str:
    sectors = (row.get("legal_sectors") or "").replace("|", " | ")
    signers = (row.get("signers") or "").split(":")[0]  # bỏ phần ID số
    header = (
        f"NGUON: thuvienphapluat.vn\n"
        f"SO_HIEU: {row.get('document_number') or ''}\n"
        f"TEN: {row.get('title') or ''}\n"
        f"LOAI: {row.get('legal_type') or ''}\n"
        f"CO_QUAN: {row.get('issuing_authority') or ''}\n"
        f"NGAY_BAN_HANH: {parse_date(row.get('issuance_date') or '')}\n"
        f"HIEU_LUC: \n"
        f"URL: {row.get('url') or ''}\n"
        f"CHU_DE: {folder}\n"
    )
    if sectors:
        header += f"LINH_VUC_GOC: {sectors}\n"
    if signers:
        header += f"NGUOI_KY: {signers[:100]}\n"
    header += "─" * 60 + "\n\n"
    return header

# ─── Manifest ─────────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"processed_ids": [], "stats": {}}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# ─── Build existing doc_numbers set để dedup ──────────────────────────────────

def build_existing_numbers() -> set[str]:
    """Thu thập SO_HIEU của tất cả file đã có để tránh trùng lặp."""
    existing: set[str] = set()
    for folder in RAW_DIR.iterdir():
        if not folder.is_dir() or folder.name in ("vectorstore", "processed", "hf_laws"):
            continue
        for fp in folder.glob("*.txt"):
            try:
                first_lines = fp.read_text(encoding="utf-8", errors="replace")[:300]
                m = re.search(r"^SO_HIEU:\s*(.+)$", first_lines, re.MULTILINE)
                if m:
                    existing.add(normalize(m.group(1).strip()))
            except Exception:
                pass
    # Cũng check root files
    for fp in RAW_DIR.glob("*.txt"):
        try:
            first_lines = fp.read_text(encoding="utf-8", errors="replace")[:300]
            m = re.search(r"^SO_HIEU:\s*(.+)$", first_lines, re.MULTILINE)
            if m:
                existing.add(normalize(m.group(1).strip()))
        except Exception:
            pass
    return existing

# ─── Core processing ──────────────────────────────────────────────────────────

def process_dataset(
    max_docs: int,
    type_filter: list[str] | None,
    dry_run: bool,
    resume: bool,
) -> dict:
    """Download + filter + convert toàn bộ dataset."""
    from datasets import load_dataset

    manifest   = load_manifest() if resume else {"processed_ids": [], "stats": {}}
    done_ids   = set(manifest.get("processed_ids", []))

    print(f"\n{'='*65}")
    print(f"  HuggingFace Dataset Loader — {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Đã xử lý trước: {len(done_ids):,} VB")
    print(f"{'='*65}\n")

    # ── Bước 1: Build dedup set từ existing data ───────────────────────────────
    if not dry_run:
        print("  [1/4] Đọc SO_HIEU đã có để tránh trùng lặp...")
        existing_numbers = build_existing_numbers()
        print(f"  → {len(existing_numbers):,} số hiệu đã có\n")
    else:
        existing_numbers = set()

    # ── Bước 2: Load metadata (nhỏ, ~82MB, load toàn bộ) ──────────────────────
    print("  [2/4] Load metadata (~82 MB)...")
    meta_ds = load_dataset(
        "vohuutridung/vietnamese-legal-documents",
        "metadata",
        split="data",
    )
    print(f"  → {len(meta_ds):,} VB trong metadata\n")

    # Filter theo loại VB
    print("  [3/4] Filter loại VB quan trọng...")
    accepted_rows: list[dict] = []
    type_stats: dict[str, int] = {}
    skip_stats: dict[str, int] = {}

    for row in meta_ds:
        if int(row["id"]) in done_ids:
            continue

        lt        = row.get("legal_type") or ""
        authority = row.get("issuing_authority") or ""
        folder    = get_folder(lt, authority)

        if folder is None:
            lt_key = normalize(lt)[:30] or "unknown"
            skip_stats[lt_key] = skip_stats.get(lt_key, 0) + 1
            continue

        # Filter user-specified types
        if type_filter:
            lt_norm = normalize(lt)
            if not any(normalize(t) in lt_norm for t in type_filter):
                continue

        # Dedup check
        doc_num = normalize(row.get("document_number") or "")
        if doc_num and doc_num in existing_numbers:
            continue

        type_stats[lt[:30]] = type_stats.get(lt[:30], 0) + 1
        accepted_rows.append({"row": row, "folder": folder})

        if len(accepted_rows) >= max_docs:
            break

    print(f"  → Chấp nhận: {len(accepted_rows):,} VB")
    print(f"  → Bỏ qua (loại thấp): {sum(skip_stats.values()):,} VB")
    print("\n  Phân bổ loại VB được chấp nhận:")
    for lt_name, cnt in sorted(type_stats.items(), key=lambda x: -x[1])[:15]:
        print(f"    {lt_name:<35}: {cnt:,}")

    if dry_run:
        print("\n  [DRY RUN] Không lưu file.")
        return {"accepted": len(accepted_rows), "type_stats": type_stats}

    if not accepted_rows:
        print("\n  Không có VB mới để xử lý.")
        return {}

    # ── Bước 4: Stream content từ cache local + save ─────────────────────────
    print(f"\n  [4/4] Stream content và lưu {len(accepted_rows):,} VB...")
    print(f"  Lưu vào  : {HF_DIR}/")
    print(f"  (Data đã cache local ~10GB — sẽ chạy nhanh từ disk)\n")

    # Build ID → meta map
    id_to_meta: dict[int, dict] = {
        int(item["row"]["id"]): item for item in accepted_rows
    }
    target_ids = set(id_to_meta.keys())

    # Load content từ cache (streaming để tiết kiệm RAM)
    content_ds = load_dataset(
        "vohuutridung/vietnamese-legal-documents",
        "content",
        split="data",
        streaming=True,
    )

    saved       = 0
    failed      = 0
    no_content  = 0
    scanned     = 0
    batch_size  = 500
    processed_ids_batch: list[int] = []
    t_start     = __import__("time").time()

    for content_row in content_ds:
        scanned += 1
        doc_id = int(content_row["id"])

        # Progress log mỗi 50,000 rows
        if scanned % 50_000 == 0:
            elapsed  = __import__("time").time() - t_start
            rate     = scanned / elapsed if elapsed > 0 else 1
            remain   = (518_255 - scanned) / rate
            pct_done = saved / len(accepted_rows) * 100 if accepted_rows else 0
            print(f"  Scan {scanned:,}/518k | "
                  f"Lưu {saved:,}/{len(accepted_rows):,} ({pct_done:.1f}%) | "
                  f"~{remain/60:.0f} phút còn lại | "
                  f"{rate:.0f} rows/s")

        if doc_id not in target_ids:
            continue

        meta_item = id_to_meta[doc_id]
        row    = meta_item["row"]
        folder = meta_item["folder"]

        content = clean_text(content_row.get("content") or "")
        if len(content) < 50:
            no_content += 1
            processed_ids_batch.append(doc_id)
            continue

        try:
            out_dir = HF_DIR / folder
            out_dir.mkdir(parents=True, exist_ok=True)
            header   = build_header(dict(row), folder)
            base     = safe_filename(row.get("document_number") or str(doc_id))
            out_path = out_dir / f"{base}.txt"
            if out_path.exists():
                out_path = out_dir / f"{base}_{doc_id}.txt"
            out_path.write_text(header + content, encoding="utf-8")
            saved += 1
            processed_ids_batch.append(doc_id)

            if saved % 1000 == 0:
                elapsed = __import__("time").time() - t_start
                print(f"  ✓ {saved:,}/{len(accepted_rows):,} VB | "
                      f"{elapsed/60:.0f} phút")
        except Exception as exc:
            print(f"  [!] Lỗi id={doc_id}: {exc}")
            failed += 1
            processed_ids_batch.append(doc_id)

        # Lưu manifest định kỳ
        if len(processed_ids_batch) >= batch_size:
            manifest["processed_ids"].extend(processed_ids_batch)
            manifest["stats"] = {
                "saved": saved, "failed": failed, "no_content": no_content,
                "last_updated": datetime.now().isoformat(),
            }
            save_manifest(manifest)
            processed_ids_batch = []

        # Dừng sớm nếu đủ
        if saved >= len(accepted_rows):
            print("  → Tìm đủ tất cả VB, dừng sớm.")
            break

    # Lưu batch cuối
    if processed_ids_batch:
        manifest["processed_ids"].extend(processed_ids_batch)
    manifest["stats"] = {
        "saved": saved, "failed": failed, "no_content": no_content,
        "total_accepted": len(accepted_rows),
        "last_updated": datetime.now().isoformat(),
    }
    save_manifest(manifest)

    return {"saved": saved, "failed": failed, "no_content": no_content}

# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Load + filter + convert HuggingFace Vietnamese Legal Dataset"
    )
    p.add_argument("--max-docs", type=int, default=999_999,
                   help="Số VB tối đa cần lưu (mặc định: tất cả)")
    p.add_argument("--types", nargs="+", default=None,
                   help="Chỉ lấy loại VB cụ thể, vd: --types luat 'nghi dinh'")
    p.add_argument("--dry-run",  action="store_true",
                   help="Chỉ đếm, không lưu file")
    p.add_argument("--resume",   action="store_true",
                   help="Tiếp tục từ lần chạy trước (dùng manifest)")
    p.add_argument("--no-dedup", action="store_true",
                   help="Bỏ qua kiểm tra trùng lặp với data đã có")
    args = p.parse_args()

    t0 = __import__("time").time()
    result = process_dataset(
        max_docs    = args.max_docs,
        type_filter = args.types,
        dry_run     = args.dry_run,
        resume      = args.resume,
    )

    elapsed = round((__import__("time").time() - t0) / 60, 1)

    print(f"\n{'='*65}")
    print(f"  HOÀN THÀNH — {elapsed} phút")
    if result:
        print(f"  Lưu thành công : {result.get('saved', 0):,} VB")
        print(f"  Không có content: {result.get('no_content', 0):,} VB")
        print(f"  Lỗi           : {result.get('failed', 0):,} VB")
    if not args.dry_run and result.get("saved", 0) > 0:
        print(f"\n  Bước tiếp theo (sau khi quyết định embedding):")
        print(f"    python -m scripts.enrich_metadata --dir hf_laws")
        print(f"    python -m scripts.analyze_data")
        print(f"    python -m scripts.ingest  # khi sẵn sàng embed")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()

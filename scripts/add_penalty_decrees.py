"""Tải có mục tiêu các Nghị định xử phạt giao thông còn thiếu trong corpus.

Các văn bản CẦN CÓ cho chatbot trả lời về mức phạt giao thông:
  - 100/2019/NĐ-CP  : Xử phạt VPHC giao thông đường bộ + đường sắt (chính)
  - 123/2021/NĐ-CP  : Sửa đổi, bổ sung NĐ 100/2019 (tăng mức phạt nồng độ cồn)
  - 168/2024/NĐ-CP  : Xử phạt giao thông mới nhất (thay thế NĐ 100)
  - 46/2016/NĐ-CP   : Xử phạt giao thông cũ (tham khảo)

Cách chạy:
    python -m scripts.add_penalty_decrees
    python -m scripts.add_penalty_decrees --ingest   # tự động chạy ingest sau khi tải
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import requests

# UTF-8 stdout
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

RAW_DIR       = Path(__file__).resolve().parent.parent / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "crawl_manifest.json"
API_URL       = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/all"
BASE_URL      = "https://vbpl.vn"
PAGE_SIZE     = 20

_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://vbpl.vn/",
    "Origin": "https://vbpl.vn",
}

# Danh sách Nghị định cần tải — theo keyword tìm kiếm và số hiệu cần khớp
TARGET_DECREES = [
    {
        "keyword": "168/2024/ND-CP xu phat giao thong",
        "so_hieu_contains": ["168/2024", "168/2024/NĐ-CP"],
        "label": "ND168_2024",
    },
    {
        "keyword": "100/2019/ND-CP xu phat hanh chinh giao thong duong bo",
        "so_hieu_contains": ["100/2019", "100/2019/NĐ-CP"],
        "label": "ND100_2019",
    },
    {
        "keyword": "123/2021/ND-CP sua doi bo sung nghi dinh 100",
        "so_hieu_contains": ["123/2021", "123/2021/NĐ-CP"],
        "label": "ND123_2021",
    },
    {
        "keyword": "xu phat vi pham hanh chinh linh vuc giao thong duong bo",
        "so_hieu_contains": ["100/2019", "123/2021", "168/2024"],
        "label": "giao_thong_phat",
        "type_codes": ["ND"],
        "max_docs": 10,
    },
]


def _save_doc(doc: dict) -> Path:
    """Lưu document vào data/raw/<label>/<so_hieu_safe>.txt."""
    label = doc.get("label", "penalty_decrees")
    out_dir = RAW_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tên file an toàn
    so_hieu_raw = doc.get("so_hieu", doc.get("id", "unknown"))
    safe = unicodedata.normalize("NFD", so_hieu_raw)
    safe = "".join(c for c in safe if unicodedata.category(c) != "Mn")
    safe = safe.replace("/", "_").replace("\\", "_").replace(" ", "_")
    safe = "".join(c for c in safe if c.isalnum() or c in ("_", "-"))[:60]

    out_path = out_dir / f"{safe}.txt"
    if out_path.exists():
        print(f"  [skip] {out_path.name} đã tồn tại")
        return out_path

    # Viết file theo format chuẩn của project
    lines = [
        f"NGUON: vbpl.vn",
        f"SO_HIEU: {doc.get('so_hieu', '')}",
        f"TEN: {doc.get('title', '')}",
        f"LOAI: {doc.get('type_name', 'Nghị định')}",
        f"CO_QUAN: {doc.get('agency', 'Chính phủ')}",
        f"NGAY_BAN_HANH: {doc.get('issue_date', '')}",
        f"HIEU_LUC: {doc.get('eff_status', '')}",
        f"URL: {doc.get('detail_url', '')}",
        "",
        "─" * 60,
        "",
        doc.get("content", ""),
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [saved] {out_path.name}  ({len(doc.get('content',''))} chars)")
    return out_path


def search_and_download(
    session: requests.Session,
    keyword: str,
    so_hieu_contains: list[str],
    label: str,
    type_codes: list[str] = None,
    max_docs: int = 5,
) -> list[Path]:
    """Tìm kiếm và tải các văn bản khớp với so_hieu_contains."""
    if type_codes is None:
        type_codes = ["ND", "NĐ", "TTLT"]  # Nghị định, Thông tư liên tịch

    print(f"\n>>> Tìm: '{keyword}'  (type={type_codes})")
    saved: list[Path] = []

    for page in range(1, 6):
        payload = {
            "keyword":     keyword,
            "matchMode":   "any_words",
            "optionDoc":   "all",
            "agencyLevel": "TRUNG_UONG",
            "pageSize":    PAGE_SIZE,
            "pageNumber":  page,
        }
        try:
            resp = session.post(API_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  [API lỗi trang {page}]: {exc}")
            break

        inner = data.get("data") or data
        items = inner.get("items") or []
        total = inner.get("total") or 0
        print(f"  Trang {page}: {len(items)} kết quả (total={total})")

        if not items:
            break

        for item in items:
            if not isinstance(item, dict):
                continue
            tc = (item.get("docType") or {}).get("code", "")
            if tc not in type_codes:
                continue

            doc_so_hieu = (item.get("docNum") or "").strip()
            title       = (item.get("title")  or "").strip()
            doc_abs     = (item.get("docAbs") or "").strip()
            doc_id      = str(item.get("id", ""))

            # Kiểm tra khớp số hiệu
            matched = any(
                s.lower().replace("/", "_") in doc_so_hieu.lower().replace("/", "_")
                or s.lower() in title.lower()
                for s in so_hieu_contains
            )
            if not matched:
                continue

            if len(doc_abs) < 200:
                print(f"  [skip-short] {doc_so_hieu}: {title[:50]} ({len(doc_abs)} chars)")
                continue

            type_name = (item.get("docType") or {}).get("name", "Nghị định")
            agency    = (item.get("agencyName") or "Chính phủ").strip()
            issue_dt  = (item.get("issueDate") or "")[:10]
            eff_name  = (item.get("effStatus") or {}).get("name", "")

            doc = {
                "id":         doc_id,
                "title":      title,
                "so_hieu":    doc_so_hieu,
                "type_name":  type_name,
                "agency":     agency,
                "issue_date": issue_dt,
                "eff_status": eff_name,
                "content":    doc_abs,
                "detail_url": f"{BASE_URL}/van-ban/chi-tiet/{doc_id}",
                "label":      label,
            }

            print(f"  [found] {doc_so_hieu}: {title[:60]}")
            path = _save_doc(doc)
            saved.append(path)
            if len(saved) >= max_docs:
                return saved

        if page * PAGE_SIZE >= total:
            break
        time.sleep(1.2)

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true",
                        help="Chạy ingest sau khi tải xong")
    args = parser.parse_args()

    print("=" * 60)
    print("  Tải Nghị định xử phạt giao thông")
    print("=" * 60)

    session = requests.Session()
    session.headers.update(_HEADERS)

    all_saved: list[Path] = []
    seen_so_hieu: set[str] = set()

    for target in TARGET_DECREES:
        paths = search_and_download(
            session,
            keyword=target["keyword"],
            so_hieu_contains=target["so_hieu_contains"],
            label=target["label"],
            type_codes=target.get("type_codes"),
            max_docs=target.get("max_docs", 5),
        )
        for p in paths:
            if p.name not in seen_so_hieu:
                seen_so_hieu.add(p.name)
                all_saved.append(p)
        time.sleep(1.5)

    print(f"\nTong: da tai {len(all_saved)} file")
    for p in all_saved:
        print(f"  {p}")

    if not all_saved:
        print("\n[WARN] Khong tim thay van ban nao!")
        print("  Co the API bi block hoac so hieu khong khop.")
        print("  Thu crawl thu cong tai: https://vbpl.vn/tim-van-ban")
        return

    if args.ingest:
        print("\n=== Chay ingest cho cac file moi ===")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "scripts.ingest", "--skip-existing"],
            capture_output=False,
        )
        print(f"Ingest exit code: {result.returncode}")


if __name__ == "__main__":
    main()

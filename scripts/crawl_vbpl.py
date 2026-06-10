"""Crawler vbpl.vn — REST API 2 bước (2025+).

API thay đổi so với phiên bản cũ:
  ✗ Cũ: search trả docAbs (full text) trực tiếp trong response
  ✓ Mới: search chỉ trả metadata → cần GET /doc/{id} để lấy full text
         Full text nằm tại: data.documentContent.content (HTML)

QUAN TRỌNG: Dùng src.vbpl_client.VBPLClient cho crawl mới.
Script này giữ lại cho tương thích, nhưng dùng VBPLClient bên dưới.

Cách chạy:
    python -m scripts.crawl_vbpl --topic all_laws      # HP+BL+Lu
    python -m scripts.crawl_vbpl --topic dat_dai       # chỉ 1 chủ đề
    python -m scripts.crawl_vbpl --dry-run             # xem list, không lưu
    python -m scripts.crawl_vbpl --max-docs 700        # giới hạn số VB

Nghị định & Thông tư:
    python -m scripts.run_supplement --only p1

Bước tiếp theo sau crawl:
    python -m scripts.post_crawl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

import requests

# UTF-8 fix for Windows console
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_URL   = "https://vbpl.vn"
API_URL    = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/all"
PAGE_SIZE  = 20
API_DELAY  = 1.5   # giây giữa mỗi request

RAW_DIR       = Path(__file__).resolve().parent.parent / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "crawl_manifest.json"

_HEADERS = {
    "Content-Type": "application/json",
    "Accept":       "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://vbpl.vn/",
    "Origin":  "https://vbpl.vn",
}

# ── Mode rộng: tất cả HP+BL+Lu (keyword rỗng → API trả về đúng thứ tự này) ──
# API trả về 600 văn bản HP/BL/Lu đầu tiên khi keyword = ""
ALL_LAWS_TOPIC: dict = {
    "label":      "Tất cả Hiến pháp + Bộ luật + Luật",
    "keywords":   [""],       # keyword rỗng → API sort HP/BL/Lu trước
    "type_codes": ["HP", "BL", "Lu"],
}

# 10 chủ đề theo từ khóa cụ thể (chế độ cũ)
TOPICS: dict[str, dict] = {
    "hien_phap": {
        "label":    "Hiến pháp",
        "keywords": ["Hien phap"],
        "type_codes": ["HP"],        # chỉ lấy Hiến pháp
    },
    "hinh_su": {
        "label":    "Bộ luật / Luật Hình sự",
        "keywords": ["Bo luat hinh su", "Luat hinh su"],
        "type_codes": ["BL", "Lu"],
    },
    "dan_su": {
        "label":    "Bộ luật / Luật Dân sự",
        "keywords": ["Bo luat dan su", "Luat dan su"],
        "type_codes": ["BL", "Lu"],
    },
    "lao_dong": {
        "label":    "Bộ luật / Luật Lao động",
        "keywords": ["Bo luat lao dong", "Luat lao dong"],
        "type_codes": ["BL", "Lu"],
    },
    "dat_dai": {
        "label":    "Luật Đất đai",
        "keywords": ["Luat dat dai"],
        "type_codes": ["Lu"],
    },
    "giao_thong": {
        "label":    "Luật Giao thông",
        "keywords": ["Luat giao thong duong bo", "Luat trat tu an toan giao thong"],
        "type_codes": ["Lu"],
    },
    "doanh_nghiep": {
        "label":    "Luật Doanh nghiệp",
        "keywords": ["Luat doanh nghiep"],
        "type_codes": ["Lu"],
    },
    "thue": {
        "label":    "Luật Thuế",
        "keywords": [
            "Luat thue gia tri gia tang",
            "Luat thue thu nhap doanh nghiep",
            "Luat thue thu nhap ca nhan",
            "Luat thue tieu thu dac biet",
        ],
        "type_codes": ["Lu"],
    },
    "bhxh": {
        "label":    "Luật BHXH & BHYT",
        "keywords": ["Luat bao hiem xa hoi", "Luat bao hiem y te"],
        "type_codes": ["Lu"],
    },
    "ngan_hang": {
        "label":    "Luật Ngân hàng",
        "keywords": ["Luat ngan hang nha nuoc", "Luat cac to chuc tin dung"],
        "type_codes": ["Lu"],
    },
}

# ─── Manifest ─────────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# ─── Text helpers ──────────────────────────────────────────────────────────────

_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAIL_WS    = re.compile(r"[ \t]+$", re.MULTILINE)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAIL_WS.sub("", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def safe_filename(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\-.]", "_", s)
    return s[:80].strip("_")

# ─── REST API ─────────────────────────────────────────────────────────────────

def _parse_item(item: dict, label: str) -> Optional[dict]:
    """
    Chuyển 1 item từ API thành dict nội bộ.
    Trả None nếu thiếu nội dung.
    """
    doc_id  = str(item.get("id", ""))
    title   = (item.get("title") or "").strip()
    doc_abs = (item.get("docAbs") or "").strip()
    doc_num = (item.get("docNum") or "").strip()

    # docNum đôi khi là "Không số" — dùng title làm fallback filename
    so_hieu = doc_num if doc_num and doc_num.lower() not in ("không số", "khong so", "") else ""

    # Metadata bổ sung
    doc_type_obj = item.get("docType") or {}
    type_code    = doc_type_obj.get("code", "")
    type_name    = doc_type_obj.get("name", "")
    agency       = (item.get("agencyName") or "").strip()
    issue_date   = (item.get("issueDate") or "")[:10]
    eff_status   = (item.get("effStatus") or {}).get("name", "")

    if not doc_id or not title or len(doc_abs) < 100:
        return None

    detail_url = f"{BASE_URL}/van-ban/chi-tiet/{doc_id}"

    return {
        "id":          doc_id,
        "title":       title,
        "so_hieu":     so_hieu,
        "type_code":   type_code,
        "type_name":   type_name,
        "agency":      agency,
        "issue_date":  issue_date,
        "eff_status":  eff_status,
        "content":     doc_abs,
        "detail_url":  detail_url,
        "label":       label,
    }


def search_api(
    keyword: str,
    type_codes: list[str],
    max_docs: int,
    session: requests.Session,
) -> list[dict]:
    """
    Phân trang REST API, trả về list doc dicts (đã parse).
    Lọc theo type_codes nếu API không hỗ trợ lọc trực tiếp.
    Dừng sớm sau 3 trang liên tiếp không có kết quả mới.
    """
    results: list[dict] = []
    seen_ids: set[str] = set()
    max_pages = max((max_docs // PAGE_SIZE) + 5, 10)
    consecutive_empty = 0

    for page_num in range(1, max_pages + 1):
        payload = {
            "keyword":     keyword,
            "matchMode":   "all_words",
            "optionDoc":   "title",
            "agencyLevel": "TRUNG_UONG",
            "pageSize":    PAGE_SIZE,
            "pageNumber":  page_num,
        }

        try:
            resp = session.post(API_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"    [!] API error (trang {page_num}): {exc}")
            break

        inner = data.get("data") or data
        items = inner.get("items") or []
        total = inner.get("total") or 0

        if not items:
            break

        new_this_page = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            tc = (item.get("docType") or {}).get("code", "")
            if type_codes and tc not in type_codes:
                continue
            parsed = _parse_item(item, "")
            if parsed and parsed["id"] not in seen_ids:
                seen_ids.add(parsed["id"])
                results.append(parsed)
                new_this_page += 1

        fetched = page_num * PAGE_SIZE
        print(f"      Trang {page_num}/{(total//PAGE_SIZE)+1}: "
              f"+{new_this_page} mới, tổng {len(results)} (API total: {total})")

        # Early stop: 3 trang liên tiếp không thêm được gì
        if new_this_page == 0:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                print(f"      → Dừng sớm (3 trang liên tiếp trống)")
                break
        else:
            consecutive_empty = 0

        if len(results) >= max_docs or fetched >= total:
            break

        time.sleep(API_DELAY)

    return results[:max_docs]

# ─── Core crawl ───────────────────────────────────────────────────────────────

def crawl_topic(
    topic_key: str,
    cfg: dict,
    manifest: dict,
    dry_run: bool = False,
    max_docs: int = 100,
) -> int:
    """Crawl 1 topic bằng VBPLClient (API 2 bước: search → GET detail)."""
    from src.vbpl_client import VBPLClient, safe_filename as _safe

    label      = cfg["label"]
    type_codes = cfg.get("type_codes", [])
    out_dir    = RAW_DIR / topic_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Chủ đề : {label}")
    print(f"  Loại   : {type_codes or 'tất cả'}")
    print(f"{'='*60}")

    client = VBPLClient()
    per_kw = max(max_docs // max(len(cfg["keywords"]), 1), 20)

    all_meta: list[dict] = []
    seen_ids: set[str]   = set()

    for keyword in cfg["keywords"]:
        print(f"\n  Tìm: \"{keyword}\"")
        results = client.search(keyword, type_codes, per_kw,
                                agency_level="TRUNG_UONG", max_empty_pages=5)
        for m in results:
            if m["id"] not in seen_ids:
                seen_ids.add(m["id"])
                all_meta.append(m)
        time.sleep(API_DELAY)

    all_meta = all_meta[:max_docs]
    print(f"\n  Tổng {len(all_meta)} văn bản (dedup, giới hạn {max_docs})")

    if dry_run:
        for m in all_meta[:30]:
            print(f"    [{m['type_code']:3}] {m['so_hieu'] or '?':25} {m['title'][:40]}")
        return 0

    if not all_meta:
        print("  [!] Không tìm thấy văn bản")
        return 0

    downloaded = 0
    for i, meta in enumerate(all_meta, 1):
        mkey = f"{topic_key}/{meta['id']}"
        if manifest.get(mkey) == "ok":
            continue

        # Lấy full text từ detail endpoint
        detail = client.fetch_with_content(meta["id"])
        if not detail or not detail.get("content") or len(detail["content"]) < 100:
            manifest[mkey] = "skip"
            save_manifest(manifest)
            continue

        file_content = VBPLClient.build_file(detail, label)
        base     = _safe(detail["so_hieu"] or detail["title"])
        out_path = out_dir / (base + ".txt")
        if out_path.exists():
            out_path = out_dir / f"{base}_{meta['id']}.txt"
        out_path.write_text(file_content, encoding="utf-8")

        manifest[mkey] = "ok"
        save_manifest(manifest)
        downloaded += 1
        print(f"  [{i}/{len(all_meta)}] ✓ {out_path.name}  ({detail['eff_status']})")

    return downloaded

# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    topic_choices = list(TOPICS) + ["all", "all_laws"]
    p = argparse.ArgumentParser(
        description="Crawler vbpl.vn — REST API (khong can browser)"
    )
    p.add_argument("--topic",    choices=topic_choices, default="all_laws",
                   help="all_laws=tat ca HP+BL+Lu (~600 VB), all=10 chu de, hoac ten chu de cu the")
    p.add_argument("--dry-run",  action="store_true", help="Liet ke, khong luu file")
    p.add_argument("--max-docs", type=int, default=700,
                   help="So van ban toi da (default: 700 cho all_laws, 100 cho chu de cu the)")
    p.add_argument("--delay",    type=float, default=API_DELAY,
                   help="Delay giua cac API request (giay)")
    return p.parse_args()


def main() -> None:
    global API_DELAY
    args = parse_args()
    API_DELAY = args.delay

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    total = 0

    if args.topic == "all_laws":
        # Mode rong: tat ca HP+BL+Lu bang keyword rong
        count = crawl_topic(
            "all_laws", ALL_LAWS_TOPIC, manifest,
            dry_run=args.dry_run,
            max_docs=args.max_docs,
        )
        total += count
    elif args.topic == "all":
        for key, cfg in TOPICS.items():
            max_per = min(args.max_docs, 100)
            count = crawl_topic(key, cfg, manifest,
                                dry_run=args.dry_run, max_docs=max_per)
            total += count
    else:
        count = crawl_topic(
            args.topic, TOPICS[args.topic], manifest,
            dry_run=args.dry_run,
            max_docs=args.max_docs,
        )
        total += count

    if not args.dry_run:
        print(f"\n{'='*60}")
        print(f"  XONG. Tong {total} van ban da luu vao data/raw/")
        if total > 0:
            print("  Buoc tiep theo: python -m scripts.ingest")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

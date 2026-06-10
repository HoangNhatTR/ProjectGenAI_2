"""Crawler Nghị định & Thông tư từ vbpl.vn REST API.

Dùng src.vbpl_client.VBPLClient (API 2 bước: search → GET /doc/{id}).

Cách chạy:
    python -m scripts.crawl_nghi_dinh                     # tất cả lĩnh vực
    python -m scripts.crawl_nghi_dinh --topic xu_phat     # chỉ NĐ xử phạt
    python -m scripts.crawl_nghi_dinh --max-docs 200      # giới hạn mỗi topic
    python -m scripts.crawl_nghi_dinh --dry-run           # xem list, không lưu

Pipeline đầy đủ (khuyến nghị):
    python -m scripts.run_supplement --only p1            # NĐ+TT + Án lệ
    python -m scripts.post_crawl                          # enrich+ingest+bm25

Kết quả:
    data/raw/nghi_dinh/<so_hieu>.txt
    data/raw/crawl_manifest.json  ← resume được
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

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_URL  = "https://vbpl.vn"
API_URL   = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/all"
PAGE_SIZE = 20
API_DELAY = 1.5

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

# ─── Topic definitions ────────────────────────────────────────────────────────
# Mỗi topic: keywords để tìm kiếm + type_codes để lọc + folder lưu

TOPICS: dict[str, dict] = {
    # ── Nghị định xử phạt hành chính ─────────────────────────────────────────
    "xu_phat": {
        "label":      "Nghị định xử phạt vi phạm hành chính",
        "keywords":   [
            "xu phat vi pham hanh chinh giao thong",
            "xu phat vi pham hanh chinh dat dai",
            "xu phat vi pham hanh chinh thue",
            "xu phat vi pham hanh chinh lao dong",
            "xu phat vi pham hanh chinh doanh nghiep",
            "xu phat vi pham hanh chinh moi truong",
            "xu phat vi pham hanh chinh xay dung",
            "xu phat vi pham hanh chinh y te",
            "xu phat vi pham hanh chinh",
        ],
        "type_codes": ["ND"],
        "folder":     "nghi_dinh",
    },

    # ── Nghị định đất đai ─────────────────────────────────────────────────────
    "dat_dai": {
        "label":      "Nghị định hướng dẫn Luật Đất đai",
        "keywords":   [
            "huong dan luat dat dai",
            "boi thuong ho tro tai dinh cu",
            "cap giay chung nhan quyen su dung dat",
            "thu hoi dat",
            "giao dat cho thue dat",
            "quy hoach su dung dat",
            "dang ky dat dai",
        ],
        "type_codes": ["ND", "TT", "TTLT"],
        "folder":     "nghi_dinh",
    },

    # ── Nghị định doanh nghiệp & đầu tư ──────────────────────────────────────
    "doanh_nghiep": {
        "label":      "Nghị định doanh nghiệp, đầu tư, kinh doanh",
        "keywords":   [
            "huong dan luat doanh nghiep",
            "dang ky doanh nghiep",
            "von dieu le",
            "huong dan luat dau tu",
            "khu cong nghiep khu kinh te",
            "dau tu nuoc ngoai",
            "hop dong kinh te",
        ],
        "type_codes": ["ND", "TT"],
        "folder":     "nghi_dinh",
    },

    # ── Nghị định thuế ────────────────────────────────────────────────────────
    "thue": {
        "label":      "Nghị định & Thông tư thuế",
        "keywords":   [
            "huong dan luat thue gia tri gia tang",
            "huong dan luat thue thu nhap doanh nghiep",
            "huong dan luat thue thu nhap ca nhan",
            "hoan thue",
            "mien giam thue",
            "khai thue nop thue",
            "hoa don chung tu",
            "kiem tra thue thanh tra thue",
        ],
        "type_codes": ["ND", "TT", "TTLT"],
        "folder":     "nghi_dinh",
    },

    # ── Thông tư lao động & BHXH ──────────────────────────────────────────────
    "lao_dong": {
        "label":      "Nghị định & Thông tư lao động, BHXH, BHYT",
        "keywords":   [
            "huong dan luat lao dong",
            "hop dong lao dong",
            "luong toi thieu vung",
            "bao hiem xa hoi",
            "bao hiem y te",
            "bao hiem that nghiep",
            "an toan lao dong ve sinh lao dong",
        ],
        "type_codes": ["ND", "TT", "TTLT"],
        "folder":     "nghi_dinh",
    },

    # ── Nghị định hình sự & tố tụng ───────────────────────────────────────────
    "hinh_su": {
        "label":      "Nghị định hình sự, thi hành án hình sự",
        "keywords":   [
            "thi hanh an hinh su",
            "giam sat giao duc",
            "dac xa an xa",
            "truoc khi giam",
            "chap hanh hinh phat",
        ],
        "type_codes": ["ND", "TT"],
        "folder":     "nghi_dinh",
    },

    # ── Nghị định dân sự & hôn nhân gia đình ──────────────────────────────────
    "dan_su": {
        "label":      "Nghị định dân sự, hôn nhân gia đình, thừa kế",
        "keywords":   [
            "ho tich",
            "hon nhan gia dinh",
            "nuoi con nuoi",
            "giam ho",
            "thua ke",
            "cong chung",
            "thi hanh an dan su",
        ],
        "type_codes": ["ND", "TT", "TTLT"],
        "folder":     "nghi_dinh",
    },

    # ── Nghị định ngân hàng & tín dụng ────────────────────────────────────────
    "ngan_hang": {
        "label":      "Nghị định ngân hàng, tín dụng, ngoại hối",
        "keywords":   [
            "to chuc tin dung",
            "cho vay ngan hang",
            "lai suat",
            "ngoai hoi ngoai te",
            "thanh toan khong dung tien mat",
            "phuong tien thanh toan",
        ],
        "type_codes": ["ND", "TT"],
        "folder":     "nghi_dinh",
    },

    # ── Nghị định nhà ở & BĐS ────────────────────────────────────────────────
    "nha_o": {
        "label":      "Nghị định nhà ở, bất động sản",
        "keywords":   [
            "huong dan luat nha o",
            "quan ly su dung nha o",
            "mua ban nha o",
            "kinh doanh bat dong san",
            "chung cu",
            "nha xa hoi",
        ],
        "type_codes": ["ND", "TT"],
        "folder":     "nghi_dinh",
    },

    # ── Thông tư hướng dẫn tố tụng ────────────────────────────────────────────
    "to_tung": {
        "label":      "Thông tư tố tụng dân sự, hình sự",
        "keywords":   [
            "to tung dan su",
            "to tung hinh su",
            "khieu nai to cao",
            "hanh chinh",
            "toa an",
            "vien kiem sat",
        ],
        "type_codes": ["TT", "TTLT"],
        "folder":     "thong_tu",
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

# ─── Helpers ──────────────────────────────────────────────────────────────────

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

# ─── API ──────────────────────────────────────────────────────────────────────

def _parse_item(item: dict) -> Optional[dict]:
    doc_id   = str(item.get("id", ""))
    title    = (item.get("title") or "").strip()
    doc_abs  = (item.get("docAbs") or "").strip()
    doc_num  = (item.get("docNum") or "").strip()
    so_hieu  = doc_num if doc_num and doc_num.lower() not in ("không số", "khong so", "") else ""

    doc_type_obj = item.get("docType") or {}
    type_code    = doc_type_obj.get("code", "")
    type_name    = doc_type_obj.get("name", "")
    agency       = (item.get("agencyName") or "").strip()
    issue_date   = (item.get("issueDate") or "")[:10]
    eff_status   = (item.get("effStatus") or {}).get("name", "")

    if not doc_id or not title or len(doc_abs) < 100:
        return None

    return {
        "id":         doc_id,
        "title":      title,
        "so_hieu":    so_hieu,
        "type_code":  type_code,
        "type_name":  type_name,
        "agency":     agency,
        "issue_date": issue_date,
        "eff_status": eff_status,
        "content":    doc_abs,
        "url":        f"{BASE_URL}/van-ban/chi-tiet/{doc_id}",
    }


def search_api(
    keyword: str,
    type_codes: list[str],
    max_docs: int,
    session: requests.Session,
) -> list[dict]:
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

        new_count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            tc = (item.get("docType") or {}).get("code", "")
            if type_codes and tc not in type_codes:
                continue
            parsed = _parse_item(item)
            if parsed and parsed["id"] not in seen_ids:
                seen_ids.add(parsed["id"])
                results.append(parsed)
                new_count += 1

        fetched = page_num * PAGE_SIZE
        print(f"      Trang {page_num}/{(total//PAGE_SIZE)+1}: "
              f"+{new_count} mới, tổng {len(results)} (API total: {total})")

        if new_count == 0:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                print("      → Dừng sớm (3 trang liên tiếp trống)")
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
    label      = cfg["label"]
    type_codes = cfg["type_codes"]
    folder     = cfg["folder"]
    out_dir    = RAW_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Lĩnh vực : {label}")
    print(f"  Loại VB  : {type_codes}")
    print(f"  Lưu vào  : data/raw/{folder}/")
    print(f"{'='*60}")

    sess = requests.Session()
    sess.headers.update(_HEADERS)

    all_docs: list[dict] = []
    per_kw = max(max_docs // max(len(cfg["keywords"]), 1), 20)

    for keyword in cfg["keywords"]:
        print(f"\n  Tìm: \"{keyword}\"")
        docs = search_api(keyword, type_codes, per_kw, sess)
        all_docs.extend(docs)
        time.sleep(API_DELAY)

    # Dedupe
    seen: set[str] = set()
    unique: list[dict] = []
    for d in all_docs:
        if d["id"] not in seen:
            seen.add(d["id"])
            unique.append(d)
    unique = unique[:max_docs]

    print(f"\n  Tổng {len(unique)} văn bản (sau dedup, giới hạn {max_docs})")

    if dry_run:
        for d in unique[:30]:
            print(f"    [{d['type_code']:6}] {d['so_hieu'] or '?':30} "
                  f"{d['title'][:50]}  ({d['eff_status']})")
        if len(unique) > 30:
            print(f"    ... và {len(unique)-30} văn bản khác")
        return 0

    if not unique:
        print("  [!] Không tìm thấy văn bản")
        return 0

    downloaded = 0
    for i, doc in enumerate(unique, 1):
        mkey = f"nd_tt/{topic_key}/{doc['id']}"
        if manifest.get(mkey) == "ok":
            print(f"  [{i}/{len(unique)}] Đã có: {doc['so_hieu'] or doc['id']} — bỏ qua")
            continue

        text = clean_text(doc["content"])
        if len(text) < 100:
            manifest[mkey] = "skip"
            save_manifest(manifest)
            continue

        header = (
            f"NGUON: vbpl.vn\n"
            f"SO_HIEU: {doc['so_hieu']}\n"
            f"TEN: {doc['title']}\n"
            f"LOAI: {doc['type_name']} ({doc['type_code']})\n"
            f"CO_QUAN: {doc['agency']}\n"
            f"NGAY_BAN_HANH: {doc['issue_date']}\n"
            f"HIEU_LUC: {doc['eff_status']}\n"
            f"URL: {doc['url']}\n"
            f"CHU_DE: {label}\n"
            f"{'─'*60}\n\n"
        )

        base     = safe_filename(doc["so_hieu"] or doc["title"])
        fname    = base + ".txt"
        out_path = out_dir / fname
        if out_path.exists():
            fname    = f"{base}_{doc['id']}.txt"
            out_path = out_dir / fname

        out_path.write_text(header + text, encoding="utf-8")
        manifest[mkey] = "ok"
        save_manifest(manifest)
        downloaded += 1
        print(f"  [{i}/{len(unique)}] ✓ {out_path.name}  ({doc['eff_status']})")

    return downloaded

# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crawler Nghị định & Thông tư từ vbpl.vn"
    )
    p.add_argument(
        "--topic",
        choices=list(TOPICS) + ["all"],
        default="all",
        help="Lĩnh vực cụ thể hoặc 'all' (mặc định)",
    )
    p.add_argument("--dry-run",  action="store_true")
    p.add_argument("--max-docs", type=int, default=150,
                   help="Số VB tối đa mỗi topic (mặc định: 150)")
    p.add_argument("--delay",    type=float, default=API_DELAY)
    return p.parse_args()


def main() -> None:
    global API_DELAY
    args = parse_args()
    API_DELAY = args.delay

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    topics_to_run = TOPICS.items() if args.topic == "all" else [(args.topic, TOPICS[args.topic])]
    total = 0

    for key, cfg in topics_to_run:
        n = crawl_topic(key, cfg, manifest, dry_run=args.dry_run, max_docs=args.max_docs)
        total += n

    if not args.dry_run:
        print(f"\n{'='*60}")
        print(f"  XONG. Tổng {total} văn bản dưới luật đã lưu.")
        if total > 0:
            print("  Bước tiếp: python -m scripts.ingest")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""Pipeline P1 + P2: crawl văn bản dưới luật + luật còn thiếu.

P2 — Luật còn thiếu + Nghị quyết HĐTP:
  1. Tìm thêm HP+BL+Lu (tăng max lên 800)
  2. Targeted search cho 7 luật cụ thể đang thiếu
  3. Crawl Nghị quyết HĐTP (type NQ)

P1 — Văn bản dưới luật:
  4. Nghị định & Thông tư (9 lĩnh vực)
  5. Án lệ (TAND Tối cao)

Sử dụng VBPLClient 2 bước: search → GET /doc/{id} lấy full text.

Cách chạy:
    python -m scripts.run_supplement                   # full P1+P2
    python -m scripts.run_supplement --only p2         # chỉ P2
    python -m scripts.run_supplement --only p1         # chỉ P1
    python -m scripts.run_supplement --dry-run         # xem list, không tải

Log: data/supplement_run.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src.vbpl_client import VBPLClient, safe_filename

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
LOG_PATH = ROOT / "data" / "supplement_run.log"

# ─── Cấu hình P2 ──────────────────────────────────────────────────────────────

# 7 luật cụ thể còn thiếu — dùng keyword tiếng Việt + doc_id trực tiếp nếu biết
# API mới 2025: keyword tiếng Việt matching tốt hơn ASCII
MISSING_LAWS = [
    # doc_id biết trước → bypass search, gọi thẳng detail endpoint
    {"doc_id": "32801", "tc": ["HP"],  "label": "Hiến pháp 2013"},
    # Các luật còn lại: search bằng keyword tiếng Việt
    {"kw": "Hộ tịch",                    "tc": ["Lu"],  "label": "Luật Hộ tịch 2014"},
    {"kw": "Thi hành án hình sự",         "tc": ["Lu"],  "label": "Luật Thi hành án Hình sự"},
    {"kw": "Tài nguyên nước",             "tc": ["Lu"],  "label": "Luật Tài nguyên nước"},
    {"kw": "Thuế thu nhập doanh nghiệp",  "tc": ["Lu"],  "label": "Luật Thuế TNDN"},
    {"kw": "Thuế thu nhập cá nhân",       "tc": ["Lu"],  "label": "Luật Thuế TNCN"},
    {"kw": "Cạnh tranh",                  "tc": ["Lu"],  "label": "Luật Cạnh tranh"},
]

# Nghị quyết HĐTP — keyword tiếng Việt
NQ_SEARCHES = [
    {"kw": "hướng dẫn áp dụng pháp luật",    "tc": ["NQ"], "label": "NQ hướng dẫn áp dụng PL"},
    {"kw": "Hội đồng Thẩm phán",              "tc": ["NQ"], "label": "NQ HĐTP TAND Tối cao"},
    {"kw": "giải thích pháp luật",            "tc": ["NQ"], "label": "NQ giải thích pháp luật"},
]

# ─── Cấu hình P1 — Nghị định & Thông tư ──────────────────────────────────────

ND_TT_TOPICS: dict[str, dict] = {
    # API mới 2025: dùng keyword tiếng Việt, không filter type (NĐ/TT nằm sâu hơn BL/Lu)
    "xu_phat": {
        "label":    "Nghị định xử phạt vi phạm hành chính",
        "keywords": [
            "xử phạt vi phạm hành chính giao thông đường bộ",
            "xử phạt vi phạm hành chính đất đai",
            "xử phạt vi phạm hành chính thuế",
            "xử phạt vi phạm hành chính lao động bảo hiểm",
            "xử phạt vi phạm hành chính môi trường",
            "xử phạt vi phạm hành chính xây dựng",
        ],
        "type_codes": [],   # không filter — để NĐ/TT nổi lên trong results
        "folder":    "nghi_dinh",
    },
    "dat_dai": {
        "label":    "Nghị định hướng dẫn Luật Đất đai",
        "keywords": [
            "hướng dẫn Luật Đất đai bồi thường hỗ trợ tái định cư",
            "cấp giấy chứng nhận quyền sử dụng đất",
            "quy hoạch sử dụng đất",
            "đăng ký bất động sản",
            "Nghị định đất đai giao đất cho thuê đất",
        ],
        "type_codes": [],
        "folder":    "nghi_dinh",
    },
    "doanh_nghiep": {
        "label":    "Nghị định doanh nghiệp, đầu tư",
        "keywords": [
            "hướng dẫn Luật Doanh nghiệp đăng ký kinh doanh",
            "hướng dẫn Luật Đầu tư khu công nghiệp",
            "đầu tư nước ngoài góp vốn mua cổ phần",
        ],
        "type_codes": [],
        "folder":    "nghi_dinh",
    },
    "thue": {
        "label":    "Nghị định & Thông tư thuế",
        "keywords": [
            "hướng dẫn Luật Thuế giá trị gia tăng",
            "hướng dẫn Luật Thuế thu nhập doanh nghiệp",
            "hướng dẫn Luật Thuế thu nhập cá nhân",
            "hoá đơn chứng từ điện tử",
            "hoàn thuế giá trị gia tăng",
        ],
        "type_codes": [],
        "folder":    "nghi_dinh",
    },
    "lao_dong": {
        "label":    "Nghị định lao động, BHXH, BHYT",
        "keywords": [
            "hướng dẫn Bộ luật Lao động hợp đồng lao động",
            "lương tối thiểu vùng",
            "bảo hiểm xã hội bắt buộc",
            "bảo hiểm y tế",
            "an toàn vệ sinh lao động",
        ],
        "type_codes": [],
        "folder":    "nghi_dinh",
    },
    "hinh_su": {
        "label":    "Nghị định hình sự, thi hành án hình sự",
        "keywords": [
            "thi hành án hình sự giám sát giáo dục",
            "đặc xá ân xá chấp hành hình phạt tù",
        ],
        "type_codes": [],
        "folder":    "nghi_dinh",
    },
    "dan_su": {
        "label":    "Nghị định dân sự, hôn nhân gia đình",
        "keywords": [
            "hộ tịch đăng ký khai sinh kết hôn",
            "nuôi con nuôi",
            "công chứng chứng thực",
            "thi hành án dân sự",
        ],
        "type_codes": [],
        "folder":    "nghi_dinh",
    },
    "ngan_hang": {
        "label":    "Nghị định ngân hàng, tài chính",
        "keywords": [
            "tổ chức tín dụng cho vay lãi suất",
            "ngoại hối ngoại tệ thanh toán",
        ],
        "type_codes": [],
        "folder":    "nghi_dinh",
    },
    "nha_o": {
        "label":    "Nghị định nhà ở, BĐS",
        "keywords": [
            "hướng dẫn Luật Nhà ở quản lý sử dụng",
            "kinh doanh bất động sản nhà ở xã hội",
        ],
        "type_codes": [],
        "folder":    "nghi_dinh",
    },
}

# ─── Manifest helpers ─────────────────────────────────────────────────────────

def load_manifest() -> dict:
    mp = RAW_DIR / "crawl_manifest.json"
    if mp.exists():
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_manifest(manifest: dict) -> None:
    mp = RAW_DIR / "crawl_manifest.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

# ─── Logging ──────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ─── Core: crawl 1 keyword/topic ──────────────────────────────────────────────

def crawl_and_save(
    client: VBPLClient,
    keyword: str,
    type_codes: list[str],
    max_docs: int,
    out_dir: Path,
    label: str,
    manifest: dict,
    mkey_prefix: str,
    dry_run: bool,
    agency_level: str = "",
    max_empty_pages: int = 10,
) -> int:
    """Search + fetch content + save. Trả về số VB mới."""
    tc_str = ", ".join(type_codes) if type_codes else "tất cả loại"
    log(f"  Tìm: \"{keyword}\" [{tc_str}]")
    meta_list = client.search(keyword, type_codes, max_docs,
                              agency_level=agency_level,
                              max_empty_pages=max_empty_pages)
    log(f"    → {len(meta_list)} kết quả")

    new_count = 0
    for meta in meta_list:
        mkey = f"{mkey_prefix}/{meta['id']}"
        if manifest.get(mkey) == "ok":
            continue

        if dry_run:
            log(f"    [DRY] {meta['so_hieu'] or meta['id']} — {meta['title'][:50]}")
            new_count += 1
            continue

        # Lấy full text từ detail endpoint
        detail = client.fetch_with_content(meta["id"])
        if not detail or not detail.get("content") or len(detail["content"]) < 100:
            log(f"    [skip] {meta['id']} — không có content")
            manifest[mkey] = "skip"
            save_manifest(manifest)
            continue

        # Ghi file
        out_dir.mkdir(parents=True, exist_ok=True)
        file_content = VBPLClient.build_file(detail, label)
        base     = safe_filename(detail["so_hieu"] or detail["title"])
        out_path = out_dir / (base + ".txt")
        if out_path.exists():
            out_path = out_dir / f"{base}_{meta['id']}.txt"
        out_path.write_text(file_content, encoding="utf-8")

        manifest[mkey] = "ok"
        save_manifest(manifest)
        new_count += 1
        log(f"    ✓ {out_path.name}  ({detail['eff_status']})")

    return new_count

# ─── P2 ───────────────────────────────────────────────────────────────────────

def run_p2(client: VBPLClient, manifest: dict, dry_run: bool) -> int:
    log("\n" + "="*60)
    log("  P2 — Luật còn thiếu + Nghị quyết HĐTP")
    log("="*60)

    total = 0
    out_laws = RAW_DIR / "all_laws"
    out_nq   = RAW_DIR / "nghi_quyet"

    # ── a. Mở rộng all_laws ────────────────────────────────────────────────────
    log("\n[P2-a] Mở rộng all_laws (HP+BL+Lu, max 800)...")
    n = crawl_and_save(client, "", ["HP","BL","Lu"], 800,
                       out_laws, "Tất cả Hiến pháp + Bộ luật + Luật",
                       manifest, "all_laws", dry_run)
    log(f"  → {n} VB mới từ all_laws mở rộng")
    total += n

    # ── b. 7 luật cụ thể ──────────────────────────────────────────────────────
    log("\n[P2-b] Targeted search 7 luật còn thiếu...")
    for spec in MISSING_LAWS:
        # Nếu biết doc_id → gọi thẳng detail endpoint
        if "doc_id" in spec:
            mkey = f"all_laws/{spec['doc_id']}"
            if manifest.get(mkey) == "ok":
                log(f"  [skip] {spec['label']} — đã có trong manifest")
                continue
            log(f"  Lấy trực tiếp doc_id={spec['doc_id']}: {spec['label']}")
            if not dry_run:
                detail = client.fetch_with_content(spec["doc_id"])
                if detail and detail.get("content") and len(detail["content"]) > 100:
                    out_laws.mkdir(parents=True, exist_ok=True)
                    fc = VBPLClient.build_file(detail, spec["label"])
                    base = safe_filename(detail["so_hieu"] or detail["title"])
                    fp   = out_laws / (base + ".txt")
                    if fp.exists():
                        fp = out_laws / f"{base}_{spec['doc_id']}.txt"
                    fp.write_text(fc, encoding="utf-8")
                    manifest[mkey] = "ok"
                    save_manifest(manifest)
                    total += 1
                    log(f"    ✓ {fp.name}")
                else:
                    log(f"    [!] Không lấy được content")
            else:
                log(f"    [DRY] {spec['label']}")
                total += 1
            continue
        # Tìm kiếm bằng keyword tiếng Việt
        n = crawl_and_save(client, spec["kw"], spec["tc"], 20,
                           out_laws, spec["label"],
                           manifest, "all_laws", dry_run)
        total += n
        if n: log(f"  → +{n} VB: {spec['label']}")

    # ── c. Nghị quyết HĐTP ────────────────────────────────────────────────────
    log("\n[P2-c] Crawl Nghị quyết HĐTP...")
    for spec in NQ_SEARCHES:
        n = crawl_and_save(client, spec["kw"], spec["tc"], 50,
                           out_nq, spec["label"],
                           manifest, "nghi_quyet", dry_run)
        total += n
        if n: log(f"  → +{n} NQ: {spec['label']}")

    log(f"\n  P2 xong. Tổng mới: {total}")
    return total

# ─── P1a: Nghị định & Thông tư ────────────────────────────────────────────────

def run_p1_nghi_dinh(client: VBPLClient, manifest: dict, dry_run: bool) -> int:
    log("\n" + "="*60)
    log("  P1a — Nghị định & Thông tư (9 lĩnh vực)")
    log("="*60)

    total = 0
    for topic_key, cfg in ND_TT_TOPICS.items():
        log(f"\n  [{topic_key.upper()}] {cfg['label']}")
        out_dir = RAW_DIR / cfg["folder"]
        for kw in cfg["keywords"]:
            n = crawl_and_save(
                client, kw, cfg["type_codes"],
                max_docs=50,
                out_dir=out_dir,
                label=cfg["label"],
                manifest=manifest,
                mkey_prefix=f"nd_tt/{topic_key}",
                dry_run=dry_run,
            )
            total += n
        log(f"  → Cộng dồn tổng mới: {total}")

    log(f"\n  P1a xong. Tổng NĐ+TT mới: {total}")
    return total

# ─── P1b: Án lệ ───────────────────────────────────────────────────────────────

def run_p1_an_le(client: VBPLClient, manifest: dict, dry_run: bool) -> int:
    log("\n" + "="*60)
    log("  P1b — Án lệ TAND Tối cao")
    log("="*60)

    out_dir = RAW_DIR / "an_le"
    total = 0

    # Án lệ thường là NQ của HĐTP hoặc có "án lệ" trong title
    searches = [
        ("an le",                 [],         "Án lệ"),
        ("Hoi dong Tham phan",    ["NQ"],     "Nghị quyết HĐTP (án lệ)"),
        ("giai phap phap ly",     [],         "Án lệ - giải pháp pháp lý"),
    ]
    for kw, tc, label in searches:
        n = crawl_and_save(client, kw, tc, 80,
                           out_dir, label,
                           manifest, "an_le", dry_run)
        total += n
        if n: log(f"  → +{n}: {label}")

    # Thử scrape từ anle.toaan.gov.vn nếu có requests
    try:
        from scripts.crawl_an_le import ToaAnCrawler, save_an_le, clean_text, OUT_DIR
        log("\n  [toaan.gov.vn] Thử crawl trực tiếp TAND trang...")
        crawler = ToaAnCrawler()
        items   = crawler.list_an_le(max_pages=10)
        log(f"  Tìm được {len(items)} án lệ")
        for info in items:
            mkey = f"an_le/toaan/{info.get('item_id') or info['url']}"
            if manifest.get(mkey) == "ok":
                continue
            content = crawler.fetch_content(info["url"])
            if not content or len(content) < 100:
                continue
            if not dry_run:
                out_path = save_an_le(
                    al_num=info.get("al_num",""), title=info["title"],
                    content=content, source_url=info["url"],
                    agency="Hội đồng Thẩm phán TAND Tối cao",
                    issue_date="", so_hieu=info.get("al_num",""),
                    out_dir=OUT_DIR,
                )
                manifest[mkey] = "ok"
                save_manifest(manifest)
                log(f"  ✓ {out_path.name}")
            total += 1
    except Exception as e:
        log(f"  [warn] toaan crawler: {e}")

    log(f"\n  P1b xong. Tổng án lệ mới: {total}")
    return total

# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Crawl P1+P2 dữ liệu pháp luật")
    p.add_argument("--only",     choices=["p1","p2","all"], default="all")
    p.add_argument("--dry-run",  action="store_true")
    p.add_argument("--max-docs", type=int, default=50,
                   help="Max docs mỗi keyword (mặc định: 50)")
    args = p.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"=== SUPPLEMENT RUN {datetime.now().isoformat()} ===\n")

    log(f"Bắt đầu: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    log(f"Mode: {args.only.upper()} {'[DRY-RUN]' if args.dry_run else ''}")
    log(f"API: vbpl.vn (2 bước: search → GET detail)")

    client   = VBPLClient(search_delay=1.5, detail_delay=1.2)
    manifest = load_manifest()
    grand    = 0
    t0       = __import__("time").time()

    if args.only in ("p2", "all"):
        grand += run_p2(client, manifest, args.dry_run)

    if args.only in ("p1", "all"):
        grand += run_p1_nghi_dinh(client, manifest, args.dry_run)
        grand += run_p1_an_le(client, manifest, args.dry_run)

    elapsed = round((__import__("time").time() - t0) / 60, 1)
    log(f"\n{'='*60}")
    log(f"  HOÀN THÀNH — {datetime.now().strftime('%H:%M:%S')}")
    log(f"  Tổng văn bản mới: {grand}")
    log(f"  Thời gian       : {elapsed} phút")
    if grand > 0 and not args.dry_run:
        log("  Bước tiếp: python -m scripts.ingest")
        log("             python -m scripts.analyze_data")
    log("="*60)


if __name__ == "__main__":
    main()

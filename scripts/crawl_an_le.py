"""Crawler Án lệ từ TAND Tối cao Việt Nam.

Nguồn chính: https://anle.toaan.gov.vn
Fallback:    vbpl.vn (nếu án lệ được publish ở đó)

Án lệ là văn bản đặc biệt:
  - Do Hội đồng Thẩm phán TAND Tối cao ban hành
  - Số hiệu: AL01, AL02, ... (hiện có ~70+ án lệ)
  - Cấu trúc: Nguồn gốc vụ án → Khái quát nội dung → Giải pháp pháp lý → Nội dung án lệ

Cách chạy:
    python -m scripts.crawl_an_le                # tải toàn bộ án lệ
    python -m scripts.crawl_an_le --dry-run      # xem danh sách
    python -m scripts.crawl_an_le --max-docs 30  # chỉ 30 án lệ đầu

Kết quả:
    data/raw/an_le/AL<N>.txt
    data/raw/crawl_manifest.json
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
from bs4 import BeautifulSoup

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ─── Config ───────────────────────────────────────────────────────────────────

TOAAN_BASE      = "https://anle.toaan.gov.vn"
TOAAN_LIST_URL  = f"{TOAAN_BASE}/Pages/tim-kiem-an-le.aspx"
TOAAN_DETAIL    = f"{TOAAN_BASE}/Pages/chi-tiet-an-le.aspx"

# Fallback: tìm trên vbpl.vn
VBPL_API_URL = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/all"

API_DELAY       = 2.0
REQUEST_TIMEOUT = 30
MAX_RETRIES     = 3

RAW_DIR       = Path(__file__).resolve().parent.parent / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "crawl_manifest.json"
OUT_DIR       = RAW_DIR / "an_le"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_API_HEADERS = {
    **_HEADERS,
    "Content-Type": "application/json",
    "Accept":       "application/json",
    "Referer":      "https://vbpl.vn/",
    "Origin":       "https://vbpl.vn",
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

# ─── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=_HEADERS,
                                timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            time.sleep(API_DELAY)
            return resp
        except requests.RequestException as exc:
            wait = API_DELAY * (2 ** attempt)
            print(f"    [retry {attempt+1}/{MAX_RETRIES}] {exc} — đợi {wait:.0f}s")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
    return None

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

# ─── Strategy 1: Crawl trực tiếp TAND Tối cao ─────────────────────────────────

class ToaAnCrawler:
    """Crawler cho anle.toaan.gov.vn."""

    def list_an_le(self, max_pages: int = 10) -> list[dict]:
        """Lấy danh sách án lệ từ trang tìm kiếm."""
        an_le_list: list[dict] = []
        print("  Tìm án lệ tại anle.toaan.gov.vn...")

        for page in range(1, max_pages + 1):
            params = {"page": str(page)}
            resp = _get(TOAAN_LIST_URL, params=params)
            if resp is None:
                print(f"    [warn] Không lấy được trang {page}")
                break

            soup  = BeautifulSoup(resp.text, "lxml")
            items = self._parse_list_page(soup)
            if not items:
                print(f"    → Hết kết quả ở trang {page}.")
                break

            an_le_list.extend(items)
            print(f"    Trang {page}: +{len(items)} án lệ (tổng {len(an_le_list)})")

            if not self._has_next(soup):
                break

        return an_le_list

    def _parse_list_page(self, soup: BeautifulSoup) -> list[dict]:
        items = []
        # Thử nhiều selector phổ biến của trang TAND
        for selector in (
            ".list-an-le li",
            ".search-result li",
            "table.tbl tbody tr",
            ".lawList li",
            "ul.LawList li",
            ".anle-item",
            "article",
        ):
            els = soup.select(selector)
            if els:
                for el in els:
                    info = self._parse_item(el)
                    if info:
                        items.append(info)
                break
        return items

    def _parse_item(self, el) -> Optional[dict]:
        try:
            link = el.select_one("a[href]")
            if not link:
                return None
            title = link.get_text(strip=True)
            href  = link.get("href", "")

            # Số án lệ: AL01, AL02,...
            al_m = re.search(r"AL[\s_-]?(\d+)", title, re.I)
            al_num = f"AL{int(al_m.group(1)):02d}" if al_m else ""

            # ItemID hoặc id trong URL
            id_m = re.search(r"[Ii]tem[Ii][Dd]=(\d+)|/(\d+)(?:$|\?|/)", href)
            item_id = (id_m.group(1) or id_m.group(2)) if id_m else ""

            if not title or not href:
                return None

            url = href if href.startswith("http") else f"{TOAAN_BASE}{href}"
            return {"al_num": al_num, "title": title, "url": url, "item_id": item_id}
        except Exception:
            return None

    def _has_next(self, soup: BeautifulSoup) -> bool:
        for selector in ("a.next", ".paging a[title*='sau']", "a:contains('Trang sau')"):
            el = soup.select_one(selector)
            if el and el.get("href"):
                return True
        return False

    def fetch_content(self, url: str) -> Optional[str]:
        resp = _get(url)
        if resp is None:
            return None
        soup = BeautifulSoup(resp.text, "lxml")

        # Thử các selector nội dung
        for selector in (
            "#anle-content",
            ".content-anle",
            "#toanvancontent",
            ".fulltext",
            "#divContent",
            ".LawDetail",
            "article.content",
            ".content-detail",
        ):
            div = soup.select_one(selector)
            if div:
                for tag in div(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                text = div.get_text("\n", strip=True)
                if len(text) > 200:
                    return text

        # Fallback: body
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        return soup.body.get_text("\n", strip=True) if soup.body else None


# ─── Strategy 2: Tìm án lệ qua vbpl.vn REST API ──────────────────────────────

class VBPLAnLeCrawler:
    """Tìm án lệ trên vbpl.vn — type code 'AL' hoặc 'QD' của HĐTP."""

    AL_KEYWORDS = [
        "an le",
        "Hoi dong Tham phan",
        "giai phap phap ly",
    ]
    # type_codes thử: có thể là "AL" hoặc không có type code riêng
    TYPE_CODES = ["AL", "NQ"]  # NQ = Nghị quyết (HĐTP ban hành án lệ bằng NQ)

    def search(self, max_docs: int = 100) -> list[dict]:
        sess = requests.Session()
        sess.headers.update(_API_HEADERS)

        all_docs: list[dict] = []
        seen: set[str] = set()

        for kw in self.AL_KEYWORDS:
            print(f"  Tìm trên vbpl.vn: \"{kw}\"...")
            for tc in self.TYPE_CODES + [""]:
                docs = self._search_api(kw, [tc] if tc else [], max_docs // 3, sess)
                for d in docs:
                    if d["id"] not in seen:
                        seen.add(d["id"])
                        all_docs.append(d)
            time.sleep(API_DELAY)

        return all_docs[:max_docs]

    def _search_api(self, keyword: str, type_codes: list[str],
                    max_docs: int, session: requests.Session) -> list[dict]:
        results: list[dict] = []
        seen_ids: set[str] = set()
        page_size = 20

        for page_num in range(1, 10):
            payload = {
                "keyword":     keyword,
                "matchMode":   "all_words",
                "optionDoc":   "title",
                "agencyLevel": "TRUNG_UONG",
                "pageSize":    page_size,
                "pageNumber":  page_num,
            }
            try:
                resp = session.post(VBPL_API_URL, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                print(f"    [!] API error: {exc}")
                break

            inner = data.get("data") or data
            items = inner.get("items") or []
            total = inner.get("total") or 0

            if not items:
                break

            for item in items:
                if not isinstance(item, dict):
                    continue
                tc = (item.get("docType") or {}).get("code", "")
                if type_codes and tc not in type_codes:
                    continue
                # Lọc thêm: phải có "án lệ" hoặc "AL" trong title hoặc là NQ của HĐTP
                title = (item.get("title") or "").lower()
                agency = (item.get("agencyName") or "").lower()
                if not (
                    "án lệ" in title
                    or " al" in title
                    or "hội đồng thẩm phán" in agency
                    or "hoi dong tham phan" in agency
                ):
                    continue

                parsed = self._parse_item(item)
                if parsed and parsed["id"] not in seen_ids:
                    seen_ids.add(parsed["id"])
                    results.append(parsed)

            fetched = page_num * page_size
            if fetched >= total or len(results) >= max_docs:
                break
            time.sleep(API_DELAY)

        return results[:max_docs]

    def _parse_item(self, item: dict) -> Optional[dict]:
        doc_id    = str(item.get("id", ""))
        title     = (item.get("title") or "").strip()
        doc_abs   = (item.get("docAbs") or "").strip()
        doc_num   = (item.get("docNum") or "").strip()
        type_obj  = item.get("docType") or {}
        issue_dt  = (item.get("issueDate") or "")[:10]
        eff_stat  = (item.get("effStatus") or {}).get("name", "")
        agency    = (item.get("agencyName") or "").strip()

        if not doc_id or len(doc_abs) < 50:
            return None

        # Trích số án lệ từ title hoặc docNum
        al_m = re.search(r"AL[\s_-]?(\d+)", title + " " + doc_num, re.I)
        al_num = f"AL{int(al_m.group(1)):02d}" if al_m else doc_num

        return {
            "id":         doc_id,
            "al_num":     al_num,
            "title":      title,
            "so_hieu":    doc_num,
            "type_code":  type_obj.get("code", ""),
            "type_name":  type_obj.get("name", ""),
            "agency":     agency,
            "issue_date": issue_dt,
            "eff_status": eff_stat,
            "content":    doc_abs,
            "url":        f"https://vbpl.vn/van-ban/chi-tiet/{doc_id}",
        }

# ─── Lưu file ─────────────────────────────────────────────────────────────────

def save_an_le(
    al_num: str,
    title: str,
    content: str,
    source_url: str,
    agency: str,
    issue_date: str,
    so_hieu: str,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    header = (
        f"NGUON: anle.toaan.gov.vn\n"
        f"SO_HIEU: {so_hieu or al_num}\n"
        f"TEN: {title}\n"
        f"LOAI: Án lệ (AL)\n"
        f"CO_QUAN: {agency or 'Hội đồng Thẩm phán TAND Tối cao'}\n"
        f"NGAY_BAN_HANH: {issue_date}\n"
        f"HIEU_LUC: Còn hiệu lực\n"
        f"URL: {source_url}\n"
        f"CHU_DE: Án lệ\n"
        f"{'─'*60}\n\n"
    )

    fname    = safe_filename(al_num or title) + ".txt"
    out_path = out_dir / fname
    out_path.write_text(header + clean_text(content), encoding="utf-8")
    return out_path

# ─── Main orchestrator ────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Crawler Án lệ TAND Tối cao")
    p.add_argument("--dry-run",  action="store_true")
    p.add_argument("--max-docs", type=int, default=100)
    p.add_argument(
        "--source",
        choices=["toaan", "vbpl", "both"],
        default="both",
        help="Nguồn crawl: toaan.gov.vn, vbpl.vn, hoặc cả hai (mặc định: both)",
    )
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    all_docs: list[dict] = []

    # ── Strategy 1: anle.toaan.gov.vn ────────────────────────────────────────
    if args.source in ("toaan", "both"):
        print("\n[1/2] Crawl anle.toaan.gov.vn...")
        crawler_toaan = ToaAnCrawler()
        toaan_list = crawler_toaan.list_an_le(max_pages=10)
        print(f"  → Tìm được {len(toaan_list)} án lệ trên TAND trang")

        for info in toaan_list:
            mkey = f"an_le/toaan/{info.get('item_id') or info['url']}"
            if manifest.get(mkey) == "ok":
                continue
            content = crawler_toaan.fetch_content(info["url"])
            if content and len(content) > 100:
                all_docs.append({
                    "al_num":     info.get("al_num", ""),
                    "title":      info["title"],
                    "so_hieu":    info.get("al_num", ""),
                    "agency":     "Hội đồng Thẩm phán TAND Tối cao",
                    "issue_date": "",
                    "content":    content,
                    "url":        info["url"],
                    "mkey":       mkey,
                })

    # ── Strategy 2: vbpl.vn REST API ─────────────────────────────────────────
    if args.source in ("vbpl", "both"):
        print("\n[2/2] Tìm kiếm án lệ trên vbpl.vn...")
        crawler_vbpl = VBPLAnLeCrawler()
        vbpl_docs = crawler_vbpl.search(max_docs=args.max_docs)
        print(f"  → Tìm được {len(vbpl_docs)} văn bản án lệ trên vbpl.vn")

        seen_ids = {d.get("mkey", "").split("/")[-1] for d in all_docs}
        for d in vbpl_docs:
            mkey = f"an_le/vbpl/{d['id']}"
            if manifest.get(mkey) != "ok" and d["id"] not in seen_ids:
                d["mkey"] = mkey
                all_docs.append(d)

    print(f"\nTổng {len(all_docs)} án lệ cần tải.")

    if args.dry_run:
        for d in all_docs:
            print(f"  {d.get('al_num') or d.get('so_hieu') or '?':8} — {d['title'][:60]}")
        return

    downloaded = 0
    for i, doc in enumerate(all_docs, 1):
        mkey = doc.get("mkey", f"an_le/unknown/{i}")
        if manifest.get(mkey) == "ok":
            continue

        content = doc.get("content", "")
        if len(clean_text(content)) < 50:
            manifest[mkey] = "skip"
            save_manifest(manifest)
            continue

        out_path = save_an_le(
            al_num     = doc.get("al_num", ""),
            title      = doc["title"],
            content    = content,
            source_url = doc["url"],
            agency     = doc.get("agency", ""),
            issue_date = doc.get("issue_date", ""),
            so_hieu    = doc.get("so_hieu", ""),
            out_dir    = OUT_DIR,
        )
        manifest[mkey] = "ok"
        save_manifest(manifest)
        downloaded += 1
        print(f"  [{i}/{len(all_docs)}] ✓ {out_path.name}")

    print(f"\nXong. Đã lưu {downloaded} án lệ vào data/raw/an_le/")
    if downloaded > 0:
        print("Bước tiếp: python -m scripts.ingest")


if __name__ == "__main__":
    main()

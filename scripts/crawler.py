"""Auto-crawler: tải văn bản pháp luật mới từ vbpl.vn.

Chỉ tải văn bản MỚI hoặc ĐÃ THAY ĐỔI kể từ lần crawl trước.
Lưu file đúng format header của ingest.py để tương thích hoàn toàn.

Cách chạy:
    python -m scripts.auto_update          # full pipeline (crawl + ingest)
    python -m scripts.crawler --dry-run    # xem có bao nhiêu VB mới, không tải
    python -m scripts.crawler --since 30   # crawl 30 ngày gần nhất (override state)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ── Constants ──────────────────────────────────────────────────────────────────

VBPL_BASE   = "https://vbpl.vn"
VBPL_SEARCH = f"{VBPL_BASE}/TW/Pages/vbpq-timkiem.aspx"
VBPL_DETAIL = f"{VBPL_BASE}/TW/Pages/vbpq-toanvan.aspx"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

REQUEST_DELAY   = 2.5   # giây giữa mỗi request — lịch sự với server
REQUEST_TIMEOUT = 30    # timeout mỗi request
MAX_RETRIES     = 3


# ── CrawlState ─────────────────────────────────────────────────────────────────

class CrawlState:
    """Lưu trữ trạng thái crawl vào JSON — biết VB nào đã tải, khi nào tải."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._data = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "last_crawled": {},  # source_name → ISO datetime
            "known_urls":   {},  # url → md5 checksum của content
            "stats":        {"total_downloaded": 0, "total_updated": 0},
        }

    def save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def last_crawled(self, source: str, default_days: int = 30) -> datetime:
        """Trả về thời điểm crawl cuối. Mặc định: `default_days` ngày trước."""
        s = self._data["last_crawled"].get(source)
        return datetime.fromisoformat(s) if s else datetime.now() - timedelta(days=default_days)

    def set_crawled_now(self, source: str):
        self._data["last_crawled"][source] = datetime.now().isoformat()

    def is_new_or_changed(self, url: str, content: str) -> bool:
        """True nếu url chưa biết hoặc nội dung đã thay đổi (checksum khác)."""
        checksum = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()
        old = self._data["known_urls"].get(url)
        if old != checksum:
            self._data["known_urls"][url] = checksum
            return True
        return False

    def add_stat(self, key: str, n: int = 1):
        self._data["stats"][key] = self._data["stats"].get(key, 0) + n

    @property
    def stats(self) -> dict:
        return self._data["stats"]


# ── HTTP helper ────────────────────────────────────────────────────────────────

def _get(url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
    """GET với retry + exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                url,
                params=params,
                headers=_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp
        except requests.RequestException as exc:
            wait = REQUEST_DELAY * (2 ** attempt)
            print(f"    [retry {attempt+1}/{MAX_RETRIES}] {exc} — đợi {wait:.0f}s")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
    return None


# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class DocInfo:
    item_id:    str
    title:      str
    doc_number: str
    doc_type:   str
    agency:     str
    issued_date: str   # YYYY-MM-DD hoặc rỗng
    url:        str
    status:     str = "Còn hiệu lực"


# ── VBPLCrawler ────────────────────────────────────────────────────────────────

class VBPLCrawler:
    """Crawler cho vbpl.vn — tìm & tải văn bản pháp luật mới."""

    SOURCE = "vbpl"

    def list_new(self, since: datetime, max_pages: int = 30) -> list[DocInfo]:
        """Lấy danh sách văn bản mới từ `since` đến hôm nay."""
        from_str = since.strftime("%d/%m/%Y")
        to_str   = datetime.now().strftime("%d/%m/%Y")
        print(f"  Tìm VB mới từ {from_str} đến {to_str} trên vbpl.vn...")

        docs: list[DocInfo] = []
        for page in range(1, max_pages + 1):
            params = {
                "type": "0",
                "s": "",
                "FromDate": from_str,
                "ToDate":   to_str,
                "DepartmentId": "",
                "DocumentTypeId": "",
                "OrganId": "",
                "Match": "0",
                "page": str(page),
            }
            resp = _get(VBPL_SEARCH, params=params)
            if resp is None:
                print(f"    [warn] Không lấy được trang {page}")
                break

            soup  = BeautifulSoup(resp.text, "lxml")
            items = self._find_items(soup)
            if not items:
                print(f"    → Hết kết quả ở trang {page}.")
                break

            for el in items:
                info = self._parse_item(el)
                if info:
                    docs.append(info)
            print(f"    Trang {page}: +{len(items)} VB (tổng {len(docs)})")

            if not self._has_next(soup):
                break

        return docs

    # ── HTML parsing helpers ──────────────────────────────────────────────────

    def _find_items(self, soup: BeautifulSoup) -> list:
        """Tìm các element chứa thông tin VB trong trang kết quả."""
        for selector in (
            ".LawList li",
            "ul.LawList li",
            ".listLaw li",
            "table.tbl-data tbody tr",
            ".lawList li",
            ".search-result li",
        ):
            items = soup.select(selector)
            if items:
                return items
        return []

    def _has_next(self, soup: BeautifulSoup) -> bool:
        """Kiểm tra còn trang tiếp theo không."""
        for selector in (
            "a.next",
            ".paging a[title*='sau']",
            ".pagination a[rel='next']",
            "a:contains('Trang sau')",
        ):
            el = soup.select_one(selector)
            if el and el.get("href"):
                return True
        return False

    def _parse_item(self, el) -> Optional[DocInfo]:
        """Parse 1 dòng kết quả → DocInfo."""
        try:
            # Link + title
            link = el.select_one(
                "a[href*='ItemID'], a[href*='item'], h3 a, h4 a, .title a, td:first-child a"
            )
            if not link:
                return None
            title = link.get_text(strip=True)
            href  = link.get("href", "")

            # ItemID
            m = re.search(r"ItemID=(\d+)", href, re.I) or re.search(r"/(\d+)(?:$|\?)", href)
            if not m:
                return None
            item_id = m.group(1)

            txt = el.get_text(" | ", strip=True)

            # Số hiệu — pattern: "XX/YYYY/XY-..."
            num_m = re.search(r"(\d{1,4}[/-]\d{4}[/-][A-ZĐ\-]+)", txt)
            doc_number = num_m.group(1) if num_m else ""

            # Loại VB
            type_m = re.search(r"(?:Loại[:\s]+)([^|•\n]+)", txt)
            doc_type = type_m.group(1).strip()[:50] if type_m else "Văn bản pháp luật"

            # Cơ quan
            agency_m = re.search(r"(?:Ban hành|Cơ quan)[:\s]+([^|•\n]+)", txt)
            agency = agency_m.group(1).strip()[:80] if agency_m else ""

            # Ngày
            date_m = re.search(r"(\d{2}/\d{2}/\d{4})", txt)
            issued = ""
            if date_m:
                d, mo, y = date_m.group(1).split("/")
                issued = f"{y}-{mo}-{d}"

            url = f"{VBPL_BASE}/TW/Pages/vbpq-toanvan.aspx?ItemID={item_id}"
            return DocInfo(
                item_id=item_id,
                title=title,
                doc_number=doc_number,
                doc_type=doc_type,
                agency=agency,
                issued_date=issued,
                url=url,
            )
        except Exception:
            return None

    # ── Fetch full text ────────────────────────────────────────────────────────

    def fetch_text(self, info: DocInfo) -> Optional[str]:
        """Tải toàn văn của 1 văn bản."""
        resp = _get(info.url)
        if resp is None:
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        # Thử nhiều selector cho vùng nội dung
        for selector in (
            "#toanvancontent",
            ".fulltext",
            ".LawDetail",
            "#divContent",
            ".content-detail",
            "article.content",
            "div[class*='content']",
        ):
            div = soup.select_one(selector)
            if div:
                for tag in div(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                text = div.get_text("\n", strip=True)
                if len(text) > 200:
                    return text

        # Fallback: lấy body text
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        return soup.body.get_text("\n", strip=True) if soup.body else None

    # ── Save ───────────────────────────────────────────────────────────────────

    def save(self, info: DocInfo, content: str, out_dir: Path) -> Path:
        """Lưu văn bản thành .txt với header format chuẩn của ingest.py."""
        out_dir.mkdir(parents=True, exist_ok=True)

        # Tên file an toàn từ số hiệu hoặc item_id
        safe = re.sub(r'[/\\?*:|"<>]', "_", info.doc_number or info.item_id)
        filename = f"{safe}.txt"
        out_path  = out_dir / filename

        header = (
            f"NGUON: vbpl.vn\n"
            f"SO_HIEU: {info.doc_number}\n"
            f"TEN: {info.title}\n"
            f"LOAI: {info.doc_type}\n"
            f"CO_QUAN: {info.agency}\n"
            f"NGAY_BAN_HANH: {info.issued_date}\n"
            f"HIEU_LUC: {info.status}\n"
            f"URL: {info.url}\n"
            f"CHU_DE: crawled\n"
            f"{'─' * 60}\n\n"
        )
        out_path.write_text(header + content, encoding="utf-8")
        return out_path


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    import argparse
    from pathlib import Path

    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    p = argparse.ArgumentParser(description="Crawl văn bản mới từ vbpl.vn")
    p.add_argument("--since", type=int, default=None,
                   help="Override: crawl N ngày gần nhất (bỏ qua crawl state)")
    p.add_argument("--dry-run", action="store_true",
                   help="Chỉ đếm, không tải về")
    p.add_argument("--max-pages", type=int, default=30)
    args = p.parse_args()

    # Paths
    root       = Path(__file__).resolve().parent.parent
    state_file = root / "data" / "crawl_state.json"
    out_dir    = root / "data" / "raw" / "crawled"

    state   = CrawlState(state_file)
    crawler = VBPLCrawler()

    since = (
        datetime.now() - timedelta(days=args.since)
        if args.since
        else state.last_crawled(VBPLCrawler.SOURCE)
    )

    docs = crawler.list_new(since, max_pages=args.max_pages)
    print(f"\nTìm được {len(docs)} văn bản mới.")

    if args.dry_run:
        for d in docs:
            print(f"  [{d.issued_date}] {d.doc_number} — {d.title[:60]}")
        return

    downloaded = 0
    for i, info in enumerate(docs, 1):
        print(f"  [{i}/{len(docs)}] {info.doc_number or info.item_id}...", end=" ", flush=True)
        text = crawler.fetch_text(info)
        if not text:
            print("SKIP (không lấy được text)")
            continue
        if state.is_new_or_changed(info.url, text):
            path = crawler.save(info, text, out_dir)
            print(f"→ {path.name}")
            state.add_stat("total_downloaded")
            downloaded += 1
        else:
            print("→ không đổi, bỏ qua")

    state.set_crawled_now(VBPLCrawler.SOURCE)
    state.save()
    print(f"\nXong. Tải mới: {downloaded} VB. State: {state_file}")


if __name__ == "__main__":
    main()

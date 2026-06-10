"""Client chung cho vbpl.vn REST API (phiên bản 2025+).

API thay đổi so với trước:
  - Search endpoint: POST /api/qtdc/public/doc/all  → trả metadata list (KHÔNG có docAbs)
  - Detail endpoint: GET  /api/qtdc/public/doc/{id} → trả đầy đủ, có:
      data.documentContent.content  (HTML full text, có thể 100k+ chars)
      data.references               (các VB liên quan)
      data.documentContentFileName  (tên file PDF gốc)

Cách dùng:
    from src.vbpl_client import VBPLClient
    client = VBPLClient()
    docs   = client.search("ho tich", type_codes=["Lu"], max_docs=20)
    for d in docs:
        text = client.fetch_content(d["id"])  # gọi detail endpoint
        ...
"""
from __future__ import annotations

import re
import time
import unicodedata
from typing import Optional

import requests
from bs4 import BeautifulSoup

MAX_RETRIES  = 4
RETRY_WAIT   = [5, 10, 20, 40]   # exponential back-off (giây)

BASE_URL   = "https://vbpl.vn"
SEARCH_URL = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/all"
DETAIL_URL = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{id}"

PAGE_SIZE   = 20
SEARCH_DLY  = 2.5   # giây giữa các search page (tăng để tránh rate-limit)
DETAIL_DLY  = 2.0   # giây giữa các detail request

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

# ─── Text helpers ─────────────────────────────────────────────────────────────

_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAIL_WS    = re.compile(r"[ \t]+$", re.MULTILINE)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAIL_WS.sub("", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def html_to_text(html: str) -> str:
    """Chuyển HTML content từ API thành plain text sạch."""
    if not html or not html.strip():
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    return clean_text(text)


def safe_filename(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\-.]", "_", s)
    return s[:80].strip("_")

# ─── VBPLClient ───────────────────────────────────────────────────────────────

class VBPLClient:
    """Client 2 bước: search → detail để lấy full text."""

    def __init__(self, search_delay: float = SEARCH_DLY,
                 detail_delay: float = DETAIL_DLY):
        self.search_delay = search_delay
        self.detail_delay = detail_delay
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        keyword: str,
        type_codes: list[str] | None = None,
        max_docs: int = 100,
        agency_level: str = "",
        max_empty_pages: int = 5,
    ) -> list[dict]:
        """Tìm kiếm và trả về list metadata (không có content).

        Lưu ý API mới (2025+):
        - Không có trường docAbs trong kết quả search
        - Cần gọi GET /doc/{id} riêng để lấy full text
        - agencyLevel="" → trả về cả VB của Bộ (NĐ, TT)
        - type_codes: filter client-side sau khi nhận response
        """
        results: list[dict] = []
        seen_ids: set[str] = set()
        max_pages = max((max_docs // PAGE_SIZE) + 5, 20)
        consecutive_empty = 0

        for page_num in range(1, max_pages + 1):
            payload: dict = {
                "keyword":   keyword,
                "matchMode": "all_words",
                "pageSize":  PAGE_SIZE,
                "pageNumber": page_num,
            }
            if agency_level:
                payload["agencyLevel"] = agency_level

            data = None
            for attempt in range(MAX_RETRIES):
                try:
                    resp = self._session.post(SEARCH_URL, json=payload, timeout=35)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as exc:
                    wait = RETRY_WAIT[min(attempt, len(RETRY_WAIT)-1)]
                    print(f"    [retry {attempt+1}/{MAX_RETRIES}] {exc} — đợi {wait}s")
                    time.sleep(wait)
            if data is None:
                print(f"    [!] Bỏ qua trang {page_num} sau {MAX_RETRIES} lần thử")
                break

            inner = data.get("data") or data
            items = (inner.get("items") or []) if isinstance(inner, dict) else []
            total = (inner.get("total") or 0) if isinstance(inner, dict) else 0

            if not items:
                break

            new_count = 0
            for item in items:
                if not isinstance(item, dict):
                    continue
                tc = (item.get("docType") or {}).get("code", "")
                # Filter client-side nếu có yêu cầu type cụ thể
                if type_codes and tc not in type_codes:
                    continue
                meta = self._parse_meta(item)
                if meta and meta["id"] not in seen_ids:
                    seen_ids.add(meta["id"])
                    results.append(meta)
                    new_count += 1

            fetched = page_num * PAGE_SIZE
            print(f"      Trang {page_num}: +{new_count} mới, tổng {len(results)} "
                  f"(API total: {total})")

            if new_count == 0:
                consecutive_empty += 1
                if consecutive_empty >= max_empty_pages:
                    print(f"      → Dừng sớm ({max_empty_pages} trang trống liên tiếp)")
                    break
            else:
                consecutive_empty = 0

            if len(results) >= max_docs or fetched >= total:
                break

            time.sleep(self.search_delay)

        return results[:max_docs]

    def _parse_meta(self, item: dict) -> Optional[dict]:
        doc_id   = str(item.get("id", ""))
        title    = (item.get("title") or "").strip()
        doc_num  = (item.get("docNum") or "").strip()
        so_hieu  = doc_num if doc_num and doc_num.lower() not in ("không số", "khong so", "") else ""
        type_obj = item.get("docType") or {}
        agency   = (item.get("agencyName") or "").strip()
        issue_dt = (item.get("issueDate") or "")[:10]
        eff_stat = (item.get("effStatus") or {}).get("name", "")

        if not doc_id or not title:
            return None
        return {
            "id":          doc_id,
            "title":       title,
            "so_hieu":     so_hieu,
            "type_code":   type_obj.get("code", ""),
            "type_name":   type_obj.get("name", ""),
            "agency":      agency,
            "issue_date":  issue_dt,
            "eff_status":  eff_stat,
            "url":         f"{BASE_URL}/van-ban/chi-tiet/{doc_id}",
        }

    # ── Detail (full text) ────────────────────────────────────────────────────

    def fetch_content(self, doc_id: str) -> Optional[str]:
        """Lấy full text từ detail endpoint. Trả về plain text hoặc None."""
        url  = DETAIL_URL.format(id=doc_id)
        data = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._session.get(url, timeout=35)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                wait = RETRY_WAIT[min(attempt, len(RETRY_WAIT)-1)]
                print(f"    [retry {attempt+1}/{MAX_RETRIES}] detail {doc_id}: {exc} — đợi {wait}s")
                time.sleep(wait)
        if data is None:
            return None

        inner = data.get("data") or {}
        if not isinstance(inner, dict):
            return None

        # Lấy content từ documentContent.content (HTML)
        doc_content = inner.get("documentContent") or {}
        html_content = doc_content.get("content", "") if isinstance(doc_content, dict) else ""

        if html_content and len(html_content) > 100:
            text = html_to_text(html_content)
            if len(text) > 100:
                time.sleep(self.detail_delay)
                return text

        # Fallback: không có content
        time.sleep(self.detail_delay)
        return None

    def fetch_with_content(self, doc_id: str) -> Optional[dict]:
        """Lấy metadata đầy đủ + references từ detail endpoint."""
        url  = DETAIL_URL.format(id=doc_id)
        data = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._session.get(url, timeout=35)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                wait = RETRY_WAIT[min(attempt, len(RETRY_WAIT)-1)]
                time.sleep(wait)
        if data is None:
            return None

        inner = data.get("data") or {}
        if not isinstance(inner, dict):
            return None

        doc_content = inner.get("documentContent") or {}
        html_content = doc_content.get("content", "") if isinstance(doc_content, dict) else ""
        plain_text = html_to_text(html_content) if html_content else ""

        # References (VB liên quan, bị thay thế...)
        refs = inner.get("references") or []
        related: list[dict] = []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            target = ref.get("targetDocument") or {}
            if isinstance(target, dict) and target.get("id"):
                related.append({
                    "id":       str(target.get("id", "")),
                    "so_hieu":  (target.get("docNum") or ""),
                    "title":    (target.get("title") or "")[:80],
                    "rel_type": (ref.get("relationType") or ref.get("type") or ""),
                })

        time.sleep(self.detail_delay)
        return {
            "id":         str(inner.get("id", "")),
            "so_hieu":    (inner.get("docNum") or ""),
            "title":      (inner.get("title") or ""),
            "type_code":  (inner.get("docType") or {}).get("code", ""),
            "type_name":  (inner.get("docType") or {}).get("name", ""),
            "agency":     (inner.get("agencyName") or ""),
            "issue_date": (inner.get("issueDate") or "")[:10],
            "eff_status": (inner.get("effStatus") or {}).get("name", ""),
            "content":    plain_text,
            "references": related,
            "url":        f"{BASE_URL}/van-ban/chi-tiet/{inner.get('id', '')}",
        }

    # ── Build header ──────────────────────────────────────────────────────────

    @staticmethod
    def build_file(doc: dict, chu_de: str) -> str:
        """Tạo nội dung file .txt với header chuẩn."""
        refs_str = ""
        for ref in doc.get("references", [])[:5]:
            refs_str += f"\n  - {ref['so_hieu'] or ref['id']}: {ref['title']} ({ref['rel_type']})"

        header = (
            f"NGUON: vbpl.vn\n"
            f"SO_HIEU: {doc.get('so_hieu','')}\n"
            f"TEN: {doc.get('title','')}\n"
            f"LOAI: {doc.get('type_name','')} ({doc.get('type_code','')})\n"
            f"CO_QUAN: {doc.get('agency','')}\n"
            f"NGAY_BAN_HANH: {doc.get('issue_date','')}\n"
            f"HIEU_LUC: {doc.get('eff_status','')}\n"
            f"URL: {doc.get('url','')}\n"
            f"CHU_DE: {chu_de}\n"
        )
        if refs_str:
            header += f"VAN_BAN_LIEN_QUAN:{refs_str}\n"
        header += "─" * 60 + "\n\n"

        return header + (doc.get("content") or "")

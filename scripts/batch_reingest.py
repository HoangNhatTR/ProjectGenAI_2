"""Re-ingest hàng loạt các văn bản hiện hành bị mất cấu trúc Điều (dạng A).

Lấy danh sách từ scan (chunk≥50, Điều<3, 2019+ hoặc luật cũ-còn-hiệu-lực), với
mỗi VB: lấy id vbpl (từ source URL hoặc search) → fetch toàn văn → chunk theo
Điều (chunker đã fix) → embed → xoá chunk cũ → ghi đầy đủ. GIỮ metadata cũ.

BỀN: resume qua progress file; bỏ qua VB lỗi; load embedder 1 lần; log từng VB.

    python -m scripts.batch_reingest --limit 2     # TEST nhỏ (2 VB)
    python -m scripts.batch_reingest               # chạy full danh sách scoped
    python -m scripts.batch_reingest --max-year 2024 --min-year 2024   # 1 năm

Chạy bằng venv Chatbot. OpenSearch phải bật. GHI INDEX — cần duyệt.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from opensearchpy import OpenSearch

from src import config
from src.chunking import _chunk_prefix, chunk_document
from src.parent_store import ParentStore
from src.schemas import DocumentMetadata, RawDocument
from src.vbpl_client import VBPLClient

ROOT = Path(__file__).resolve().parent.parent
PROGRESS = ROOT / "data" / "batch_reingest_progress.json"
LAW_TYPES = ["Bộ luật", "Bộ luật (BL)", "Luật", "Luật (Lu)", "Pháp lệnh",
             "Nghị định", "Nghị định (NĐ)"]
KEEP_OLD = {"52/2014/QH13", "50/2014/QH13", "15/2012/QH13", "65/2014/QH13",
            "67/2014/QH13", "43/2013/QH13", "45/2013/QH13", "40/2013/QH13"}
SKIP = {"144/2021/NĐ-CP"}  # bản tiếng Anh, nguồn không phải vbpl

cli = OpenSearch(hosts=[config.OPENSEARCH_URL], timeout=300)
IDX = config.OPENSEARCH_INDEX


def scan_flagged():
    flagged, after = [], None
    while True:
        comp = {"size": 500, "sources": [{"dn": {"terms": {"field": "doc_number"}}}]}
        if after:
            comp["after"] = after
        res = cli.search(index=IDX, body={
            "size": 0, "query": {"terms": {"doc_type": LAW_TYPES}},
            "aggs": {"g": {"composite": comp,
                           "aggregations": {"a": {"cardinality": {"field": "article"}}}}}})
        g = res["aggregations"]["g"]
        for b in g["buckets"]:
            if b["doc_count"] >= 50 and b["a"]["value"] < 3:
                flagged.append(b["key"]["dn"])
        after = g.get("after_key")
        if not after or not g["buckets"]:
            return flagged


def first_chunk_source(dn: str) -> dict | None:
    r = cli.search(index=IDX, body={"size": 1, "query": {"term": {"doc_number": dn}},
                                    "_source": ["source", "doc_type", "doc_number", "title",
                                                "issued_date", "effective_date", "status",
                                                "linh_vuc", "co_quan", "folder"]})
    h = r["hits"]["hits"]
    return h[0]["_source"] if h else None


def resolve_vbpl_id(client: VBPLClient, source: str, dn: str) -> str | None:
    m = re.search(r"/chi-tiet/([^/?#]+)", source or "")
    seg = m.group(1) if m else ""
    if re.fullmatch(r"\d+", seg) or re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]+", seg):
        return seg
    docs = client.search(dn, max_docs=20)
    t = next((d for d in docs if dn.lower() in (d["so_hieu"] or "").lower()), None)
    return t["id"] if t else None


def build_meta(s: dict) -> DocumentMetadata:
    return DocumentMetadata(
        source=s["source"], doc_type=s.get("doc_type"), doc_number=s.get("doc_number"),
        title=s.get("title"), issued_date=s.get("issued_date"),
        effective_date=s.get("effective_date"), status=s.get("status"),
        linh_vuc=s.get("linh_vuc"), co_quan=s.get("co_quan"), folder=s.get("folder"))


def load_progress() -> dict:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    return {"done": {}, "skipped": {}}


def save_progress(p: dict):
    PROGRESS.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-year", type=int, default=2012)
    ap.add_argument("--max-year", type=int, default=2026)
    args = ap.parse_args()

    print("Quét danh sách...", flush=True)
    flagged = [dn for dn in scan_flagged() if dn not in SKIP]
    # lọc theo năm + giữ luật cũ then chốt
    todo = []
    for dn in flagged:
        s = first_chunk_source(dn)
        if not s:
            continue
        yr = (s.get("issued_date") or "")[:4]
        y = int(yr) if yr.isdigit() else 0
        if (args.min_year <= y <= args.max_year) or dn in KEEP_OLD:
            todo.append((dn, s))
    todo.sort(key=lambda x: -(int((x[1].get("issued_date") or "0000")[:4] or 0)
                              if (x[1].get("issued_date") or "")[:4].isdigit() else 0))

    prog = load_progress()
    todo = [(dn, s) for dn, s in todo if dn not in prog["done"]]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Cần xử lý: {len(todo)} VB (đã xong trước đó: {len(prog['done'])})\n", flush=True)
    if not todo:
        print("Không còn VB nào. Xong.")
        return

    from src.embedding import Embedder
    from src.opensearch_store import OpenSearchVectorStore
    embedder = Embedder(config.EMBEDDING_MODEL)
    vstore = OpenSearchVectorStore(config.OPENSEARCH_URL, config.OPENSEARCH_INDEX)
    client = VBPLClient()
    ps = ParentStore(config.PARENT_STORE_PATH)

    t_all = time.time()
    for i, (dn, s) in enumerate(todo, 1):
        try:
            vid = resolve_vbpl_id(client, s["source"], dn)
            if not vid:
                prog["skipped"][dn] = "no_id"; save_progress(prog)
                print(f"[{i}/{len(todo)}] {dn:<18} SKIP (no id)", flush=True); continue
            full = client.fetch_with_content(vid)
            if not full or not (full.get("content") or "").strip() or len(full["content"]) < 500:
                prog["skipped"][dn] = "no_content"; save_progress(prog)
                print(f"[{i}/{len(todo)}] {dn:<18} SKIP (no content)", flush=True); continue
            if dn.split("/")[0] not in (full.get("so_hieu") or ""):
                prog["skipped"][dn] = f"mismatch:{full.get('so_hieu')}"; save_progress(prog)
                print(f"[{i}/{len(todo)}] {dn:<18} SKIP (id→{full.get('so_hieu')})", flush=True); continue

            doc = RawDocument(text=full["content"], metadata=build_meta(s))
            # dọn parent cũ cùng prefix
            prefix = _chunk_prefix(doc)
            with sqlite3.connect(ps.db_path) as conn:
                conn.execute("DELETE FROM parents WHERE id LIKE ?", (prefix + "%",)); conn.commit()
            chunks = chunk_document(doc, parent_store=ps)
            arts = len({c.article for c in chunks if c.article})
            embs = embedder.encode([c.text for c in chunks])
            cli.delete_by_query(index=IDX, refresh=False, conflicts="proceed",
                                body={"query": {"term": {"doc_number": dn}}})
            vstore.add(chunks, embs)

            prog["done"][dn] = {"chunks": len(chunks), "articles": arts}
            save_progress(prog)
            el = time.time() - t_all
            print(f"[{i}/{len(todo)}] {dn:<18} OK {len(chunks):>4} chunk / {arts:>3} Điều "
                  f"| {full['title'][:34]} | {el/60:.0f}m", flush=True)
        except Exception as exc:
            prog["skipped"][dn] = f"error:{exc}"; save_progress(prog)
            print(f"[{i}/{len(todo)}] {dn:<18} ERROR {exc}", flush=True)

    cli.indices.refresh(index=IDX)
    done = len(prog["done"]); sk = len(prog["skipped"])
    print(f"\n=== XONG lượt này. Tổng done={done}, skipped={sk}, {(time.time()-t_all)/60:.0f} phút ===")


if __name__ == "__main__":
    main()

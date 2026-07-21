# -*- coding: utf-8 -*-
"""Re-extract semantic KG NĐ 168/2024 theo CHUNK KHOẢN (sửa lỗi cắt 8000 chars).

Điều dài > 7500 chars được tách theo khoản, mỗi chunk kèm header Điều để giữ
ngữ cảnh; kết quả các chunk được MERGE phía client (dedupe offense/subject theo
name, penalty giữ thứ tự + offset index) rồi ghi 1 LẦN/điều — tránh đè
penalty id `{article_id}::penalty_{i}`.

Chạy (cwd = ProjectGenAI_2):
    python reextract_nd168_chunked.py --dry-run     # chỉ in chunk plan
    python reextract_nd168_chunked.py               # chạy thật
    python reextract_nd168_chunked.py --articles 6,7  # chỉ vài điều
"""
import argparse
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"c:\Users\HP\Desktop\HoangNhatpr\Project2\ProjectGenAI_2")
import os

os.chdir(r"c:\Users\HP\Desktop\HoangNhatpr\Project2\ProjectGenAI_2")
# kieai fallback dùng slug đã verify hoạt động (docs.kie.ai); KHÔNG dùng
# default deepseek-chat vì base_url template mới không có slug đó.
os.environ.setdefault("KIEAI_SEMANTIC_MODEL", "gemini-2.5-flash")

from src.chunking import _iter_articles
from src.kg.neo4j_client import Neo4jClient
from src.kg.semantic_extractor import SemanticExtractor
from src.kg.structural_extractor import (
    _article_id,
    _law_id_from_source,
    _normalize_for_kg,
)
from src.parsing import load_document
from scripts.build_semantic_kg import find_raw_file
from src import config

DN = "168/2024/NĐ-CP"
MAX_CHUNK = 7500          # < max_text_chars=8000 của extractor
PROVIDERS = ["gemini", "kieai"]
DELAYS = {"gemini": 4.1, "kieai": 1.0}

_KHOAN_RE = re.compile(r"(?m)^(?=\d{1,2}\.\s)")
_DIEM_RE = re.compile(r"(?m)^(?=[a-zđ]\)\s)")


def split_article(text: str) -> list[str]:
    """Tách điều dài thành chunks ≤ MAX_CHUNK, ranh giới là khoản (fallback điểm)."""
    if len(text) <= MAX_CHUNK:
        return [text]
    parts = _KHOAN_RE.split(text)
    header = parts[0].strip()[:400]  # "Điều X. Tiêu đề..." trước khoản 1
    blocks: list[str] = []
    for blk in parts[1:]:
        if len(blk) > MAX_CHUNK - len(header):
            # khoản khổng lồ → tách tiếp theo điểm a) b) c)
            sub = _DIEM_RE.split(blk)
            head2 = sub[0]  # "9. Phạt tiền từ ..." mở đầu khoản
            cur = head2
            for piece in sub[1:]:
                if len(cur) + len(piece) > MAX_CHUNK - len(header) - 50:
                    blocks.append(cur)
                    cur = head2.rstrip() + " (tiếp)\n" + piece
                else:
                    cur += piece
            blocks.append(cur)
        else:
            blocks.append(blk)
    chunks: list[str] = []
    cur = ""
    for blk in blocks:
        if cur and len(cur) + len(blk) > MAX_CHUNK - len(header):
            chunks.append(header + "\n" + cur)
            cur = blk
        else:
            cur += blk if cur else blk
    if cur:
        chunks.append(header + "\n" + cur)
    return chunks


def merge_results(results) -> dict:
    """Merge extraction nhiều chunk: dedupe offense/subject, offset penalty idx."""
    offenses, subjects, penalties, relations = [], [], [], []
    seen_off, seen_sub = set(), set()
    for res in results:
        offset = len(penalties)
        for o in res.offenses:
            name = (o.get("name") if isinstance(o, dict) else str(o) or "").strip()
            if name and name not in seen_off:
                seen_off.add(name)
                offenses.append(o if isinstance(o, dict) else {"name": name})
        for s2 in res.subjects:
            name = (s2.get("name") if isinstance(s2, dict) else str(s2) or "").strip()
            if name and name not in seen_sub:
                seen_sub.add(name)
                subjects.append(s2 if isinstance(s2, dict) else {"name": name})
        penalties.extend(res.penalties)
        for rel in res.relations:
            if not isinstance(rel, dict):
                continue
            idxs = [i + offset for i in (rel.get("penalty_indices") or [])
                    if isinstance(i, int)]
            relations.append({**rel, "penalty_indices": idxs})
    return dict(offenses=offenses, penalties=penalties,
                subjects=subjects, relations=relations)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--articles", help="chỉ các điều này, vd 6,7")
    args = ap.parse_args()
    only = set(args.articles.split(",")) if args.articles else None

    found = find_raw_file(DN, config.RAW_DIR)
    if not found:
        sys.exit("Không tìm thấy file raw NĐ 168")
    path, meta = found
    doc = load_document(path, meta)
    norm = _normalize_for_kg(doc.text)
    law_id = _law_id_from_source(meta.source)

    arts = [(num, t) for _c, num, t in _iter_articles(norm) if num]
    if only:
        arts = [(n, t) for n, t in arts if str(n) in only]
    plan = [(n, t, split_article(t)) for n, t in arts]
    n_calls = sum(len(c) for _, _, c in plan)
    print(f"NĐ 168: {len(plan)} điều → {n_calls} LLM calls "
          f"({sum(1 for _, t, _ in plan if len(t) > MAX_CHUNK)} điều phải tách)")
    if args.dry_run:
        for n, t, chunks in plan:
            if len(chunks) > 1:
                print(f"  Điều {n}: {len(t):,} chars → {len(chunks)} chunks "
                      f"({[len(c) for c in chunks]})")
        return

    client = Neo4jClient.from_env()
    pidx = 0
    cache: dict[str, SemanticExtractor] = {}

    def get_ex():
        name = PROVIDERS[pidx]
        if name not in cache:
            cache[name] = SemanticExtractor(provider=name)
            print(f"  ⚡ Init provider: {name} (model: {cache[name].model})", flush=True)
        return cache[name], DELAYS.get(name, 4.0)

    get_ex()
    t_start = time.time()
    n_ok = n_err = 0
    for i, (num, text, chunks) in enumerate(plan, 1):
        art_id = _article_id(law_id, str(num))
        results = []
        err = None
        for j, ch in enumerate(chunks):
            while True:
                ex, delay = get_ex()
                t0 = time.time()
                res = ex.extract(art_id, ch)
                if res.quota_exhausted and pidx + 1 < len(PROVIDERS):
                    pidx += 1
                    print(f"    ⚡ Quota {PROVIDERS[pidx-1]} hết → {PROVIDERS[pidx]}", flush=True)
                    continue
                break
            if res.error:
                err = res.error
                break
            results.append(res)
            sleep = delay - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
        if err:
            n_err += 1
            print(f"  [{i:3}] Điều {num}: ❌ {err[:90]}", flush=True)
            continue
        merged = merge_results(results)
        try:
            client.write_semantic_extraction(article_id=art_id, **merged)
            client.mark_article_done(art_id)
        except Exception as exc:
            n_err += 1
            print(f"  [{i:3}] Điều {num}: ⚠ write lỗi: {exc}", flush=True)
            continue
        n_ok += 1
        print(f"  [{i:3}] Điều {num}: ★ {len(chunks)}ch O={len(merged['offenses'])} "
              f"P={len(merged['penalties'])} S={len(merged['subjects'])}", flush=True)

    print(f"\nXong: {n_ok} OK / {n_err} lỗi — {(time.time()-t_start)/60:.1f} phút")
    client.close()


if __name__ == "__main__":
    main()

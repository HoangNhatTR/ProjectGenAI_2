# -*- coding: utf-8 -*-
"""Phase 1 (chunked): Semantic KG extraction — bản tổng quát của
reextract_nd168_chunked.py cho NHIỀU luật, tránh bug max_text_chars=8000
cắt cụt Điều dài (nghị định xử phạt hành chính hay dính lỗi này).

Điều dài > MAX_CHUNK được tách theo khoản (fallback điểm a/b/c), mỗi chunk
kèm header Điều giữ ngữ cảnh; các chunk được MERGE phía client rồi ghi
1 LẦN/điều (tránh đè penalty id `{article_id}::penalty_{i}`).

Chạy (cwd = ProjectGenAI_2):
    python -m scripts.build_semantic_kg_chunked --dry-run
        # auto-detect mọi luật có Article semantic_done IS NULL, chỉ in kế hoạch
    python -m scripts.build_semantic_kg_chunked
        # chạy thật cho toàn bộ luật còn thiếu
    python -m scripts.build_semantic_kg_chunked --laws "54/2019/QH14,08/2022/QH15"
        # chỉ định luật cụ thể
    python -m scripts.build_semantic_kg_chunked --provider kieai,gemini,groq --delay 3
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Optional

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.chunking import _iter_articles
from src.kg.neo4j_client import Neo4jClient
from src.kg.semantic_extractor import SemanticExtractor
from src.kg.structural_extractor import _article_id, _law_id_from_source, _normalize_for_kg
from src.parsing import load_document
from scripts.ingest import iter_raw_files

MAX_CHUNK = 7500  # < max_text_chars=8000 của SemanticExtractor


def build_raw_index(raw_dir) -> dict:
    """Quét data/raw MỘT LẦN, map doc_number -> (path, meta).

    find_raw_file() gốc quét lại toàn bộ ~51k file cho mỗi luật — quá chậm
    khi xử lý nhiều luật cùng lúc.
    """
    index: dict[str, tuple] = {}
    for path, meta in iter_raw_files(raw_dir):
        if meta.doc_number and meta.doc_number not in index:
            index[meta.doc_number] = (path, meta)
    return index
DEFAULT_CHAIN = "kieai,gemini,groq"
DELAYS = {"kieai": 3.0, "gemini": 4.1, "groq": 10.0, "groq-8b": 7.5, "openrouter": 4.0, "router9": 10.0}

_KHOAN_RE = re.compile(r"(?m)^(?=\d{1,2}\.\s)")
_DIEM_RE = re.compile(r"(?m)^(?=[a-zđ]\)\s)")


def split_article(text: str) -> list[str]:
    """Tách điều dài thành chunks <= MAX_CHUNK, ranh giới là khoản (fallback điểm)."""
    if len(text) <= MAX_CHUNK:
        return [text]
    parts = _KHOAN_RE.split(text)
    header = parts[0].strip()[:400]
    blocks: list[str] = []
    for blk in parts[1:]:
        if len(blk) > MAX_CHUNK - len(header):
            sub = _DIEM_RE.split(blk)
            head2 = sub[0]
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
            idxs = [i + offset for i in (rel.get("penalty_indices") or []) if isinstance(i, int)]
            relations.append({**rel, "penalty_indices": idxs})
    return dict(offenses=offenses, penalties=penalties, subjects=subjects, relations=relations)


def _split_entry(entry: str) -> tuple[str, Optional[str]]:
    if ":" in entry:
        base, model = entry.split(":", 1)
        if base in DELAYS:
            return base, model
    return entry, None


def autodetect_laws(client: Neo4jClient) -> list[str]:
    """Luật có ít nhất 1 Article chưa semantic_done."""
    with client.session() as s:
        rows = s.run(
            """
            MATCH (l:Law)-[:HAS_ARTICLE]->(a:Article)
            WHERE a.semantic_done IS NULL
            RETURN DISTINCT l.doc_number AS docnum
            """
        ).data()
    return [r["docnum"] for r in rows if r["docnum"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laws", default=None, help="Comma-separated doc_number; mặc định auto-detect")
    ap.add_argument("--provider", default=DEFAULT_CHAIN, help="Provider chain, comma-separated")
    ap.add_argument("--delay", type=float, default=None, help="Override throttle giây/call")
    ap.add_argument("--limit-articles", type=int, default=None, help="Test: N điều đầu/luật")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = Neo4jClient.from_env()
    client.create_constraints()

    laws_list = [s.strip() for s in args.laws.split(",")] if args.laws else autodetect_laws(client)
    if not laws_list:
        print("Không có luật nào cần semantic extraction (mọi Article đã semantic_done).")
        client.close()
        return

    providers_chain = [p.strip() for p in args.provider.split(",") if p.strip()]
    chain = [_split_entry(e) for e in providers_chain]
    for base, _m in chain:
        if base not in DELAYS:
            sys.exit(f"⚠ Unknown provider '{base}', valid: {list(DELAYS.keys())}")

    print(f"Semantic KG (chunked) — {len(laws_list)} luật: {laws_list}")
    print(f"Provider chain: {' -> '.join(providers_chain)}\n")

    existing_articles: set[str] = set()
    with client.session() as s:
        rows = s.run("MATCH (a:Article) WHERE a.semantic_done IS NOT NULL RETURN a.id AS id").data()
        existing_articles = {r["id"] for r in rows}
    print(f"{len(existing_articles)} Article đã semantic_done, sẽ skip.\n")

    print("Đang index data/raw (1 lần)...")
    raw_index = build_raw_index(config.RAW_DIR)
    print(f"  -> {len(raw_index)} văn bản có doc_number.\n")

    pidx = 0
    cache: dict[str, SemanticExtractor] = {}

    def get_ex():
        key = providers_chain[pidx]
        base, model = chain[pidx]
        if key not in cache:
            cache[key] = SemanticExtractor(provider=base, model=model)
            print(f"  ⚡ Init provider: {key} (model: {cache[key].model})", flush=True)
        return cache[key], (args.delay if args.delay else DELAYS[base])

    get_ex()

    t_start = time.time()
    n_ok = n_err = n_skip = 0
    n_calls_total = 0

    for law_idx, doc_number in enumerate(laws_list, 1):
        found = raw_index.get(doc_number)
        if found is None:
            print(f"[{law_idx}/{len(laws_list)}] {doc_number}: ⚠ không tìm thấy file raw, skip luật.")
            continue
        path, meta = found
        try:
            doc = load_document(path, meta)
        except Exception as exc:
            print(f"[{law_idx}/{len(laws_list)}] {doc_number}: ⚠ lỗi parse: {exc}")
            continue

        norm = _normalize_for_kg(doc.text)
        law_id = _law_id_from_source(meta.source)
        arts = [(num, t) for _c, num, t in _iter_articles(norm) if num]
        if args.limit_articles:
            arts = arts[: args.limit_articles]

        plan = []
        for num, text in arts:
            art_id = _article_id(law_id, str(num))
            if art_id in existing_articles:
                n_skip += 1
                continue
            plan.append((num, art_id, text, split_article(text)))

        n_calls = sum(len(c) for *_, c in plan)
        n_long = sum(1 for *_, c in plan if len(c) > 1)
        print(f"\n[{law_idx}/{len(laws_list)}] {doc_number}: {len(plan)} điều cần extract "
              f"-> {n_calls} LLM calls ({n_long} điều bị tách chunk)")

        if args.dry_run:
            for num, _aid, text, chunks in plan:
                if len(chunks) > 1:
                    print(f"    Điều {num}: {len(text):,} chars -> {len(chunks)} chunks "
                          f"({[len(c) for c in chunks]})")
            n_calls_total += n_calls
            continue

        for i, (num, art_id, _text, chunks) in enumerate(plan, 1):
            results = []
            err = None
            for ch in chunks:
                while True:
                    ex, delay = get_ex()
                    t0 = time.time()
                    res = ex.extract(art_id, ch)
                    if res.quota_exhausted and pidx + 1 < len(providers_chain):
                        pidx += 1
                        print(f"    ⚡ Quota {providers_chain[pidx-1]} hết -> {providers_chain[pidx]}", flush=True)
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
                print(f"  [{i:3}/{len(plan)}] Điều {num}: ❌ {err[:90]}", flush=True)
                continue
            merged = merge_results(results)
            try:
                if merged["offenses"] or merged["penalties"] or merged["subjects"]:
                    client.write_semantic_extraction(article_id=art_id, **merged)
                client.mark_article_done(art_id)
            except Exception as exc:
                n_err += 1
                print(f"  [{i:3}/{len(plan)}] Điều {num}: ⚠ write lỗi: {exc}", flush=True)
                continue
            n_ok += 1
            print(f"  [{i:3}/{len(plan)}] Điều {num}: * {len(chunks)}ch O={len(merged['offenses'])} "
                  f"P={len(merged['penalties'])} S={len(merged['subjects'])}", flush=True)

    if args.dry_run:
        print(f"\nTổng cộng: {n_calls_total} LLM calls dự kiến cho {len(laws_list)} luật.")
        client.close()
        return

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Xong: {n_ok} OK / {n_err} lỗi / {n_skip} đã skip — {elapsed/60:.1f} phút")

    stats = client.stats()
    print("\nStats Neo4j:")
    for label, n in stats["nodes"].items():
        print(f"  ({label}): {n}")
    for rtype, n in stats["relations"].items():
        print(f"  -[{rtype}]->: {n}")
    client.close()


if __name__ == "__main__":
    main()

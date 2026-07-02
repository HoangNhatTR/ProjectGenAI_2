"""Eval harness ĐO CHẤT LƯỢNG RETRIEVAL (#6) — KHÔNG qua LLM generate.

Đo retrieve+rerank trực tiếp với gold = văn bản nguồn ĐÚNG → hit@k, MRR. Nhanh,
TẤT ĐỊNH, re-run được để so sánh TRƯỚC/SAU mỗi thay đổi ranking (tránh đoán mò
heuristic như bài học bảng-tóm-tắt).

    python -m scripts.eval_retrieval                      # full gold, top-k=10
    python -m scripts.eval_retrieval --top-k 5 --category GT,HS
    python -m scripts.eval_retrieval --limit 5            # smoke nhanh
    python -m scripts.eval_retrieval --tag baseline       # lưu snapshot để diff
    python -m scripts.eval_retrieval --compare baseline   # so sánh với snapshot

Grading: HIT nếu MỘT chunk trong top-k có doc_number chứa MỘT trong expect_doc.
Coverage check: với mỗi expect_doc, đếm chunk trong corpus → phân biệt MISS do
ranking (doc CÓ trong corpus nhưng không lọt top-k) vs do thiếu dữ liệu (0 chunk).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.embedding import Embedder
from src.reranker import rerank as _rerank

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = ROOT / "tests" / "eval_retrieval_gold.json"
REPORT_DIR = ROOT / "data" / "eval_retrieval"


# ── Build retriever (vector + BM25 + parent, KG optional) ─────────────────────

def build_retriever(use_kg: bool):
    from src.retriever import Retriever

    embedder = Embedder(config.EMBEDDING_MODEL)
    if config.VECTOR_BACKEND == "opensearch":
        from src.opensearch_store import OpenSearchLexicalIndex, OpenSearchVectorStore
        vstore = OpenSearchVectorStore(config.OPENSEARCH_URL, config.OPENSEARCH_INDEX)
        bm25 = OpenSearchLexicalIndex(vstore)
    else:
        from src.vectorstore import VectorStore
        vstore = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)
        bm25 = None

    parent_store = None
    if config.PARENT_STORE_PATH.exists():
        from src.parent_store import ParentStore
        parent_store = ParentStore(config.PARENT_STORE_PATH)

    kg_retriever = None
    if use_kg:
        try:
            from src.kg.kg_retriever import KGRetriever
            kg_retriever = KGRetriever()
        except Exception as e:
            print(f"[eval] KG không khả dụng, bỏ qua: {e}")

    return Retriever(
        embedder, vstore, bm25=bm25, kg_retriever=kg_retriever, parent_store=parent_store,
    ), vstore


# ── Grading ───────────────────────────────────────────────────────────────────

def _chunk_match(chunk, gold: dict) -> bool:
    """HIT ở cấp LUẬT: khớp doc_number/source HOẶC điều HOẶC tên luật trong title.

    Corpus phục vụ bộ luật qua nhiều bản (gốc + các Văn bản hợp nhất, số hiệu
    trùng/đổi qua năm) → chấm theo số hiệu cứng là sai. title của VBHN đều chứa
    tên luật gốc ('...hợp nhất Bộ luật Hình sự...') nên expect_title bắt được cả
    bản gốc lẫn mọi bản hợp nhất. expect_article ghim đúng điều (vd Điều 173).
    """
    docnum = (chunk.metadata.doc_number or "").lower()
    src = (chunk.metadata.source or "").lower()
    for d in gold.get("expect_doc", []):
        dl = d.lower()
        if dl in docnum or dl in src:
            return True
    title = (chunk.metadata.title or "").lower()
    for t in gold.get("expect_title", []):
        if t.lower() in title:
            return True
    art = (chunk.article or "").lower()
    for a in gold.get("expect_article", []):
        if a.lower() in art:
            return True
    return False


def first_hit_rank(contexts, gold: dict) -> int:
    """Hạng (1-based) của chunk đầu tiên khớp gold; 0 nếu không có."""
    for i, rc in enumerate(contexts, 1):
        if _chunk_match(rc.chunk, gold):
            return i
    return 0


def coverage(vstore, gold: dict) -> int:
    """Số chunk trong corpus khớp gold (doc_number hoặc title) — sanity 'có dữ liệu'."""
    cli = vstore._connect()
    should = [{"wildcard": {"doc_number": f"*{d}*"}} for d in gold.get("expect_doc", [])]
    should += [{"match_phrase": {"title": t}} for t in gold.get("expect_title", [])]
    if not should:
        return -1
    try:
        res = cli.count(index=vstore.index_name, body={
            "query": {"bool": {"should": should, "minimum_should_match": 1}}
        })
        return int(res["count"])
    except Exception:
        return -1


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Eval chất lượng retrieval (#6)")
    p.add_argument("--gold", default=str(DEFAULT_GOLD))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--category", default=None, help="Lọc nhóm, vd GT,HS")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--kg", action="store_true", help="Bật KG branch (cần Neo4j)")
    p.add_argument("--hyde", action="store_true", help="Bật HyDE (cần LLM client)")
    p.add_argument("--multi-query", action="store_true",
                   help="Bật multi-query fusion — paraphrase + RRF gộp (cần LLM client)")
    p.add_argument("--ce-threshold", type=float, default=0.04,
                   help="Smart-skip CrossEncoder khi top RRF score ≥ ngưỡng (mirror pipeline)")
    p.add_argument("--tag", default=None, help="Lưu snapshot kết quả với tên này")
    p.add_argument("--compare", default=None, help="So sánh với snapshot đã lưu")
    args = p.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    questions = gold["questions"]
    if args.category:
        cats = {c.strip().upper() for c in args.category.split(",")}
        questions = [q for q in questions if q["category"].upper() in cats]
    if args.limit:
        questions = questions[:args.limit]

    print(f"[eval] {len(questions)} câu | top-k={args.top_k} | "
          f"backend={config.VECTOR_BACKEND} | kg={args.kg} hyde={args.hyde} "
          f"mq={args.multi_query}")
    retriever, vstore = build_retriever(args.kg)

    # HyDE / multi-query cần LLM client (fast model) — wire khi được yêu cầu
    if args.hyde or args.multi_query:
        from src.llm_client import create_client
        from src.pipeline import provider_credentials
        _key, _host = provider_credentials(config.LLM_PROVIDER)
        retriever.llm_client = create_client(config.LLM_PROVIDER, _key, _host)
        retriever.hyde_model = config.HYDE_MODEL
        retriever.mq_model = config.MULTI_QUERY_MODEL

    K = args.top_k
    rows = []
    cov_cache: dict[str, int] = {}
    t_start = time.time()

    for n, q in enumerate(questions, 1):
        if not (q.get("expect_doc") or q.get("expect_title") or q.get("expect_article")):
            continue
        key = "|".join(q.get("expect_doc", []) + q.get("expect_title", []))
        if key not in cov_cache:
            cov_cache[key] = coverage(vstore, q)

        t0 = time.time()
        contexts = retriever.retrieve(
            q["question"], top_k=K, use_kg=args.kg,
            use_hyde=args.hyde, use_parent_expansion=True,
            use_multi_query=args.multi_query,
        )
        if contexts:
            use_ce = contexts[0].score < args.ce_threshold
            contexts = _rerank(q["question"], contexts, top_k=K, use_cross_encoder=use_ce)
        dt = time.time() - t0

        rank = first_hit_rank(contexts, q)
        top1_doc = (contexts[0].chunk.metadata.doc_number if contexts else "") or "—"
        rows.append({
            "id": q["id"], "category": q["category"], "question": q["question"],
            "expect_doc": q.get("expect_doc", []), "rank": rank, "corpus_count": cov_cache[key],
            "top1_doc": top1_doc, "secs": round(dt, 2),
        })
        status = f"hit@{rank}" if rank else ("MISS-no-data" if cov_cache[key] == 0 else "MISS")
        print(f"  [{n:>2}/{len(questions)}] {q['id']:<7} {status:<13} "
              f"top1={top1_doc:<18} ({dt:.1f}s) {q['question'][:50]}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def metrics(items):
        if not items:
            return {"n": 0, "hit@1": 0, "hit@3": 0, "hit@5": 0, f"hit@{K}": 0, "mrr": 0}
        n = len(items)
        h1 = sum(1 for r in items if 0 < r["rank"] <= 1)
        h3 = sum(1 for r in items if 0 < r["rank"] <= 3)
        h5 = sum(1 for r in items if 0 < r["rank"] <= 5)
        hk = sum(1 for r in items if r["rank"] > 0)
        mrr = sum((1.0 / r["rank"]) for r in items if r["rank"] > 0) / n
        return {"n": n, "hit@1": h1/n, "hit@3": h3/n, "hit@5": h5/n, f"hit@{K}": hk/n, "mrr": mrr}

    overall = metrics(rows)
    by_cat = {c: metrics([r for r in rows if r["category"] == c])
              for c in sorted({r["category"] for r in rows})}
    no_data = [r for r in rows if r["corpus_count"] == 0]

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"KẾT QUẢ ({overall['n']} câu, {time.time()-t_start:.0f}s tổng)")
    print("=" * 64)
    hdr = f"{'nhóm':<8}{'n':>4}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{('hit@'+str(K)):>8}{'MRR':>8}"
    print(hdr)
    for c, m in by_cat.items():
        print(f"{c:<8}{m['n']:>4}{m['hit@1']:>8.2f}{m['hit@3']:>8.2f}{m['hit@5']:>8.2f}{m[f'hit@{K}']:>8.2f}{m['mrr']:>8.3f}")
    print("-" * 64)
    print(f"{'TỔNG':<8}{overall['n']:>4}{overall['hit@1']:>8.2f}{overall['hit@3']:>8.2f}{overall['hit@5']:>8.2f}{overall[f'hit@{K}']:>8.2f}{overall['mrr']:>8.3f}")
    if no_data:
        print(f"\n⚠ {len(no_data)} câu MISS do THIẾU DỮ LIỆU (doc không có trong corpus): "
              f"{', '.join(r['id'] for r in no_data)}")

    # ── Persist ───────────────────────────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = {"overall": overall, "by_cat": by_cat, "rows": rows,
                "top_k": K, "ts": time.time()}
    (REPORT_DIR / "latest.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(REPORT_DIR / "report.md", snapshot, by_cat, overall, no_data, K)
    print(f"\n[eval] report → {REPORT_DIR/'report.md'}  |  snapshot → {REPORT_DIR/'latest.json'}")

    if args.tag:
        (REPORT_DIR / f"snap_{args.tag}.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] đã lưu snapshot '{args.tag}'")

    if args.compare:
        _compare(REPORT_DIR / f"snap_{args.compare}.json", overall, by_cat, K)


def _write_report(path, snapshot, by_cat, overall, no_data, K):
    lines = ["# Eval Retrieval Report", "",
             f"- top-k: **{K}**  |  số câu: **{overall['n']}**",
             f"- hit@1={overall['hit@1']:.2f}  hit@5={overall['hit@5']:.2f}  "
             f"hit@{K}={overall[f'hit@{K}']:.2f}  MRR={overall['mrr']:.3f}", "",
             "## Theo nhóm", "",
             f"| nhóm | n | hit@1 | hit@3 | hit@5 | hit@{K} | MRR |",
             "|---|---|---|---|---|---|---|"]
    for c, m in by_cat.items():
        lines.append(f"| {c} | {m['n']} | {m['hit@1']:.2f} | {m['hit@3']:.2f} | "
                     f"{m['hit@5']:.2f} | {m[f'hit@{K}']:.2f} | {m['mrr']:.3f} |")
    lines += ["", "## Chi tiết từng câu", "",
              "| id | nhóm | rank | top1_doc | corpus | câu |",
              "|---|---|---|---|---|---|"]
    for r in snapshot["rows"]:
        rk = r["rank"] or "MISS"
        lines.append(f"| {r['id']} | {r['category']} | {rk} | {r['top1_doc']} | "
                     f"{r['corpus_count']} | {r['question'][:60]} |")
    if no_data:
        lines += ["", f"> ⚠ Thiếu dữ liệu: {', '.join(r['id'] for r in no_data)}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def _compare(snap_path: Path, overall, by_cat, K):
    if not snap_path.exists():
        print(f"\n[eval] không thấy snapshot để so sánh: {snap_path}")
        return
    old = json.loads(snap_path.read_text(encoding="utf-8"))
    o = old["overall"]
    print("\n" + "=" * 64)
    print(f"SO SÁNH với snapshot (Δ = mới − cũ)")
    print("=" * 64)
    for m in ("hit@1", "hit@3", "hit@5", f"hit@{K}", "mrr"):
        if m in o and m in overall:
            d = overall[m] - o[m]
            arrow = "▲" if d > 0 else ("▼" if d < 0 else "=")
            print(f"  {m:<8} {o[m]:.3f} → {overall[m]:.3f}  ({arrow} {d:+.3f})")


if __name__ == "__main__":
    main()

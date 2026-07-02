"""Phase 0: Build Structural KG cho toàn bộ data/raw/all_laws/.

Pipeline:
  1. Iterate raw files (reuse scripts.ingest.iter_raw_files)
  2. parse + extract structural entities (no LLM)
  3. Bulk upsert vào Neo4j Aura

Chạy:
    python -m scripts.build_structural_kg
    python -m scripts.build_structural_kg --topic all_laws --limit 10
    python -m scripts.build_structural_kg --reset       # xoá KG cũ trước khi build
    python -m scripts.build_structural_kg --dry-run     # chỉ in stats, không write
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config  # load_dotenv() runs here
from src.kg.neo4j_client import Neo4jClient
from src.kg.structural_extractor import (
    extract_structural,
    dedup_citations,
    resolve_amendments,
)
from src.parsing import load_document
from scripts.ingest import iter_raw_files


BATCH_SIZE = 200  # số doc/batch khi flush vào Neo4j


def main() -> None:
    p = argparse.ArgumentParser(description="Build Structural KG vào Neo4j")
    p.add_argument("--topic", default=None, help="Chỉ build 1 chủ đề")
    p.add_argument("--limit", type=int, default=None, help="Giới hạn N file (test)")
    p.add_argument("--laws", default=None,
                   help="Chỉ build các doc_number này (comma-separated). Bỏ qua file khác.")
    p.add_argument("--top15", action="store_true",
                   help="Chỉ build 15 luật trong DEFAULT_TOP_LAWS (benchmark graph, vừa Aura Free).")
    p.add_argument("--reset", action="store_true", help="Xoá KG cũ trước khi build")
    p.add_argument("--dry-run", action="store_true",
                   help="Chỉ extract, không write Neo4j (xem stats)")
    args = p.parse_args()

    # Filter theo doc_number: --top15 dùng whitelist của semantic build (DRY), --laws tự nhập
    laws_filter: set[str] | None = None
    if args.top15:
        from scripts.build_semantic_kg import DEFAULT_TOP_LAWS
        laws_filter = set(DEFAULT_TOP_LAWS)
    elif args.laws:
        laws_filter = {s.strip() for s in args.laws.split(",") if s.strip()}
    if laws_filter is not None:
        print(f"Chỉ build {len(laws_filter)} luật (filter doc_number)")

    client = None
    if not args.dry_run:
        print("Kết nối Neo4j Aura...")
        client = Neo4jClient.from_env()
        if args.reset:
            print("  Reset KG cũ...")
            client.reset_all()
        client.create_constraints()
        print("  ✓ Constraints OK")

    # ── Buffer cho bulk upsert ──────────────────────────────────────────────
    law_buf: list[dict] = []
    article_buf: list[dict] = []
    clause_buf: list[dict] = []
    cites_buf: list[dict] = []
    amend_pairs: list[tuple[str, str]] = []  # (src_law_id, target_doc_number)
    docnum_to_law_id: dict[str, str] = {}    # build trong khi duyệt
    all_article_ids: set[str] = set()

    n_docs = n_articles = n_clauses = n_cites = 0
    n_errors = 0

    def flush() -> None:
        if not client:
            law_buf.clear()
            article_buf.clear()
            clause_buf.clear()
            cites_buf.clear()
            return
        if law_buf:
            client.upsert_laws(law_buf)
        if article_buf:
            client.upsert_articles(article_buf)
        if clause_buf:
            client.upsert_clauses(clause_buf)
        # Lọc citations: chỉ giữ target Article đã có trong KG
        good_cites = dedup_citations(cites_buf, all_article_ids)
        if good_cites:
            client.add_internal_citations(good_cites)
        law_buf.clear()
        article_buf.clear()
        clause_buf.clear()
        cites_buf.clear()

    # ── Main loop ───────────────────────────────────────────────────────────
    print(f"\nĐang quét data/raw/{'all topics' if not args.topic else args.topic}...")
    for i, (path, meta) in enumerate(iter_raw_files(config.RAW_DIR, topic_filter=args.topic)):
        if args.limit and i >= args.limit:
            break
        if laws_filter is not None and meta.doc_number not in laws_filter:
            continue
        try:
            doc = load_document(path, meta)
            result = extract_structural(doc)
        except Exception as exc:
            print(f"  [!] Lỗi {path.name}: {exc}")
            n_errors += 1
            continue

        law_node = result["law_node"]
        law_buf.append(law_node)
        article_buf.extend(result["article_nodes"])
        clause_buf.extend(result["clause_nodes"])
        cites_buf.extend(result["internal_cites"])

        for a in result["article_nodes"]:
            all_article_ids.add(a["id"])

        # Track amendment targets
        if meta.doc_number:
            docnum_to_law_id[meta.doc_number] = law_node["id"]
        for raw_target in result["amend_targets_raw"]:
            amend_pairs.append((law_node["id"], raw_target))

        n_docs += 1
        n_articles += len(result["article_nodes"])
        n_clauses += len(result["clause_nodes"])
        n_cites += len(result["internal_cites"])

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] {n_docs} docs, {n_articles} Điều, {n_clauses} Khoản")

        if len(law_buf) >= BATCH_SIZE:
            flush()

    # Flush phần còn lại
    flush()

    # ── Amendment edges (sau khi đã có hết Law trong KG) ────────────────────
    n_amendments = 0
    if client and amend_pairs:
        amend_edges = resolve_amendments(amend_pairs, docnum_to_law_id)
        if amend_edges:
            client.add_amendment_edges(amend_edges)
            n_amendments = len(amend_edges)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Đã xử lý: {n_docs} luật ({n_errors} lỗi)")
    print(f"  Articles: {n_articles}")
    print(f"  Clauses:  {n_clauses}")
    print(f"  Internal citations: {n_cites} (chưa lọc)")
    print(f"  Amendment edges:    {n_amendments}")

    if client:
        print("\nStats từ Neo4j:")
        stats = client.stats()
        for label, n in stats["nodes"].items():
            print(f"  ({label}): {n}")
        for rtype, n in stats["relations"].items():
            print(f"  -[{rtype}]->: {n}")
        client.close()
    else:
        print("\n(dry-run, không write vào Neo4j)")


if __name__ == "__main__":
    main()

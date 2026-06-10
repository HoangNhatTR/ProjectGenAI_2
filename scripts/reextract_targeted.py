"""Re-extract CHỈ các Article cụ thể (không quét cả 3.238 articles).

Dùng cho cleanup workflow: sau khi clear edges của các Article anomalous,
script này re-extract chính xác chúng với prompt cải tiến.

Chạy:
    # Default: re-extract 15 Article anomalous đã được clear trong cleanup
    python -m scripts.reextract_targeted

    # Custom list
    python -m scripts.reextract_targeted --laws 100/2015/QH13 --articles 7,12,14,51

    # Auto: tự tìm Article đã clear (không có edge PENALIZES/IMPOSES) trong top 15 luật
    python -m scripts.reextract_targeted --auto
"""
from __future__ import annotations

import argparse
import sys
import time

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
from scripts.build_semantic_kg import DEFAULT_TOP_LAWS, DELAYS, find_raw_file


# Default 15 anomalous Articles bị clear trong cleanup (theo log cleanup_kg.py)
DEFAULT_TARGETS = [
    ("100/2015/QH13", [7, 12, 14, 51, 52, 76, 123, 151, 175, 178, 311]),
    ("52/2014/QH13",  [5]),
    ("15/2023/QH15",  [7]),
    ("19/2023/QH15",  [10]),
    ("36/2024/QH15",  [33]),
]


def find_cleared_articles(client: Neo4jClient) -> list[tuple[str, int]]:
    """Tự tìm Article trong top 15 luật mà KHÔNG có edge semantic.

    Article có edge PENALIZES/IMPOSES → đã extract, skip.
    Article không có edge → cần extract (bao gồm cả Article định nghĩa rỗng).
    """
    with client.session() as s:
        rows = s.run(
            """
            MATCH (l:Law)-[:HAS_ARTICLE]->(a:Article)
            WHERE l.doc_number IN $laws
              AND NOT EXISTS { MATCH (a)-[:HAS_CLAUSE]->(:Clause)-[:PENALIZES|IMPOSES]->() }
              AND NOT EXISTS { MATCH (a)-[:PENALIZES|IMPOSES]->() }
            RETURN l.doc_number AS law, a.number AS num
            ORDER BY law, num
            """,
            laws=DEFAULT_TOP_LAWS,
        ).data()
    return [(r["law"], r["num"]) for r in rows]


def main() -> None:
    p = argparse.ArgumentParser(description="Re-extract Article cụ thể")
    p.add_argument("--laws", help="Comma-separated doc_numbers (vd: 100/2015/QH13,52/2014/QH13)")
    p.add_argument("--articles", help="Comma-separated article numbers (vd: 12,51,76)")
    p.add_argument("--auto", action="store_true",
                   help="Tự tìm Article đã cleared trong top 15 luật")
    p.add_argument("--provider", default="gemini,groq,groq-8b",
                   help="Provider chain (default: gemini,groq,groq-8b)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # ── Build targets list ────────────────────────────────────────────
    client = Neo4jClient.from_env()

    if args.auto:
        targets = find_cleared_articles(client)
        print(f"AUTO mode: tìm thấy {len(targets)} Article không có semantic edges trong top 15 luật.")
    elif args.laws and args.articles:
        law_nums = [s.strip() for s in args.laws.split(",")]
        art_nums = [int(x.strip()) for x in args.articles.split(",")]
        targets = [(law, n) for law in law_nums for n in art_nums]
    else:
        # Default: 15 anomalous
        targets = []
        for law, arts in DEFAULT_TARGETS:
            for n in arts:
                targets.append((law, n))
        print(f"Dùng default 15 anomalous Articles.")

    print(f"\nSẽ re-extract {len(targets)} Article:")
    for law, num in targets[:20]:
        print(f"  {law:<18s} Điều {num}")
    if len(targets) > 20:
        print(f"  ... và {len(targets) - 20} nữa")
    print()

    if args.dry_run:
        client.close()
        return

    # ── Group by law để load file 1 lần ───────────────────────────────
    by_law: dict[str, list[int]] = {}
    for law, num in targets:
        by_law.setdefault(law, []).append(num)

    # ── Init extractor chain ──────────────────────────────────────────
    providers = [p.strip() for p in args.provider.split(",")]
    extractors_cache: dict[str, SemanticExtractor] = {}
    provider_idx = 0

    def get_extractor():
        p_name = providers[provider_idx]
        if p_name not in extractors_cache:
            extractors_cache[p_name] = SemanticExtractor(provider=p_name)
            print(f"⚡ Init provider: {p_name} (model: {extractors_cache[p_name].model})")
        return extractors_cache[p_name], p_name, DELAYS.get(p_name, 4.0)

    _ = get_extractor()

    n_ok = 0
    n_err = 0
    n_offense = n_penalty = n_subject = 0
    t_start = time.time()

    # ── Loop ──────────────────────────────────────────────────────────
    for law_num, art_nums in by_law.items():
        print(f"\n=== Luật {law_num} — {len(art_nums)} điều ===")
        found = find_raw_file(law_num, config.RAW_DIR)
        if not found:
            print(f"  ⚠ Không tìm thấy file raw cho {law_num}, skip.")
            continue
        path, meta = found

        try:
            doc = load_document(path, meta)
        except Exception as exc:
            print(f"  ⚠ Lỗi parse: {exc}")
            continue

        normalized = _normalize_for_kg(doc.text)
        law_id = _law_id_from_source(meta.source)
        target_set = set(str(n) for n in art_nums)

        for _chap, a, text in _iter_articles(normalized):
            if a not in target_set:
                continue

            art_id = _article_id(law_id, a)
            ex, current_p, delay = get_extractor()
            t0 = time.time()
            result = ex.extract(art_id, text)
            elapsed = time.time() - t0

            if result.quota_exhausted:
                if provider_idx + 1 < len(providers):
                    provider_idx += 1
                    print(f"    ⚡ Quota {current_p} hết → switch sang {providers[provider_idx]}")
                    ex, current_p, delay = get_extractor()
                    t0 = time.time()
                    result = ex.extract(art_id, text)
                    elapsed = time.time() - t0
                else:
                    print(f"    ❌ Hết tất cả providers")
                    break

            if result.error:
                n_err += 1
                print(f"  Điều {a}: ❌ {result.error[:100]}")
            else:
                n_ok += 1
                n_offense += len(result.offenses)
                n_penalty += len(result.penalties)
                n_subject += len(result.subjects)
                marker = "·" if result.is_empty else "★"
                print(f"  Điều {a}: {marker} O={len(result.offenses)} P={len(result.penalties)} S={len(result.subjects)} ({elapsed:.1f}s)")

                if not result.is_empty:
                    try:
                        client.write_semantic_extraction(
                            article_id=art_id,
                            offenses=result.offenses,
                            penalties=result.penalties,
                            subjects=result.subjects,
                            relations=result.relations,
                        )
                    except Exception as exc:
                        print(f"        ⚠ Lỗi write Neo4j: {exc}")

            # Throttle
            sleep_needed = delay - elapsed
            if sleep_needed > 0:
                time.sleep(sleep_needed)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Total: {n_ok + n_err} điều ({n_ok} OK, {n_err} lỗi)")
    print(f"Extracted: {n_offense} Offenses, {n_penalty} Penalties, {n_subject} Subjects")
    print(f"Time: {elapsed/60:.1f} phút")
    client.close()


if __name__ == "__main__":
    main()

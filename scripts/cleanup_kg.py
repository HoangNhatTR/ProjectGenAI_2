"""Cleanup KG Phase 1 noise:

  1. Xoá Offense generic (phân loại tội phạm, không phải hành vi cụ thể)
  2. Xoá edges PENALIZES/IMPOSES/APPLIES_TO/PUNISHED_BY của Article có
     extraction bất thường (>= 15 Offense — thường là Điều định nghĩa)
     → để Phase 1 re-run sẽ extract lại đúng
  3. Merge case-insensitive duplicate Subject
  4. Xoá orphan Offense/Penalty/Subject (không còn edge nào)

Chạy:
    python -m scripts.cleanup_kg              # full cleanup
    python -m scripts.cleanup_kg --dry-run    # chỉ in stats
"""
from __future__ import annotations

import argparse
import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.kg.neo4j_client import Neo4jClient


# Generic Offense names — không phải hành vi cụ thể, sẽ xoá
GENERIC_OFFENSES = [
    "phạm tội",
    "Phạm tội",
    "tội phạm",
    "Tội phạm",
    "tội phạm nghiêm trọng",
    "tội phạm ít nghiêm trọng",
    "tội phạm rất nghiêm trọng",
    "tội phạm đặc biệt nghiêm trọng",
    "phạm tội ít nghiêm trọng",
    "phạm tội nghiêm trọng",
    "phạm tội rất nghiêm trọng",
    "phạm tội đặc biệt nghiêm trọng",
    "phạm tội thuộc một trong các trường hợp",
    "phạm tội thuộc một trong các trường hợp sau đây",
    "hành vi vi phạm",
    "vi phạm pháp luật",
    "vi phạm quy định",
    "cấu thành tội phạm",
    "trách nhiệm hình sự",
    "miễn trách nhiệm",
]

# Ngưỡng anomaly: Article có >= N Offense → có thể bị extract sai
ANOMALY_OFFENSE_THRESHOLD = 15


def section(title: str) -> None:
    print("\n" + "═" * 75)
    print(" " + title)
    print("═" * 75)


def main() -> None:
    p = argparse.ArgumentParser(description="Cleanup KG noise")
    p.add_argument("--dry-run", action="store_true", help="Chỉ in, không xoá")
    p.add_argument("--anomaly-threshold", type=int, default=ANOMALY_OFFENSE_THRESHOLD,
                   help="Article có >=N Offense sẽ clear (default 15)")
    args = p.parse_args()

    c = Neo4jClient.from_env()

    with c.session() as s:
        # ──────────────────────────────────────────────
        # PHASE A: Xoá generic Offense
        # ──────────────────────────────────────────────
        section("PHASE A: Xoá Generic Offense (phân loại, không phải hành vi)")

        rows = s.run(
            "MATCH (o:Offense) WHERE o.name IN $names "
            "RETURN o.name AS name, count{ (o)<-[]-() } AS n_in",
            names=GENERIC_OFFENSES,
        ).data()
        if not rows:
            print("  ✓ Không có Generic Offense — đã clean trước đó.")
        else:
            print(f"  Sẽ xoá {len(rows)} Generic Offense:")
            for r in rows:
                print(f"    • \"{r['name']}\" ({r['n_in']} edges)")

            if not args.dry_run:
                result = s.run(
                    "MATCH (o:Offense) WHERE o.name IN $names DETACH DELETE o",
                    names=GENERIC_OFFENSES,
                )
                summary = result.consume()
                print(f"  ✓ Đã xoá {summary.counters.nodes_deleted} nodes, "
                      f"{summary.counters.relationships_deleted} edges.")

        # ──────────────────────────────────────────────
        # PHASE B: Clear semantic edges của Article anomalous
        # ──────────────────────────────────────────────
        section(f"PHASE B: Clear semantic edges của Article >= {args.anomaly_threshold} Offense")

        # Tìm Article có nhiều Offense
        anomalous = s.run("""
            MATCH (a:Article)
            OPTIONAL MATCH (a)-[:HAS_CLAUSE]->(:Clause)-[:PENALIZES]->(o1:Offense)
            OPTIONAL MATCH (a)-[:PENALIZES]->(o2:Offense)
            WITH a, count(DISTINCT o1) + count(DISTINCT o2) AS n_offense
            WHERE n_offense >= $threshold
            MATCH (l:Law)-[:HAS_ARTICLE]->(a)
            RETURN a.id AS aid, l.doc_number AS law, a.number AS dieu, n_offense
            ORDER BY n_offense DESC
        """, threshold=args.anomaly_threshold).data()

        print(f"  Tìm thấy {len(anomalous)} Article có >= {args.anomaly_threshold} Offense:")
        for r in anomalous[:20]:
            print(f"    {r['law']:<18s} Điều {r['dieu']:<4} → {r['n_offense']} Offense")
        if len(anomalous) > 20:
            print(f"    ... và {len(anomalous) - 20} Article nữa")

        if anomalous and not args.dry_run:
            article_ids = [r["aid"] for r in anomalous]
            # Xoá tất cả edges semantic từ các Article này
            result = s.run("""
                MATCH (a:Article) WHERE a.id IN $aids
                OPTIONAL MATCH (a)-[r1:PENALIZES|IMPOSES]->()
                OPTIONAL MATCH (a)-[:HAS_CLAUSE]->(:Clause)-[r2:PENALIZES|IMPOSES]->()
                DELETE r1, r2
            """, aids=article_ids)
            summary = result.consume()
            print(f"  ✓ Đã xoá {summary.counters.relationships_deleted} edges. "
                  f"Phase 1 re-run sẽ extract lại các Article này.")

        # ──────────────────────────────────────────────
        # PHASE C: Merge duplicate Subject case-insensitive
        # ──────────────────────────────────────────────
        section("PHASE C: Merge duplicate Subject (case-insensitive)")

        dups = s.run("""
            MATCH (sub:Subject)
            WITH toLower(sub.name) AS key, collect(sub) AS subs, count(*) AS n
            WHERE n > 1
            RETURN key, subs, n
            ORDER BY n DESC
        """).data()

        print(f"  Tìm thấy {len(dups)} nhóm duplicate:")
        merge_count = 0
        for d in dups[:10]:
            variants = [sub["name"] for sub in d["subs"]]
            print(f"    \"{d['key']}\" ({d['n']} variants): {variants}")

        if dups and not args.dry_run:
            # Manual merge: chọn canonical (variant đầu tiên), reattach edges, delete others
            for d in dups:
                subs = d["subs"]
                canonical = subs[0]
                others = subs[1:]
                canonical_name = canonical["name"]
                for other in others:
                    other_name = other["name"]
                    # Reattach edges incoming
                    s.run("""
                        MATCH (other:Subject {name: $oname})<-[r:APPLIES_TO]-(o)
                        MATCH (can:Subject {name: $cname})
                        MERGE (o)-[:APPLIES_TO]->(can)
                        DELETE r
                    """, oname=other_name, cname=canonical_name)
                    # Delete the duplicate
                    s.run("MATCH (n:Subject {name: $n}) DETACH DELETE n", n=other_name)
                    merge_count += 1
            print(f"  ✓ Đã merge {merge_count} duplicate Subject vào {len(dups)} canonical.")

        # ──────────────────────────────────────────────
        # PHASE D: Xoá orphan nodes (không còn edge nào)
        # ──────────────────────────────────────────────
        section("PHASE D: Xoá orphan Offense/Penalty/Subject")

        for label in ("Offense", "Penalty", "Subject"):
            n_orphan = s.run(
                f"MATCH (n:{label}) WHERE NOT (n)--() RETURN count(n) AS n"
            ).single()["n"]
            print(f"  Orphan {label}: {n_orphan}")
            if n_orphan and not args.dry_run:
                result = s.run(
                    f"MATCH (n:{label}) WHERE NOT (n)--() DETACH DELETE n"
                )
                summary = result.consume()
                print(f"    ✓ Đã xoá {summary.counters.nodes_deleted}")

        # ──────────────────────────────────────────────
        # SUMMARY
        # ──────────────────────────────────────────────
        section("KG SAU CLEANUP")
        stats = c.stats()
        print("\n  Nodes:")
        for label, n in stats["nodes"].items():
            print(f"    ({label:<10})  {n:>8,}")
        print("\n  Edges:")
        for rtype, n in stats["relations"].items():
            print(f"    -[:{rtype:<14}]->  {n:>8,}")

    c.close()
    print("\n✓ Cleanup done. Run `python -m scripts.build_semantic_kg --skip-existing --provider router9,gemini,groq,groq-8b` để re-extract.")


if __name__ == "__main__":
    main()

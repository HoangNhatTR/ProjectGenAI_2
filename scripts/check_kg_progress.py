"""Quick check tiến độ Phase 1 — query Neo4j stats.

Chạy bất cứ lúc nào (không gián đoạn Phase 1):
    python -m scripts.check_kg_progress
"""
from __future__ import annotations

import sys
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.kg.neo4j_client import Neo4jClient


def main() -> None:
    client = Neo4jClient.from_env()
    stats = client.stats()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] KG Stats")
    print("-" * 50)

    # Phase 0 (structural — sẽ không tăng)
    print("Phase 0 (structural):")
    for k in ("Law", "Article", "Clause"):
        v = stats["nodes"].get(k, 0)
        print(f"  {k:10s}: {v:>10,}")

    # Phase 1 (semantic — đang tăng)
    print("\nPhase 1 (semantic — đang chạy):")
    for k in ("Offense", "Penalty", "Subject"):
        v = stats["nodes"].get(k, 0)
        print(f"  {k:10s}: {v:>10,}")

    # Edges từ Phase 1
    print("\nEdges Phase 1:")
    for k in ("PENALIZES", "IMPOSES", "APPLIES_TO", "PUNISHED_BY"):
        v = stats["relations"].get(k, 0)
        print(f"  {k:14s}: {v:>10,}")

    # Estimate progress
    print("\n" + "=" * 50)
    n_offense = stats["nodes"].get("Offense", 0)
    n_penalty = stats["nodes"].get("Penalty", 0)
    if n_offense + n_penalty == 0:
        print("⏳ Phase 1 vẫn đang khởi động hoặc đang xử lý các Điều chưa có Offense.")
    else:
        # Rough estimate: top 10 luật ~2650 điều, ~30% có Offense
        # → expect ~800 Offense nodes total (vì dedup theo name)
        # Mỗi 100 Offense ≈ 13% progress
        approx_pct = min(100, n_offense * 100 // 100)  # very rough
        print(f"📈 Đã extract ~{n_offense} Offense / ~Penalty {n_penalty}")
        print(f"   (Phase 1 hoàn thành khi không tăng thêm sau 2-3 lần check)")

    client.close()


if __name__ == "__main__":
    main()

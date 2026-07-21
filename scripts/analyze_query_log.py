"""Phân tích data/query_log.db — dùng để ƯU TIÊN P2.1 (backfill hiệu lực) theo
luật THẬT SỰ được hỏi nhiều, và xem phân bố nhãn độ tin cậy (P2.3) trên
traffic thật thay vì chỉ trên gold set nhỏ. Xem docs/tro-ly-luat-su/lo-trinh-cong-viec.md P2.1.

    python -m scripts.analyze_query_log                # toàn bộ log
    python -m scripts.analyze_query_log --days 7        # 7 ngày gần nhất
    python -m scripts.analyze_query_log --top 30
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import Counter

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src.query_log import DB_PATH


def main() -> None:
    p = argparse.ArgumentParser(description="Phân tích log truy vấn chat")
    p.add_argument("--days", type=int, default=None, help="Chỉ lấy N ngày gần nhất")
    p.add_argument("--top", type=int, default=20, help="Số câu hỏi phổ biến hiển thị")
    p.add_argument("--db", default=str(DB_PATH))
    args = p.parse_args()

    if not DB_PATH.exists() and args.db == str(DB_PATH):
        print(f"[analyze_query_log] chưa có log tại {DB_PATH} — chưa có câu hỏi nào được ghi.")
        return

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    where = ""
    params: tuple = ()
    if args.days:
        where = "WHERE ts >= ?"
        params = (time.time() - args.days * 86400,)

    rows = [dict(r) for r in conn.execute(f"SELECT * FROM query_log {where} ORDER BY ts", params)]
    conn.close()

    if not rows:
        print("[analyze_query_log] không có dòng nào khớp bộ lọc.")
        return

    print(f"[analyze_query_log] {len(rows)} câu hỏi"
          f"{f' ({args.days} ngày gần nhất)' if args.days else ' (toàn bộ log)'}")

    # ── Phân bố nhãn độ tin cậy ───────────────────────────────────────────────
    conf_counts = Counter(r["confidence_label"] or "(không áp dụng)" for r in rows)
    print("\n== Phân bố nhãn độ tin cậy ==")
    for label, n in conf_counts.most_common():
        print(f"  {label:<20}{n:>6}  ({n/len(rows):.0%})")

    # ── Phân bố rag_mode / model ─────────────────────────────────────────────
    mode_counts = Counter(r["rag_mode"] or "?" for r in rows)
    print("\n== Phân bố rag_mode ==")
    for mode, n in mode_counts.most_common():
        print(f"  {mode:<20}{n:>6}")

    # ── Câu hỏi phổ biến (nguyên văn, để tay soi lại luật nào hay được hỏi) ──
    q_counts = Counter(r["question"] for r in rows)
    print(f"\n== Top {args.top} câu hỏi (nguyên văn, trùng lặp) ==")
    for q, n in q_counts.most_common(args.top):
        print(f"  {n:>4}x  {q[:80]}")

    # ── Thời gian phản hồi ────────────────────────────────────────────────────
    elapsed = [r["elapsed_s"] for r in rows if r["elapsed_s"] is not None]
    if elapsed:
        elapsed.sort()
        p50 = elapsed[len(elapsed) // 2]
        p90 = elapsed[int(len(elapsed) * 0.9)]
        print(f"\n== Thời gian phản hồi == p50={p50:.1f}s  p90={p90:.1f}s  max={max(elapsed):.1f}s")


if __name__ == "__main__":
    main()

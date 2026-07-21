"""Log truy vấn chat (câu hỏi, mode, model, độ tin cậy, thời gian) — dữ liệu
CẦN THIẾT để ưu tiên P2.1 (backfill hiệu lực) theo luật thật sự được hỏi nhiều
thay vì đoán, và để biết P2.3 (nhãn độ tin cậy) đang phân bố thế nào trên
traffic thật. Xem docs/tro-ly-luat-su/lo-trinh-cong-viec.md P2.1.

Ghi KHÔNG BAO GIỜ được làm hỏng/chậm câu trả lời cho người dùng — mọi lỗi ghi
log bị NUỐT (log ra console, không raise) vì đây là dữ liệu phân tích phụ,
không phải luồng chính.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

from . import config
from .schemas import Answer

DB_PATH = config.DATA_DIR / "query_log.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL NOT NULL,
    question        TEXT NOT NULL,
    rag_mode        TEXT,
    model           TEXT,
    n_citations     INTEGER,
    confidence_label TEXT,
    top1_score      REAL,
    n_sources       INTEGER,
    elapsed_s       REAL
);
CREATE INDEX IF NOT EXISTS idx_query_log_ts ON query_log(ts);
"""


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def log_query(
    question: str, rag_mode: str, model: str, answer: Optional[Answer], elapsed_s: float,
    db_path: Path = DB_PATH,
) -> None:
    """Ghi 1 dòng — best-effort, KHÔNG BAO GIỜ raise (xem docstring module)."""
    try:
        conf = answer.confidence if answer else None
        with _connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO query_log
                  (ts, question, rag_mode, model, n_citations, confidence_label,
                   top1_score, n_sources, elapsed_s)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    time.time(), (question or "")[:500], rag_mode, model,
                    len(answer.citations) if answer else 0,
                    conf.label if conf else None,
                    conf.top1_score if conf else None,
                    conf.n_sources if conf else None,
                    elapsed_s,
                ),
            )
            conn.commit()
    except Exception as exc:  # noqa: BLE001 — logging never breaks the response
        print(f"[query_log] ghi lỗi (bỏ qua, không ảnh hưởng câu trả lời): {exc}", flush=True)

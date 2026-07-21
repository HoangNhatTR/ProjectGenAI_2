"""Test log truy vấn (src/query_log.py) — offline, SQLite tmp_path."""
from __future__ import annotations

import sqlite3

from src.confidence import ConfidenceInfo
from src.query_log import log_query
from src.schemas import Answer, Citation


def _rows(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM query_log ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def test_log_query_inserts_row_with_confidence(tmp_path):
    db_path = tmp_path / "q.db"
    answer = Answer(
        question="Vượt đèn đỏ phạt bao nhiêu?", answer="4-6 triệu đồng",
        citations=[Citation(source="x", snippet="", ref=1)],
        confidence=ConfidenceInfo(
            label="cao", label_vi="Cao", reasons_vi=["ok"],
            top1_score=0.9, n_sources=2, citation_pass_rate=1.0, has_expired_source=False,
        ),
    )
    log_query("Vượt đèn đỏ phạt bao nhiêu?", "legal-ai-graph", "legal-ai-graph", answer, 1.23, db_path=db_path)

    rows = _rows(db_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["question"] == "Vượt đèn đỏ phạt bao nhiêu?"
    assert r["rag_mode"] == "legal-ai-graph"
    assert r["n_citations"] == 1
    assert r["confidence_label"] == "cao"
    assert r["top1_score"] == 0.9
    assert r["n_sources"] == 2
    assert r["elapsed_s"] == 1.23


def test_log_query_handles_none_answer(tmp_path):
    db_path = tmp_path / "q.db"
    log_query("câu hỏi", "legal-ai-graph", "m", None, 0.5, db_path=db_path)
    rows = _rows(db_path)
    assert len(rows) == 1
    assert rows[0]["n_citations"] == 0
    assert rows[0]["confidence_label"] is None


def test_log_query_handles_answer_without_confidence(tmp_path):
    db_path = tmp_path / "q.db"
    answer = Answer(question="q", answer="a", citations=[])
    log_query("q", "legal-ai-graph", "m", answer, 0.1, db_path=db_path)
    rows = _rows(db_path)
    assert rows[0]["confidence_label"] is None
    assert rows[0]["top1_score"] is None


def test_log_query_never_raises_on_bad_path(monkeypatch, capsys):
    import src.query_log as mod

    def _boom(db_path):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(mod, "_connect", _boom)
    # KHÔNG được raise — đây là lời hứa cốt lõi của module (logging phụ,
    # không được làm hỏng câu trả lời chính).
    log_query("q", "legal-ai-graph", "m", None, 0.1)
    assert "ghi lỗi" in capsys.readouterr().out


def test_log_query_accumulates_multiple_rows(tmp_path):
    db_path = tmp_path / "q.db"
    for i in range(3):
        log_query(f"câu {i}", "legal-ai-graph", "m", None, 0.1, db_path=db_path)
    assert len(_rows(db_path)) == 3

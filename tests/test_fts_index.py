"""Tests cho ChromaFTSIndex — nhánh lexical qua FTS5 của Chroma."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.fts_index import ChromaFTSIndex, _build_match

_LOCAL_DB = Path(__file__).resolve().parent.parent / "data" / "vectorstore" / "chroma.sqlite3"


def test_build_match_cum_2_tu_va_tu_dai():
    m = _build_match("xe máy vượt đèn đỏ")
    assert '"xe máy"' in m
    assert '"đèn đỏ"' in m
    assert '"vượt"' in m          # từ đơn ≥4 ký tự
    assert '"xe"' not in m.replace('"xe máy"', "")  # từ 2 ký tự không đứng riêng
    assert " OR " in m


def test_build_match_rong():
    assert _build_match("") == ""
    assert _build_match("  ") == ""


def test_khong_co_db_thi_khong_available(tmp_path):
    fts = ChromaFTSIndex(tmp_path)
    assert fts.is_available() is False
    assert fts.query("đèn đỏ") == []


@pytest.mark.skipif(not _LOCAL_DB.exists(), reason="không có chroma.sqlite3 local")
def test_query_tren_store_that():
    fts = ChromaFTSIndex(_LOCAL_DB.parent)
    assert fts.is_available()
    hits = fts.query("xe máy vượt đèn đỏ bị phạt bao nhiêu tiền", top_k=5)
    assert hits, "phải có kết quả lexical"
    # Xếp hạng giảm dần theo score
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    # Chunk có text + metadata source
    c0 = hits[0][0]
    assert c0.text and c0.metadata.source

"""Unit tests cho opensearch_store — phần thuần logic, không cần server."""
from __future__ import annotations

import pytest

from src.opensearch_store import (
    EMBED_DIM,
    INDEX_MAPPING,
    _hit_to_chunk,
    _where_to_filter,
    chunk_to_doc,
)
from src.schemas import Chunk, DocumentMetadata


# ── _where_to_filter ───────────────────────────────────────────────────────────

def test_where_none_va_rong():
    assert _where_to_filter(None) is None
    assert _where_to_filter({}) is None


def test_where_eq():
    f = _where_to_filter({"source": {"$eq": "https://x/1"}})
    assert f == {"term": {"source": "https://x/1"}}


def test_where_shorthand():
    f = _where_to_filter({"article": "Điều 8"})
    assert f == {"term": {"article": "Điều 8"}}


def test_where_in():
    f = _where_to_filter({"source": {"$in": ["a", "b"]}})
    assert f == {"terms": {"source": ["a", "b"]}}


def test_where_and_long_nhau():
    f = _where_to_filter({"$and": [
        {"source": {"$eq": "u"}},
        {"article": {"$eq": "Điều 8"}},
    ]})
    assert f == {"bool": {"filter": [
        {"term": {"source": "u"}},
        {"term": {"article": "Điều 8"}},
    ]}}


def test_where_toan_tu_la_thi_raise():
    with pytest.raises(ValueError):
        _where_to_filter({"x": {"$gte": 5}})


# ── chunk_to_doc / _hit_to_chunk roundtrip ─────────────────────────────────────

def _chunk() -> Chunk:
    return Chunk(
        chunk_id="abc_0001",
        text="[168/2024/NĐ-CP — Điều 7 — Khoản 2]\na) Không chấp hành đèn tín hiệu...",
        article="Điều 7",
        clause="Khoản 2",
        point="Điểm a",
        parent_id="abc_p_7_k_2",
        metadata=DocumentMetadata(
            source="https://x/168", doc_number="168/2024/NĐ-CP",
            status="Còn hiệu lực", issued_date="2024-12-26", folder="nghi_dinh",
        ),
    )


def test_chunk_to_doc_du_truong():
    doc = chunk_to_doc(_chunk(), embedding=[0.1] * EMBED_DIM)
    assert doc["article"] == "Điều 7"
    assert doc["point"] == "Điểm a"
    assert doc["parent_id"] == "abc_p_7_k_2"
    assert doc["doc_number"] == "168/2024/NĐ-CP"
    assert len(doc["embedding"]) == EMBED_DIM


def test_roundtrip_hit_to_chunk():
    doc = chunk_to_doc(_chunk(), embedding=[0.1] * EMBED_DIM)
    hit = {"_id": "abc_0001", "_score": 0.93, "_source": doc}
    c = _hit_to_chunk(hit)
    assert c.chunk_id == "abc_0001"
    assert c.article == "Điều 7" and c.point == "Điểm a"
    assert c.parent_id == "abc_p_7_k_2"
    assert c.metadata.status == "Còn hiệu lực"
    assert "đèn tín hiệu" in c.text


# ── Mapping sanity ─────────────────────────────────────────────────────────────

def test_mapping_knn_lucene_cosine():
    emb = INDEX_MAPPING["mappings"]["properties"]["embedding"]
    assert emb["dimension"] == 1024
    assert emb["method"]["engine"] == "lucene"
    assert emb["method"]["space_type"] == "cosinesimil"
    assert INDEX_MAPPING["settings"]["index"]["knn"] is True

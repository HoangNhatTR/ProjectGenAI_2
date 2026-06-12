"""OpenSearch backend — vector (kNN) + lexical (BM25) trong MỘT index.

Thay thế cặp Chroma + FTS khi VECTOR_BACKEND=opensearch:
  - OpenSearchVectorStore : cùng interface VectorStore (query/get_by_filter/
    count/add/reset/distinct_sources) → Retriever không đổi
  - OpenSearchLexicalIndex: cùng interface BM25Index (is_available/query)

Index dùng HNSW engine "lucene": đọc qua page cache của OS — KHÔNG cần
load toàn bộ vectors vào RAM như faiss native → chạy được trên máy 16GB
với 4.9M vectors × 1024 dim.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

from loguru import logger

from .schemas import Chunk, RetrievedChunk
from .vectorstore import _chroma_to_chunk

EMBED_DIM = 1024  # BGE-M3

INDEX_MAPPING: dict = {
    "settings": {
        "index": {
            "knn": True,
            "number_of_shards": 2,
            "number_of_replicas": 0,
        }
    },
    "mappings": {
        "properties": {
            "text":      {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": EMBED_DIM,
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",          # page-cache friendly, không bắt RAM
                    "space_type": "cosinesimil",
                    # m=16/efc=100: cân bằng tốc độ build (4.9M vectors trên
                    # máy cá nhân) vs recall — recall@10 vẫn >0.95 cho 1024-dim
                    "parameters": {"m": 16, "ef_construction": 100},
                },
            },
            # Metadata — keyword để filter chính xác
            "source":     {"type": "keyword"},
            "article":    {"type": "keyword"},
            "clause":     {"type": "keyword"},
            "point":      {"type": "keyword"},
            "parent_id":  {"type": "keyword"},
            "doc_type":   {"type": "keyword"},
            "doc_number": {"type": "keyword"},
            "title":      {"type": "text"},
            "issued_date":    {"type": "keyword"},
            "effective_date": {"type": "keyword"},
            "status":     {"type": "keyword"},
            "folder":     {"type": "keyword"},
            "co_quan":    {"type": "keyword"},
            "linh_vuc":   {"type": "keyword"},
        }
    },
}

_META_FIELDS = [
    "source", "article", "clause", "point", "parent_id", "doc_type",
    "doc_number", "title", "issued_date", "effective_date", "status",
    "folder", "co_quan", "linh_vuc",
]


def _where_to_filter(where: Optional[dict]) -> Optional[dict]:
    """Dịch Chroma-style `where` → OpenSearch bool filter.

    Hỗ trợ đúng các pattern Retriever đang dùng:
      {"field": {"$eq": v}} | {"field": {"$in": [...]}} | {"$and": [w1, w2]}
      và shorthand {"field": value}
    """
    if not where:
        return None
    if "$and" in where:
        clauses = [_where_to_filter(w) for w in where["$and"]]
        return {"bool": {"filter": [c for c in clauses if c]}}

    filters: list[dict] = []
    for field, cond in where.items():
        if isinstance(cond, dict):
            if "$eq" in cond:
                filters.append({"term": {field: cond["$eq"]}})
            elif "$in" in cond:
                filters.append({"terms": {field: list(cond["$in"])}})
            else:
                raise ValueError(f"Toán tử where chưa hỗ trợ: {cond}")
        else:
            filters.append({"term": {field: cond}})
    if len(filters) == 1:
        return filters[0]
    return {"bool": {"filter": filters}}


def _hit_to_chunk(hit: dict) -> Chunk:
    src = dict(hit.get("_source") or {})
    text = src.pop("text", "") or ""
    src.pop("embedding", None)
    return _chroma_to_chunk(hit["_id"], text, src)


def chunk_to_doc(chunk: Chunk, embedding: Optional[list[float]] = None) -> dict:
    """Chunk → document body cho OpenSearch (dùng cả ở ingest)."""
    doc: dict[str, Any] = {"text": chunk.text, "source": chunk.metadata.source}
    if embedding is not None:
        doc["embedding"] = embedding
    for f in ("article", "clause", "point", "parent_id"):
        v = getattr(chunk, f)
        if v:
            doc[f] = v
    for f in ("doc_type", "doc_number", "title", "issued_date", "effective_date",
              "status", "folder", "co_quan", "linh_vuc"):
        v = getattr(chunk.metadata, f)
        if v:
            doc[f] = v
    return doc


class OpenSearchVectorStore:
    """Vector store trên OpenSearch — interface tương thích VectorStore."""

    def __init__(self, url: str, index_name: str):
        self.url = url
        self.index_name = index_name
        self._client = None

    def _connect(self):
        if self._client is None:
            from opensearchpy import OpenSearch
            self._client = OpenSearch(
                hosts=[self.url], timeout=60,
                max_retries=2, retry_on_timeout=True,
            )
        return self._client

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def ensure_index(self) -> None:
        cli = self._connect()
        if not cli.indices.exists(index=self.index_name):
            cli.indices.create(index=self.index_name, body=INDEX_MAPPING)
            logger.info(f"OpenSearch: đã tạo index '{self.index_name}'")

    def reset(self) -> None:
        cli = self._connect()
        if cli.indices.exists(index=self.index_name):
            cli.indices.delete(index=self.index_name)
        self.ensure_index()

    def count(self) -> int:
        cli = self._connect()
        if not cli.indices.exists(index=self.index_name):
            return 0
        return int(cli.count(index=self.index_name)["count"])

    # ── Write ──────────────────────────────────────────────────────────────────

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        from opensearchpy import helpers
        cli = self._connect()
        actions = (
            {"_op_type": "index", "_index": self.index_name, "_id": c.chunk_id,
             "_source": chunk_to_doc(c, e)}
            for c, e in zip(chunks, embeddings)
        )
        helpers.bulk(cli, actions, chunk_size=500, request_timeout=120)

    # ── Read (interface Retriever dùng) ────────────────────────────────────────

    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        cli = self._connect()
        knn: dict[str, Any] = {"vector": list(embedding), "k": top_k}
        flt = _where_to_filter(where)
        if flt:
            knn["filter"] = flt
        body = {
            "size": top_k,
            "_source": {"excludes": ["embedding"]},
            "query": {"knn": {"embedding": knn}},
        }
        res = cli.search(index=self.index_name, body=body)
        return [
            RetrievedChunk(chunk=_hit_to_chunk(h), score=float(h["_score"]))
            for h in res["hits"]["hits"]
        ]

    def get_by_filter(self, where: dict, limit: int = 50) -> list[Chunk]:
        cli = self._connect()
        flt = _where_to_filter(where) or {"match_all": {}}
        body = {
            "size": limit,
            "_source": {"excludes": ["embedding"]},
            "query": {"bool": {"filter": [flt]}},
        }
        res = cli.search(index=self.index_name, body=body)
        return [_hit_to_chunk(h) for h in res["hits"]["hits"]]

    # ── Scan (ingest resume / BM25 rebuild compat) ─────────────────────────────

    def distinct_sources(self) -> set[str]:
        cli = self._connect()
        sources: set[str] = set()
        after = None
        while True:
            comp: dict[str, Any] = {
                "size": 5000,
                "sources": [{"src": {"terms": {"field": "source"}}}],
            }
            if after:
                comp["after"] = after
            res = cli.search(index=self.index_name, body={
                "size": 0, "aggs": {"by_src": {"composite": comp}},
            })
            agg = res["aggregations"]["by_src"]
            for b in agg["buckets"]:
                sources.add(b["key"]["src"])
            after = agg.get("after_key")
            if not after or not agg["buckets"]:
                return sources

    def iter_all_chunks(self, batch_size: int = 1000) -> Iterator[Chunk]:
        from opensearchpy import helpers
        cli = self._connect()
        for hit in helpers.scan(
            cli, index=self.index_name,
            query={"_source": {"excludes": ["embedding"]}, "query": {"match_all": {}}},
            size=batch_size,
        ):
            yield _hit_to_chunk(hit)


class OpenSearchLexicalIndex:
    """Nhánh lexical (BM25 native của Lucene) — interface giống BM25Index."""

    def __init__(self, store: OpenSearchVectorStore):
        self.store = store

    def is_available(self) -> bool:
        try:
            return self.store.count() > 0
        except Exception as exc:
            logger.warning(f"OpenSearch lexical không khả dụng: {exc}")
            return False

    def query(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        if not query.strip():
            return []
        try:
            cli = self.store._connect()
            body = {
                "size": top_k,
                "_source": {"excludes": ["embedding"]},
                "query": {
                    "bool": {
                        "must": [{"match": {"text": {"query": query}}}],
                        # Cụm khớp nguyên văn được cộng điểm
                        "should": [{"match_phrase": {"text": {"query": query, "slop": 2, "boost": 2.0}}}],
                    }
                },
            }
            res = cli.search(index=self.store.index_name, body=body)
            return [
                (_hit_to_chunk(h), float(h["_score"]))
                for h in res["hits"]["hits"]
            ]
        except Exception as exc:
            logger.warning(f"OpenSearch BM25 query lỗi: {exc}")
            return []

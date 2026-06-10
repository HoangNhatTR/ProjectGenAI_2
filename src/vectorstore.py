from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import chromadb
from chromadb.config import Settings

from .schemas import Chunk, DocumentMetadata, RetrievedChunk


def _chunk_to_chroma_meta(chunk: Chunk) -> dict:
    """Flatten Chunk metadata cho Chroma. Bỏ field None vì Chroma không nhận."""
    meta: dict[str, str] = {"source": chunk.metadata.source}
    if chunk.article:
        meta["article"] = chunk.article
    if chunk.clause:
        meta["clause"] = chunk.clause
    if chunk.point:
        meta["point"] = chunk.point
    if chunk.parent_id:
        meta["parent_id"] = chunk.parent_id
    for field in ("doc_type", "doc_number", "title", "issued_date", "effective_date", "status"):
        value = getattr(chunk.metadata, field)
        if value:
            meta[field] = value
    return meta


def _chroma_to_chunk(chunk_id: str, text: str, meta: dict) -> Chunk:
    doc_meta = DocumentMetadata(
        source=meta.get("source", ""),
        doc_type=meta.get("doc_type"),
        doc_number=meta.get("doc_number"),
        title=meta.get("title"),
        issued_date=meta.get("issued_date"),
        effective_date=meta.get("effective_date"),
        status=meta.get("status"),
    )
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        article=meta.get("article"),
        clause=meta.get("clause"),
        point=meta.get("point"),
        metadata=doc_meta,
        parent_id=meta.get("parent_id"),
    )


class VectorStore:
    """Wrapper quanh Chroma. Có thể swap sang Qdrant/Weaviate khi lên prod."""

    def __init__(self, persist_dir: Path, collection_name: str):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._client: Optional[chromadb.api.ClientAPI] = None
        self._collection = None

    def _connect(self) -> None:
        if self._collection is not None:
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks + embeddings + metadata vào collection."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) và embeddings ({len(embeddings)}) không khớp"
            )
        self._connect()
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[_chunk_to_chroma_meta(c) for c in chunks],
        )

    def get_by_filter(self, where: dict, limit: int = 50) -> list[Chunk]:
        """Lấy chunks khớp metadata filter (không cần semantic search).

        Dùng cho KG retrieval: cho 1 (source_url, article_label) → lấy hết chunks tương ứng.
        Chroma `where` syntax: {"$and": [{"field": value}, ...]} hoặc {"field": {"$eq": value}}.
        """
        self._connect()
        try:
            result = self._collection.get(where=where, limit=limit)
        except Exception:
            return []
        chunks: list[Chunk] = []
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        for cid, text, meta in zip(ids, docs, metas):
            chunks.append(_chroma_to_chunk(cid, text or "", meta or {}))
        return chunks

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """Tìm top-k chunk gần nhất, có thể lọc theo metadata (doc_type, năm,...).

        Score trả về là cosine similarity ∈ [-1, 1] (đã chuyển từ distance).
        """
        self._connect()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            chunk = _chroma_to_chunk(chunk_id, text, meta)
            retrieved.append(RetrievedChunk(chunk=chunk, score=1.0 - float(dist)))
        return retrieved

    def count(self) -> int:
        self._connect()
        return self._collection.count()

    def iter_all_chunks(self, batch_size: int = 1000) -> Iterator[Chunk]:
        """Yield mọi chunk trong collection (dùng để rebuild BM25 sau ingest).

        Phân trang qua `limit`/`offset` để không OOM với collection lớn.
        """
        self._connect()
        offset = 0
        while True:
            result = self._collection.get(limit=batch_size, offset=offset)
            ids = result.get("ids") or []
            if not ids:
                return
            docs = result.get("documents") or []
            metas = result.get("metadatas") or []
            for chunk_id, text, meta in zip(ids, docs, metas):
                yield _chroma_to_chunk(chunk_id, text or "", meta or {})
            if len(ids) < batch_size:
                return
            offset += len(ids)

    def reset(self) -> None:
        """Xoá toàn bộ collection (dùng khi reindex)."""
        self._connect()
        self._client.delete_collection(name=self.collection_name)
        self._collection = None
        self._connect()

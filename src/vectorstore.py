from __future__ import annotations

from loguru import logger

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
    for field in ("doc_type", "doc_number", "title", "issued_date", "effective_date",
                  "status", "linh_vuc", "co_quan", "folder"):
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
        linh_vuc=meta.get("linh_vuc"),
        co_quan=meta.get("co_quan"),
        folder=meta.get("folder"),
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
        except Exception as exc:
            logger.warning(f"Chroma get_by_filter lỗi (where={where}): {exc}")
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

    # ── Full scan ──────────────────────────────────────────────────────────────
    # Chroma .get(offset=N) quét lại từ đầu mỗi trang → O(n²) toàn cục.
    # Với store hàng triệu chunks, API pagination mất hàng chục phút đến hàng giờ.
    # → Đường nhanh: đọc thẳng chroma.sqlite3 (1 pass ORDER BY, vài chục giây).

    def distinct_sources(self) -> set[str]:
        """Tập metadata.source của mọi chunk — dùng cho ingest --skip-existing.

        Đọc thẳng SQLite (giây) thay vì iter toàn bộ chunks (chục phút).
        Fallback API scan nếu schema Chroma thay đổi.
        """
        db = self.persist_dir / "chroma.sqlite3"
        if db.exists():
            try:
                import sqlite3
                con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
                try:
                    rows = con.execute(
                        "SELECT DISTINCT string_value FROM embedding_metadata "
                        "WHERE key = 'source'"
                    ).fetchall()
                finally:
                    con.close()
                return {r[0] for r in rows if r[0]}
            except Exception as exc:
                logger.warning(f"distinct_sources: sqlite trực tiếp lỗi ({exc}) — fallback API scan")
        return {c.metadata.source for c in self.iter_all_chunks()}

    def iter_all_chunks(self, batch_size: int = 1000) -> Iterator[Chunk]:
        """Yield mọi chunk trong collection (dùng để rebuild BM25 sau ingest).

        Thử đường nhanh SQLite (1 pass); fallback API pagination nếu lỗi
        ngay từ đầu. Lỗi GIỮA chừng stream sqlite sẽ raise (không fallback
        để tránh yield trùng chunk).
        """
        db = self.persist_dir / "chroma.sqlite3"
        if db.exists():
            gen = None
            try:
                gen = self._iter_all_chunks_sqlite(db)
                first = next(gen, None)  # lỗi schema lộ ra ngay tại đây
            except Exception as exc:
                logger.warning(f"iter_all_chunks: sqlite trực tiếp lỗi ({exc}) — fallback API pagination")
            else:
                if first is not None:
                    yield first
                    yield from gen
                return
        yield from self._iter_all_chunks_paged(batch_size)

    def _iter_all_chunks_sqlite(self, db: Path) -> Iterator[Chunk]:
        """Stream chunks từ chroma.sqlite3 — 1 pass, không offset re-scan."""
        import sqlite3
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        try:
            cur = con.execute(
                "SELECT e.embedding_id, m.key, m.string_value, m.int_value, "
                "       m.float_value, m.bool_value "
                "FROM embedding_metadata m "
                "JOIN embeddings e ON e.id = m.id "
                "ORDER BY m.id"
            )
            cur_id: Optional[str] = None
            meta: dict = {}
            text = ""
            for eid, key, sv, iv, fv, bv in cur:
                if eid != cur_id:
                    if cur_id is not None:
                        yield _chroma_to_chunk(cur_id, text, meta)
                    cur_id, meta, text = eid, {}, ""
                if key == "chroma:document":
                    text = sv or ""
                else:
                    val = sv if sv is not None else (
                        iv if iv is not None else (fv if fv is not None else bv)
                    )
                    meta[key] = val
            if cur_id is not None:
                yield _chroma_to_chunk(cur_id, text, meta)
        finally:
            con.close()

    def _iter_all_chunks_paged(self, batch_size: int = 1000) -> Iterator[Chunk]:
        """Fallback: phân trang qua API `limit`/`offset` (chậm với store lớn)."""
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

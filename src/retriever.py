from __future__ import annotations

from typing import Optional

from .bm25_index import BM25Index
from .embedding import Embedder
from .schemas import Chunk, RetrievedChunk
from .vectorstore import VectorStore


DEFAULT_RRF_K = 60  # smoothing constant trong Reciprocal Rank Fusion


class Retriever:
    """Tầng trên VectorStore. Hybrid: vector (semantic) + BM25 (keyword) → RRF.

    Nếu `bm25` không được pass HOẶC chưa có file index → fallback vector-only.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        bm25: Optional[BM25Index] = None,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict] = None,
        min_score: Optional[float] = None,
    ) -> list[RetrievedChunk]:
        """Lấy top-k chunks liên quan tới query.

        Args:
            query: câu hỏi của user.
            top_k: số chunk cuối cùng trả về.
            filters: Chroma `where` filter (chỉ áp dụng cho vector branch).
            min_score: ngưỡng cosine, CHỈ áp dụng cho vector-only mode.
                Hybrid mode dùng RRF score (scale khác) nên min_score bị bỏ qua.
        """
        if not query.strip():
            return []

        # Vector branch
        query_embedding = self.embedder.encode([query])[0]
        # Lấy nhiều candidate hơn top_k để RRF có vùng giao cắt
        candidate_k = max(top_k * 2, top_k + 5)
        vector_results = self.store.query(
            query_embedding, top_k=candidate_k, where=filters
        )

        # Vector-only mode (BM25 chưa sẵn sàng)
        if self.bm25 is None or not self.bm25.is_available():
            results = vector_results[:top_k]
            if min_score is not None:
                results = [r for r in results if r.score >= min_score]
            return results

        # Hybrid: BM25 → RRF fuse với vector
        try:
            bm25_results = self.bm25.query(query, top_k=candidate_k)
        except Exception:
            # BM25 lỗi → degrade về vector-only thay vì crash
            results = vector_results[:top_k]
            if min_score is not None:
                results = [r for r in results if r.score >= min_score]
            return results

        return self._rrf_fuse(bm25_results, vector_results, top_k=top_k)

    def _rrf_fuse(
        self,
        bm25_results: list[tuple[Chunk, float]],
        vector_results: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion: score = Σ 1 / (k + rank_i).

        RRF score không so sánh được với cosine — không apply min_score ở đây.
        """
        scores: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}

        for rank, (chunk, _bm25_score) in enumerate(bm25_results):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            chunks_by_id.setdefault(cid, chunk)

        for rank, r in enumerate(vector_results):
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            chunks_by_id.setdefault(cid, r.chunk)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            RetrievedChunk(chunk=chunks_by_id[cid], score=score)
            for cid, score in ranked
        ]

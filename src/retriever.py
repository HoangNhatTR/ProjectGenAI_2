from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

KG_TIMEOUT_S = float(os.getenv("KG_TIMEOUT_S", "3.0"))  # giây tối đa cho Neo4j query (Aura cold-call hay vượt 3s → override qua env)
# Hệ số nhánh KG trong RRF. Đặt 1.0 (ngang vector/BM25, KHÔNG boost) để KG khớp
# nhầm không hất văng kết quả vector đúng — đo thật cho thấy 1.5 làm Graph RAG tụt
# trên câu hỏi dạng tình huống (KG noise được đẩy lên top, mất điều đúng). 1.0 vẫn
# giữ được các ca KG thắng (G9→Đ330, B5→Đ134). Override qua env KG_WEIGHT.
KG_WEIGHT = float(os.getenv("KG_WEIGHT", "1.0"))
HYDE_TIMEOUT_S = 8.0  # giây tối đa cho HyDE LLM call
BM25_TIMEOUT_S = 10.0  # giây tối đa chờ BM25 branch (chạy nền song song vector)

# Câu hỏi đã chứa trích dẫn luật cụ thể (Điều/Khoản/Điểm/số hiệu VB) thì retrieval
# đã chính xác → HyDE (thêm 1 lần gọi LLM, tới 8s) không đáng. Chỉ bật HyDE cho
# câu hỏi mơ hồ, ngôn ngữ tự nhiên.
_CITATION_RE = re.compile(
    r"(?:Điều|điều)\s+\d+"
    r"|(?:Khoản|khoản)\s+\d+"
    r"|(?:Điểm|điểm)\s+[a-zA-Z]\b"
    r"|\d+/\d{4}/[A-ZĐÂĂÊÔƠƯ\-]+"
    r"|(?:Nghị\s+định|Thông\s+tư|Bộ\s+luật|Luật)\s+(?:số\s+)?\d+",
)


def _hyde_worth_it(query: str) -> bool:
    """False khi câu hỏi đã có trích dẫn luật cụ thể → bỏ HyDE cho nhanh."""
    return _CITATION_RE.search(query) is None

from .bm25_index import BM25Index
from .embedding import Embedder
from .schemas import Chunk, RetrievedChunk
from .vectorstore import VectorStore

if TYPE_CHECKING:
    from .kg.kg_retriever import KGRetriever
    from .parent_store import ParentStore


DEFAULT_RRF_K = 60  # smoothing constant trong Reciprocal Rank Fusion

_HYDE_PROMPT = (
    "Viết 1 đoạn văn bản pháp luật Việt Nam ngắn (50-80 từ) dạng điều khoản, "
    "có thể trả lời câu hỏi sau. Chỉ viết nội dung pháp lý, không giải thích:\n\n"
    "Câu hỏi: {query}"
)


class Retriever:
    """Tầng trên VectorStore. Hybrid: vector + BM25 + KG → RRF fusion (3 branches).

    Cải tiến:
      - Parent-Child expansion: child chunk → full Điều text khi generate
      - HyDE: sinh hypothetical document → embed → retrieve (tăng recall)

    Degrade gracefully:
      - Không bm25 → vector-only (+ KG nếu có)
      - Không kg → vector + BM25 (như cũ)
      - Không parent_store → trả về child text trực tiếp (backward compat)
      - Không llm_client → tắt HyDE tự động
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        bm25: Optional[BM25Index] = None,
        kg_retriever: Optional["KGRetriever"] = None,
        rrf_k: int = DEFAULT_RRF_K,
        parent_store: Optional["ParentStore"] = None,
        llm_client: Optional[Any] = None,
        hyde_model: Optional[str] = None,
    ):
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.kg_retriever = kg_retriever
        self.rrf_k = rrf_k
        self.parent_store = parent_store
        self.llm_client = llm_client
        self.hyde_model = hyde_model
        # Persistent executor — tránh tạo/hủy thread pool trên mỗi query
        self._kg_executor = ThreadPoolExecutor(max_workers=1)
        self._hyde_executor = ThreadPoolExecutor(max_workers=1)
        self._bm25_executor = ThreadPoolExecutor(max_workers=1)
        # REFERS_TO expansion sau fusion: bám điều ANCHOR trong kết quả đã fuse (nơi
        # vector tìm được), đi theo cạnh dẫn chiếu kéo thêm điều liên quan. Env-toggle.
        self.expand_refers = os.getenv("KG_EXPAND_REFERS", "1") not in ("0", "false", "False", "")

    # _RRF_SCALE không còn dùng cho min_score filter; để lại cho backward compat.
    _RRF_SCALE = 20.0

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict] = None,
        min_score: Optional[float] = None,
        use_kg: bool = True,
        allowed_sources: Optional[list[str]] = None,
        use_hyde: bool = False,
        use_parent_expansion: bool = True,
        rrf_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Lấy top-k chunks liên quan tới query.

        Args:
            query: câu hỏi của user.
            top_k: số chunk cuối cùng trả về.
            filters: Chroma `where` filter tùy ý (chỉ áp dụng cho vector branch).
            min_score: ngưỡng relevance [0,1].
            use_kg: True để dùng KG branch (Graph-RAG); False → RAG-only.
            allowed_sources: nếu set, chỉ giữ chunks có metadata.source trong list này.
            use_hyde: True → sinh hypothetical document trước khi embed (tăng recall).
                      Chỉ hoạt động khi llm_client và hyde_model được set.
            use_parent_expansion: True → sau RRF, expand child chunks về parent text.
                      Chỉ hoạt động khi parent_store được set.
            rrf_k: override smoothing constant RRF cho riêng query này
                      (không mutate self.rrf_k — an toàn khi dùng chung instance).
        """
        if not query.strip():
            return []

        # Build composite filter cho vector branch
        vec_filter = filters
        if allowed_sources:
            src_filter = {"source": {"$in": list(allowed_sources)}}
            vec_filter = src_filter if not filters else {"$and": [filters, src_filter]}

        # ── Xác định branch khả dụng & kick off BM25 + KG song song ────────────
        # BM25 và KG độc lập với embedding → chạy nền trong khi main thread lo
        # HyDE + embed + vector query, cắt latency tổng (trước đây tuần tự).
        candidate_k = max(top_k * 3, top_k + 10)
        bm25_available = self.bm25 is not None and self.bm25.is_available()
        kg_available = use_kg and (self.kg_retriever is not None)

        bm25_future = (
            self._bm25_executor.submit(self._bm25_branch, query, candidate_k, allowed_sources)
            if bm25_available else None
        )
        kg_future = (
            self._kg_executor.submit(self._kg_branch, query, candidate_k)
            if kg_available else None
        )

        # ── Branch 1: Vector search (+ optional HyDE) ở main thread ────────────
        hyde_ok = (
            use_hyde and self.llm_client is not None and self.hyde_model
            and _hyde_worth_it(query)
        )
        if hyde_ok:
            hyde_text = self._hyde_query(query)
            embed_text = hyde_text if hyde_text else query
        else:
            embed_text = query

        query_embedding = self.embedder.encode([embed_text])[0]
        vector_results = self.store.query(
            query_embedding, top_k=candidate_k, where=vec_filter
        )

        # Vector-only mode (không có nhánh nào khác chạy nền)
        if not bm25_available and not kg_available:
            results = vector_results[:top_k]
            if min_score is not None:
                results = [r for r in results if r.score >= min_score]
            if use_parent_expansion and self.parent_store:
                results = self._expand_to_parents(results)
            return results

        # ── Branch 2: thu BM25 (đã chạy nền) ───────────────────────────────────
        bm25_results: list[tuple[Chunk, float]] = []
        if bm25_future is not None:
            try:
                bm25_results = bm25_future.result(timeout=BM25_TIMEOUT_S)
            except _FuturesTimeout:
                logger.warning(f"BM25 branch timeout sau {BM25_TIMEOUT_S}s, bỏ qua BM25")
            except Exception as exc:
                logger.warning(f"BM25 collect lỗi, bỏ qua BM25: {exc}")

        # ── Branch 3: thu KG (đã chạy nền, hard timeout) ───────────────────────
        kg_chunks: list[Chunk] = []
        if kg_future is not None:
            try:
                kg_chunks = kg_future.result(timeout=KG_TIMEOUT_S)
            except _FuturesTimeout:
                logger.warning(f"KG branch timeout sau {KG_TIMEOUT_S}s, bỏ qua KG")
                kg_chunks = []
            except Exception as exc:
                logger.warning(f"KG branch lỗi, bỏ qua KG: {exc}")
                kg_chunks = []
            if allowed_sources and kg_chunks:
                allowed = set(allowed_sources)
                kg_chunks = [c for c in kg_chunks if c.metadata.source in allowed]

        do_expand = kg_available and self.expand_refers and self.kg_retriever is not None
        fuse_k = (top_k * 2 + 5) if do_expand else top_k
        results = self._rrf_fuse(
            bm25_results=bm25_results,
            vector_results=vector_results,
            kg_chunks=kg_chunks,
            top_k=fuse_k,
            rrf_k=rrf_k,
        )

        # ── REFERS_TO expansion: bám điều anchor đã fuse, kéo điều dẫn chiếu ────
        if do_expand:
            results = self._expand_refers_post(results, top_k)
        else:
            results = results[:top_k]

        # ── Parent expansion: thay child text bằng full Điều context ───────────
        if use_parent_expansion and self.parent_store:
            results = self._expand_to_parents(results)

        return results

    def _expand_refers_post(
        self, results: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Dedup kết quả theo (source, Điều), rồi bám top-3 điều anchor đi theo cạnh
        REFERS_TO kéo điều dẫn chiếu (chưa có) vào các slot còn trống. Cut về top_k.

        Vá đúng 2 vấn đề đã đo: (1) trùng chunk cùng 1 Điều lấp hết slot → dedup theo
        Điều; (2) expansion cũ chỉ bám hit KG → giờ bám điều anchor trong KẾT QUẢ FUSE
        (thường do vector tìm). Điều dẫn chiếu nhận điểm dưới mọi điều gốc.
        """
        def _key(rc: RetrievedChunk):
            return (rc.chunk.metadata.source or "", rc.chunk.article or "")

        # Dedup theo Điều, giữ chunk điểm cao nhất (results đã sort giảm dần).
        deduped: list[RetrievedChunk] = []
        seen: set = set()
        for rc in results:
            k = _key(rc)
            if k in seen:
                continue
            seen.add(k)
            deduped.append(rc)

        if len(deduped) >= top_k:
            # Vẫn thử expand nếu còn slot sau cut; nhưng đã đủ điều phân biệt.
            base = deduped[:top_k]
        else:
            base = deduped

        # Seed = top-3 điều anchor (source, number).
        seeds: list[tuple[str, int]] = []
        for rc in base[:3]:
            art = rc.chunk.article or ""
            src = rc.chunk.metadata.source or ""
            if art.startswith("Điều ") and src:
                try:
                    seeds.append((src, int(art.split()[1])))
                except (IndexError, ValueError):
                    pass
        refs = self.kg_retriever.refers_from(seeds) if seeds else []
        if not refs:
            return base[:top_k]

        existing = {_key(rc) for rc in base}
        min_score = min((rc.score for rc in base), default=0.01)
        added: list[RetrievedChunk] = []
        for src, num, _title in refs:
            label = f"Điều {num}"
            if (src, label) in existing:
                continue
            where = {"$and": [{"source": {"$eq": src}}, {"article": {"$eq": label}}]}
            try:
                chunks = self.store.get_by_filter(where, limit=1)
            except Exception:
                chunks = []
            if not chunks:
                continue
            existing.add((src, label))
            added.append(RetrievedChunk(chunk=chunks[0], score=min_score * 0.5))

        return (base + added)[:top_k]

    # ── HyDE ───────────────────────────────────────────────────────────────────

    def _hyde_query(self, query: str) -> Optional[str]:
        """Sinh hypothetical legal document cho câu hỏi → dùng làm query embedding.

        Chạy trong thread riêng với timeout để không block nếu LLM chậm.
        """
        def _call() -> str:
            response = self.llm_client.chat(
                model=self.hyde_model,
                messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
                options={"temperature": 0.1, "num_ctx": 512},
            )
            return response["message"]["content"].strip()

        try:
            fut = self._hyde_executor.submit(_call)
            return fut.result(timeout=HYDE_TIMEOUT_S)
        except Exception as exc:
            logger.warning(f"HyDE lỗi/timeout, dùng query gốc: {exc}")
            return None

    # ── Parent expansion ───────────────────────────────────────────────────────

    def _expand_to_parents(self, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Thay text của child chunks bằng full Điều text từ ParentStore.

        Dedup: nếu nhiều child chunks cùng parent_id, chỉ giữ lần đầu (score cao nhất).
        Backward compat: chunk không có parent_id → giữ nguyên.
        """
        if not results:
            return results

        # Collect tất cả parent_ids để batch fetch
        parent_ids = list({r.chunk.parent_id for r in results if r.chunk.parent_id})
        if not parent_ids:
            return results

        parent_texts = self.parent_store.get_batch(parent_ids)

        seen_parents: set[str] = set()
        expanded: list[RetrievedChunk] = []

        for r in results:
            pid = r.chunk.parent_id
            if pid and pid in parent_texts:
                if pid in seen_parents:
                    continue  # bỏ duplicate — giữ instance score cao nhất
                seen_parents.add(pid)
                expanded_chunk = r.chunk.model_copy(update={"text": parent_texts[pid]})
                expanded.append(RetrievedChunk(chunk=expanded_chunk, score=r.score))
            else:
                expanded.append(r)

        return expanded

    # ── BM25 branch ──────────────────────────────────────────────────────────────

    def _bm25_branch(
        self, query: str, candidate_k: int, allowed_sources: Optional[list[str]]
    ) -> list[tuple[Chunk, float]]:
        """BM25 lexical search (query gốc — không HyDE). Chạy nền song song vector."""
        try:
            bm25_raw = self.bm25.query(query, top_k=candidate_k * 2)
            if allowed_sources:
                allowed = set(allowed_sources)
                bm25_raw = [(c, s) for c, s in bm25_raw if c.metadata.source in allowed]
            return bm25_raw[:candidate_k]
        except Exception as exc:
            logger.warning(f"BM25 branch lỗi, degrade về vector-only: {exc}")
            return []

    # ── KG branch ──────────────────────────────────────────────────────────────

    def _kg_branch(self, query: str, top_k: int) -> list[Chunk]:
        """Query KG → map về chunks trong vectorstore qua (source_url, article_label)."""
        hits = self.kg_retriever.retrieve(query, top_k=top_k)
        if not hits:
            return []

        ordered_chunks: list[Chunk] = []
        seen_ids: set[str] = set()

        for hit in hits:
            if not hit.source_url:
                continue
            where = {
                "$and": [
                    {"source": {"$eq": hit.source_url}},
                    {"article": {"$eq": hit.article_label}},
                ]
            }
            chunks = self.store.get_by_filter(where, limit=5)
            for c in chunks:
                if c.chunk_id in seen_ids:
                    continue
                seen_ids.add(c.chunk_id)
                ordered_chunks.append(c)
                if len(ordered_chunks) >= top_k:
                    return ordered_chunks
        return ordered_chunks

    # ── RRF fusion ─────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        bm25_results: list[tuple[Chunk, float]],
        vector_results: list[RetrievedChunk],
        kg_chunks: list[Chunk],
        top_k: int,
        rrf_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion 3 branches: score = Σ 1 / (k + rank_i)."""
        k = rrf_k if rrf_k is not None else self.rrf_k
        scores: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}

        for rank, (chunk, _) in enumerate(bm25_results):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunks_by_id.setdefault(cid, chunk)

        for rank, r in enumerate(vector_results):
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunks_by_id.setdefault(cid, r.chunk)

        kg_weight = KG_WEIGHT  # boost structured grounding (override qua env KG_WEIGHT)
        for rank, c in enumerate(kg_chunks):
            cid = c.chunk_id
            scores[cid] = scores.get(cid, 0.0) + kg_weight / (k + rank + 1)
            chunks_by_id.setdefault(cid, c)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            RetrievedChunk(chunk=chunks_by_id[cid], score=score)
            for cid, score in ranked
        ]

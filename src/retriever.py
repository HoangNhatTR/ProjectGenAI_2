from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

KG_TIMEOUT_S = float(os.getenv("KG_TIMEOUT_S", "3.0"))  # giây tối đa cho Neo4j query (Aura cold-call hay vượt 3s → override qua env)
# ── Trọng số 3 nhánh trong RRF fusion (override qua .env) ─────────────────────
# score(chunk) = Σ_i  w_i / (RRF_K + rank_i + 1), i ∈ {vector, bm25, kg}.
# Mặc định cả ba = 1.0 (bình đẳng). Tăng w của nhánh nào = tin nhánh đó hơn.
# ⚠ ĐỔI TRỌNG SỐ PHẢI ĐO BẰNG scripts.eval_retrieval (harness 34 câu) — bài học
# dự án: chỉnh ranking theo cảm tính hay hại nhóm khác (KG_WEIGHT=1.5 từng làm
# Graph RAG tụt: KG noise bị đẩy lên top, mất điều đúng; đã hạ về 1.0).
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "1.0"))  # ngữ nghĩa (BGE-M3)
BM25_WEIGHT   = float(os.getenv("BM25_WEIGHT", "1.0"))    # từ khóa (lexical)
KG_WEIGHT     = float(os.getenv("KG_WEIGHT", "1.0"))      # đồ thị (Neo4j)
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

# ── Multi-query fusion (RAG-fusion) ────────────────────────────────────────────
# Sinh N cách diễn đạt khác của câu hỏi → retrieve từng câu → RRF gộp giữa các
# query. Giải đúng ca corpus phát biểu khác lời user: "vượt đèn đỏ" (user) vs
# "không chấp hành hiệu lệnh của đèn tín hiệu" (NĐ 168) — 1 query đơn thua từ vựng.

_MQ_PROMPT = (
    "Viết lại câu hỏi pháp luật sau thành {n} cách diễn đạt khác nhau để tìm "
    "văn bản pháp luật Việt Nam, dùng thuật ngữ pháp lý chính xác thay cho từ "
    "đời thường (ví dụ: 'vượt đèn đỏ' → 'không chấp hành hiệu lệnh của đèn tín "
    "hiệu giao thông'; 'xe máy' → 'xe mô tô, xe gắn máy'; 'bằng lái' → 'giấy "
    "phép lái xe'; 'đi ngược chiều' → 'đi ngược chiều của đường một chiều'). "
    "Mỗi cách trên 1 dòng, không đánh số, không giải thích.\n\n"
    "Câu hỏi: {query}"
)
MQ_TIMEOUT_S = 8.0  # giây tối đa cho LLM sinh paraphrase
MULTI_QUERY_N = int(os.getenv("MULTI_QUERY_N", "2"))  # số paraphrase (ngoài câu gốc)
# LLM paraphrase: TẮT mặc định 2026-07-09 (user yêu cầu giảm latency). Bước này
# gọi LLM qua mạng ~5-15s (nút thắt Retrieve), giá trị recall thêm ÍT so với
# rule expansion tất định (_RULE_SYNONYMS) vốn đã cứu ca "xe máy vượt đèn đỏ".
# Bật lại: MQ_LLM_PARAPHRASE=true. Khi tắt, multi-query = gốc + rule variant.
MQ_LLM_PARAPHRASE = os.getenv("MQ_LLM_PARAPHRASE", "false").lower() == "true"
# Recall guard: top-N đầu bảng của MỖI query variant được bảo lưu suất trong
# pool sau fusion. RRF đồng thuận pha loãng chunk chỉ xuất hiện ở 1 list
# (đo 2026-07-08: rule query tìm đúng Đ7K7 NĐ168 top-1 nhưng fused loại nó
# vì 3 list kia không có) — reranker sẽ quyết thứ hạng cuối, guard chỉ đảm
# bảo ứng viên mạnh của từng cách diễn đạt CÓ MẶT để được chấm.
MQ_HEAD_KEEP = int(os.getenv("MQ_HEAD_KEEP", "3"))

# ── Demote TẦNG RETRIEVAL (trước RRF) ─────────────────────────────────────────
# Nhãn 'Hết hiệu lực toàn bộ' chỉ demote ở RERANK (×0.5) là QUÁ MUỘN: pool RRF
# thô đã bị VB cổ lấp chỗ trước khi reranker được nhìn thấy ứng viên đúng (đo
# 2026-07-08: query đèn tín hiệu hoàn hảo → Đ7K7 NĐ168 hạng 18/20, trên nó là
# 17 chunk của ~10 NĐ phạt GT 1995-2016 cùng nguyên văn — backfill status từ
# vbpl 2026-07-08 đã gắn nhãn thật cho các VB này). Demote áp vào TỪNG NHÁNH
# (vector + BM25) rồi re-sort → RRF xếp hạng trên danh sách đã lành mạnh.
# Tắt: RETRIEVAL_EXPIRED_FACTOR=1.0 (railway theo RAILWAY_MISMATCH_FACTOR).
RETRIEVAL_EXPIRED_FACTOR = float(os.getenv("RETRIEVAL_EXPIRED_FACTOR", "0.5"))
# Sai miền đường thủy/hàng hải: VB xử phạt đường thủy (139/2021 HIỆN HÀNH nên
# không gắn nhãn hết hiệu lực được) cũng nói về "đèn tín hiệu" → lọt pool câu
# hỏi đường bộ. Nhận diện theo TITLE văn bản, chỉ demote khi query không nhắc
# gì tới sông nước. Tắt: WATERWAY_MISMATCH_FACTOR=1.0
WATERWAY_MISMATCH_FACTOR = float(os.getenv("WATERWAY_MISMATCH_FACTOR", "0.8"))
# 'thủy' (dấu trên u) vs 'thuỷ' (dấu trên y) — corpus có cả hai cách bỏ dấu
_WATERWAY_RE = re.compile(
    r"đường\s+th(?:ủy|uỷ)|hàng\s+hải|tàu\s+thuyền|thuyền|phà\b|cảng\b", re.IGNORECASE
)


def _stale_factor(query_lower: str, chunk: "Chunk") -> float:
    """Hệ số demote tầng retrieval: VB hết hiệu lực toàn bộ + sai miền
    (đường ngang/đường sắt/đường thủy) cho query không thuộc miền đó."""
    from .reranker import _RAILWAY_RE, _railway_mismatch_factor

    f = 1.0
    status = (chunk.metadata.status or "").lower()
    if "hết hiệu lực" in status and "toàn bộ" in status:
        f *= RETRIEVAL_EXPIRED_FACTOR
    f *= _railway_mismatch_factor(query_lower, chunk)
    # Sai miền theo TITLE văn bản (header chunk có thể không nhắc miền):
    # VB chuyên đường sắt / đường thủy cho query không thuộc miền đó.
    title = (chunk.metadata.title or "").lower()
    if title and not _RAILWAY_RE.search(query_lower) and _RAILWAY_RE.search(title):
        f *= WATERWAY_MISMATCH_FACTOR
    if WATERWAY_MISMATCH_FACTOR < 1.0 and not _WATERWAY_RE.search(query_lower):
        if _WATERWAY_RE.search(title):
            f *= WATERWAY_MISMATCH_FACTOR
    return f

# ── Rule-based legal synonym expansion (TẤT ĐỊNH) ─────────────────────────────
# LLM paraphrase (dù temperature=0) qua API vẫn KHÔNG tất định giữa các lần
# chạy → thỉnh thoảng thiếu biến thể thuật ngữ luật làm rớt recall (đo
# 2026-07-08: "Xe máy vượt đèn đỏ thì..." cùng câu lúc PASS lúc FAIL tùy
# paraphrase). Bảng thay thế này áp bằng CODE → biến thể pháp lý LUÔN có mặt
# trong multi-query, không phụ thuộc "tâm trạng" LLM.
_RULE_SYNONYMS: list[tuple[str, str]] = [
    ("vượt đèn đỏ", "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"),
    ("xe máy", "xe mô tô, xe gắn máy"),
    ("bằng lái", "giấy phép lái xe"),
    ("đi ngược chiều", "đi ngược chiều của đường một chiều"),
    ("nồng độ cồn", "nồng độ cồn trong máu hoặc hơi thở"),
]


def _rule_expand(query: str) -> Optional[str]:
    """Thay từ đời thường bằng thuật ngữ pháp lý; None nếu không có gì để thay."""
    out = query
    hit = False
    for coll, legal in _RULE_SYNONYMS:
        if coll in out.lower():
            out = re.sub(re.escape(coll), legal, out, flags=re.IGNORECASE)
            hit = True
    return out if hit else None


def _parse_paraphrases(raw: str, original: str, n: int) -> list[str]:
    """Tách output LLM thành list paraphrase sạch (bỏ đánh số/bullet, trùng, quá ngắn)."""
    out: list[str] = []
    orig_norm = original.strip().lower()
    for line in (raw or "").splitlines():
        line = re.sub(r"^[\s\-•*]+", "", line.strip())
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if len(line) < 8 or line.lower() == orig_norm:
            continue
        if line not in out:
            out.append(line)
        if len(out) >= n:
            break
    return out


def _fuse_ranked_lists(
    lists: list[list[RetrievedChunk]], top_k: int, k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """RRF gộp nhiều danh sách đã xếp hạng (fusion GIỮA các query, không phải branch).

    Score = MEAN-RRF (chia số danh sách): thứ hạng y hệt sum-RRF nhưng thang điểm
    giữ nguyên như single-query. Sum-RRF từng làm top score vượt ce_threshold
    (0.04) → smart-skip TẮT NHẦM CrossEncoder trên mọi query multi-query
    (phát hiện 2026-07-08: đúng chunk NĐ 168 vào pool nhưng CE không chạy
    để kéo lên top).
    """
    scores: dict[str, float] = {}
    chunks_by_id: dict = {}
    for lst in lists:
        for rank, r in enumerate(lst):
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunks_by_id.setdefault(cid, r.chunk)
    n = max(1, len(lists))
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    return [RetrievedChunk(chunk=chunks_by_id[cid], score=s / n) for cid, s in ranked]


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
        mq_model: Optional[str] = None,
    ):
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.kg_retriever = kg_retriever
        self.rrf_k = rrf_k
        self.parent_store = parent_store
        self.llm_client = llm_client
        self.hyde_model = hyde_model
        # Model sinh paraphrase cho multi-query fusion (fallback: hyde_model)
        self.mq_model = mq_model
        # Persistent executor — tránh tạo/hủy thread pool trên mỗi query
        self._kg_executor = ThreadPoolExecutor(max_workers=1)
        self._hyde_executor = ThreadPoolExecutor(max_workers=1)
        # 4 workers: multi-query chạy tối đa 4 retrieve song song (gốc + rule +
        # 2 paraphrase), mỗi cái submit 1 BM25 job — ít worker sẽ serialize
        # khiến job sau dễ hụt BM25_TIMEOUT_S.
        self._bm25_executor = ThreadPoolExecutor(max_workers=4)
        # Multi-query: retrieve câu gốc + rule + các paraphrase song song (I/O
        # OpenSearch overlap; encode BGE-M3 chia core nhưng wall-time ≤ tuần tự).
        self._mq_executor = ThreadPoolExecutor(max_workers=4)
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
        use_multi_query: bool = False,
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
            use_multi_query: True → RAG-fusion: sinh paraphrase + RRF gộp giữa
                      các query. Cần llm_client; tốn +1 LLM call + N lần retrieve.
        """
        if not query.strip():
            return []

        # ── Multi-query fusion: câu đã chứa trích dẫn cụ thể (Điều/số hiệu VB)
        # thì retrieval vốn chính xác → khỏi paraphrase (cùng gate với HyDE)
        if use_multi_query and self.llm_client is not None and _hyde_worth_it(query):
            fused = self._retrieve_multi(
                query, top_k, filters=filters, min_score=min_score, use_kg=use_kg,
                allowed_sources=allowed_sources, use_hyde=use_hyde,
                use_parent_expansion=use_parent_expansion, rrf_k=rrf_k,
            )
            if fused is not None:
                return fused

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

        # Demote tầng retrieval: VB hết hiệu lực toàn bộ / đường ngang sai miền
        # xuống đáy nhánh vector TRƯỚC khi RRF nhìn thấy thứ hạng.
        _ql = query.lower()
        vector_results = sorted(
            (
                RetrievedChunk(chunk=r.chunk, score=r.score * _stale_factor(_ql, r.chunk))
                for r in vector_results
            ),
            key=lambda r: -r.score,
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
        if bm25_results:
            # Cùng demote như nhánh vector — RRF dùng THỨ HẠNG nên phải re-sort
            bm25_results = sorted(
                ((c, s * _stale_factor(_ql, c)) for c, s in bm25_results),
                key=lambda t: -t[1],
            )

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

    # ── Multi-query fusion ─────────────────────────────────────────────────────

    def _paraphrase_queries(self, query: str, n: int) -> list[str]:
        """Sinh n cách diễn đạt khác của query bằng fast model (timeout cứng)."""
        model = self.mq_model or self.hyde_model
        if self.llm_client is None or not model:
            return []

        def _call() -> str:
            # temperature=0: paraphrase TẤT ĐỊNH — cùng câu hỏi ra cùng pool,
            # câu trả lời ổn định giữa các lần (prompt đã ép n cách KHÁC NHAU
            # nên temp 0 vẫn đủ đa dạng; temp 0.4 cũ làm ranking dao động).
            response = self.llm_client.chat(
                model=model,
                messages=[{"role": "user", "content": _MQ_PROMPT.format(n=n, query=query)}],
                options={"temperature": 0.0, "num_ctx": 512},
            )
            return response["message"]["content"]

        try:
            raw = self._hyde_executor.submit(_call).result(timeout=MQ_TIMEOUT_S)
        except Exception as exc:
            logger.warning(f"Multi-query paraphrase lỗi/timeout, retrieve thường: {exc}")
            return []
        return _parse_paraphrases(raw, query, n)

    def _retrieve_multi(
        self,
        query: str,
        top_k: int,
        *,
        filters: Optional[dict],
        min_score: Optional[float],
        use_kg: bool,
        allowed_sources: Optional[list[str]],
        use_hyde: bool,
        use_parent_expansion: bool,
        rrf_k: Optional[int],
    ) -> Optional[list[RetrievedChunk]]:
        """RAG-fusion: retrieve câu gốc + paraphrases → RRF gộp GIỮA các query.

        Trả None khi không sinh được paraphrase → caller fallback retrieve thường.
        Paraphrase chạy KHÔNG KG (KG bám keyword câu gốc là đủ, đỡ N lần Neo4j)
        và không parent expansion (expand 1 lần SAU khi fuse — expand trước sẽ
        dedup theo parent_id làm lệch rank giữa các list).
        """
        common: dict[str, Any] = dict(
            filters=filters, min_score=min_score, allowed_sources=allowed_sources,
            use_multi_query=False, use_parent_expansion=False, rrf_k=rrf_k,
        )

        # Retrieve câu GỐC chạy nền NGAY — song song với LLM sinh paraphrase
        # (hai việc độc lập; tuần tự như cũ phí ~4-8s chờ LLM rồi mới retrieve).
        orig_future = self._mq_executor.submit(
            self.retrieve, query, top_k=top_k, use_kg=use_kg, use_hyde=use_hyde, **common,
        )

        # Biến thể RULE (thuật ngữ luật) — TẤT ĐỊNH, không phụ thuộc LLM, chạy
        # song song luôn. Lưới an toàn recall cho các cặp từ đời thường↔pháp lý
        # đã biết ("xe máy" → "xe mô tô, xe gắn máy") — LLM paraphrase lúc sinh
        # ra biến thể này lúc không (đo 2026-07-08: cùng câu lúc PASS lúc FAIL).
        rule_q = _rule_expand(query)
        rule_future = (
            self._mq_executor.submit(
                self.retrieve, rule_q, top_k=top_k, use_kg=False, use_hyde=False, **common,
            )
            if rule_q else None
        )

        # LLM paraphrase (tùy chọn — mặc định TẮT để giảm latency ~5-15s)
        paraphrases = self._paraphrase_queries(query, MULTI_QUERY_N) if MQ_LLM_PARAPHRASE else []
        # Bỏ paraphrase trùng biến thể rule (đỡ 1 lần retrieve vô ích)
        if rule_q:
            paraphrases = [p for p in paraphrases if p.strip().lower() != rule_q.strip().lower()]
        if not paraphrases and rule_future is None:
            # Không có biến thể nào → dùng luôn kết quả câu gốc đang chạy nền
            # (tương đương đường retrieve thường, đừng bỏ phí rồi retrieve lại).
            try:
                base = orig_future.result()
            except Exception as exc:
                logger.warning(f"Multi-query: retrieve câu gốc lỗi: {exc}")
                return None
            if use_parent_expansion and self.parent_store:
                base = self._expand_to_parents(base)
            return base
        logger.info(
            f"Multi-query: +{len(paraphrases)} paraphrase"
            f"{' +1 rule' if rule_q else ''} → RRF fusion (song song)"
        )

        # Các paraphrase retrieve song song với nhau
        pq_futures = [
            self._mq_executor.submit(
                self.retrieve, pq, top_k=top_k, use_kg=False, use_hyde=False, **common,
            )
            for pq in paraphrases
        ]
        pending = [(orig_future, "gốc")]
        if rule_future is not None:
            pending.append((rule_future, "rule"))
        pending += [(f, "paraphrase") for f in pq_futures]

        lists: list[list[RetrievedChunk]] = []
        for fut, label in pending:
            try:
                lists.append(fut.result())
            except Exception as exc:
                logger.warning(f"Multi-query: retrieve câu {label} lỗi, bỏ qua list: {exc}")
        if not lists:
            return None

        _k = rrf_k if rrf_k is not None else self.rrf_k
        fused = _fuse_ranked_lists(lists, top_k, k=_k)

        # Recall guard (MQ_HEAD_KEEP): top đầu mỗi list vào pool với điểm
        # single-list KHÔNG chia n — ngang chunk đồng thuận tuyệt đối, đảm bảo
        # ứng viên mạnh của từng cách diễn đạt được reranker chấm.
        if MQ_HEAD_KEEP > 0 and len(lists) > 1:
            seen = {r.chunk.chunk_id for r in fused}
            extras: list[RetrievedChunk] = []
            for lst in lists:
                for rank, r in enumerate(lst[:MQ_HEAD_KEEP]):
                    if r.chunk.chunk_id not in seen:
                        seen.add(r.chunk.chunk_id)
                        extras.append(
                            RetrievedChunk(chunk=r.chunk, score=1.0 / (_k + rank + 1))
                        )
            if extras:
                fused = sorted(fused + extras, key=lambda r: -r.score)[:top_k]

        if use_parent_expansion and self.parent_store:
            fused = self._expand_to_parents(fused)
        return fused

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

    def expand_parents(self, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Expand child chunks về full Điều SAU rerank (pipeline gọi).

        No-op khi không có parent_store. Dedup theo parent_id — caller cắt
        top_k SAU khi gọi để vẫn đủ số Điều phân biệt.
        """
        if not self.parent_store:
            return results
        return self._expand_to_parents(results)

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
            scores[cid] = scores.get(cid, 0.0) + BM25_WEIGHT / (k + rank + 1)
            chunks_by_id.setdefault(cid, chunk)

        for rank, r in enumerate(vector_results):
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + VECTOR_WEIGHT / (k + rank + 1)
            chunks_by_id.setdefault(cid, r.chunk)

        for rank, c in enumerate(kg_chunks):
            cid = c.chunk_id
            scores[cid] = scores.get(cid, 0.0) + KG_WEIGHT / (k + rank + 1)
            chunks_by_id.setdefault(cid, c)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            RetrievedChunk(chunk=chunks_by_id[cid], score=score)
            for cid, score in ranked
        ]

"""KG-augmented retrieval — branch thứ 3 trong hybrid pipeline (cùng với vector + BM25).

Pipeline:
  query → KG traversal → list of (Article_id_kg, source_url, article_number, score)
  → caller map sang chunks trong vectorstore qua VectorStore.get_by_filter()
  → đưa vào RRF fusion cùng vector + BM25 results

Mục tiêu của Tích hợp C: giúp retrieve những điều luật có TỪNG ĐƯỢC trích xuất
như là chứa hành vi pháp lý liên quan tới query — kể cả khi vector + BM25 miss.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

from .. import config
from .neo4j_client import Neo4jClient

# ─── Semantic offense match (embedding đồ thị) ───────────────────────────────
# Bật/tắt qua env KG_SEMANTIC (mặc định bật). Ngưỡng cosine tối thiểu để coi là
# khớp Offense; cao quá → bỏ sót, thấp quá → nhiễu. Cache do
# scripts/build_offense_embeddings.py sinh.
_KG_SEMANTIC = os.getenv("KG_SEMANTIC", "1") not in ("0", "false", "False", "")
_KG_SEMANTIC_MIN_COS = float(os.getenv("KG_SEMANTIC_MIN_COS", "0.6"))
_KG_SEMANTIC_TOPN = int(os.getenv("KG_SEMANTIC_TOPN", "8"))

# REFERS_TO expansion: sau khi tìm được điều liên quan, đi theo cạnh DẪN CHIẾU kéo
# thêm điều được viện dẫn vào pool (tăng recall cho câu cần nhiều điều liên quan).
# Chỉ mở rộng top-N hit trực tiếp; điểm điều dẫn chiếu = điểm cha × decay (< điều gốc).
_KG_EXPAND_REFERS = os.getenv("KG_EXPAND_REFERS", "1") not in ("0", "false", "False", "")
_KG_EXPAND_TOPN = int(os.getenv("KG_EXPAND_TOPN", "3"))       # số hit gốc đem expand
_KG_EXPAND_DECAY = float(os.getenv("KG_EXPAND_DECAY", "0.5"))  # hệ số giảm điểm điều dẫn chiếu
_KG_EXPAND_MAX_PER = int(os.getenv("KG_EXPAND_MAX_PER", "8"))  # cap số điều dẫn chiếu / điều gốc (chống hub)

# Nhiều NGUỒN semantic — mỗi nguồn embed một loại nội dung node của đồ thị, ánh xạ
# về Article. base = điểm nền (offense cụ thể hơn title chút → cao hơn nhẹ).
#   (tên, file vectors, file meta, base_weight)
_SEMANTIC_SOURCES = [
    ("offense", config.DATA_DIR / "kg" / "offense_vectors.npy",
     config.DATA_DIR / "kg" / "offense_meta.json", 2.2),
    ("article_title", config.DATA_DIR / "kg" / "article_title_vectors.npy",
     config.DATA_DIR / "kg" / "article_title_meta.json", 2.0),
]

# Cache nạp 1 lần / tiến trình: list[(name, vectors, labels, articles, base)].
_SEM_CACHES: Optional[list] = None
_SEM_CACHES_TRIED = False
_QUERY_EMBEDDER = None                        # Embedder singleton dùng cho query


def _load_semantic_caches():
    """Nạp mọi nguồn semantic có sẵn trên đĩa; [] nếu chưa build nguồn nào."""
    global _SEM_CACHES, _SEM_CACHES_TRIED
    if _SEM_CACHES is not None:
        return _SEM_CACHES
    _SEM_CACHES_TRIED = True
    caches = []
    for name, vec_path, meta_path, base in _SEMANTIC_SOURCES:
        if not (vec_path.exists() and meta_path.exists()):
            continue
        vecs = np.load(vec_path).astype(np.float32)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        caches.append((name, vecs, meta["names"], meta["articles"], base))
        logger.info(f"KG semantic: nạp {vecs.shape[0]} '{name}' vectors.")
    if not caches:
        logger.warning(
            "KG semantic: chưa có cache embeddings nào "
            "(chạy: python -m scripts.build_offense_embeddings). Bỏ qua nhánh semantic."
        )
    _SEM_CACHES = caches
    return caches


def _get_query_embedder(injected=None):
    """Trả Embedder để encode query. Ưu tiên cái được inject; nếu không, singleton."""
    global _QUERY_EMBEDDER
    if injected is not None:
        return injected
    if _QUERY_EMBEDDER is None:
        from ..embedding import Embedder
        _QUERY_EMBEDDER = Embedder(config.EMBEDDING_MODEL)
    return _QUERY_EMBEDDER


@dataclass
class KGHit:
    article_id_kg: str           # vd "vbpl_96122::dieu_173"
    source_url: str              # URL của Law
    article_number: int          # 173
    article_label: str           # "Điều 173" — để khớp với metadata trong vectorstore
    score: float                 # ranking score (cao hơn = liên quan hơn)
    reason: str                  # "via_offense" | "via_article_title" | "via_clause_text"
    matched_text: str = ""       # text matched (offense name, title...)


class KGRetriever:
    """Truy vấn KG để tìm Article relevant tới câu hỏi.

    Lazy-connect — chỉ kết nối Neo4j khi gọi retrieve() lần đầu.
    """

    def __init__(self, max_keywords: int = 5, embedder=None, semantic: Optional[bool] = None):
        self._client: Optional[Neo4jClient] = None
        self.max_keywords = max_keywords
        # embedder: nên truyền embedder dùng chung (tránh load model 2 lần). None →
        # nhánh semantic tự lazy-load 1 singleton khi cần.
        self._embedder = embedder
        # semantic: None → theo env KG_SEMANTIC; True/False → ép bật/tắt (để A/B test).
        self.semantic = _KG_SEMANTIC if semantic is None else semantic

    def _get_client(self) -> Neo4jClient:
        if self._client is None:
            self._client = Neo4jClient.from_env()
        return self._client

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        """Tách query thành keywords có ý nghĩa, loại stop words tiếng Việt.

        - Lowercase, tách theo whitespace + dấu câu
        - Bỏ words < 3 ký tự
        - Bỏ stop words (đại từ, từ nối, hỏi)
        - Cũng emit cụm 2-3 từ liên tiếp (n-grams) để match cụm có nghĩa
        """
        import re
        VI_STOPWORDS = {
            "là", "có", "được", "không", "và", "hay", "hoặc", "của", "cho", "với",
            "vào", "ra", "tại", "trong", "trên", "dưới", "bị", "thì", "đã", "sẽ",
            "đang", "này", "đó", "kia", "nào", "ai", "gì", "sao", "thế", "vì",
            "nên", "mà", "nhưng", "rằng", "khi", "lúc", "ngay", "đều", "cũng",
            "phải", "cần", "muốn", "biết", "làm", "đi", "đến", "lại", "rồi",
            "như", "theo", "qua", "từ", "tới", "về", "bằng", "nhằm", "đối",
            "các", "những", "mọi", "mỗi", "vài", "một", "hai", "ba",
            "tôi", "bạn", "chúng", "họ", "mình", "ta", "anh", "chị",
            "bao", "nhiêu", "tiền", "tiền?", "bao_nhiêu",  # specific to our domain queries
            "ra", "sao?", "thế_nào", "như_thế_nào",
        }

        # Cụm pháp lý CHUNG CHUNG: xuất hiện trong rất nhiều tên tội/điều luật nên
        # khớp lung tung (vd "phạm tội" dính "phạm tội có tổ chức", "phạm tội vì
        # vụ lợi"...). Loại để KG chỉ khớp cụm ĐẶC TRƯNG cho hành vi.
        VI_PHRASE_STOPWORDS = {
            "phạm tội", "tội phạm", "trách nhiệm", "hình sự", "trách nhiệm hình",
            "trách nhiệm hình sự", "truy cứu", "truy cứu trách nhiệm",
            "truy cứu trách nhiệm hình", "cứu trách nhiệm", "người khác",
            "cho người", "cho người khác", "xử lý", "quy định", "pháp luật",
            "có tổ chức", "phạm tội có", "phạm tội có tổ chức", "vì vụ lợi",
            "bị phạt", "phạt tù", "khung hình", "hình phạt", "khung hình phạt",
            # ── Từ ngữ KỂ TÌNH HUỐNG (không phải tên tội) — gây khớp nhầm Offense
            #    ở luật khác (vd "cưỡng chế"→Luật Thuế Đ46). Loại để KG bám tên tội.
            "cưỡng chế", "xô xát", "người dân", "bảo vệ", "công ty", "công an",
            "khách hàng", "cán bộ", "hồ sơ", "dự án", "giám đốc", "nhân viên",
            "hợp thức", "đẩy ngã", "giật còng",
        }

        # Lowercase + tách
        text = query.lower().strip()
        # Remove punctuation except space and Vietnamese letters
        text = re.sub(r"[?!.,;:\"'(){}\[\]]", " ", text)
        words = [w for w in text.split() if len(w) >= 3 and w not in VI_STOPWORDS]

        keywords: list[str] = list(words)  # individual words

        # 2-grams + 3-grams: gộp cụm để match Offense names phổ biến
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                if phrase in VI_PHRASE_STOPWORDS:
                    continue
                keywords.append(phrase)

        # Dedupe, giữ ngắn nhất 3 chars
        seen: set[str] = set()
        out: list[str] = []
        for k in keywords:
            if k in seen or len(k) < 3 or k in VI_PHRASE_STOPWORDS:
                continue
            seen.add(k)
            out.append(k)
        return out[:15]  # cap để Cypher không quá nặng

    def retrieve(self, query: str, top_k: int = 10) -> list[KGHit]:
        """Tìm top-k Article relevant qua KG.

        Chiến lược: extract keywords từ query, tìm match TỪNG keyword với
        Offense/Article title/Clause text. Cộng dồn score theo số match.

        Stop words tiếng Việt phổ biến bị loại để giảm noise.
        """
        if not query.strip():
            return []

        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        # Cypher trả về kw đã khớp → score tính ở Python theo độ cụ thể (số từ).
        # Quan trọng: branch Offense CHỈ nhận cụm ≥2 từ (tên tội luôn đa từ:
        # "gây thương tích", "trộm cắp", "cho vay lãi nặng"...) → loại nhiễu từ
        # đơn như "vay"→"họ hụi", "đánh"→"đánh tráo".
        cypher = """
            // ── Branch 1a: qua Offense → Clause → Article ────────────────────
            UNWIND $keywords AS kw
            WITH kw WHERE size(split(kw, ' ')) >= 2
            MATCH (o:Offense) WHERE toLower(o.name) CONTAINS kw
            MATCH (c:Clause)-[:PENALIZES]->(o)
            MATCH (a:Article)-[:HAS_CLAUSE]->(c)
            MATCH (l:Law)-[:HAS_ARTICLE]->(a)
            RETURN a.id            AS article_id_kg,
                   a.number        AS article_number,
                   l.source        AS source_url,
                   o.name          AS matched_text,
                   kw              AS matched_kw,
                   'via_offense'   AS reason

            UNION ALL

            // ── Branch 1b: qua Offense → Article (direct PENALIZES) ──────────
            UNWIND $keywords AS kw
            WITH kw WHERE size(split(kw, ' ')) >= 2
            MATCH (o:Offense) WHERE toLower(o.name) CONTAINS kw
            MATCH (a:Article)-[:PENALIZES]->(o)
            MATCH (l:Law)-[:HAS_ARTICLE]->(a)
            RETURN a.id            AS article_id_kg,
                   a.number        AS article_number,
                   l.source        AS source_url,
                   o.name          AS matched_text,
                   kw              AS matched_kw,
                   'via_offense'   AS reason

            UNION ALL

            // ── Branch 2: qua Article title ─────────────────────────────────
            UNWIND $keywords AS kw
            MATCH (a:Article)<-[:HAS_ARTICLE]-(l:Law)
            WHERE toLower(coalesce(a.title, '')) CONTAINS kw
            RETURN a.id            AS article_id_kg,
                   a.number        AS article_number,
                   l.source        AS source_url,
                   a.title         AS matched_text,
                   kw              AS matched_kw,
                   'via_article_title' AS reason

            UNION ALL

            // ── Branch 3: qua Clause text_preview ───────────────────────────
            UNWIND $keywords AS kw
            MATCH (c:Clause)<-[:HAS_CLAUSE]-(a:Article)<-[:HAS_ARTICLE]-(l:Law)
            WHERE toLower(coalesce(c.text_preview, '')) CONTAINS kw
            RETURN a.id            AS article_id_kg,
                   a.number        AS article_number,
                   l.source        AS source_url,
                   substring(c.text_preview, 0, 100) AS matched_text,
                   kw              AS matched_kw,
                   'via_clause_text' AS reason
        """

        try:
            with self._get_client().session() as s:
                rows = s.run(cypher, keywords=keywords).data()
        except Exception as exc:
            logger.warning(f"Neo4j query lỗi, KG trả rỗng: {exc}")
            return []

        # Score = trọng số theo nhánh × số từ của cụm khớp (cụm dài = cụ thể hơn
        # = điểm cao hơn). "gây thương tích" (3 từ) ⇒ 9 đè "đánh" (đã bị loại).
        BASE = {"via_offense": 3.0, "via_article_title": 2.0, "via_clause_text": 1.0}

        # Dedupe theo article_id_kg, giữ score cao nhất
        best: dict[str, KGHit] = {}
        for r in rows:
            aid = r["article_id_kg"]
            article_num = r["article_number"]
            if not article_num:
                continue
            matched_kw = r.get("matched_kw") or ""
            n_words = len(matched_kw.split()) or 1
            score = BASE.get(r["reason"], 1.0) * n_words
            if aid in best and best[aid].score >= score:
                continue
            best[aid] = KGHit(
                article_id_kg=aid,
                source_url=r["source_url"] or "",
                article_number=int(article_num),
                article_label=f"Điều {article_num}",
                score=score,
                reason=r["reason"],
                matched_text=(r["matched_text"] or "")[:200],
            )

        # ── Nhánh SEMANTIC: khớp Offense bằng cosine (embedding đồ thị) ─────────
        # Bù cho lexical khi câu KHÔNG chứa tên tội nguyên văn (câu kể chuyện dài).
        # Dùng query gốc để giữ trọn ngữ cảnh hành vi.
        if self.semantic:
            self._add_semantic_hits(query, best)

        # Sort by score
        ranked = sorted(best.values(), key=lambda h: -h.score)
        return ranked[:top_k]

    def refers_from(self, seeds: list[tuple[str, int]]) -> list[tuple[str, int, str]]:
        """Theo cạnh REFERS_TO từ các điều SEED (source, number) → điều được viện dẫn.

        Seed lấy từ KẾT QUẢ ĐÃ FUSE (nơi điều anchor thật sự nằm — thường do vector
        tìm), không chỉ từ hit của KG. REFERS_TO là intra-law nên điều đích cùng luật.
        Trả [(source, number, title)], cap _KG_EXPAND_MAX_PER / seed (chống hub).
        """
        if not seeds:
            return []
        params = [{"src": s, "num": int(n)} for s, n in seeds]
        cypher = """
            UNWIND $seeds AS seed
            MATCH (l:Law)-[:HAS_ARTICLE]->(a:Article)-[:REFERS_TO]->(b:Article)<-[:HAS_ARTICLE]-(l)
            WHERE l.source = seed.src AND a.number = seed.num AND b.number IS NOT NULL
            RETURN seed.src AS src, seed.num AS frm, b.number AS num, b.title AS title
        """
        try:
            with self._get_client().session() as s:
                rows = s.run(cypher, seeds=params).data()
        except Exception as exc:
            logger.warning(f"KG REFERS_TO expand lỗi, bỏ qua: {exc}")
            return []
        out, per_seed, seen = [], {}, set()
        for r in rows:
            key = (r["src"], r["frm"])
            if per_seed.get(key, 0) >= _KG_EXPAND_MAX_PER:
                continue
            tgt = (r["src"], int(r["num"]))
            if tgt in seen:
                continue
            seen.add(tgt)
            per_seed[key] = per_seed.get(key, 0) + 1
            out.append((r["src"] or "", int(r["num"]), r["title"] or ""))
        return out

    def _add_semantic_hits(self, query: str, best: dict[str, KGHit]) -> None:
        """Khớp query với các nguồn embedding của đồ thị (cosine) → thêm KGHit.

        Hai nguồn: tên Offense + tiêu đề Article (title = tên tội canonical). Điểm
        semantic = base + 3.0×cos → nằm DƯỚI lexical via_offense mạnh (≥6) nhưng vẫn
        đóng góp khi lexical không khớp gì. Dedupe theo article_id, giữ score cao nhất.
        """
        caches = _load_semantic_caches()
        if not caches:
            return
        try:
            embedder = _get_query_embedder(self._embedder)
            q = np.asarray(embedder.encode([query])[0], dtype=np.float32)
        except Exception as exc:
            logger.warning(f"KG semantic: encode query lỗi, bỏ qua: {exc}")
            return

        for name, vecs, labels, articles, base in caches:
            # vectors đã L2-normalize + query normalize → dot = cosine.
            sims = vecs @ q
            order = np.argsort(-sims)[: _KG_SEMANTIC_TOPN]
            for idx in order:
                cos = float(sims[idx])
                if cos < _KG_SEMANTIC_MIN_COS:
                    break  # đã sort giảm dần → các sau còn thấp hơn
                score = base + 3.0 * cos
                label = labels[idx]
                for aid, num, src in articles[idx]:
                    if not num:
                        continue
                    if aid in best and best[aid].score >= score:
                        continue
                    best[aid] = KGHit(
                        article_id_kg=aid,
                        source_url=src or "",
                        article_number=int(num),
                        article_label=f"Điều {num}",
                        score=score,
                        reason=f"via_{name}_semantic",
                        matched_text=f"{label} (cos={cos:.2f})",
                    )

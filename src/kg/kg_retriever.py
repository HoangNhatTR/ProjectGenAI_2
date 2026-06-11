"""KG-augmented retrieval — branch thứ 3 trong hybrid pipeline (cùng với vector + BM25).

Pipeline:
  query → KG traversal → list of (Article_id_kg, source_url, article_number, score)
  → caller map sang chunks trong vectorstore qua VectorStore.get_by_filter()
  → đưa vào RRF fusion cùng vector + BM25 results

Mục tiêu của Tích hợp C: giúp retrieve những điều luật có TỪNG ĐƯỢC trích xuất
như là chứa hành vi pháp lý liên quan tới query — kể cả khi vector + BM25 miss.
"""
from __future__ import annotations

from loguru import logger

from dataclasses import dataclass
from typing import Optional

from .neo4j_client import Neo4jClient


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

    def __init__(self, max_keywords: int = 5):
        self._client: Optional[Neo4jClient] = None
        self.max_keywords = max_keywords

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
                keywords.append(phrase)

        # Dedupe, giữ ngắn nhất 3 chars
        seen: set[str] = set()
        out: list[str] = []
        for k in keywords:
            if k in seen or len(k) < 3:
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

        # Cypher chấp nhận list keywords, mỗi keyword check riêng → tính score
        cypher = """
            // ── Branch 1a: qua Offense → Clause → Article ────────────────────
            UNWIND $keywords AS kw
            MATCH (o:Offense) WHERE toLower(o.name) CONTAINS kw
            MATCH (c:Clause)-[:PENALIZES]->(o)
            MATCH (a:Article)-[:HAS_CLAUSE]->(c)
            MATCH (l:Law)-[:HAS_ARTICLE]->(a)
            WITH a, l, o, kw
            RETURN a.id            AS article_id_kg,
                   a.number        AS article_number,
                   l.source        AS source_url,
                   o.name          AS matched_text,
                   'via_offense'   AS reason,
                   3.0             AS score

            UNION ALL

            // ── Branch 1b: qua Offense → Article (direct PENALIZES) ──────────
            UNWIND $keywords AS kw
            MATCH (o:Offense) WHERE toLower(o.name) CONTAINS kw
            MATCH (a:Article)-[:PENALIZES]->(o)
            MATCH (l:Law)-[:HAS_ARTICLE]->(a)
            RETURN a.id            AS article_id_kg,
                   a.number        AS article_number,
                   l.source        AS source_url,
                   o.name          AS matched_text,
                   'via_offense'   AS reason,
                   3.0             AS score

            UNION ALL

            // ── Branch 2: qua Article title ─────────────────────────────────
            UNWIND $keywords AS kw
            MATCH (a:Article)<-[:HAS_ARTICLE]-(l:Law)
            WHERE toLower(coalesce(a.title, '')) CONTAINS kw
            RETURN a.id            AS article_id_kg,
                   a.number        AS article_number,
                   l.source        AS source_url,
                   a.title         AS matched_text,
                   'via_article_title' AS reason,
                   2.0             AS score

            UNION ALL

            // ── Branch 3: qua Clause text_preview ───────────────────────────
            UNWIND $keywords AS kw
            MATCH (c:Clause)<-[:HAS_CLAUSE]-(a:Article)<-[:HAS_ARTICLE]-(l:Law)
            WHERE toLower(coalesce(c.text_preview, '')) CONTAINS kw
            RETURN a.id            AS article_id_kg,
                   a.number        AS article_number,
                   l.source        AS source_url,
                   substring(c.text_preview, 0, 100) AS matched_text,
                   'via_clause_text' AS reason,
                   1.0             AS score
        """

        try:
            with self._get_client().session() as s:
                rows = s.run(cypher, keywords=keywords).data()
        except Exception as exc:
            logger.warning(f"Neo4j query lỗi, KG trả rỗng: {exc}")
            return []

        # Dedupe theo article_id_kg, giữ score cao nhất
        best: dict[str, KGHit] = {}
        for r in rows:
            aid = r["article_id_kg"]
            article_num = r["article_number"]
            if not article_num:
                continue
            score = float(r["score"])
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

        # Sort by score
        ranked = sorted(best.values(), key=lambda h: -h.score)
        return ranked[:top_k]

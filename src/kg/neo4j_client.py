"""Wrapper quanh neo4j-python-driver — singleton, helper methods cho legal KG."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from neo4j import Driver, GraphDatabase, Session


class Neo4jClient:
    """Singleton driver. Đọc credentials từ env (load_dotenv() ở config.py)."""

    _instance: Optional["Neo4jClient"] = None

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.database = database
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    @classmethod
    def from_env(cls) -> "Neo4jClient":
        if cls._instance is not None:
            return cls._instance
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        pw = os.getenv("NEO4J_PASSWORD")
        db = os.getenv("NEO4J_DATABASE", "neo4j")
        if not (uri and user and pw):
            raise RuntimeError(
                "Thiếu NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD trong .env"
            )
        cls._instance = cls(uri, user, pw, db)
        return cls._instance

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._driver.session(database=self.database) as s:
            yield s

    def close(self) -> None:
        self._driver.close()
        Neo4jClient._instance = None

    # ─── Helpers cho legal KG ─────────────────────────────────────────────────

    def reset_all(self) -> None:
        """Xoá toàn bộ nodes và relationships. Dùng cho rebuild from scratch."""
        with self.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    def create_constraints(self) -> None:
        """Unique constraints để MERGE idempotent + tăng tốc lookup."""
        with self.session() as s:
            s.run("CREATE CONSTRAINT law_id IF NOT EXISTS FOR (l:Law) REQUIRE l.id IS UNIQUE")
            s.run("CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE")
            s.run("CREATE CONSTRAINT clause_id IF NOT EXISTS FOR (c:Clause) REQUIRE c.id IS UNIQUE")
            s.run("CREATE CONSTRAINT offense_name IF NOT EXISTS FOR (o:Offense) REQUIRE o.name IS UNIQUE")
            s.run("CREATE CONSTRAINT subject_name IF NOT EXISTS FOR (s:Subject) REQUIRE s.name IS UNIQUE")

    def stats(self) -> dict[str, int]:
        """Đếm node theo label + relation theo type."""
        with self.session() as s:
            node_counts = s.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC"
            ).data()
            rel_counts = s.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY n DESC"
            ).data()
        return {
            "nodes": {r["label"]: r["n"] for r in node_counts if r["label"]},
            "relations": {r["type"]: r["n"] for r in rel_counts},
        }

    # ─── Bulk write (UNWIND batch) ───────────────────────────────────────────

    def upsert_laws(self, laws: list[dict[str, Any]]) -> None:
        """MERGE batch Law nodes."""
        if not laws:
            return
        with self.session() as s:
            s.run(
                """
                UNWIND $rows AS row
                MERGE (l:Law {id: row.id})
                SET l += row.props
                """,
                rows=laws,
            )

    def upsert_articles(self, articles: list[dict[str, Any]]) -> None:
        """MERGE Article + HAS_ARTICLE edge tới Law."""
        if not articles:
            return
        with self.session() as s:
            s.run(
                """
                UNWIND $rows AS row
                MATCH (l:Law {id: row.law_id})
                MERGE (a:Article {id: row.id})
                SET a += row.props
                MERGE (l)-[:HAS_ARTICLE]->(a)
                """,
                rows=articles,
            )

    def upsert_clauses(self, clauses: list[dict[str, Any]]) -> None:
        """MERGE Clause + HAS_CLAUSE edge tới Article."""
        if not clauses:
            return
        with self.session() as s:
            s.run(
                """
                UNWIND $rows AS row
                MATCH (a:Article {id: row.article_id})
                MERGE (c:Clause {id: row.id})
                SET c += row.props
                MERGE (a)-[:HAS_CLAUSE]->(c)
                """,
                rows=clauses,
            )

    def add_internal_citations(self, edges: list[dict[str, str]]) -> None:
        """Tạo REFERS_TO edges giữa Articles trong cùng Law."""
        if not edges:
            return
        with self.session() as s:
            s.run(
                """
                UNWIND $rows AS row
                MATCH (src:Article {id: row.src})
                MATCH (dst:Article {id: row.dst})
                MERGE (src)-[:REFERS_TO]->(dst)
                """,
                rows=edges,
            )

    def add_amendment_edges(self, edges: list[dict[str, str]]) -> None:
        """Tạo AMENDS edges giữa các Law."""
        if not edges:
            return
        with self.session() as s:
            s.run(
                """
                UNWIND $rows AS row
                MATCH (src:Law {id: row.src})
                MATCH (dst:Law {id: row.dst})
                MERGE (src)-[:AMENDS]->(dst)
                """,
                rows=edges,
            )

    # ─── Phase 1: Semantic entities ──────────────────────────────────────────

    def write_semantic_extraction(
        self,
        article_id: str,
        offenses: list[dict],
        penalties: list[dict],
        subjects: list[dict],
        relations: list[dict],
    ) -> None:
        """Ghi 1 batch extract semantic vào Neo4j cho 1 Article.

        - Offense + Subject MERGE theo name (dedupe global)
        - Penalty CREATE mới (mỗi mức phạt 1 node — không dedupe global vì
          cùng "phạt tiền 800k" có thể xuất hiện ở nhiều luật khác nhau)
        - Relations từ Clause của article tới Offense/Penalty/Subject
        """
        with self.session() as s:
            # 1. Offense MERGE theo name
            if offenses:
                s.run(
                    """
                    UNWIND $rows AS row
                    MERGE (o:Offense {name: row.name})
                    SET o.description = coalesce(o.description, row.description)
                    """,
                    rows=offenses,
                )

            # 2. Subject MERGE theo name
            if subjects:
                s.run(
                    """
                    UNWIND $rows AS row
                    MERGE (sub:Subject {name: row.name})
                    SET sub.type = coalesce(sub.type, row.type)
                    """,
                    rows=subjects,
                )

            # 3. Penalty: tạo id deterministic = article_id + index để idempotent
            penalty_rows = []
            for i, p in enumerate(penalties):
                pid = f"{article_id}::penalty_{i}"
                penalty_rows.append({
                    "id": pid,
                    "description": p.get("description", "")[:300],
                    "type": p.get("type", "khac"),
                    "amount_min": p.get("amount_min"),
                    "amount_max": p.get("amount_max"),
                    "duration_min": p.get("duration_min"),
                    "duration_max": p.get("duration_max"),
                })
            if penalty_rows:
                s.run(
                    """
                    UNWIND $rows AS row
                    MERGE (p:Penalty {id: row.id})
                    SET p.description  = row.description,
                        p.type         = row.type,
                        p.amount_min   = row.amount_min,
                        p.amount_max   = row.amount_max,
                        p.duration_min = row.duration_min,
                        p.duration_max = row.duration_max
                    """,
                    rows=penalty_rows,
                )

            # 4. Build relations: Clause -> Offense / Penalty / Subject
            # Mapping article_id + clause_num -> clause_id
            for rel in relations:
                clause_num = str(rel.get("clause", "0")).strip()
                clause_id = f"{article_id}::khoan_{clause_num}" if clause_num != "0" else None

                offense_names = rel.get("offense_names") or []
                penalty_idxs  = rel.get("penalty_indices") or []
                subject_names = rel.get("subject_names") or []

                penalty_ids = [
                    f"{article_id}::penalty_{i}"
                    for i in penalty_idxs
                    if isinstance(i, int) and 0 <= i < len(penalties)
                ]

                # Edge gốc: từ Clause (nếu có), fallback từ Article
                if clause_id:
                    # Verify clause exists; if not, fall back to article
                    exists = s.run(
                        "MATCH (c:Clause {id: $cid}) RETURN c.id AS id LIMIT 1",
                        cid=clause_id,
                    ).single()
                    if not exists:
                        clause_id = None

                src_query = (
                    "MATCH (src:Clause {id: $src_id})"
                    if clause_id
                    else "MATCH (src:Article {id: $src_id})"
                )
                src_id = clause_id or article_id

                if offense_names:
                    s.run(
                        f"""
                        {src_query}
                        UNWIND $names AS oname
                        MATCH (o:Offense {{name: oname}})
                        MERGE (src)-[:PENALIZES]->(o)
                        """,
                        src_id=src_id, names=offense_names,
                    )

                if penalty_ids:
                    s.run(
                        f"""
                        {src_query}
                        UNWIND $pids AS pid
                        MATCH (p:Penalty {{id: pid}})
                        MERGE (src)-[:IMPOSES]->(p)
                        """,
                        src_id=src_id, pids=penalty_ids,
                    )

                if subject_names and offense_names:
                    # Offense -> APPLIES_TO -> Subject (mức ngữ nghĩa)
                    s.run(
                        """
                        UNWIND $onames AS oname
                        MATCH (o:Offense {name: oname})
                        UNWIND $snames AS sname
                        MATCH (sub:Subject {name: sname})
                        MERGE (o)-[:APPLIES_TO]->(sub)
                        """,
                        onames=offense_names, snames=subject_names,
                    )

                # Shortcut: Offense -> PUNISHED_BY -> Penalty (1-hop convenience)
                if offense_names and penalty_ids:
                    s.run(
                        """
                        UNWIND $onames AS oname
                        MATCH (o:Offense {name: oname})
                        UNWIND $pids AS pid
                        MATCH (p:Penalty {id: pid})
                        MERGE (o)-[:PUNISHED_BY]->(p)
                        """,
                        onames=offense_names, pids=penalty_ids,
                    )

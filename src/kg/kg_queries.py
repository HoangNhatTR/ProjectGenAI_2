"""Predefined Cypher queries + helper API cho KG Explorer UI và Agent tool.

Mục tiêu: che giấu Cypher complexity, cung cấp API Python sạch sẽ cho Streamlit
và LegalToolRegistry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .neo4j_client import Neo4jClient


# ─── Predefined queries cho UI dropdown ──────────────────────────────────────

@dataclass
class QueryTemplate:
    name: str
    description: str
    cypher: str
    params_schema: dict[str, str]  # param_name -> description / default

    def needs_params(self) -> bool:
        return bool(self.params_schema)


PREDEFINED_QUERIES: dict[str, QueryTemplate] = {
    "top_laws_by_articles": QueryTemplate(
        name="Top luật có nhiều Điều nhất",
        description="Liệt kê các luật xếp theo số lượng Điều, top 20.",
        cypher="""
            MATCH (l:Law)-[:HAS_ARTICLE]->(a:Article)
            RETURN coalesce(l.title, '?')      AS luat,
                   coalesce(l.doc_number, '?') AS so_hieu,
                   coalesce(l.issued_date, '?') AS ngay,
                   count(a)                     AS so_dieu
            ORDER BY so_dieu DESC LIMIT 20
        """,
        params_schema={},
    ),

    "most_cited_articles": QueryTemplate(
        name="Điều được dẫn chiếu nhiều nhất",
        description="Tìm Điều quan trọng (được nhiều Điều khác trỏ tới).",
        cypher="""
            MATCH (a:Article)<-[r:REFERS_TO]-()
            WITH a, count(r) AS so_lan
            MATCH (l:Law)-[:HAS_ARTICLE]->(a)
            RETURN coalesce(l.doc_number, '?') AS so_hieu_luat,
                   a.number                    AS dieu_so,
                   coalesce(a.title, '')       AS tieu_de,
                   so_lan
            ORDER BY so_lan DESC LIMIT 20
        """,
        params_schema={},
    ),

    "search_offense_by_keyword": QueryTemplate(
        name="Tìm hành vi vi phạm theo keyword",
        description="Tìm Offense + Penalty áp dụng cho 1 hành vi.",
        cypher="""
            MATCH (o:Offense)
            WHERE toLower(o.name) CONTAINS toLower($keyword)
            OPTIONAL MATCH (o)-[:PUNISHED_BY]->(p:Penalty)
            RETURN o.name             AS hanh_vi,
                   o.description       AS mo_ta,
                   collect(DISTINCT p.description) AS hinh_phat
            LIMIT 20
        """,
        params_schema={"keyword": "Từ khoá hành vi (vd: 'trộm', 'vượt đèn đỏ')"},
    ),

    "offenses_with_jail_min": QueryTemplate(
        name="Hành vi có phạt tù tối thiểu ≥ N tháng",
        description="Tìm các tội nghiêm trọng có khung phạt tù.",
        cypher="""
            MATCH (o:Offense)-[:PUNISHED_BY]->(p:Penalty)
            WHERE p.type = 'phat_tu' AND p.duration_min >= $min_months
            RETURN o.name             AS hanh_vi,
                   p.description       AS muc_phat,
                   p.duration_min      AS thang_min,
                   p.duration_max      AS thang_max
            ORDER BY p.duration_min DESC LIMIT 30
        """,
        params_schema={"min_months": "Số tháng tù tối thiểu (vd: 60 = 5 năm)"},
    ),

    "law_subjects": QueryTemplate(
        name="Chủ thể áp dụng cho 1 hành vi",
        description="Hành vi nào áp dụng cho chủ thể nào.",
        cypher="""
            MATCH (o:Offense)-[:APPLIES_TO]->(s:Subject)
            WHERE toLower(s.name) CONTAINS toLower($subject_keyword)
            RETURN s.name             AS chu_the,
                   collect(DISTINCT o.name)[..15] AS hanh_vi_lien_quan
            LIMIT 10
        """,
        params_schema={"subject_keyword": "Keyword chủ thể (vd: 'pháp nhân', 'xe máy')"},
    ),

    "law_amends_chain": QueryTemplate(
        name="Chuỗi sửa đổi của 1 luật",
        description="Xem luật nào sửa luật nào (chuỗi AMENDS).",
        cypher="""
            MATCH path = (l:Law)-[:AMENDS*1..3]->(target:Law)
            WHERE coalesce(l.doc_number, '') = $doc_number
               OR coalesce(target.doc_number, '') = $doc_number
            RETURN [n IN nodes(path) | coalesce(n.doc_number, n.title)] AS chain
            LIMIT 10
        """,
        params_schema={"doc_number": "Số hiệu văn bản (vd: '100/2015/QH13')"},
    ),

    "internal_citations_for_law": QueryTemplate(
        name="Mạng dẫn chiếu nội bộ trong 1 luật",
        description="Xem các Điều dẫn chiếu lẫn nhau trong cùng 1 luật.",
        cypher="""
            MATCH (l:Law)-[:HAS_ARTICLE]->(a1:Article)-[:REFERS_TO]->(a2:Article)
            WHERE coalesce(l.doc_number, '') = $doc_number
            RETURN a1.number AS dieu_nguon,
                   a2.number AS dieu_dich,
                   coalesce(a2.title, '') AS tieu_de_dich
            ORDER BY a1.number, a2.number LIMIT 100
        """,
        params_schema={"doc_number": "Số hiệu luật (vd: '100/2015/QH13')"},
    ),

    "article_full_context": QueryTemplate(
        name="Xem chi tiết 1 Điều (Khoản + dẫn chiếu)",
        description="Lấy đầy đủ Khoản con + Điều liên quan của 1 Điều.",
        cypher="""
            MATCH (l:Law {doc_number: $doc_number})-[:HAS_ARTICLE]->(a:Article {number: toInteger($article_no)})
            OPTIONAL MATCH (a)-[:HAS_CLAUSE]->(c:Clause)
            OPTIONAL MATCH (a)-[:REFERS_TO]->(ref:Article)<-[:HAS_ARTICLE]-(refLaw:Law)
            RETURN a.number AS dieu,
                   coalesce(a.title, '') AS tieu_de,
                   collect(DISTINCT {khoan: c.number, noi_dung: c.text_preview})[..10] AS cac_khoan,
                   collect(DISTINCT refLaw.doc_number + ' Điều ' + ref.number)[..10] AS dan_chieu
        """,
        params_schema={
            "doc_number": "Số hiệu luật (vd: '100/2015/QH13')",
            "article_no": "Số điều (vd: '260')",
        },
    ),
}


# ─── Helper API ──────────────────────────────────────────────────────────────

def _client() -> Neo4jClient:
    return Neo4jClient.from_env()


def execute_cypher(cypher: str, params: Optional[dict] = None) -> list[dict]:
    """Execute Cypher and return list of records as dicts."""
    with _client().session() as s:
        result = s.run(cypher, parameters=params or {})
        return [dict(r) for r in result]


def get_stats() -> dict[str, Any]:
    """Lấy stats nodes + relations từ KG."""
    return _client().stats()


def get_subgraph_around(
    center_label: str,
    center_property: str,
    center_value: str,
    depth: int = 2,
    limit: int = 50,
) -> list[dict]:
    """Lấy subgraph quanh 1 node để visualize.

    Returns list of dicts {source, target, src_label, tgt_label, rel_type}.
    """
    cypher = f"""
        MATCH (center:{center_label} {{{center_property}: $val}})
        MATCH path = (center)-[*1..{depth}]-(neighbor)
        WITH relationships(path) AS rels
        UNWIND rels AS r
        WITH DISTINCT r
        RETURN
          coalesce(startNode(r).name, startNode(r).id, startNode(r).title, 'unknown') AS source,
          labels(startNode(r))[0] AS src_label,
          coalesce(endNode(r).name, endNode(r).id, endNode(r).title, 'unknown') AS target,
          labels(endNode(r))[0] AS tgt_label,
          type(r) AS rel_type
        LIMIT {limit}
    """
    return execute_cypher(cypher, {"val": center_value})


def search_by_name(
    label: str,
    name_keyword: str,
    limit: int = 20,
) -> list[dict]:
    """Search nodes by name contains (case-insensitive). Dùng cho auto-complete."""
    cypher = f"""
        MATCH (n:{label})
        WHERE toLower(coalesce(n.name, n.title, n.id, '')) CONTAINS toLower($kw)
        RETURN coalesce(n.name, n.title, n.id) AS name,
               labels(n)[0] AS label
        LIMIT {limit}
    """
    return execute_cypher(cypher, {"kw": name_keyword})


# ─── Tool for agent (Tích hợp B sẽ dùng) ──────────────────────────────────────

def search_offense_to_penalty(offense_keyword: str) -> dict:
    """1-shot lookup: hành vi → hình phạt → chủ thể.

    Dùng làm "knowledge_graph_lookup" tool cho agent.
    """
    cypher = """
        MATCH (o:Offense)
        WHERE toLower(o.name) CONTAINS toLower($kw)
        OPTIONAL MATCH (o)-[:PUNISHED_BY]->(p:Penalty)
        OPTIONAL MATCH (o)-[:APPLIES_TO]->(s:Subject)
        OPTIONAL MATCH (c:Clause)-[:PENALIZES]->(o)
        OPTIONAL MATCH (l:Law)-[:HAS_ARTICLE]->(:Article)-[:HAS_CLAUSE]->(c)
        WITH o,
             collect(DISTINCT p.description)              AS penalties,
             collect(DISTINCT s.name)                      AS subjects,
             collect(DISTINCT l.doc_number + ' ' + l.title) AS source_laws
        RETURN o.name AS offense,
               o.description AS description,
               penalties, subjects, source_laws
        LIMIT 5
    """
    rows = execute_cypher(cypher, {"kw": offense_keyword})
    return {"keyword": offense_keyword, "results": rows}

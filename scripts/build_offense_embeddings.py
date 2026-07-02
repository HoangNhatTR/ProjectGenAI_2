"""Embed toàn bộ tên Offense trong KG → cache để khớp NGỮ NGHĨA (semantic offense match).

Hiện KG khớp Offense bằng CHUỖI (`CONTAINS`) → câu kể chuyện dài không chứa tên tội
nguyên văn thì trượt. Script này embed mọi `Offense.name` bằng chính BGE-M3 (cùng không
gian với chunk vector) + lưu kèm ánh xạ Offense → Article(s) để nhánh KG cosine trỏ thẳng
được tới điều luật.

Output (cache đọc bởi src/kg/kg_retriever.py):
  data/kg/offense_vectors.npy   — ma trận N×1024 float32 (đã normalize)
  data/kg/offense_meta.json     — {"names": [...], "articles": [[[aid,num,src],...], ...]}

Chạy:
  $env:PYTHONUTF8="1"; ../Chatbot/Scripts/python.exe -m scripts.build_offense_embeddings
"""
from __future__ import annotations

import json
import sys
import time

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import numpy as np

from src import config
from src.embedding import Embedder
from src.kg.neo4j_client import Neo4jClient

OUT_DIR = config.DATA_DIR / "kg"
VEC_PATH = OUT_DIR / "offense_vectors.npy"
META_PATH = OUT_DIR / "offense_meta.json"
TITLE_VEC_PATH = OUT_DIR / "article_title_vectors.npy"
TITLE_META_PATH = OUT_DIR / "article_title_meta.json"

# Tiêu đề Article = TÊN TỘI CANONICAL ("Tội cố ý gây thương tích..."). Embed để
# khớp ngữ nghĩa thẳng tới Điều — bù nơi tên Offense gán nhãn lệch/thiếu.
TITLE_CYPHER = """
    MATCH (l:Law)-[:HAS_ARTICLE]->(a:Article)
    WHERE a.title IS NOT NULL AND a.number IS NOT NULL
    RETURN a.title AS title, a.id AS aid, a.number AS num, l.source AS src
"""

# Lấy Offense + Article xử phạt nó qua CẢ HAI đường: qua Clause và PENALIZES trực tiếp.
CYPHER = """
    MATCH (a:Article)-[:HAS_CLAUSE]->(:Clause)-[:PENALIZES]->(o:Offense),
          (l:Law)-[:HAS_ARTICLE]->(a)
    WHERE o.name IS NOT NULL AND a.number IS NOT NULL
    RETURN o.name AS name, a.id AS aid, a.number AS num, l.source AS src
    UNION
    MATCH (a:Article)-[:PENALIZES]->(o:Offense),
          (l:Law)-[:HAS_ARTICLE]->(a)
    WHERE o.name IS NOT NULL AND a.number IS NOT NULL
    RETURN o.name AS name, a.id AS aid, a.number AS num, l.source AS src
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Truy vấn Neo4j: Offense → Article...")
    client = Neo4jClient.from_env()
    with client.session() as s:
        rows = s.run(CYPHER).data()
    print(f"  {len(rows):,} cặp (offense, article).")

    # Gom theo tên Offense (duy nhất) → tập article (aid, num, src) khử trùng.
    name_to_articles: dict[str, list[list]] = {}
    for r in rows:
        name = (r["name"] or "").strip()
        if not name:
            continue
        entry = [r["aid"], int(r["num"]), r["src"] or ""]
        bucket = name_to_articles.setdefault(name, [])
        if entry not in bucket:
            bucket.append(entry)

    names = list(name_to_articles.keys())
    articles = [name_to_articles[n] for n in names]
    print(f"  {len(names):,} tên Offense duy nhất (TB {sum(len(a) for a in articles)/max(len(names),1):.1f} điều/offense).")

    print("Embed bằng BGE-M3...")
    emb = Embedder(config.EMBEDDING_MODEL)
    t0 = time.time()
    vecs = np.asarray(emb.encode(names), dtype=np.float32)  # đã L2-normalize
    print(f"  Embed xong {vecs.shape} trong {time.time()-t0:.0f}s.")

    np.save(VEC_PATH, vecs)
    META_PATH.write_text(
        json.dumps({"names": names, "articles": articles}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Đã lưu offense:\n  {VEC_PATH}\n  {META_PATH}")

    # ── Nguồn 2: Article title ────────────────────────────────────────────────
    print("\nTruy vấn Neo4j: Article title...")
    with client.session() as s:
        trows = s.run(TITLE_CYPHER).data()
    # Mỗi Article = 1 entry; meta dùng CHUNG format {"names","articles"} với offense:
    # names = list tiêu đề, articles[i] = [[aid,num,src]] (một phần tử).
    titles, t_articles, seen = [], [], set()
    for r in trows:
        aid = r["aid"]
        title = (r["title"] or "").strip()
        if not title or aid in seen:
            continue
        seen.add(aid)
        titles.append(title)
        t_articles.append([[aid, int(r["num"]), r["src"] or ""]])
    print(f"  {len(titles):,} Article có title. Embed...")
    t0 = time.time()
    tvecs = np.asarray(emb.encode(titles), dtype=np.float32)
    print(f"  Embed xong {tvecs.shape} trong {time.time()-t0:.0f}s.")
    np.save(TITLE_VEC_PATH, tvecs)
    TITLE_META_PATH.write_text(
        json.dumps({"names": titles, "articles": t_articles}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Đã lưu article_title:\n  {TITLE_VEC_PATH}\n  {TITLE_META_PATH}")
    print("\nXong 2 nguồn. Nhánh KG semantic tự nạp (bật env KG_SEMANTIC=1, mặc định bật).")


if __name__ == "__main__":
    main()

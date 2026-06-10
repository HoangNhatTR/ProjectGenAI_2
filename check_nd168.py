"""Kiểm tra NĐ 168 có trong vectorstore và có được retrieve không."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.vectorstore import VectorStore
from src.embedding import Embedder
from src import config

e = Embedder(config.EMBEDDING_MODEL)
v = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)
print(f"Total chunks: {v.count()}")

# 1. Tìm chunk từ NĐ 168 trong store
r = v._collection.get(where={"source": {"$contains": "ND168"}}, limit=5)
print(f"\n=== Chunks có source chứa 'ND168': {len(r['ids'])} ===")
for i, (doc_id, meta, doc) in enumerate(zip(r["ids"], r["metadatas"], r["documents"]), 1):
    print(f"[{i}] id={doc_id[:20]} | src={meta.get('source','')}")
    print(f"     {doc[:120].replace(chr(10),' ')}")

# 2. Query semantic với câu hỏi tiếng Việt
q = "xe máy không đội mũ bảo hiểm bị phạt bao nhiêu tiền"
r2 = v._collection.query(query_texts=[q], n_results=8, include=["documents","metadatas","distances"])
print(f"\n=== Top 8 kết quả cho: '{q}' ===")
for i, (doc, meta, dist) in enumerate(zip(r2["documents"][0], r2["metadatas"][0], r2["distances"][0]), 1):
    src = meta.get("source","").replace("\\","/").split("/")[-1][:45]
    print(f"[{i}] score={1-dist:.3f} | {src}")
    print(f"     {doc[:130].replace(chr(10),' ')}")

# 3. Query thẳng về mức phạt trong NĐ 168
q2 = "mức phạt Điều 8 Nghị định 168 giao thông mũ bảo hiểm xe mô tô"
r3 = v._collection.query(query_texts=[q2], n_results=5, include=["documents","metadatas","distances"])
print(f"\n=== Top 5 kết quả cho: '{q2}' ===")
for i, (doc, meta, dist) in enumerate(zip(r3["documents"][0], r3["metadatas"][0], r3["distances"][0]), 1):
    src = meta.get("source","").replace("\\","/").split("/")[-1][:45]
    print(f"[{i}] score={1-dist:.3f} | {src}")
    print(f"     {doc[:130].replace(chr(10),' ')}")

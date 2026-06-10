"""Check retrieval quality."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from src.vectorstore import VectorStore
from src.embedding import Embedder
from src.retriever import Retriever
from src import config

print("=== Init ===")
e = Embedder(config.EMBEDDING_MODEL)
v = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)
r = Retriever(e, v)
print(f"Total chunks: {v.count()}")

queries = [
    "muc phat xe may khong doi mu bao hiem",
    "xu phat khong doi mu bao hiem xe may",
    "Nghi dinh 100 2019 phat giao thong mu bao hiem",
    "100/2019/ND-CP",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"Query: {q}")
    print('='*60)
    chunks = r.retrieve(q, top_k=5)
    for i, c in enumerate(chunks, 1):
        src = c.chunk.metadata.source.replace("\\","/").split("/")[-1].replace(".txt","")[:40]
        art = (c.chunk.article or "")[:30]
        txt = c.chunk.text[:150].replace("\n"," ")
        print(f"  [{i}] score={c.score:.3f} | {src} | {art}")
        print(f"       {txt}")

# Kiem tra xem ND100 co trong store khong
print("\n=== Search for ND100/2019 ===")
results = v.collection.query(
    query_texts=["100/2019/ND-CP xu phat hanh chinh giao thong"],
    n_results=5,
    include=["documents","metadatas"]
)
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
    src = meta.get("source","").replace("\\","/").split("/")[-1][:50]
    print(f"  [{i}] source={src}")
    print(f"       {doc[:120].replace(chr(10),' ')}")

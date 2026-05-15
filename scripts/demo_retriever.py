"""Demo Retriever — wrapper trên Embedder + VectorStore.

Yêu cầu: đã chạy `python -m scripts.demo_vectorstore` trước để có 72 chunks trong store.

    python -m scripts.demo_retriever
"""
from __future__ import annotations

from src import config
from src.embedding import Embedder
from src.retriever import Retriever
from src.schemas import RetrievedChunk
from src.vectorstore import VectorStore


def show(results: list[RetrievedChunk]) -> None:
    if not results:
        print("  (không có kết quả)")
        return
    for rank, r in enumerate(results, 1):
        tag = " | ".join(filter(None, [r.chunk.article, r.chunk.clause])) or "(preamble)"
        preview = r.chunk.text.replace("\n", " ")[:120]
        print(f"  #{rank}  score={r.score:.3f}  [{r.chunk.metadata.source} → {tag}]")
        print(f"        {preview}...")


def main() -> None:
    embedder = Embedder(config.EMBEDDING_MODEL)
    store = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)
    retriever = Retriever(embedder, store)

    print(f"Vectorstore có {store.count()} chunks")
    print(f"TOP_K mặc định  : {config.TOP_K}")
    print()

    # 1) Query bình thường
    print("=" * 70)
    print("1. QUERY BÌNH THƯỜNG (không filter)")
    print("=" * 70)
    q = "Người uống rượu lái xe ô tô bị phạt bao nhiêu?"
    print(f"Q: {q}")
    show(retriever.retrieve(q, top_k=config.TOP_K))

    # 2) Filter theo source — chỉ tìm trong file Bộ luật Hình sự
    print("\n" + "=" * 70)
    print("2. FILTER theo source — chỉ tìm trong Bộ luật Hình sự")
    print("=" * 70)
    q = "Tội cố ý gây thương tích bị phạt thế nào?"
    print(f"Q: {q}")
    show(retriever.retrieve(
        q,
        top_k=3,
        filters={"source": "bo_luat_hinh_su.txt"},
    ))

    # 3) Filter ngược — cũng query đó nhưng tìm trong file luật giao thông
    #    → mong đợi không tìm thấy gì liên quan (hoặc score thấp)
    print("\n" + "=" * 70)
    print("3. FILTER ngược — cùng query nhưng tìm trong file luật giao thông")
    print("=" * 70)
    print(f"Q: {q}")
    show(retriever.retrieve(
        q,
        top_k=3,
        filters={"source": "luat_giao_thong_2025.txt"},
    ))

    # 4) min_score threshold — chỉ giữ kết quả có cosine ≥ 0.5
    print("\n" + "=" * 70)
    print("4. min_score=0.5 — bỏ kết quả không đủ tin cậy")
    print("=" * 70)
    q = "Cách nấu phở bò ngon nhất"  # query lạc đề
    print(f"Q: {q}  (lạc đề — kỳ vọng bị filter sạch)")
    show(retriever.retrieve(q, top_k=5, min_score=0.5))

    # 5) Cùng query với min_score=None — để so sánh
    print("\n[so sánh] cùng query nhưng KHÔNG min_score:")
    show(retriever.retrieve(q, top_k=5))


if __name__ == "__main__":
    main()

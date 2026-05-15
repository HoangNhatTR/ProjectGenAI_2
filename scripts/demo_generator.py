"""Demo end-to-end RAG: retrieve → generate với Ollama local.

Yêu cầu:
- Đã chạy `python -m scripts.demo_vectorstore` để có 72 chunks trong store
- Ollama server đang chạy (kiểm tra: `ollama list`)

    python -m scripts.demo_generator
"""
from __future__ import annotations

import time

from src import config
from src.embedding import Embedder
from src.generator import Generator
from src.retriever import Retriever
from src.vectorstore import VectorStore


QUESTIONS = [
    "Vượt đèn đỏ ô tô bị phạt bao nhiêu tiền?",
    "Tội cố ý gây thương tích bị xử lý thế nào?",
    "Trừ bao nhiêu điểm nếu tôi vượt quá tốc độ 15 km/h?",
    "Cách nấu phở bò ngon?",  # lạc đề — kỳ vọng "không tìm thấy căn cứ"
]


def ask(retriever: Retriever, generator: Generator, question: str, top_k: int = 5) -> None:
    print("\n" + "=" * 70)
    print(f"❓ {question}")
    print("=" * 70)

    t0 = time.time()
    contexts = retriever.retrieve(question, top_k=top_k, min_score=0.3)
    t_retr = time.time() - t0
    print(f"\n📚 Retrieved {len(contexts)} chunks (top_k={top_k}, min_score=0.3)  [{t_retr:.2f}s]")
    for i, r in enumerate(contexts, 1):
        tag = " | ".join(filter(None, [r.chunk.article, r.chunk.clause])) or "(preamble)"
        print(f"   [{i}] score={r.score:.3f}  {r.chunk.metadata.source} → {tag}")

    t0 = time.time()
    answer = generator.generate(question, contexts)
    t_gen = time.time() - t0
    print(f"\n💬 Trả lời  [{t_gen:.1f}s, model={generator.model}]:")
    print(answer.answer)

    if answer.citations:
        print(f"\n🔗 Citations ({len(answer.citations)}):")
        for i, cit in enumerate(answer.citations, 1):
            tag = " | ".join(filter(None, [cit.article, cit.clause])) or "(preamble)"
            print(f"   [{i}] {cit.source} → {tag}")


def main() -> None:
    print(f"Vectorstore     : {config.VECTORSTORE_DIR}")
    print(f"Embedding model : {config.EMBEDDING_MODEL}")
    print(f"LLM provider    : {config.LLM_PROVIDER}")
    print(f"LLM model       : {config.LLM_MODEL}")
    print(f"Ollama host     : {config.OLLAMA_HOST}")

    embedder = Embedder(config.EMBEDDING_MODEL)
    store = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)
    retriever = Retriever(embedder, store)
    generator = Generator(
        model=config.LLM_MODEL,
        host=config.OLLAMA_HOST,
        temperature=config.LLM_TEMPERATURE,
    )

    print(f"\nVectorstore có {store.count()} chunks. Bắt đầu hỏi.")

    for q in QUESTIONS:
        ask(retriever, generator, q, top_k=config.TOP_K)


if __name__ == "__main__":
    main()

"""Chạy 7 câu hỏi CHUỖI qua full pipeline (LLM KieAI) ở 2 chế độ, giữ history
giữa các lượt — xuất transcript song song để so Graph RAG vs RAG Top 15.

Lưu ý: bộ nhớ hội thoại (history/state) GIỐNG nhau ở 2 chế độ — khác biệt chỉ
đến từ KG truy xuất từng lượt. Script này cho ra câu trả lời thật để đọc cạnh nhau.

Chạy (Windows):
    $env:VECTORSTORE_DIR="...\\data\\vectorstore_top15"; python -m scripts.chained_transcript
"""
from __future__ import annotations

import os
import sys

# ── Override env TRƯỚC khi import config (load_dotenv không override os.environ) ──
os.environ.setdefault("COLLECTION_NAME", "legal_top15")
os.environ["LLM_PROVIDER"] = os.getenv("TRANSCRIPT_PROVIDER", "groq")
os.environ["LLM_MODEL"] = os.getenv("TRANSCRIPT_MODEL", "llama-3.3-70b-versatile")
os.environ["ROUTER_MODEL"] = os.environ["LLM_MODEL"]
os.environ["USE_HYDE"] = "false"

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import json
import time
from pathlib import Path

import src.retriever as _retr_mod
_retr_mod.KG_TIMEOUT_S = float(os.getenv("KG_TIMEOUT_S", "8.0"))  # Aura cloud cold-call cần >3s

TURN_DELAY = float(os.getenv("TURN_DELAY", "0"))  # giãn nhịp/lượt để né rate limit (Groq 12k TPM)

from src import config
from src.bm25_index import BM25Index
from src.embedding import Embedder
from src.generator import Generator
from src.kg.kg_retriever import KGRetriever
from src.pipeline import LegalPipeline, provider_credentials
from src.retriever import Retriever
from src.router import SmartRouter
from src.vectorstore import VectorStore

QUESTIONS = [
    ("Q1", "Hợp đồng mua bán căn hộ hình thành trong tương lai cần điều kiện gì để có hiệu lực và chủ đầu tư có bắt buộc bảo lãnh ngân hàng không?"),
    ("Q2", "Chủ đầu tư chậm bàn giao nhà 12 tháng thì người mua có quyền đơn phương chấm dứt hợp đồng và đòi phạt, bồi thường không?"),
    ("Q3", "Lý do chậm là bị Nhà nước thu hồi một phần đất dự án — đây có được coi là sự kiện bất khả kháng để miễn trách nhiệm không?"),
    ("Q4", "Cùng lúc công ty nợ lương người lao động 6 tháng và nợ bảo hiểm xã hội — người lao động có quyền gì, và việc này ảnh hưởng thế nào tới khả năng thực hiện nghĩa vụ với khách hàng?"),
    ("Q5", "Nhà đầu tư góp vốn muốn rút vốn khi công ty sắp mất khả năng thanh toán — rút được không và đứng ở thứ tự ưu tiên thanh toán nào so với khách hàng và người lao động?"),
    ("Q6", "Giám đốc dùng tiền đặt cọc của khách hàng sai mục đích có phạm tội hình sự không, và trách nhiệm hình sự của giám đốc quan hệ thế nào với trách nhiệm dân sự của công ty?"),
    ("Q7", "Công ty phá sản: thứ tự ưu tiên thanh toán giữa khách hàng, người lao động và cơ quan thuế ra sao, ai được kiện ai, và giám đốc có chịu trách nhiệm cá nhân không?"),
]


def build_pipeline() -> LegalPipeline:
    embedder = Embedder(config.EMBEDDING_MODEL)
    vstore = VectorStore(config.VECTORSTORE_DIR, config.COLLECTION_NAME)
    bm = BM25Index(config.DATA_DIR / "bm25" / "top15_index.json")
    try:
        kg = KGRetriever()
    except Exception as e:
        print(f"[!] KG không khả dụng: {e}")
        kg = None
    retriever = Retriever(embedder, vstore, bm25=bm, kg_retriever=kg)

    api_key, host = provider_credentials(config.LLM_PROVIDER)
    generator = Generator(model=config.LLM_MODEL, host=host,
                          temperature=config.LLM_TEMPERATURE,
                          provider=config.LLM_PROVIDER, api_key=api_key)
    try:
        from src.intent_classifier import IntentClassifier
        clf = IntentClassifier(embedder=embedder)
    except Exception as e:
        print(f"[!] IntentClassifier lỗi ({e}) — router không classifier")
        clf = None
    router = SmartRouter(model=config.ROUTER_MODEL, host=host,
                         provider=config.LLM_PROVIDER, api_key=api_key, classifier=clf)
    from src.tools import LegalToolRegistry
    tools = LegalToolRegistry(retriever=retriever, ollama_client=generator.get_client(),
                              model=config.LLM_MODEL)

    man = json.loads((config.DATA_DIR / "comparison" / "top15_laws" / "manifest.json").read_text(encoding="utf-8"))
    top15 = [l["source_url"] for l in man["laws"] if l.get("source_url")]

    print(f"Store={vstore.count():,} | BM25={bm.count():,} | KG={'on' if kg else 'off'} "
          f"| LLM={config.LLM_MODEL}@{config.LLM_PROVIDER}")
    return LegalPipeline(retriever=retriever, generator=generator, router=router,
                         tool_registry=tools, top15_urls=top15)


def run_mode(pipe: LegalPipeline, mode: str) -> list[dict]:
    history: list[dict] = []
    turns = []
    for qid, q in QUESTIONS:
        t0 = time.time()
        try:
            ans = pipe.run(q, history=history, rag_mode=mode, top_k=5, web_search_enabled=False)
            text = ans.answer or ""
            cites = []
            for c in (ans.citations or [])[:6]:
                lbl = getattr(c, "article", None) or getattr(c, "label", None) or ""
                src = getattr(c, "source", "") or getattr(getattr(c, "metadata", None), "source", "")
                cites.append(f"{str(src).split('/')[-1]}|{lbl}")
        except Exception as e:
            text, cites = f"[LỖI: {e}]", []
        dt = time.time() - t0
        print(f"  [{mode}] {qid} ({dt:.1f}s, {len(text)} ký tự)")
        turns.append({"qid": qid, "q": q, "answer": text, "cites": cites})
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": text})
        if TURN_DELAY:
            time.sleep(TURN_DELAY)
    return turns


def main() -> None:
    pipe = build_pipeline()
    print("\n=== GRAPH RAG ===")
    gr = run_mode(pipe, "graph_rag")
    print("\n=== RAG TOP 15 ===")
    rag = run_mode(pipe, "rag_top10")

    out = Path("chained_transcript.md")
    with out.open("w", encoding="utf-8") as f:
        f.write("# Transcript hội thoại chuỗi — Graph RAG vs RAG Top 15 (LLM: KieAI)\n\n")
        f.write("> Bộ nhớ hội thoại giống nhau ở 2 chế độ; khác biệt đến từ KG truy xuất.\n\n")
        for g, r in zip(gr, rag):
            f.write(f"## {g['qid']}. {g['q']}\n\n")
            f.write(f"### 🕸️ Graph RAG\n\nNguồn: `{', '.join(g['cites']) or '—'}`\n\n{g['answer']}\n\n")
            f.write(f"### 📚 RAG Top 15\n\nNguồn: `{', '.join(r['cites']) or '—'}`\n\n{r['answer']}\n\n---\n\n")
    print(f"\nĐã ghi transcript → {out.resolve()}")


if __name__ == "__main__":
    main()

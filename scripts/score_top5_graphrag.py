"""Chấm KHÁCH QUAN top-5 retrieval: RAG vs Graph RAG trên legal_top15.

Không phụ thuộc LLM. Với mỗi câu hỏi có ĐÁP ÁN CHUẨN (điều luật + đúng bộ luật),
kiểm tra điều luật đó có nằm trong top-5 chunk truy xuất hay không, ở cả 2 chế độ.

Tập câu = "sân nhà" của Graph RAG (tên tội khẩu ngữ, corpus nhiều biến thể) +
vài câu đối chứng để trung thực.

Chạy (PowerShell):
  $env:VECTORSTORE_DIR="data/vectorstore_top15"; $env:KG_TIMEOUT_S="8"
  ../Chatbot/Scripts/python.exe -m scripts.score_top5_graphrag
"""
from __future__ import annotations

import sys
import time

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.bm25_index import BM25Index
from src.embedding import Embedder
from src.kg.kg_retriever import KGRetriever
from src.retriever import Retriever
from src.vectorstore import VectorStore

COLL = "legal_top15"
BM25_PATH = config.DATA_DIR / "bm25" / "top15_index.json"
TOP_K = 5

# Định danh bộ luật trong source_url (đuôi id vbpl)
BLHS = "96122"   # Bộ luật Hình sự 100/2015
BLDS = "95942"   # Bộ luật Dân sự 91/2015

# id, câu hỏi, điều luật chuẩn, mã bộ luật phải khớp, nhóm
QUESTIONS = [
    # ── Sân nhà Graph RAG: tên tội khẩu ngữ, nhiều biến thể na ná ──
    ("S1", "Chống người thi hành công vụ bị xử lý ra sao?",                       "Điều 330", BLHS, "crime"),
    ("S2", "Đánh người gây thương tích thì bị truy cứu trách nhiệm hình sự khi nào?", "Điều 134", BLHS, "crime"),
    ("S3", "Cướp tài sản bị xử lý như thế nào?",                                  "Điều 168", BLHS, "crime"),
    ("S4", "Mua bán trái phép chất ma túy bị phạt thế nào?",                      "Điều 251", BLHS, "crime"),
    ("S5", "Làm giả con dấu, tài liệu của cơ quan, tổ chức bị xử lý thế nào?",    "Điều 341", BLHS, "crime"),
    ("S6", "Cho vay lãi nặng trong giao dịch dân sự (tín dụng đen) bị xử lý ra sao?", "Điều 201", BLHS, "crime"),
    ("S7", "Tội lừa đảo chiếm đoạt tài sản bị xử phạt thế nào?",                  "Điều 174", BLHS, "crime"),
    ("S8", "Lạm dụng tín nhiệm chiếm đoạt tài sản bị xử lý ra sao?",              "Điều 175", BLHS, "crime"),
    # ── Đối chứng: câu định nghĩa / phổ thông, kỳ vọng RAG đã đủ ──
    ("C1", "Thời hiệu khởi kiện để yêu cầu Tòa án giải quyết tranh chấp dân sự là bao lâu?", "Điều 154", BLDS, "control"),
    ("C2", "Năng lực hành vi dân sự của cá nhân là gì?",                          "Điều 19",  BLDS, "control"),
    # ── Câu KỂ CHUYỆN dài (không nêu thẳng tên tội) — nơi lexical KG bó tay,
    #    semantic offense match kỳ vọng cứu. Lấy từ Benchmark_TinhHuong.
    ("N1", "Khi công an đến cưỡng chế, bảo vệ công ty chống đối, giật còng và đẩy ngã cán bộ thì bị xử lý ra sao?", "Điều 330", BLHS, "narrative"),
    ("N2", "Trong lúc xô xát, một bảo vệ đánh người dân gây thương tích thì bị truy cứu trách nhiệm hình sự khi nào?", "Điều 134", BLHS, "narrative"),
    ("N3", "Giám đốc nhận tiền đặt cọc của khách hàng rồi dùng sai mục đích và bỏ trốn thì phạm tội gì?", "Điều 175", BLHS, "narrative"),
]


def hit(chunks, art, law_kw) -> tuple[bool, list[str]]:
    """Có chunk nào trong top-5 đúng điều luật + đúng bộ luật không."""
    found = False
    labels = []
    for c in chunks:
        src = (c.chunk.metadata.source or "")
        a = c.chunk.article or "-"
        labels.append(f"{a}@{src.split('/')[-1][:6]}")
        if a == art and law_kw in src:
            found = True
    return found, labels


def main():
    print("Đang init (embedder + store + KG)...")
    embedder = Embedder(config.EMBEDDING_MODEL)
    store = VectorStore(config.VECTORSTORE_DIR, COLL)
    if store.count() == 0:
        sys.exit(f"Collection '{COLL}' rỗng. Đặt VECTORSTORE_DIR=data/vectorstore_top15 ?")
    bm = BM25Index(BM25_PATH)
    kg = KGRetriever(embedder=embedder)  # dùng chung embedder (nhánh semantic cần encode query)
    print(f"  KG semantic: {'BẬT' if kg.semantic else 'TẮT'}")
    rag = Retriever(embedder=embedder, store=store, bm25=bm, kg_retriever=None)
    gr = Retriever(embedder=embedder, store=store, bm25=bm, kg_retriever=kg)
    print(f"Store: {store.count():,} chunks | BM25: {bm.count():,} | KG ready")
    # Warm-up KG: nuốt cold-call đầu tiên để câu S1 không bị phạt oan timeout
    try:
        kg.retrieve("chống người thi hành công vụ", top_k=5)
        print("KG warm-up xong.\n")
    except Exception as e:
        print(f"KG warm-up lỗi: {e}\n")

    rows = []
    for qid, query, art, law, grp in QUESTIONS:
        rag_c = rag.retrieve(query, top_k=TOP_K, use_kg=False)
        t0 = time.time()
        gr_c = gr.retrieve(query, top_k=TOP_K, use_kg=True)
        gr_ms = (time.time() - t0) * 1000
        rag_hit, rag_lbl = hit(rag_c, art, law)
        gr_hit, gr_lbl = hit(gr_c, art, law)
        rows.append((qid, grp, query, art, rag_hit, gr_hit, rag_lbl, gr_lbl, gr_ms))

        flag = ""
        if gr_hit and not rag_hit:
            flag = "  🏆 GRAPH RAG THẮNG"
        elif rag_hit and not gr_hit:
            flag = "  ⚠️ RAG hơn"
        print("=" * 78)
        print(f"  {qid} [{grp}]  cần: {art} ({law})  {query}{flag}")
        print(f"    RAG   top5: {'✅' if rag_hit else '❌'}  {rag_lbl}")
        print(f"    GRAPH top5: {'✅' if gr_hit else '❌'}  {gr_lbl}  ({gr_ms:.0f}ms)")

    print("\n" + "=" * 78)
    print("  📊 TỔNG KẾT (top-5 có chứa điều luật chuẩn?)")
    print("=" * 78)
    print(f"  {'Q':<5}{'Nhóm':<9}{'Cần':<11}{'RAG':>5}{'GRAPH':>7}  Ghi chú")
    for r in rows:
        note = ""
        if r[5] and not r[4]:
            note = "Graph RAG thắng"
        elif r[4] and not r[5]:
            note = "RAG hơn"
        elif r[4] and r[5]:
            note = "hòa (đều đúng)"
        else:
            note = "cả hai trượt"
        print(f"  {r[0]:<5}{r[1]:<9}{r[3]:<11}{'✅' if r[4] else '❌':>5}{'✅' if r[5] else '❌':>7}  {note}")

    for grp, label in (("crime", "Câu tên tội (sân nhà Graph RAG)"), ("control", "Câu đối chứng"),
                       ("narrative", "Câu kể chuyện dài (lexical KG hay trượt)")):
        sub = [r for r in rows if r[1] == grp]
        if not sub:
            continue
        rag_n = sum(1 for r in sub if r[4])
        gr_n = sum(1 for r in sub if r[5])
        print(f"\n  {label}: RAG {rag_n}/{len(sub)}  |  Graph RAG {gr_n}/{len(sub)}")
    all_rag = sum(1 for r in rows if r[4])
    all_gr = sum(1 for r in rows if r[5])
    wins = sum(1 for r in rows if r[5] and not r[4])
    losses = sum(1 for r in rows if r[4] and not r[5])
    print(f"\n  TỔNG: RAG {all_rag}/{len(rows)}  |  Graph RAG {all_gr}/{len(rows)}"
          f"   (Graph thắng riêng {wins}, thua riêng {losses})")


if __name__ == "__main__":
    main()

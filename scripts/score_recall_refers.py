"""Đo RECALL@k cho câu hỏi NHIỀU ĐIỀU — để lượng hoá lợi ích REFERS_TO expansion.

Khác `score_top5_graphrag.py` (gold MỘT điều): ở đây mỗi câu có gold là MỘT TẬP điều
luật liên quan. Tập gold lấy từ chính cấu trúc dẫn chiếu của luật: một điều "anchor"
(KG khớp được qua title) + các điều mà nó REFERS_TO — đây CHÍNH LÀ đáp án pháp lý
(vd "đình chỉ vụ án" Điều 248 BLTTHS căn cứ vào Điều 16/29/91/132/155/157).

Đo recall = |gold ∩ top-k| / |gold| ở 2 chế độ, CÙNG một retriever, chỉ bật/tắt
REFERS_TO expansion (kg.expand_refers) → cô lập đúng tác dụng của cạnh quan hệ.

Chạy:
  $env:VECTORSTORE_DIR="data/vectorstore_top15"; $env:KG_TIMEOUT_S="8"
  ../Chatbot/Scripts/python.exe -m scripts.score_recall_refers
"""
from __future__ import annotations

import sys

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
TOP_K = 15

# id, câu hỏi, mã luật (đuôi id vbpl), tập gold (số điều: anchor + các điều REFERS_TO).
# Gold = đáp án pháp lý thật (điều anchor dẫn chiếu tới các điều này trong văn bản).
QUESTIONS = [
    ("R1", "Trình tự, thủ tục cấp Giấy chứng nhận quyền sử dụng đất, quyền sở hữu tài sản gắn liền với đất liên quan đến những quy định nào?",
     "177815", {138, 118, 137, 139, 140, 141, 176, 195, 196}),
    ("R2", "Đình chỉ vụ án hình sự được thực hiện theo những căn cứ và điều luật nào?",
     "96172", {248, 16, 29, 91, 132, 155, 157}),
    ("R3", "Việc xác định cha, mẹ, con có yếu tố nước ngoài áp dụng những quy định nào?",
     "36870", {128, 88, 89, 90, 97, 98, 99}),
    ("R4", "Tội vi phạm quy định về quản lý, bảo vệ động vật hoang dã liên quan đến những điều luật nào?",
     "96122", {234, 79, 242, 244}),
    ("R5", "Tội vi phạm quy định về xây dựng gây hậu quả nghiêm trọng liên quan đến những điều luật nào?",
     "96122", {298, 224, 225, 281}),
    ("R6", "Giao đất, cho thuê đất thông qua đấu giá quyền sử dụng đất áp dụng những quy định nào?",
     "177815", {125, 119, 120, 122, 124, 126, 217}),
]


def found_nums(chunks, law_kw, gold) -> set:
    out = set()
    for c in chunks:
        src = c.chunk.metadata.source or ""
        art = c.chunk.article or ""
        if law_kw not in src or not art.startswith("Điều "):
            continue
        try:
            n = int(art.split()[1])
        except (IndexError, ValueError):
            continue
        if n in gold:
            out.add(n)
    return out


def main():
    print("Init...")
    embedder = Embedder(config.EMBEDDING_MODEL)
    store = VectorStore(config.VECTORSTORE_DIR, COLL)
    if store.count() == 0:
        sys.exit("Collection rỗng — đặt VECTORSTORE_DIR=data/vectorstore_top15 ?")
    bm = BM25Index(BM25_PATH)
    kg = KGRetriever(embedder=embedder)
    gr = Retriever(embedder=embedder, store=store, bm25=bm, kg_retriever=kg)
    kg.retrieve("đình chỉ vụ án")  # warm-up KG
    print(f"Store {store.count():,} | top_k={TOP_K}\n")

    rows = []
    for qid, query, law, gold in QUESTIONS:
        res = {}
        for label, flag in (("OFF", False), ("ON", True)):
            gr.expand_refers = flag
            chunks = gr.retrieve(query, top_k=TOP_K, use_kg=True)
            hit = found_nums(chunks, law, gold)
            res[label] = (len(hit), hit)
        rows.append((qid, gold, res))
        off_n, off_set = res["OFF"]
        on_n, on_set = res["ON"]
        gained = sorted(on_set - off_set)
        print("=" * 78)
        print(f"  {qid} (gold {len(gold)} điều)  {query[:60]}")
        print(f"    B-OFF recall@{TOP_K}: {off_n}/{len(gold)} = {off_n/len(gold):.0%}  {sorted(off_set)}")
        print(f"    B-ON  recall@{TOP_K}: {on_n}/{len(gold)} = {on_n/len(gold):.0%}  {sorted(on_set)}")
        if gained:
            print(f"    🟢 REFERS_TO kéo thêm: {gained}")

    print("\n" + "=" * 78)
    print("  📊 TỔNG KẾT RECALL@{}  (gold nhiều điều)".format(TOP_K))
    print("=" * 78)
    print(f"  {'Q':<5}{'gold':>5}{'B-OFF':>9}{'B-ON':>9}{'Δ':>6}")
    tot_off = tot_on = tot_gold = 0
    for qid, gold, res in rows:
        off_n = res["OFF"][0]; on_n = res["ON"][0]
        tot_off += off_n; tot_on += on_n; tot_gold += len(gold)
        print(f"  {qid:<5}{len(gold):>5}{off_n:>9}{on_n:>9}{on_n-off_n:>+6}")
    print("-" * 34)
    print(f"  Micro-recall  B-OFF {tot_off}/{tot_gold}={tot_off/tot_gold:.0%}   "
          f"B-ON {tot_on}/{tot_gold}={tot_on/tot_gold:.0%}   (+{tot_on-tot_off} điều)")


if __name__ == "__main__":
    main()

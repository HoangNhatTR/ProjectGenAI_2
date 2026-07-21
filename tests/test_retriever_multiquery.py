"""Unit tests cho multi-query fusion (RAG-fusion) — paraphrase + RRF giữa các query."""
from __future__ import annotations

from src.retriever import Retriever, _fuse_ranked_lists, _parse_paraphrases
from src.schemas import Chunk, DocumentMetadata, RetrievedChunk


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=f"text {cid}",
        metadata=DocumentMetadata(source=f"https://vbpl.vn/{cid}"),
    )


def _rc(cid: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk=_chunk(cid), score=score)


# ── _parse_paraphrases ─────────────────────────────────────────────────────────

def test_parse_bo_danh_so_va_bullet():
    raw = "1. Mức xử phạt hành vi không chấp hành đèn tín hiệu\n- Phạt tiền lỗi vượt đèn đỏ xe máy"
    out = _parse_paraphrases(raw, "vượt đèn đỏ phạt bao nhiêu?", 2)
    assert out == [
        "Mức xử phạt hành vi không chấp hành đèn tín hiệu",
        "Phạt tiền lỗi vượt đèn đỏ xe máy",
    ]


def test_parse_loai_trung_va_cau_goc():
    original = "vượt đèn đỏ phạt bao nhiêu?"
    raw = f"{original}\nMức phạt không chấp hành đèn tín hiệu\nMức phạt không chấp hành đèn tín hiệu"
    out = _parse_paraphrases(raw, original, 3)
    assert out == ["Mức phạt không chấp hành đèn tín hiệu"]


def test_parse_cap_n_va_bo_dong_ngan():
    raw = "ok\nCâu diễn đạt thứ nhất về xử phạt\nCâu diễn đạt thứ hai về xử phạt\nCâu diễn đạt thứ ba về xử phạt"
    out = _parse_paraphrases(raw, "q", 2)
    assert len(out) == 2  # "ok" quá ngắn bị bỏ, cap ở n=2


# ── _fuse_ranked_lists ─────────────────────────────────────────────────────────

def test_fuse_chunk_xuat_hien_nhieu_list_thang():
    lists = [
        [_rc("A"), _rc("B")],
        [_rc("B"), _rc("C")],
        [_rc("C"), _rc("B")],
    ]
    fused = _fuse_ranked_lists(lists, top_k=3)
    assert fused[0].chunk.chunk_id == "B"  # có mặt cả 3 list
    assert len(fused) == 3


def test_fuse_ton_trong_top_k():
    lists = [[_rc(f"c{i}") for i in range(10)]]
    assert len(_fuse_ranked_lists(lists, top_k=4)) == 4


# ── retrieve(use_multi_query=True) integration với stubs ──────────────────────

class _StubEmbedder:
    def encode(self, texts):
        return [[0.0, 0.0]] * len(texts)


class _StubStore:
    """Trả kết quả theo hàng đợi — mỗi lần query() lấy 1 list kế tiếp."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = 0

    def query(self, embedding, top_k, where=None):
        self.calls += 1
        return self.batches.pop(0) if self.batches else []


class _StubLLM:
    def __init__(self, content: str = "", raise_exc: bool = False):
        self.content = content
        self.raise_exc = raise_exc
        self.calls = 0

    def chat(self, **kw):
        self.calls += 1
        if self.raise_exc:
            raise RuntimeError("LLM down")
        return {"message": {"content": self.content}}


def test_multi_query_fuse_4_lan_retrieve(monkeypatch):
    """Gốc + 1 rule ('vượt đèn đỏ' có trong _RULE_SYNONYMS) + 2 paraphrase = 4."""
    monkeypatch.setattr("src.retriever.MQ_LLM_PARAPHRASE", True)
    store = _StubStore([
        [_rc("A"), _rc("B")],   # câu gốc
        [_rc("B"), _rc("C")],   # rule variant
        [_rc("C"), _rc("B")],   # paraphrase 1
        [_rc("B"), _rc("D")],   # paraphrase 2
    ])
    llm = _StubLLM("Mức phạt không chấp hành đèn tín hiệu\nXử phạt lỗi vượt đèn đỏ xe máy")
    r = Retriever(embedder=_StubEmbedder(), store=store, llm_client=llm, mq_model="fast")

    results = r.retrieve("vượt đèn đỏ phạt bao nhiêu", top_k=3, use_kg=False,
                         use_multi_query=True)

    assert store.calls == 4          # gốc + rule + 2 paraphrase
    assert llm.calls == 1
    assert results[0].chunk.chunk_id == "B"


def test_multi_query_llm_loi_van_con_rule_variant():
    """LLM lỗi nhưng câu khớp rule → vẫn fuse gốc + rule (recall tất định)."""
    store = _StubStore([
        [_rc("A"), _rc("B")],   # câu gốc
        [_rc("B"), _rc("C")],   # rule variant
    ])
    llm = _StubLLM(raise_exc=True)
    r = Retriever(embedder=_StubEmbedder(), store=store, llm_client=llm, mq_model="fast")

    results = r.retrieve("vượt đèn đỏ phạt bao nhiêu", top_k=3, use_kg=False,
                         use_multi_query=True)

    assert store.calls == 2          # gốc + rule (không có paraphrase)
    assert results[0].chunk.chunk_id == "B"  # B có mặt cả 2 list


def test_multi_query_llm_loi_khong_rule_fallback_retrieve_thuong():
    """LLM lỗi + câu KHÔNG khớp rule nào → dùng kết quả câu gốc như cũ."""
    store = _StubStore([[_rc("A"), _rc("B")]])
    llm = _StubLLM(raise_exc=True)
    r = Retriever(embedder=_StubEmbedder(), store=store, llm_client=llm, mq_model="fast")

    results = r.retrieve("thủ tục ly hôn đơn phương cần giấy tờ gì", top_k=2,
                         use_kg=False, use_multi_query=True)

    assert store.calls == 1          # chỉ retrieve câu gốc
    assert [x.chunk.chunk_id for x in results] == ["A", "B"]


def test_multi_query_bo_qua_khi_cau_da_co_trich_dan():
    """Câu chứa 'Điều N + số hiệu VB' → khỏi paraphrase (cùng gate với HyDE)."""
    store = _StubStore([[_rc("A")]])
    llm = _StubLLM("paraphrase nào đó dài đủ tám ký tự")
    r = Retriever(embedder=_StubEmbedder(), store=store, llm_client=llm, mq_model="fast")

    r.retrieve("Điều 8 Nghị định 168/2024/NĐ-CP quy định gì?", top_k=1,
               use_kg=False, use_multi_query=True)

    assert llm.calls == 0
    assert store.calls == 1


def test_multi_query_tat_mac_dinh():
    store = _StubStore([[_rc("A")]])
    llm = _StubLLM("paraphrase nào đó dài đủ tám ký tự")
    r = Retriever(embedder=_StubEmbedder(), store=store, llm_client=llm, mq_model="fast")

    r.retrieve("vượt đèn đỏ phạt bao nhiêu", top_k=1, use_kg=False)

    assert llm.calls == 0
    assert store.calls == 1


def test_recall_guard_giu_ngoi_sao_mot_list(monkeypatch):
    """Chunk top-1 của CHỈ 1 list (rule/paraphrase) phải có mặt trong fused —
    RRF đồng thuận thuần từng loại nó (bug Đ7K7 2026-07-08)."""
    monkeypatch.setattr("src.retriever.MQ_LLM_PARAPHRASE", True)
    # 4 list: "STAR" chỉ đứng đầu list cuối; các list khác đồng thuận X/Y/Z
    consensus = [_rc("X"), _rc("Y"), _rc("Z")]
    store = _StubStore([
        list(consensus),                 # gốc
        list(consensus),                 # rule ("vượt đèn đỏ" khớp synonym)
        list(consensus),                 # paraphrase 1
        [_rc("STAR")] + list(consensus), # paraphrase 2 — STAR chỉ ở đây
    ])
    llm = _StubLLM("Mức phạt không chấp hành đèn tín hiệu\nXử phạt lỗi vượt đèn đỏ xe máy")
    r = Retriever(embedder=_StubEmbedder(), store=store, llm_client=llm, mq_model="fast")

    results = r.retrieve("vượt đèn đỏ phạt bao nhiêu", top_k=4, use_kg=False,
                         use_multi_query=True)

    ids = [x.chunk.chunk_id for x in results]
    assert "STAR" in ids, f"recall guard phải giữ STAR trong pool, got {ids}"


def test_mac_dinh_khong_goi_llm_paraphrase_chi_rule(monkeypatch):
    """Mặc định MQ_LLM_PARAPHRASE=off: KHÔNG gọi LLM, chỉ gốc + rule variant."""
    monkeypatch.setattr("src.retriever.MQ_LLM_PARAPHRASE", False)
    store = _StubStore([
        [_rc("A"), _rc("B")],   # câu gốc
        [_rc("B"), _rc("C")],   # rule variant ("xe máy"→"xe mô tô, xe gắn máy")
    ])
    llm = _StubLLM("paraphrase KHÔNG được dùng tới")
    r = Retriever(embedder=_StubEmbedder(), store=store, llm_client=llm, mq_model="fast")

    results = r.retrieve("xe máy vượt đèn đỏ phạt bao nhiêu", top_k=3, use_kg=False,
                         use_multi_query=True)

    assert llm.calls == 0            # KHÔNG gọi LLM paraphrase
    assert store.calls == 2          # gốc + rule
    assert results[0].chunk.chunk_id == "B"  # B đồng thuận cả 2 list

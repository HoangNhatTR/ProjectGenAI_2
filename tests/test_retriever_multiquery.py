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


def test_multi_query_fuse_3_lan_retrieve():
    store = _StubStore([
        [_rc("A"), _rc("B")],   # câu gốc
        [_rc("B"), _rc("C")],   # paraphrase 1
        [_rc("C"), _rc("B")],   # paraphrase 2
    ])
    llm = _StubLLM("Mức phạt không chấp hành đèn tín hiệu\nXử phạt lỗi vượt đèn đỏ xe máy")
    r = Retriever(embedder=_StubEmbedder(), store=store, llm_client=llm, mq_model="fast")

    results = r.retrieve("vượt đèn đỏ phạt bao nhiêu", top_k=3, use_kg=False,
                         use_multi_query=True)

    assert store.calls == 3          # gốc + 2 paraphrase
    assert llm.calls == 1
    assert results[0].chunk.chunk_id == "B"


def test_multi_query_llm_loi_fallback_retrieve_thuong():
    store = _StubStore([[_rc("A"), _rc("B")]])
    llm = _StubLLM(raise_exc=True)
    r = Retriever(embedder=_StubEmbedder(), store=store, llm_client=llm, mq_model="fast")

    results = r.retrieve("vượt đèn đỏ phạt bao nhiêu", top_k=2, use_kg=False,
                         use_multi_query=True)

    assert store.calls == 1          # chỉ retrieve thường
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

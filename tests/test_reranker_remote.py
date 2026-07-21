"""Tests remote rerank API path: dùng score API, map index, fallback khi lỗi."""
from __future__ import annotations

import json

import src.reranker as reranker
from src.reranker import rerank, remote_rerank_available
from src.schemas import Chunk, DocumentMetadata, RetrievedChunk


def _rc(cid: str, text: str = "nội dung", score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=cid, text=text,
                    metadata=DocumentMetadata(source=f"https://vbpl.vn/{cid}")),
        score=score,
    )


class _FakeResp:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _enable_remote(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_API_URL", "https://fake/v1/rerank")
    monkeypatch.setattr(reranker, "RERANK_API_KEY", "sk-test")
    # Pin kind — .env của máy dev có thể đặt RERANK_API_KIND=hf làm lệch test
    monkeypatch.setattr(reranker, "RERANK_API_KIND", "cohere")


def test_remote_available_flag(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_API_URL", "")
    monkeypatch.setattr(reranker, "RERANK_API_KEY", "")
    assert not remote_rerank_available()
    _enable_remote(monkeypatch)
    assert remote_rerank_available()


def test_remote_scores_quyet_dinh_thu_hang(monkeypatch):
    """Score API cao hơn phải thắng, kể cả khi RRF score gốc thấp hơn."""
    _enable_remote(monkeypatch)

    def fake_post(url, headers=None, json=None, timeout=None):
        # doc thứ 2 (index=1) relevance cao nhất; API trả đã sort theo score
        return _FakeResp({"results": [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.10},
        ]})

    monkeypatch.setattr("requests.post", fake_post)
    low_rrf_but_relevant = _rc("b", score=0.1)
    high_rrf_irrelevant = _rc("a", score=0.9)
    out = rerank("câu hỏi", [high_rrf_irrelevant, low_rrf_but_relevant], top_k=2)
    assert out[0].chunk.chunk_id == "b"
    assert out[0].score > out[1].score


def test_remote_ket_hop_railway_demote(monkeypatch):
    """Hệ số railway demote vẫn áp lên score API (chuỗi factor giữ nguyên)."""
    _enable_remote(monkeypatch)
    rail_text = ("[03/VBHN — Điều 47. Xử phạt quy tắc giao thông tại đường ngang, "
                 "cầu chung]\n5. Phạt tiền...")

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp({"results": [
            {"index": 0, "relevance_score": 0.70},   # chunk đường ngang
            {"index": 1, "relevance_score": 0.65},   # chunk đường bộ
        ]})

    monkeypatch.setattr("requests.post", fake_post)
    rail = _rc("rail", text=rail_text)
    road = _rc("road", text="Điều 6. Xử phạt xe ô tô vi phạm quy tắc giao thông đường bộ")
    out = rerank("ô tô vượt đèn đỏ phạt bao nhiêu", [rail, road], top_k=2)
    # 0.70 × RAILWAY_MISMATCH_FACTOR(0.8) = 0.56 < 0.65 → road thắng
    assert out[0].chunk.chunk_id == "road"


def test_remote_loi_fallback_khong_vo(monkeypatch):
    """API lỗi → không crash, rơi về rule-based (CE local tắt trong test env)."""
    _enable_remote(monkeypatch)

    def fake_post(url, headers=None, json=None, timeout=None):
        raise ConnectionError("mạng rớt")

    monkeypatch.setattr("requests.post", fake_post)
    # Chặn CE local load model thật trong test
    monkeypatch.setattr(reranker, "_ce_available", False)
    monkeypatch.setattr(reranker, "_cross_encoder", None)

    a, b = _rc("a", score=0.9), _rc("b", score=0.1)
    out = rerank("câu hỏi", [a, b], top_k=2)
    assert out[0].chunk.chunk_id == "a"  # rule-based giữ thứ tự theo score gốc


def test_hf_format_parse_score(monkeypatch):
    """kind=hf: parse [{label,score}] per pair, score đã sigmoid dùng thẳng."""
    _enable_remote(monkeypatch)
    monkeypatch.setattr(reranker, "RERANK_API_KIND", "hf")

    def fake_post(url, headers=None, json=None, timeout=None):
        assert "inputs" in json and json["inputs"][0]["text_pair"]
        # Shape thực tế HF: 1 list ngoài, list trong = N dict theo thứ tự cặp
        return _FakeResp([[
            {"label": "LABEL_0", "score": 0.15},
            {"label": "LABEL_0", "score": 0.92},
        ]])

    monkeypatch.setattr("requests.post", fake_post)
    a, b = _rc("a", score=0.9), _rc("b", score=0.1)
    out = rerank("câu hỏi", [a, b], top_k=2)
    assert out[0].chunk.chunk_id == "b"  # score HF 0.92 thắng 0.15


def test_hf_format_sai_shape_fallback(monkeypatch):
    """kind=hf: response sai shape → fallback rule-based, không crash."""
    _enable_remote(monkeypatch)
    monkeypatch.setattr(reranker, "RERANK_API_KIND", "hf")

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp({"error": "loading"})

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(reranker, "_ce_available", False)
    monkeypatch.setattr(reranker, "_cross_encoder", None)
    a, b = _rc("a", score=0.9), _rc("b", score=0.1)
    out = rerank("câu hỏi", [a, b], top_k=2)
    assert out[0].chunk.chunk_id == "a"


def test_remote_thieu_ket_qua_fallback(monkeypatch):
    """API trả thiếu kết quả (không đủ số doc) → coi như lỗi, fallback."""
    _enable_remote(monkeypatch)

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp({"results": [{"index": 0, "relevance_score": 0.9}]})

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(reranker, "_ce_available", False)
    monkeypatch.setattr(reranker, "_cross_encoder", None)

    a, b = _rc("a", score=0.9), _rc("b", score=0.1)
    out = rerank("câu hỏi", [a, b], top_k=2)
    assert len(out) == 2 and out[0].chunk.chunk_id == "a"

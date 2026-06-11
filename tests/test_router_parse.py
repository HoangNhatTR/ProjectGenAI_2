"""Unit tests cho SmartRouter: parse decision + fallback khi LLM lỗi/JSON xấu."""
from __future__ import annotations

from src.router import RouterDecision, SmartRouter


class _FakeLLM:
    """Client giả trả về content cố định (hoặc raise)."""

    def __init__(self, content: str = "", raise_exc: bool = False):
        self.content = content
        self.raise_exc = raise_exc

    def chat(self, model, messages, format="", options=None):
        if self.raise_exc:
            raise RuntimeError("LLM down")
        return {"message": {"content": self.content}}


def _router(client: _FakeLLM) -> SmartRouter:
    r = SmartRouter(model="fake-model")
    r._client = client  # inject — bỏ qua _connect
    return r


# ── _parse_decision (không cần LLM) ───────────────────────────────────────────

def test_parse_retrieve_hop_le():
    r = _router(_FakeLLM())
    d = r._parse_decision(
        {"action": "retrieve", "intent": "legal", "search_query": "mũ bảo hiểm"},
        question="đội mũ bảo hiểm",
    )
    assert d.action == "retrieve"
    assert d.search_query == "mũ bảo hiểm"


def test_parse_action_la_nguoc_ve_retrieve():
    d = _router(_FakeLLM())._parse_decision(
        {"action": "do_something_weird", "intent": "legal"}, question="q",
    )
    assert d.action == "retrieve"
    assert d.search_query == "q"  # fallback về câu hỏi gốc


def test_parse_tool_khong_hop_le_ve_retrieve():
    d = _router(_FakeLLM())._parse_decision(
        {"action": "use_tool", "intent": "legal", "tool_name": "hack_the_db"},
        question="q",
    )
    assert d.action == "retrieve"


def test_parse_intent_compare_tu_dong_gan_tool():
    d = _router(_FakeLLM())._parse_decision(
        {"action": "use_tool", "intent": "compare", "tool_name": "web_search",
         "tool_query": "luật A|luật B"},
        question="so sánh A và B",
    )
    assert d.tool_name == "compare_regulations"


def test_parse_answer_direct_thieu_response_ve_retrieve():
    d = _router(_FakeLLM())._parse_decision(
        {"action": "answer_direct", "intent": "chitchat"}, question="xin chào",
    )
    assert d.action == "retrieve"


def test_parse_answer_direct_hop_le():
    d = _router(_FakeLLM())._parse_decision(
        {"action": "answer_direct", "intent": "chitchat",
         "direct_response": "Chào bạn!"},
        question="xin chào",
    )
    assert d.action == "answer_direct"
    assert d.direct_response == "Chào bạn!"


# ── route() end-to-end với fake client ─────────────────────────────────────────

def test_route_llm_loi_fallback_retrieve():
    r = _router(_FakeLLM(raise_exc=True))
    d = r.route("xe máy vượt đèn đỏ phạt bao nhiêu?")
    assert isinstance(d, RouterDecision)
    assert d.action == "retrieve"
    assert d.search_query == "xe máy vượt đèn đỏ phạt bao nhiêu?"


def test_route_json_hong_fallback_retrieve():
    r = _router(_FakeLLM(content="đây không phải JSON"))
    d = r.route("câu hỏi pháp lý")
    assert d.action == "retrieve"


def test_route_json_hop_le():
    r = _router(_FakeLLM(content='{"action": "retrieve", "intent": "legal", "search_query": "tốc độ tối đa"}'))
    d = r.route("tốc độ tối đa trong khu dân cư?")
    assert d.action == "retrieve"
    assert d.search_query == "tốc độ tối đa"

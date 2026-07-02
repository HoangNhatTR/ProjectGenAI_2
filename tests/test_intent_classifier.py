"""Unit tests cho IntentClassifier (tất định) + tích hợp guard trong SmartRouter.

Phần lớn chạy rule-only (embedder=None). Embedding gate test bằng FakeEmbedder.
"""
from __future__ import annotations

from src.intent_classifier import IntentClassifier, has_validation_target
from src.router import RouterDecision, SmartRouter


# ── Rule layer (embedder=None) ────────────────────────────────────────────────

def test_chitchat_ngan():
    c = IntentClassifier(embedder=None)
    r = c.classify("Xin chào bạn")
    assert r.intent == "chitchat" and r.action == "answer_direct" and r.confident


def test_chitchat_khong_nuot_cau_phap_ly_dai():
    c = IntentClassifier(embedder=None)
    # Bắt đầu bằng "chào" nhưng là câu pháp lý dài → KHÔNG phải chitchat
    r = c.classify("Chào bạn, cho hỏi mức phạt khi vượt đèn đỏ với xe máy là bao nhiêu tiền")
    assert r.intent != "chitchat"


def test_validate_chi_khi_co_van_ban():
    c = IntentClassifier(embedder=None)
    # Có trỏ rõ "bản án này" → validate
    r1 = c.classify("Phân tích bản án này giúp tôi")
    assert r1.intent == "validate" and r1.tool_name == "validate_document"
    # Có document đính kèm qua history flag
    r2 = c.classify("phân tích giúp tôi", has_document=True)
    assert r2.intent == "validate"


def test_validate_khong_misfire_khi_khong_co_van_ban():
    c = IntentClassifier(embedder=None)
    # Câu dày trích dẫn formal nhưng KHÔNG có văn bản → KHÔNG được ra validate
    q = ("Căn cứ Điều 250 Bộ luật Hình sự 2015, áp dụng điểm c khoản 1, "
         "phân tích cấu thành tội phạm và khung hình phạt")
    r = c.classify(q, has_document=False)
    assert r.intent != "validate"


def test_draft():
    c = IntentClassifier(embedder=None)
    r = c.classify("Soạn giúp tôi đơn ly hôn")
    assert r.intent == "draft" and r.needs_llm_query  # cần LLM dựng tool_query


def test_compare():
    c = IntentClassifier(embedder=None)
    r = c.classify("So sánh mức phạt vượt đèn đỏ giữa xe máy và ô tô")
    assert r.intent == "compare" and r.needs_llm_query


def test_calculate_can_2_hanh_vi():
    c = IntentClassifier(embedder=None)
    r = c.classify("Tôi vừa vượt đèn đỏ vừa không đội mũ bảo hiểm thì tổng cộng phạt bao nhiêu")
    assert r.intent == "calculate" and r.tool_name == "calculate_fine"


def test_mot_hanh_vi_khong_phai_calculate():
    c = IntentClassifier(embedder=None)
    r = c.classify("Vượt đèn đỏ phạt bao nhiêu tiền")
    assert r.intent != "calculate"


def test_web_search():
    c = IntentClassifier(embedder=None)
    r = c.classify("Tìm nghị định mới nhất về xử phạt giao thông trên internet")
    assert r.intent == "web_search"


def test_web_search_tat_thi_khong_chon():
    c = IntentClassifier(embedder=None)
    r = c.classify("nghị định mới nhất về giao thông", web_search_enabled=False)
    assert r.intent != "web_search"


def test_has_validation_target_helper():
    assert has_validation_target("phân tích bản án này", None) is True
    assert has_validation_target("vượt đèn đỏ phạt bao nhiêu", None) is False
    assert has_validation_target("phân tích giúp", [{"role": "system", "content": "file: a.pdf"}]) is True


# ── Embedding gate (FakeEmbedder) ─────────────────────────────────────────────

class _FakeEmbedder:
    """Trả vector cố định: query trùng prototype → cosine=1; khác → 0."""
    def __init__(self, query_vec):
        self.query_vec = query_vec
    def encode(self, texts):
        out = []
        for _ in texts:
            out.append(list(self.query_vec))
        return out


def test_embedding_gate_legal_khi_giong_prototype():
    # prototypes và query cùng vector [1,0] → cosine 1.0 ≥ ngưỡng → legal/retrieve
    c = IntentClassifier(embedder=_FakeEmbedder([1.0, 0.0]))
    r = c.classify("một câu hỏi pháp lý chung chung không khớp rule")
    assert r.action == "retrieve" and r.intent in ("legal", "consulting") and r.confident


def test_embedding_gate_consulting_khi_ngoi_thu_nhat():
    c = IntentClassifier(embedder=_FakeEmbedder([1.0, 0.0]))
    r = c.classify("Tôi bị công ty cho nghỉ việc không báo trước thì phải làm sao đây")
    assert r.intent == "consulting"


# ── Tích hợp SmartRouter ──────────────────────────────────────────────────────

class _ExplodingLLM:
    """Client raise nếu bị gọi — chứng minh classifier đã bỏ qua LLM."""
    def chat(self, *a, **k):
        raise AssertionError("LLM KHÔNG được gọi khi classifier đã chắc chắn")


def test_router_classifier_bo_qua_llm():
    clf = IntentClassifier(embedder=None)
    r = SmartRouter(model="fake", classifier=clf)
    r._client = _ExplodingLLM()
    # calculate rõ ràng → deterministic, không gọi LLM
    d = r.route("Tôi vừa vượt đèn đỏ vừa không đội mũ thì tổng cộng phạt bao nhiêu")
    assert d.action == "use_tool" and d.tool_name == "calculate_fine"
    assert d.raw == "classifier"


class _ValidateLLM:
    """Client luôn trả intent=validate (mô phỏng misfire)."""
    def chat(self, *a, **k):
        return {"message": {"content":
            '{"action":"use_tool","intent":"validate","tool_name":"validate_document",'
            '"tool_query":"x"}'}}


def test_router_guard_chong_validate_misfire():
    # classifier=None → đi đường LLM; LLM ra validate nhưng KHÔNG có văn bản
    r = SmartRouter(model="fake")
    r._client = _ValidateLLM()
    d = r.route("Căn cứ Điều 250 BLHS phân tích cấu thành tội phạm và khung hình phạt")
    assert d.action == "retrieve", "guard phải ép validate→retrieve khi không có văn bản"


def test_router_guard_giu_validate_khi_co_van_ban():
    r = SmartRouter(model="fake")
    r._client = _ValidateLLM()
    d = r.route("phân tích văn bản này", history=[{"role": "system", "content": "file: ban_an.pdf"}])
    assert d.action == "use_tool" and d.tool_name == "validate_document"

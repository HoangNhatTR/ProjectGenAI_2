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


# ── Follow-up gate (cần has_history) ──────────────────────────────────────────

def test_followup_anaphora_khi_co_history():
    c = IntentClassifier(embedder=None)
    # Đúng ca bug thật: "thế nếu ô tô vượt thì sao ?" từng bị retrieve nguyên văn
    r = c.classify("thế nếu ô tô vượt thì sao ?", has_history=True)
    assert r.intent == "followup" and r.needs_llm_query


def test_followup_cau_ngan_khi_co_history():
    c = IntentClassifier(embedder=None)
    # Câu ngắn elliptical (26 ký tự) — không tự chứa chủ đề
    r = c.classify("tôi vượt ở nút giao thông", has_history=True)
    assert r.intent == "followup" and r.needs_llm_query


def test_khong_followup_khi_khong_co_history():
    c = IntentClassifier(embedder=None)
    # Lượt ĐẦU hội thoại: câu y hệt không được ra followup (không có gì để rewrite)
    r = c.classify("thế nếu ô tô vượt thì sao ?", has_history=False)
    assert r.intent != "followup"


def test_cau_dai_tu_dung_khong_followup_du_co_history():
    c = IntentClassifier(embedder=None)
    q = "Người điều khiển xe ô tô vượt đèn đỏ tại nút giao thông thì bị phạt bao nhiêu tiền"
    r = c.classify(q, has_history=True)
    assert r.intent != "followup"


def test_followup_khong_nuot_rule_nguy_hiem():
    c = IntentClassifier(embedder=None)
    # Câu ngắn + history nhưng khớp rule validate/compare → rule thắng
    r1 = c.classify("phân tích bản án này", has_history=True)
    assert r1.intent == "validate"
    r2 = c.classify("so sánh với ô tô thì sao", has_history=True)
    assert r2.intent == "compare"


def test_router_followup_rewrite_chuyen_dung():
    """Follow-up → LLM chuyên dụng viết lại (text thuần) → retrieve query độc lập."""
    clf = IntentClassifier(embedder=None)
    r = SmartRouter(model="fake", classifier=clf)

    class _RewriteLLM:
        # _rewrite_followup gọi chat KHÔNG format=json → trả text thuần
        def chat(self, *a, **k):
            return {"message": {"content": "ô tô vượt đèn đỏ bị phạt bao nhiêu tiền"}}

    r._client = _RewriteLLM()
    d = r.route(
        "thế nếu ô tô vượt thì sao ?",
        history=[
            {"role": "user", "content": "Vượt đèn đỏ với xe máy bị phạt bao nhiêu tiền?"},
            {"role": "assistant", "content": "Mức phạt 4-6 triệu đồng theo NĐ 168/2024."},
        ],
    )
    assert d.action == "retrieve" and d.intent == "followup"
    assert d.search_query == "ô tô vượt đèn đỏ bị phạt bao nhiêu tiền"
    assert d.raw == "followup_rewrite"


def test_router_followup_rewrite_loi_dung_cau_goc():
    """LLM rewrite lỗi → dùng câu gốc (KHÔNG đổi sang tool, KHÔNG crash)."""
    clf = IntentClassifier(embedder=None)
    r = SmartRouter(model="fake", classifier=clf)

    class _BrokenLLM:
        def chat(self, *a, **k):
            raise RuntimeError("LLM down")

    r._client = _BrokenLLM()
    d = r.route("thế đối với ô tô thì sao ?",
                history=[{"role": "user", "content": "xe máy vượt đèn đỏ phạt bao nhiêu?"},
                         {"role": "assistant", "content": "4-6 triệu."}])
    assert d.action == "retrieve" and d.intent == "followup"
    assert d.search_query == "thế đối với ô tô thì sao ?"  # fallback câu gốc


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


def test_router_followup_khong_bao_gio_ra_tool():
    """Dù LLM nền có xu hướng compare, follow-up ĐI ĐƯỜNG rewrite chuyên dụng
    (text thuần) nên KHÔNG bao giờ ra use_tool/compare."""
    clf = IntentClassifier(embedder=None)
    r = SmartRouter(model="fake", classifier=clf)

    class _TextLLM:
        def chat(self, *a, **k):
            # _rewrite_followup gọi không format=json → trả text độc lập
            return {"message": {"content": "ô tô vượt đèn đỏ bị phạt bao nhiêu tiền"}}

    r._client = _TextLLM()
    d = r.route("thế đối với ô tô thì sao ?",
                history=[{"role": "user", "content": "xe máy vượt đèn đỏ thì bị phạt như thế nào?"},
                         {"role": "assistant", "content": "Phạt 4-6 triệu theo NĐ 168/2024."}])
    assert d.action == "retrieve" and d.intent == "followup"
    assert d.tool_name is None
    assert "ô tô" in (d.search_query or "").lower()

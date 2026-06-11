"""Unit tests cho api.py + src/pipeline.py: auth, per-request generator, guardrails."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

import api
from src import config
from src.generator import Generator
from src.guardrails import apply_guardrails
from src.pipeline import make_generator, resolve_model_provider
from src.schemas import Answer


# ── _require_auth ──────────────────────────────────────────────────────────────

def test_auth_tat_khi_khong_set_key(monkeypatch):
    monkeypatch.setattr(config, "API_AUTH_KEY", "")
    api._require_auth(None)  # không raise


def test_auth_dung_token(monkeypatch):
    monkeypatch.setattr(config, "API_AUTH_KEY", "secret123")
    api._require_auth("Bearer secret123")  # không raise


def test_auth_sai_token_hoac_thieu(monkeypatch):
    monkeypatch.setattr(config, "API_AUTH_KEY", "secret123")
    with pytest.raises(HTTPException) as ei:
        api._require_auth("Bearer wrong")
    assert ei.value.status_code == 401
    with pytest.raises(HTTPException):
        api._require_auth(None)


# ── pipeline.make_generator ────────────────────────────────────────────────────

def _base_generator() -> Generator:
    gen = Generator(
        model="base-model",
        host=config.ROUTER9_BASE_URL,
        temperature=0.2,
        top_p=1.0,
        provider="router9",
        api_key=config.ROUTER9_API_KEY,
    )
    gen._client = object()  # giả lập client đã connect
    return gen


def test_make_generator_khong_mutate_base():
    base = _base_generator()

    gen = make_generator(
        base,
        provider="router9",
        model="cc/claude-sonnet-4-6",
        temperature=0.9,
    )

    assert gen is not base
    assert gen.model == "cc/claude-sonnet-4-6"
    assert gen.temperature == 0.9
    # Base giữ nguyên — không bị request ghi đè
    assert base.model == "base-model"
    assert base.temperature == 0.2
    # Cùng provider/host/key → tái dùng client đã connect
    assert gen._client is base._client


def test_make_generator_doi_provider_khong_reuse_client():
    base = _base_generator()

    gen = make_generator(base, provider="kieai", model="deepseek-chat")

    assert gen.provider == "kieai"
    assert gen.model == "deepseek-chat"
    # Provider khác → không tái dùng client cũ
    assert gen._client is not base._client
    assert base.provider == "router9"


def test_make_generator_mac_dinh_giu_config_base():
    base = _base_generator()

    gen = make_generator(base)

    assert gen.model == base.model
    assert gen.temperature == base.temperature
    assert gen._client is base._client


def test_resolve_model_provider_theo_prefix():
    assert resolve_model_provider("cc/claude-sonnet-4-6") == "router9"
    assert resolve_model_provider("gh/gpt-4o-mini") == "router9"
    assert resolve_model_provider("deepseek-chat") == "kieai"


# ── apply_guardrails: warn_no_evidence ─────────────────────────────────────────

def test_guardrails_tool_khong_canh_bao_thieu_can_cu():
    ans = Answer(question="So sánh quy định A và B", answer="Kết quả so sánh chi tiết.", citations=[])
    out = apply_guardrails(ans, [], warn_no_evidence=False)
    assert "Cơ sở dữ liệu chưa có văn bản" not in out.answer
    # Vẫn có disclaimer chung
    assert "không thay thế" in out.answer


def test_guardrails_mac_dinh_canh_bao_khi_khong_context():
    ans = Answer(question="Câu hỏi bất kỳ", answer="Trả lời không nêu nguồn.", citations=[])
    out = apply_guardrails(ans, [])
    assert "Cơ sở dữ liệu chưa có văn bản" in out.answer


# ── _format_answer / _format_citations ────────────────────────────────────────

def test_format_citations_rong():
    assert api._format_citations([]) == ""


def test_format_answer_khong_citation_giu_nguyen_text():
    ans = Answer(question="q", answer="Nội dung trả lời.", citations=[])
    assert api._format_answer(ans) == "Nội dung trả lời."

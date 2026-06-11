"""Integration tests cho API endpoints (TestClient, không cần load agent/LLM)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api
from src import config


@pytest.fixture()
def client(monkeypatch):
    # Đảm bảo agent rỗng (không chạy lifespan → không load model)
    monkeypatch.setattr(api, "_agent", {})
    # TestClient không dùng context manager → lifespan KHÔNG chạy
    return TestClient(api.app)


def test_health_khong_can_auth(client, monkeypatch):
    monkeypatch.setattr(config, "API_AUTH_KEY", "secret")
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_yeu_cau_auth_khi_bat(client, monkeypatch):
    monkeypatch.setattr(config, "API_AUTH_KEY", "secret")
    assert client.get("/v1/models").status_code == 401
    r = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "legal-ai-graph" in ids


def test_models_khong_can_auth_khi_tat(client, monkeypatch):
    monkeypatch.setattr(config, "API_AUTH_KEY", "")
    assert client.get("/v1/models").status_code == 200


def test_get_model_404(client, monkeypatch):
    monkeypatch.setattr(config, "API_AUTH_KEY", "")
    assert client.get("/v1/models/khong-ton-tai").status_code == 404


def test_chat_503_khi_agent_chua_san_sang(client, monkeypatch):
    monkeypatch.setattr(config, "API_AUTH_KEY", "")
    r = client.post(
        "/v1/chat/completions",
        json={"model": "legal-ai", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503


def test_chat_401_truoc_khi_check_agent(client, monkeypatch):
    """Auth phải chặn trước — không lộ trạng thái server cho request không key."""
    monkeypatch.setattr(config, "API_AUTH_KEY", "secret")
    r = client.post(
        "/v1/chat/completions",
        json={"model": "legal-ai", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401


def test_export_chan_path_traversal(client, monkeypatch):
    monkeypatch.setattr(config, "API_AUTH_KEY", "")
    assert client.get("/v1/export/..%2F..%2F.env").status_code in (400, 404)
    assert client.get("/v1/export/khongphai.txt").status_code == 400

"""Smoke tests cho generator (phần không cần LLM).

Logic LLM-driven (router, extract_memory_facts, generate) yêu cầu mock Ollama
nên không test trực tiếp ở đây.
"""
from src.generator import _build_system_prompt, _extract_json


def test_system_prompt_without_memory_has_no_memory_block():
    prompt = _build_system_prompt("")
    assert "Thông tin đã ghi nhớ" not in prompt
    assert "TRỢ LÝ TƯ VẤN pháp luật Việt Nam" in prompt


def test_system_prompt_injects_memory_text():
    mem = "Thông tin đã ghi nhớ về người dùng:\n- Là tài xế xe máy"
    prompt = _build_system_prompt(mem)
    assert "Là tài xế xe máy" in prompt


def test_extract_json_plain():
    assert _extract_json('{"facts": ["a", "b"]}') == {"facts": ["a", "b"]}


def test_extract_json_with_code_fence():
    text = '```json\n{"facts": ["x"]}\n```'
    assert _extract_json(text) == {"facts": ["x"]}


def test_extract_json_with_prefix():
    text = 'Đây là kết quả: {"facts": []} thank you'
    assert _extract_json(text) == {"facts": []}


def test_extract_json_invalid_returns_none():
    assert _extract_json("không phải json") is None

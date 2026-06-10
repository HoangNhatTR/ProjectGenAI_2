"""Shared utilities dùng chung cho các module của Legal AI Agent."""
from __future__ import annotations

import json
import re
from typing import Optional


def extract_json(text: str) -> Optional[dict]:
    """Parse JSON từ LLM response — xử lý markdown fences và partial JSON.

    Dùng bởi: router.py, generator.py, planner.py
    """
    text = text.strip()
    # Strip markdown code fences nếu có
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: tìm object JSON đầu tiên trong text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

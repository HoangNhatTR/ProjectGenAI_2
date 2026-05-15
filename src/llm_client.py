"""LLM client wrappers: GeminiClient và GroqClient — drop-in replacements cho ollama.Client.

Interface chat() giống hệt Ollama để các module Generator, Router, Tools,
Planner không cần thay đổi logic gọi LLM.
"""
from __future__ import annotations

from typing import Any, Optional


class GeminiClient:
    """Wrapper quanh google.genai, giả lập interface ollama.Client."""

    def __init__(self, api_key: str):
        from google import genai
        self._client = genai.Client(api_key=api_key)

    def chat(
        self,
        model: str,
        messages: list[dict],
        format: str = "",
        options: Optional[dict] = None,
    ) -> dict:
        from google.genai import types

        options = options or {}
        temperature = float(options.get("temperature", 0.2))

        system_instruction: Optional[str] = None
        contents: list[Any] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=content)])
                )
            elif role == "assistant":
                contents.append(
                    types.Content(role="model", parts=[types.Part(text=content)])
                )

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if format == "json":
            config_kwargs["response_mime_type"] = "application/json"

        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return {"message": {"content": response.text}}


class GroqClient:
    """Wrapper quanh Groq API, giả lập interface ollama.Client."""

    def __init__(self, api_key: str):
        from groq import Groq
        self._client = Groq(api_key=api_key)

    def chat(
        self,
        model: str,
        messages: list[dict],
        format: str = "",
        options: Optional[dict] = None,
    ) -> dict:
        options = options or {}
        temperature = float(options.get("temperature", 0.2))

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        return {"message": {"content": text}}

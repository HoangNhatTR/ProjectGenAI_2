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

    def stream_chat(
        self,
        model: str,
        messages: list[dict],
        options: Optional[dict] = None,
    ):
        """Yield các chunk text khi Gemini trả về (streaming)."""
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
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=content)]))

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        for chunk in self._client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        ):
            if chunk.text:
                yield chunk.text


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


class OpenRouterClient:
    """Wrapper cho OpenRouter (openrouter.ai) — OpenAI-compatible API."""

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Cần cài 'openai' package: pip install openai>=1.30.0") from e
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/legal-ai-agent",
                "X-Title": "Legal AI Agent VN",
            },
            timeout=60.0,
        )

    def chat(
        self,
        model: str,
        messages: list[dict],
        format: str = "",
        options: Optional[dict] = None,
    ) -> dict:
        import time
        options = options or {}
        temperature = float(options.get("temperature", 0.2))

        # Tắt thinking mode cho Qwen3: inject /no_think vào system message
        if any(x in model.lower() for x in ("qwen3", "qwq")):
            msgs = list(messages)
            if msgs and msgs[0].get("role") == "system":
                msgs[0] = {**msgs[0], "content": "/no_think\n" + msgs[0]["content"]}
            else:
                msgs = [{"role": "system", "content": "/no_think"}] + msgs
            messages = msgs

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if format == "json":
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["max_tokens"] = int(options.get("max_tokens", 512))
        elif "max_tokens" in options:
            kwargs["max_tokens"] = int(options["max_tokens"])

        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""
                return {"message": {"content": text}}
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                # Nếu lỗi json_object không được hỗ trợ → thử lại không có response_format
                if "response_format" in kwargs and "json" in err_str:
                    kwargs.pop("response_format", None)
                    continue
                # Lỗi kết nối tạm thời → retry sau 2s
                if any(x in err_str for x in ("connection", "timeout", "502", "503", "529")):
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        raise last_exc

    def stream_chat(
        self,
        model: str,
        messages: list[dict],
        options: Optional[dict] = None,
    ):
        """Yield các chunk text khi model trả về."""
        options = options or {}
        temperature = float(options.get("temperature", 0.2))
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        for chunk in self._client.chat.completions.create(**kwargs):
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


class Router9Client:
    """Wrapper cho 9Router (OpenAI-compatible local proxy ở localhost:20128).

    Lợi điểm: 9Router tự fallback giữa 40+ providers, có free claude-sonnet-4.5
    qua Kiro AI, RTK token compression giảm 20-40% tokens.
    """

    def __init__(self, api_key: str, base_url: str = "http://localhost:20128/v1"):
        # Reuse openai SDK vì 9Router OpenAI-compatible
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Cần cài 'openai' package: pip install openai>=1.30.0"
            ) from e
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        model: str,
        messages: list[dict],
        format: str = "",
        options: Optional[dict] = None,
    ) -> dict:
        options = options or {}
        temperature = float(options.get("temperature", 0.2))
        top_p       = float(options.get("top_p", 1.0))

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,   # 9Router mặc định stream, phải tắt tường minh
        }
        if format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "response_format" in kwargs and "json_object" in str(exc).lower():
                kwargs.pop("response_format", None)
                response = self._client.chat.completions.create(**kwargs)
            else:
                raise

        text = response.choices[0].message.content or ""
        return {"message": {"content": text}}

    def stream_chat(
        self,
        model: str,
        messages: list[dict],
        options: Optional[dict] = None,
    ):
        """Yield các chunk text khi model trả về (Server-Sent Events)."""
        options = options or {}
        temperature = float(options.get("temperature", 0.2))
        top_p       = float(options.get("top_p", 1.0))
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        for chunk in self._client.chat.completions.create(**kwargs):
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

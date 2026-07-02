"""LLM client wrappers — drop-in replacements cho ollama.Client.

Interface chat() giống hệt Ollama để các module Generator, Router, Tools,
Planner không cần thay đổi logic gọi LLM.

Dùng create_client() làm factory duy nhất — KHÔNG tự if/elif provider ở nơi
khác (Generator và Router từng có 2 bản copy lệch nhau: router thiếu kieai
→ rơi vào nhánh ollama → crash trên môi trường không cài ollama).
"""
from __future__ import annotations

from typing import Any, Optional


def _is_transient(exc: Exception) -> bool:
    """Lỗi tạm thời (đáng retry): rate limit, quá tải, timeout, 5xx."""
    msg = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (408, 409, 429, 500, 502, 503, 504):
        return True
    keys = ("rate limit", "overload", "timeout", "timed out", "temporarily",
            "too many requests", "empty response", "choices=none", "503", "502",
            "500", "429", "connection", "unavailable")
    return any(k in msg for k in keys)


def create_client(
    provider: str,
    api_key: str = "",
    host: str = "http://localhost:11434",
) -> Any:
    """Factory chung: provider → LLM client cùng interface chat()."""
    if provider == "gemini":
        return GeminiClient(api_key=api_key or "")
    if provider == "groq":
        return GroqClient(api_key=api_key or "")
    if provider == "router9":
        return Router9Client(api_key=api_key or "", base_url=host)
    if provider == "openrouter":
        return OpenRouterClient(api_key=api_key or "", base_url=host)
    if provider == "kieai":
        return KieAIClient(api_key=api_key or "", base_url=host)
    # Mặc định: Ollama local
    from ollama import Client
    return Client(host=host)


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
        # timeout + max_retries: tránh treo VÔ HẠN khi máy ngủ / rớt mạng giữa call
        # (trước đây thiếu → process sống mà CPU=0, đứng im hàng giờ). APITimeoutError
        # → _call_with_retry coi là transient → backoff retry, giống Router9/KieAI client.
        self._client = Groq(api_key=api_key, timeout=90.0, max_retries=0)

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
            kwargs["max_tokens"] = int(options.get("max_tokens", 2048))
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
        # timeout=90s: tránh treo vô hạn khi mất kết nối (vd máy ngủ) — sẽ raise
        # APITimeoutError → _call_with_retry coi là transient → backoff retry.
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=90.0)

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


class KieAIClient:
    """Wrapper cho Kie AI — OpenAI-compatible API. Hỗ trợ 2 kiểu base URL:

    1. Gateway gộp chung (cũ): https://kieai.erweima.ai/api/v1 — 1 endpoint cho
       mọi model, nhưng thực tế CHỈ nhận deepseek-chat (model khác → "Operation
       not found"); endpoint này hay bảo trì.
    2. Endpoint CHÍNH THỨC theo TỪNG MODEL: https://api.kie.ai/{model}/v1 — mỗi
       model 1 base URL riêng (slug nằm trong path). Đã xác nhận chạy với key
       thật: gemini-3-pro, gemini-3-flash, gpt-5-2 (xác minh 2026-06-22).
       Slug = tên trên URL trang model (kie.ai/gemini-3-pro → "gemini-3-pro").

    Nếu base_url chứa placeholder "{model}" → tự dựng URL theo model + cache 1
    client/model. Ngược lại → 1 client duy nhất (tương thích gateway cũ).
    Xem danh mục: https://docs.kie.ai/market/chat
    """

    BASE_URL = "https://kieai.erweima.ai/api/v1"
    _MAX_RETRIES = 4  # tổng số lần thử khi KieAI trả rỗng/nghẽn (1 lần đầu + 3 retry)

    def __init__(self, api_key: str, base_url: str = BASE_URL):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Cần cài 'openai' package: pip install openai>=1.30.0") from e
        self._api_key  = api_key
        self._base_url = base_url
        self._OpenAI   = OpenAI
        # WAF api.kie.ai CHẶN User-Agent mặc định của OpenAI SDK ("OpenAI/Python…")
        # → 403 "Your request was blocked." (xác minh 2026-06-22). Ghi đè UA trung
        # tính để qua. Các header X-Stainless-* không bị chặn.
        self._headers = {"User-Agent": "kie-legalai/1.0"}
        # base_url có "{model}" → endpoint chính thức theo từng model (cache riêng).
        self._per_model: bool = "{model}" in base_url
        self._clients: dict[str, Any] = {}
        if not self._per_model:
            # timeout=90s: tránh treo vô hạn khi mất kết nối (giống Router9Client).
            self._clients[""] = OpenAI(
                api_key=api_key, base_url=base_url, timeout=90.0,
                default_headers=self._headers,
            )

    def _client_for(self, model: str):
        """Trả client OpenAI cho model — dựng base_url theo model nếu cần."""
        cache_key = model if self._per_model else ""
        client = self._clients.get(cache_key)
        if client is None:
            url = self._base_url.format(model=model) if self._per_model else self._base_url
            client = self._OpenAI(
                api_key=self._api_key, base_url=url, timeout=90.0,
                default_headers=self._headers,
            )
            self._clients[cache_key] = client
        return client

    def chat(
        self,
        model: str,
        messages: list[dict],
        format: str = "",
        options: Optional[dict] = None,
    ) -> dict:
        options     = options or {}
        temperature = float(options.get("temperature", 0.2))
        top_p       = float(options.get("top_p", 1.0))
        client      = self._client_for(model)
        kwargs: dict[str, Any] = {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "top_p":       top_p,
            "stream":      False,
        }
        if format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        # KieAI đôi khi trả HTTP 200 với choices=None khi bị rate limit / quá tải
        # (đặc biệt với prompt RAG dài). Retry + backoff để tự phục hồi thay vì
        # ném lỗi cho người dùng. Mỗi lần thử là 1 request mới (không stream).
        import time as _time
        import random as _random
        last_err: Optional[Exception] = None
        for attempt in range(self._MAX_RETRIES):
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                if "response_format" in kwargs and "json_object" in str(exc).lower():
                    # model không hỗ trợ json_object → bỏ rồi thử lại NGAY (không tính lượt)
                    kwargs.pop("response_format", None)
                    continue
                if not _is_transient(exc) or attempt == self._MAX_RETRIES - 1:
                    raise
                last_err = exc
            else:
                if response and response.choices:
                    return {"message": {"content": response.choices[0].message.content or ""}}
                last_err = RuntimeError(
                    "KieAI empty response (choices=None) — rate limit / overload"
                )
            # backoff: 1.5s, 3s, 6s (+ jitter) — đủ để KieAI hạ tải qua cơn nghẽn
            _time.sleep(min(1.5 * (2 ** attempt), 12.0) + _random.uniform(0, 0.5))
        raise last_err or RuntimeError("KieAI empty response (choices=None) — rate limit / overload")

    def stream_chat(
        self,
        model: str,
        messages: list[dict],
        options: Optional[dict] = None,
    ):
        options     = options or {}
        temperature = float(options.get("temperature", 0.2))
        top_p       = float(options.get("top_p", 1.0))
        client      = self._client_for(model)
        kwargs: dict[str, Any] = {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "top_p":       top_p,
            "stream":      True,
        }
        import time as _time
        import random as _random
        # Retry CHỈ khi chưa phát ký tự nào (an toàn — không lặp nội dung). Nếu
        # KieAI nghẽn trả stream rỗng / lỗi giữa lúc thiết lập → thử lại; đã phát
        # chữ rồi mà đứt thì raise (không thể tua lại stream).
        last_err: Optional[Exception] = None
        for attempt in range(self._MAX_RETRIES):
            emitted = False
            try:
                for chunk in client.chat.completions.create(**kwargs):
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        emitted = True
                        yield delta
            except Exception as exc:
                if emitted or not _is_transient(exc) or attempt == self._MAX_RETRIES - 1:
                    raise
                last_err = exc
            else:
                if emitted:
                    return
                last_err = RuntimeError(
                    "KieAI empty stream (choices=None) — rate limit / overload"
                )
            _time.sleep(min(1.5 * (2 ** attempt), 12.0) + _random.uniform(0, 0.5))
        if last_err:
            raise last_err

from __future__ import annotations

import logging
import re
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from src.llm.base import LLMClient, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class OpenAICompatibleClient(LLMClient):
    """Minimal OpenAI-compatible chat client with optional bearer auth."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 1800,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        max_json_tokens: int | None = None,
        json_retry_cap: int | None = None,
        supports_response_format: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("OpenAI-compatible base_url is required")
        if not model:
            raise ValueError("OpenAI-compatible model is required")
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or ""
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self.max_json_tokens = max_json_tokens
        self.json_retry_cap = json_retry_cap or max_json_tokens
        self.supports_response_format = supports_response_format
        self._extra_body = dict(extra_body or {})
        self._session = requests.Session()
        self._session.trust_env = False
        adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        extra_body = dict(self._extra_body)
        call_extra_body = kwargs.pop("extra_body", None)
        if isinstance(call_extra_body, dict):
            extra_body.update(call_extra_body)
        if not self.supports_response_format:
            kwargs.pop("response_format", None)
        if kwargs:
            payload.update(kwargs)
        if extra_body:
            payload.update(extra_body)

        delay = self._retry_backoff
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._session.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                break
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                is_transient = isinstance(exc, (requests.ConnectionError, requests.Timeout)) or status in {
                    408,
                    409,
                    403,
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                if not is_transient or attempt == self._max_retries:
                    raise
                logger.warning(
                    "OpenAI-compatible transient error (attempt %d/%d, status=%s): %s; retrying in %.1fs",
                    attempt + 1,
                    self._max_retries + 1,
                    status,
                    str(exc)[:120],
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        delta = choice.get("delta") or {}
        content = message.get("content") or delta.get("content") or choice.get("text") or ""
        if "<think>" in content or "</think>" in content:
            content = re.sub(r"<think>[\s\S]*?</think>", "", content)
            content = re.sub(r"</?think>", "", content).strip()

        usage_payload = data.get("usage")
        usage = (
            {
                "prompt_tokens": usage_payload.get("prompt_tokens", 0),
                "completion_tokens": usage_payload.get("completion_tokens", 0),
                "total_tokens": usage_payload.get("total_tokens", 0),
            }
            if isinstance(usage_payload, dict)
            else {}
        )
        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            usage=usage,
            raw=data,
        )

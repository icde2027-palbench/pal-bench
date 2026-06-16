from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str | list[dict[str, Any]]  # str for text; list for multimodal (OpenAI format)


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class LLMClient(ABC):
    """抽象 LLM 客户端，所有模型实现继承此类。"""

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """发送对话请求，返回 LLMResponse。"""

    def simple(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_empty_retries: int = 3,
        **kwargs: Any,
    ) -> str:
        """便捷方法：单轮对话，直接返回文本内容。

        空响应（iChat 偶发）会自动重试最多 ``max_empty_retries`` 次，
        每次等待 5 秒后重发。
        """
        messages = [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=prompt),
        ]
        for attempt in range(max_empty_retries + 1):
            content = self.chat(
                messages, temperature=temperature, max_tokens=max_tokens, **kwargs
            ).content
            if content and content.strip():
                return content
            if attempt < max_empty_retries:
                logger.warning(
                    "LLM returned empty response (attempt %d/%d), retrying in 5s…",
                    attempt + 1, max_empty_retries + 1,
                )
                time.sleep(5)
        # 最终仍为空，返回空字符串让上层的 fallback 逻辑处理
        logger.error("LLM returned empty response after %d retries", max_empty_retries + 1)
        return ""

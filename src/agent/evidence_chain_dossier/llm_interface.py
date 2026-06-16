"""Budget-tracked LLM interface used by PAL-TRACE."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import defaultdict
from typing import Any

from src.llm import create_llm_for_role
from src.llm.base import LLMMessage

logger = logging.getLogger(__name__)

_TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


class BudgetTracker:
    """Thread-safe LLM call budget tracker."""

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self._used = 0
        self._lock = threading.Lock()
        self.parse_failures = 0

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self._used)

    def can_call(self, n: int = 1) -> bool:
        return self._used + n <= self.max_calls

    def record_call(self, n: int = 1) -> None:
        with self._lock:
            self._used += n

    def record_parse_failure(self) -> None:
        with self._lock:
            self.parse_failures += 1


class DossierLLM:
    """PAL-TRACE LLM wrapper with call budgeting and JSON repair."""

    def __init__(self, llm: Any | None = None, max_calls: int = 160):
        self.llm = llm or create_llm_for_role("agent_llm")
        self.budget = BudgetTracker(max_calls)
        self._usage_totals = _empty_usage_bucket()
        self._usage_by_stage: defaultdict[str, dict[str, int]] = defaultdict(_empty_usage_bucket)
        self._call_records: list[dict[str, object]] = []

    def call_json(
        self,
        prompt: str,
        system: str,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        retries: int = 1,
        stage: str = "unknown",
    ) -> dict | list | None:
        current_max_tokens = max_tokens
        retry_cap = int(getattr(self.llm, "json_retry_cap", 32768) or 32768)
        configured_json_tokens = getattr(self.llm, "max_json_tokens", None)
        if configured_json_tokens is not None:
            current_max_tokens = max(current_max_tokens, int(configured_json_tokens))

        for attempt in range(1 + retries + 2):
            if not self.budget.can_call():
                logger.warning(
                    "Budget exhausted (%d/%d used), skipping call",
                    self.budget.used,
                    self.budget.max_calls,
                )
                return None

            self.budget.record_call()
            try:
                actual_prompt = prompt
                if attempt > 0:
                    actual_prompt = (
                        prompt
                        + "\n\n[IMPORTANT: Previous response was invalid. "
                        "Output ONLY one valid JSON object. No markdown.]"
                    )
                response_obj = self._chat(
                    actual_prompt,
                    system,
                    temperature=temperature,
                    max_tokens=current_max_tokens,
                    response_format={"type": "json_object"},
                )
                response = response_obj.content or ""
                finish_reason = _finish_reason(response_obj.raw)
                self._record_usage(stage, response_obj.usage, finish_reason=finish_reason)
            except Exception as exc:
                logger.error("LLM JSON call failed (attempt %d): %s", attempt + 1, exc)
                self.budget.record_parse_failure()
                continue

            if finish_reason == "length":
                current_max_tokens = min(current_max_tokens * 2, retry_cap)
                self.budget.record_parse_failure()
                continue
            if not response.strip():
                self.budget.record_parse_failure()
                continue

            parsed = _parse_json_response(response)
            if parsed is not None:
                return parsed
            self.budget.record_parse_failure()

        return None

    def call_text(
        self,
        prompt: str,
        system: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        stage: str = "unknown",
    ) -> str:
        if not self.budget.can_call():
            logger.warning(
                "Budget exhausted (%d/%d used), skipping call",
                self.budget.used,
                self.budget.max_calls,
            )
            return ""

        self.budget.record_call()
        try:
            for attempt in range(4):
                response_obj = self._chat(
                    prompt,
                    system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._record_usage(stage, response_obj.usage)
                content = response_obj.content or ""
                if content.strip():
                    return content
                if attempt < 3:
                    time.sleep(5)
        except Exception as exc:
            logger.error("LLM text call failed: %s", exc)
        return ""

    def usage_summary(self) -> dict[str, object]:
        return {
            **_bucket_with_coverage(self._usage_totals),
            "usage_by_stage": {
                stage: _bucket_with_coverage(bucket)
                for stage, bucket in sorted(self._usage_by_stage.items())
            },
            "call_records": list(self._call_records),
        }

    def _chat(self, prompt: str, system: str, **kwargs: Any):
        messages = [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=prompt),
        ]
        try:
            if (
                "response_format" in kwargs
                and getattr(self.llm, "supports_response_format", True) is False
            ):
                raise TypeError("response_format disabled for this LLM client")
            return self.llm.chat(messages, **kwargs)
        except TypeError:
            kwargs.pop("response_format", None)
            return self.llm.chat(messages, **kwargs)

    def _record_usage(
        self,
        stage: str,
        usage: dict[str, int] | None,
        *,
        finish_reason: str = "",
    ) -> None:
        stage = stage or "unknown"
        bucket = self._usage_by_stage[stage]
        record: dict[str, object] = {"stage": stage, "finish_reason": finish_reason}
        has_usage = False
        for key in _TOKEN_KEYS:
            value = usage.get(key) if isinstance(usage, dict) else None
            if isinstance(value, (int, float)):
                token_value = int(value)
                self._usage_totals[key] += token_value
                bucket[key] += token_value
                record[key] = token_value
                has_usage = True
            else:
                record[key] = None
        if has_usage:
            self._usage_totals["calls_with_usage"] += 1
            bucket["calls_with_usage"] += 1
        else:
            self._usage_totals["calls_without_usage"] += 1
            bucket["calls_without_usage"] += 1
        self._call_records.append(record)


def _empty_usage_bucket() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls_with_usage": 0,
        "calls_without_usage": 0,
    }


def _extract_json(text: str) -> str | None:
    if not text or not text.strip():
        return None
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if fence_match:
        return fence_match.group(1).strip()
    if text.startswith(("{", "[")):
        return text
    for i, ch in enumerate(text):
        if ch not in ("{", "["):
            continue
        depth = 0
        close_ch = "}" if ch == "{" else "]"
        for j in range(i, len(text)):
            if text[j] == ch:
                depth += 1
            elif text[j] == close_ch:
                depth -= 1
                if depth == 0:
                    return text[i : j + 1]
        return text[i:]
    return None


def _parse_json_response(text: str) -> dict | list | None:
    json_str = _extract_json(text) or text.strip()
    for candidate in (json_str, _repair_common_json_defects(json_str)):
        try:
            value = json.loads(candidate)
            if isinstance(value, (dict, list)):
                return value
        except json.JSONDecodeError:
            pass
        first = _load_first_json_value(candidate)
        if first is not None:
            return first
    return None


def _repair_common_json_defects(text: str) -> str:
    return re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)\s*:', r'"\1":', text)


def _load_first_json_value(text: str) -> dict | list | None:
    try:
        value, _ = json.JSONDecoder().raw_decode(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, (dict, list)) else None


def _finish_reason(raw: Any) -> str:
    if isinstance(raw, dict):
        choices = raw.get("choices") or []
        if choices and isinstance(choices[0], dict):
            return str(choices[0].get("finish_reason") or "")
    return ""


def _bucket_with_coverage(bucket: dict[str, int]) -> dict[str, int | float | None]:
    out: dict[str, int | float | None] = dict(bucket)
    calls_with = int(out.get("calls_with_usage") or 0)
    calls_without = int(out.get("calls_without_usage") or 0)
    denom = calls_with + calls_without
    out["token_logging_coverage"] = (calls_with / denom) if denom else None
    return out

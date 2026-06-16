from src.llm.base import LLMClient, LLMMessage, LLMResponse
from src.llm.openai_compatible import OpenAICompatibleClient
from src.llm.registry import create_llm_for_role, load_model_registry, resolve_role

__all__ = [
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "OpenAICompatibleClient",
    "create_llm",
    "create_llm_for_role",
    "load_model_registry",
    "resolve_role",
]


def create_llm(cfg: dict) -> LLMClient:
    """Create an LLM client from a public model config.

    PAL-Bench's release code intentionally supports only generic
    OpenAI-compatible chat APIs. Site-local gateways used during development are
    not part of the public artifact.
    """

    llm_cfg = dict(cfg.get("llm", {}))
    provider = llm_cfg.get("provider", "openai_compatible")
    if provider != "openai_compatible":
        raise ValueError(
            f"Unsupported LLM provider in the public release: {provider!r}. "
            "Use provider='openai_compatible'."
        )

    return OpenAICompatibleClient(
        api_key=llm_cfg.get("api_key") or None,
        base_url=llm_cfg.get("base_url", ""),
        model=llm_cfg.get("model", ""),
        timeout=int(llm_cfg.get("timeout", 1800)),
        max_retries=int(llm_cfg.get("max_retries", 3)),
        retry_backoff=float(llm_cfg.get("retry_backoff", 2.0)),
        max_json_tokens=(
            int(llm_cfg["max_json_tokens"])
            if llm_cfg.get("max_json_tokens") is not None
            else None
        ),
        json_retry_cap=(
            int(llm_cfg["json_retry_cap"])
            if llm_cfg.get("json_retry_cap") is not None
            else None
        ),
        supports_response_format=bool(llm_cfg.get("supports_response_format", True)),
        extra_body=(
            llm_cfg.get("extra_body")
            or (
                {"chat_template_kwargs": llm_cfg["chat_template_kwargs"]}
                if llm_cfg.get("chat_template_kwargs") is not None
                else None
            )
        ),
    )

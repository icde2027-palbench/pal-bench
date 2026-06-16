"""Central model registry — single source of truth for provider + model config.

Usage::

    from src.llm.registry import create_llm_for_role, resolve_role

    # Create an LLM client for a named role
    llm = create_llm_for_role("pipeline_llm")

    # With per-call overrides (e.g. longer timeout for Step 6)
    llm = create_llm_for_role("pipeline_llm", timeout=600)

    # Get raw config dict (compatible with create_llm)
    cfg = resolve_role("pipeline_image")
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default registry path (relative to project root)
_DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "models.yaml"
_EXAMPLE_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "models.example.yaml"
)
_ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}$")

_registry: dict | None = None


def load_model_registry(path: str | Path | None = None) -> dict:
    """Load and cache the model registry from ``configs/models.yaml``.

    The registry is loaded once per process and cached globally.
    Pass *path* explicitly only in tests or unusual layouts.
    """
    global _registry
    if _registry is not None:
        return _registry

    if path:
        registry_path = Path(path)
    else:
        env_path = os.environ.get("PALBENCH_MODEL_REGISTRY")
        registry_path = Path(env_path) if env_path else _DEFAULT_REGISTRY_PATH
        if not registry_path.exists() and _EXAMPLE_REGISTRY_PATH.exists():
            registry_path = _EXAMPLE_REGISTRY_PATH
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. "
            "Expected configs/models.yaml or configs/models.example.yaml in the project root."
        )

    with open(registry_path, "r", encoding="utf-8") as fh:
        _registry = _expand_env(yaml.safe_load(fh) or {})

    logger.debug(
        "Loaded model registry with providers=%s, roles=%s",
        list(_registry.get("providers", {}).keys()),
        list(_registry.get("roles", {}).keys()),
    )
    return _registry


def _expand_env(value: Any) -> Any:
    """Expand ${ENV_VAR} and ${ENV_VAR:default} strings in loaded YAML."""

    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, str):
        match = _ENV_PATTERN.match(value)
        if match:
            name, default = match.groups()
            return os.environ.get(name, default or "")
    return value


def reset_registry() -> None:
    """Clear the cached registry (for testing)."""
    global _registry
    _registry = None


def resolve_role(role: str, **overrides: Any) -> dict:
    """Resolve a role name to a config dict compatible with :func:`create_llm`.

    Returns ``{"llm": {"provider": "...", "model": "...", ...}}``.

    Any keyword *overrides* (e.g. ``timeout=600``) are merged on top,
    taking highest priority.
    """
    reg = load_model_registry()
    roles = reg.get("roles", {})
    if role not in roles:
        available = ", ".join(sorted(roles.keys())) or "(none)"
        raise KeyError(
            f"Unknown role '{role}'. Available roles: {available}"
        )

    role_cfg = dict(roles[role])
    provider_name = role_cfg.pop("provider")

    providers = reg.get("providers", {})
    provider_cfg = dict(providers.get(provider_name, {}))

    # Merge order: provider defaults < role config < caller overrides
    merged = {**provider_cfg, **role_cfg, "provider": provider_name}
    if overrides:
        merged.update(overrides)

    return {"llm": merged}


def create_llm_for_role(role: str, **overrides: Any):
    """Shortcut: resolve *role* and return a ready-to-use :class:`LLMClient`.

    Equivalent to ``create_llm(resolve_role(role, **overrides))``.
    """
    # Import here to avoid circular imports (registry ← __init__ ← registry)
    from src.llm import create_llm

    return create_llm(resolve_role(role, **overrides))

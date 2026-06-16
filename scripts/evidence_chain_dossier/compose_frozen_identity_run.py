#!/usr/bin/env python3
"""Compose an ECD run with frozen identity and replacement owner facts.

This utility is for paired experiments where identity/person rows must remain
bit-for-bit stable while owner-fact or evidence-grounding layers are varied.
It writes evaluator-compatible ``predicted_profile.json`` files under a new
run root without re-running any LLM identity phase.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.evidence_chain_dossier.config import ECDConfig
from src.agent.evidence_chain_dossier.evidence_citation_optimizer import (
    optimize_evidence_citations,
)
from src.agent.evidence_chain_dossier.inventory import build_inventory
from src.utils.io import load_json, save_json

_TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
_CALL_KEYS = ("calls_with_usage", "calls_without_usage")


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = load_json(path)
    return data if isinstance(data, dict) else {}


def _users_from_root(root: Path) -> list[str]:
    return sorted(
        path.parent.name
        for path in root.glob("user_*/predicted_profile.json")
        if path.parent.is_dir()
    )


def _split_users(value: str | None, stable_root: Path) -> list[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return _users_from_root(stable_root)


def _n_llm(stats: dict[str, Any], profile: dict[str, Any]) -> int:
    source = stats or (profile.get("run_stats") or {})
    try:
        return int(source.get("n_llm_calls", 0) or source.get("budget_used", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _numeric(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _int_numeric(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _json_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def _stage_usage(stats: dict[str, Any], stage: str) -> dict[str, Any]:
    usage_by_stage = stats.get("usage_by_stage")
    if not isinstance(usage_by_stage, dict):
        return {}
    usage = usage_by_stage.get(stage)
    return usage if isinstance(usage, dict) else {}


def _stage_call_count(usage: dict[str, Any]) -> int:
    return _int_numeric(usage.get("calls_with_usage")) + _int_numeric(usage.get("calls_without_usage"))


def _merge_usage_by_stage(
    stable_usage: dict[str, Any],
    owner_detail_usage: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    if isinstance(stable_usage, dict):
        for stage, usage in stable_usage.items():
            if isinstance(usage, dict):
                merged[stage] = copy.deepcopy(usage)
    if owner_detail_usage:
        target = merged.setdefault("owner_detail_miner", {})
        for key in (*_TOKEN_KEYS, *_CALL_KEYS):
            target[key] = _json_number(_numeric(target.get(key)) + _numeric(owner_detail_usage.get(key)))
        target["token_logging_coverage"] = _coverage(
            _int_numeric(target.get("calls_with_usage")),
            _int_numeric(target.get("calls_without_usage")),
        )
    return merged


def _coverage(calls_with: int, calls_without: int) -> float | None:
    denom = calls_with + calls_without
    return (calls_with / denom) if denom else None


def _combine_token_stats(stable_stats: dict[str, Any], owner_detail_usage: dict[str, Any]) -> dict[str, Any]:
    has_any_usage = any(isinstance(stable_stats.get(key), (int, float)) for key in (*_TOKEN_KEYS, *_CALL_KEYS))
    has_any_usage = has_any_usage or any(
        isinstance(owner_detail_usage.get(key), (int, float)) for key in (*_TOKEN_KEYS, *_CALL_KEYS)
    )
    if not has_any_usage:
        return {}

    combined: dict[str, Any] = {}
    for key in _TOKEN_KEYS:
        combined[key] = _numeric(stable_stats.get(key)) + _numeric(owner_detail_usage.get(key))
        if float(combined[key]).is_integer():
            combined[key] = int(combined[key])
    for key in _CALL_KEYS:
        combined[key] = _int_numeric(stable_stats.get(key)) + _int_numeric(owner_detail_usage.get(key))
    combined["token_logging_coverage"] = _coverage(
        _int_numeric(combined.get("calls_with_usage")),
        _int_numeric(combined.get("calls_without_usage")),
    )
    combined["usage_by_stage"] = _merge_usage_by_stage(
        stable_stats.get("usage_by_stage") if isinstance(stable_stats.get("usage_by_stage"), dict) else {},
        owner_detail_usage,
    )
    return combined


def _compose_one(
    *,
    user_id: str,
    stable_root: Path,
    owner_root: Path,
    users_root: Path,
    output_root: Path,
    config: ECDConfig,
    apply_eco: bool,
    owner_extra_llm_calls: int,
) -> dict[str, Any]:
    stable_dir = stable_root / user_id
    owner_dir = owner_root / user_id
    output_dir = output_root / user_id
    stable_profile_path = stable_dir / "predicted_profile.json"
    owner_profile_path = owner_dir / "predicted_profile.json"
    if not stable_profile_path.exists():
        raise FileNotFoundError(f"Missing stable profile: {stable_profile_path}")
    if not owner_profile_path.exists():
        raise FileNotFoundError(f"Missing owner-source profile: {owner_profile_path}")

    stable_profile = load_json(stable_profile_path)
    owner_profile = load_json(owner_profile_path)
    if not isinstance(stable_profile, dict) or not isinstance(owner_profile, dict):
        raise ValueError(f"Expected profile JSON objects for {user_id}")

    stable_stats = _load_optional_json(stable_dir / "run_stats.json") or dict(stable_profile.get("run_stats") or {})
    owner_stats = _load_optional_json(owner_dir / "run_stats.json") or dict(owner_profile.get("run_stats") or {})
    stable_n_llm = _n_llm(stable_stats, stable_profile)
    owner_n_llm = _n_llm(owner_stats, owner_profile)
    owner_detail_usage = _stage_usage(owner_stats, "owner_detail_miner")
    owner_detail_calls = _stage_call_count(owner_detail_usage)
    charged_owner_llm_calls = owner_detail_calls or max(0, int(owner_extra_llm_calls))
    stable_total_time = _numeric(stable_stats.get("total_time_s"))
    owner_detail_time = _numeric(owner_stats.get("owner_detail_miner_time_s"))

    composed = copy.deepcopy(stable_profile)
    stable_owner = stable_profile.get("owner") if isinstance(stable_profile.get("owner"), dict) else {}
    source_owner = owner_profile.get("owner") if isinstance(owner_profile.get("owner"), dict) else {}
    composed["owner"] = copy.deepcopy(source_owner)
    composed.setdefault("owner", {})
    composed["owner"]["face_id"] = stable_owner.get("face_id") or source_owner.get("face_id") or ""
    composed["owner"]["facts"] = copy.deepcopy(source_owner.get("facts") or [])
    composed["persons"] = copy.deepcopy(stable_profile.get("persons") or [])
    composed["framework"] = stable_profile.get("framework") or "evidence_chain_dossier"
    composed["variant"] = "ecd_frozen_identity_owner_source_eco" if apply_eco else "ecd_frozen_identity_owner_source"

    diagnostics: dict[str, Any] = {
        "compose_frozen_identity": {
            "enabled": True,
            "user_id": user_id,
            "stable_identity_root": str(stable_root),
            "owner_source_root": str(owner_root),
            "stable_identity_profile": str(stable_profile_path),
            "owner_source_profile": str(owner_profile_path),
            "n_stable_persons": len(stable_profile.get("persons") or []),
            "n_owner_source_facts": len(source_owner.get("facts") or []),
            "stable_identity_n_llm_calls": stable_n_llm,
            "owner_source_n_llm_calls": owner_n_llm,
            "owner_extra_llm_calls": int(owner_extra_llm_calls),
            "charged_owner_llm_calls": charged_owner_llm_calls,
            "owner_source_owner_detail_miner_time_s": round(owner_detail_time, 3),
            "owner_detail_usage": copy.deepcopy(owner_detail_usage),
        }
    }

    if apply_eco:
        album_path = users_root / user_id / f"{user_id}_agent_album.json"
        album = load_json(album_path)
        if not isinstance(album, dict):
            raise ValueError(f"Expected album JSON object at {album_path}")
        inventory = build_inventory(album)
        # Resolved identities are intentionally empty here: the compose run is
        # owner-evidence focused and must not perturb frozen person evidence.
        composed, eco_diagnostics = optimize_evidence_citations(
            profile=composed,
            inventory=inventory,
            resolved=[],
            config=config,
        )
        diagnostics["evidence_citation_optimizer"] = eco_diagnostics
    else:
        diagnostics["evidence_citation_optimizer"] = {"enabled": False}

    run_stats = copy.deepcopy(stable_stats)
    combined_n_llm = stable_n_llm + charged_owner_llm_calls
    combined_token_stats = _combine_token_stats(stable_stats, owner_detail_usage)
    run_stats.update(
        {
            "n_llm_calls": combined_n_llm,
            "budget_used": combined_n_llm,
            "total_time_s": round(stable_total_time + owner_detail_time, 3),
            "compose_mode": "frozen_identity_owner_source",
            "stable_identity_n_llm_calls": stable_n_llm,
            "owner_source_n_llm_calls": owner_n_llm,
            "owner_extra_llm_calls": int(owner_extra_llm_calls),
            "charged_owner_llm_calls": charged_owner_llm_calls,
            "stable_identity_total_time_s": round(stable_total_time, 3),
            "owner_source_owner_detail_miner_time_s": round(owner_detail_time, 3),
            "evidence_citation_optimizer_applied": bool(apply_eco),
            **combined_token_stats,
        }
    )
    composed["run_stats"] = run_stats

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(composed, output_dir / "predicted_profile.json")
    save_json(run_stats, output_dir / "run_stats.json")
    save_json(diagnostics, output_dir / "diagnostics.json")
    return {
        "user_id": user_id,
        "n_persons": len(composed.get("persons") or []),
        "n_owner_facts": len((composed.get("owner") or {}).get("facts") or []),
        "n_llm_calls": combined_n_llm,
        "eco_owner_updates": (
            diagnostics.get("evidence_citation_optimizer", {}).get("n_owner_updates")
            if apply_eco
            else 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose frozen-identity ECD profiles.")
    parser.add_argument("--stable-root", required=True, help="Run root providing frozen persons/identity.")
    parser.add_argument("--owner-root", required=True, help="Run root providing owner facts.")
    parser.add_argument("--output-root", required=True, help="Output run root.")
    parser.add_argument("--users-root", default="data/full/users")
    parser.add_argument("--users", default=None, help="Optional comma-separated user ids.")
    parser.add_argument("--config", default=None, help="ECD config for ECO thresholds.")
    parser.add_argument("--apply-eco", action="store_true", help="Apply owner-only ECO after composition.")
    parser.add_argument(
        "--owner-extra-llm-calls",
        type=int,
        default=1,
        help="Additional LLM calls charged for importing owner-detail mining.",
    )
    args = parser.parse_args()

    stable_root = Path(args.stable_root)
    owner_root = Path(args.owner_root)
    output_root = Path(args.output_root)
    users_root = Path(args.users_root)
    config = ECDConfig.from_path(args.config) if args.config else ECDConfig()
    users = _split_users(args.users, stable_root)
    if not users:
        raise SystemExit("No users to compose.")

    rows = []
    for user_id in users:
        row = _compose_one(
            user_id=user_id,
            stable_root=stable_root,
            owner_root=owner_root,
            users_root=users_root,
            output_root=output_root,
            config=config,
            apply_eco=bool(args.apply_eco),
            owner_extra_llm_calls=int(args.owner_extra_llm_calls),
        )
        rows.append(row)
        print(
            f"[compose] {user_id}: {row['n_persons']} persons, {row['n_owner_facts']} owner facts, "
            f"{row['n_llm_calls']} charged calls, eco_owner_updates={row['eco_owner_updates']}"
        )
    save_json(
        {
            "stable_root": str(stable_root),
            "owner_root": str(owner_root),
            "output_root": str(output_root),
            "users": users,
            "apply_eco": bool(args.apply_eco),
            "owner_extra_llm_calls": int(args.owner_extra_llm_calls),
            "rows": rows,
        },
        output_root / "compose_summary.json",
    )
    print(f"[compose] Done: {len(rows)} users -> {output_root}")


if __name__ == "__main__":
    main()

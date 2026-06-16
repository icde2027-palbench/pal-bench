#!/usr/bin/env python3
"""Estimate LLM judge calls, tokens, and cost for PAL-Bench formal evaluation."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import load_json, save_json


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _target_counts(users: list[str], users_root: Path) -> dict[str, Any]:
    by_family: Counter[str] = Counter()
    per_user: dict[str, dict[str, int]] = {}
    for user_id in users:
        gt_path = users_root / user_id / f"{user_id}_eval_gt.json"
        gt = load_json(gt_path)
        counts: Counter[str] = Counter(str(t.get("target_type") or "unknown") for t in gt.get("evaluation_targets") or [])
        per_user[user_id] = dict(counts)
        by_family.update(counts)
    return {"by_family": dict(by_family), "per_user": per_user}


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate official judge budget")
    parser.add_argument("--planned-methods", required=True, help="Comma-separated method ids")
    parser.add_argument("--users", required=True, help="Comma-separated user ids")
    parser.add_argument("--users-root", default="data/full/users")
    parser.add_argument("--judge-role", default="eval_judge")
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-passes-per-target", type=float, default=1.0)
    parser.add_argument("--avg-value-input-tokens", type=int, default=900)
    parser.add_argument("--avg-value-output-tokens", type=int, default=90)
    parser.add_argument("--avg-evidence-input-tokens", type=int, default=1700)
    parser.add_argument("--avg-evidence-output-tokens", type=int, default=140)
    parser.add_argument("--input-price-per-mtok", type=float, default=0.0)
    parser.add_argument("--output-price-per-mtok", type=float, default=0.0)
    parser.add_argument("--phase05-cap-usd", type=float, default=0.0)
    parser.add_argument("--full50-cap-usd", type=float, default=0.0)
    args = parser.parse_args()

    methods = _split_csv(args.planned_methods)
    users = _split_csv(args.users)
    users_root = Path(args.users_root)
    counts = _target_counts(users, users_root)
    by_family = counts["by_family"]
    n_methods = len(methods)
    n_targets = sum(by_family.values())
    n_owner = int(by_family.get("owner_fact_atom", 0))

    # Current formal evaluator makes one semantic owner-value call per owner
    # target and one evidence-faithfulness call per target in the upper bound.
    value_calls_per_method = n_owner
    evidence_calls_per_method = n_targets
    value_calls_total = int(round(value_calls_per_method * n_methods * args.judge_passes_per_target))
    evidence_calls_total = int(round(evidence_calls_per_method * n_methods * args.judge_passes_per_target))
    total_calls = value_calls_total + evidence_calls_total
    input_tokens = (
        value_calls_total * int(args.avg_value_input_tokens)
        + evidence_calls_total * int(args.avg_evidence_input_tokens)
    )
    output_tokens = (
        value_calls_total * int(args.avg_value_output_tokens)
        + evidence_calls_total * int(args.avg_evidence_output_tokens)
    )
    estimated_cost = (
        input_tokens / 1_000_000 * float(args.input_price_per_mtok)
        + output_tokens / 1_000_000 * float(args.output_price_per_mtok)
    )

    per_method = {}
    for method in methods:
        per_method[method] = {
            "value_calls_upper_bound": value_calls_per_method,
            "evidence_calls_upper_bound": evidence_calls_per_method,
            "total_calls_upper_bound": value_calls_per_method + evidence_calls_per_method,
        }

    result = {
        "schema_version": "judge_budget_estimate.v1",
        "users_root": str(users_root),
        "judge_role": args.judge_role,
        "methods": methods,
        "n_users": len(users),
        "target_counts": by_family,
        "per_user_target_counts": counts["per_user"],
        "call_model": {
            "owner_value_calls": "one per owner_fact_atom per method",
            "evidence_calls": "upper bound one per target per method after precheck",
            "judge_passes_per_target": args.judge_passes_per_target,
        },
        "per_method": per_method,
        "totals": {
            "value_calls_upper_bound": value_calls_total,
            "evidence_calls_upper_bound": evidence_calls_total,
            "judge_calls_upper_bound": total_calls,
            "input_tokens_estimate": input_tokens,
            "output_tokens_estimate": output_tokens,
            "estimated_cost_usd": round(estimated_cost, 4),
        },
        "spending_caps": {
            "phase05_cap_usd": args.phase05_cap_usd,
            "full50_cap_usd": args.full50_cap_usd,
        },
        "fallback_policy": [
            "Use the predeclared stratified sample before reducing official judge scope.",
            "Reduce cross-family reliability sample size before changing eval_judge.",
            "If reliability or budget is insufficient, demote EFS to diagnostic.",
        ],
    }
    save_json(result, args.output)
    print(f"[judge-budget] calls<={total_calls} cost=${estimated_cost:.4f} -> {args.output}")


if __name__ == "__main__":
    main()

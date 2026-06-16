#!/usr/bin/env python3
"""Paired user-level bootstrap comparisons for formal eval aggregates."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import load_json, save_json


DEFAULT_METRICS = ["OFR", "PIR", "PIR-hard", "PRR-ID", "EFS", "ECE", "nLLM"]


def _load_per_user(path: str | Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    rows = {}
    for row in data.get("per_user") or []:
        user_id = str(row.get("user_id") or "")
        metrics = row.get("metrics") or {}
        if user_id:
            rows[user_id] = metrics
    return rows


def _paired_bootstrap(a: list[float], b: list[float], *, n: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    observed = mean(diffs) if diffs else 0.0
    samples = []
    m = len(diffs)
    for _ in range(n):
        sample = [diffs[rng.randrange(m)] for _ in range(m)]
        samples.append(mean(sample))
    samples.sort()
    low = samples[int(0.025 * (n - 1))]
    high = samples[int(0.975 * (n - 1))]
    if observed == 0:
        p_two = 1.0
    else:
        prop_le_zero = sum(1 for value in samples if value <= 0.0) / max(1, n)
        prop_ge_zero = sum(1 for value in samples if value >= 0.0) / max(1, n)
        p_two = min(1.0, 2.0 * min(prop_le_zero, prop_ge_zero))
    return {
        "mean_delta": round(observed, 6),
        "ci_low": round(low, 6),
        "ci_high": round(high, 6),
        "paired_p_two_sided_bootstrap": round(p_two, 6),
        "n_pairs": m,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap over users")
    parser.add_argument("--baseline", required=True, help="baseline_name=path/to/aggregate_formal.json")
    parser.add_argument("--candidate", required=True, help="candidate_name=path/to/aggregate_formal.json")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_name, base_path = args.baseline.split("=", 1)
    cand_name, cand_path = args.candidate.split("=", 1)
    base = _load_per_user(base_path)
    cand = _load_per_user(cand_path)
    users = sorted(set(base) & set(cand))
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    rows = []
    for metric in metrics:
        a = []
        b = []
        for user_id in users:
            av = cand[user_id].get(metric)
            bv = base[user_id].get(metric)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                a.append(float(av))
                b.append(float(bv))
        if not a:
            continue
        rows.append(
            {
                "comparison": f"{cand_name} - {base_name}",
                "metric": metric,
                **_paired_bootstrap(a, b, n=int(args.n_bootstrap), seed=int(args.seed) + len(rows)),
            }
        )
    result = {
        "schema_version": "paired_bootstrap.v1",
        "baseline": {"name": base_name, "aggregate": base_path},
        "candidate": {"name": cand_name, "aggregate": cand_path},
        "users": users,
        "n_bootstrap": args.n_bootstrap,
        "rows": rows,
    }
    save_json(result, args.output)
    print(f"[bootstrap] {cand_name} vs {base_name}: {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()

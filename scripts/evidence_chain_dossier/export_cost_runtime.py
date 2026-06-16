#!/usr/bin/env python3
"""Export token, runtime, parse-failure, and judge-call diagnostics."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import load_json, save_json


def _parse_method(value: str) -> tuple[str, str, Path, Path]:
    parts = value.split("=", 3)
    if len(parts) != 4:
        raise ValueError(f"Expected method=label=run_root=eval_dir, got {value}")
    return parts[0], parts[1], Path(parts[2]), Path(parts[3])


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sum_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) if values else None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _load_run_stats(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run_root.glob("user_*/run_stats.json")):
        row = load_json(path)
        row["user_id"] = path.parent.name
        rows.append(row)
    return rows


def _numeric_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return _mean(values)


def _parse_fail_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    failures = sum(float(row.get("parse_failures") or 0.0) for row in rows)
    calls = sum(float(row.get("n_llm_calls") or row.get("budget_used") or 0.0) for row in rows)
    if calls <= 0:
        return 0.0 if failures <= 0 else None
    return failures / calls


def _summarize(method: str, label: str, run_root: Path, eval_dir: Path) -> dict[str, Any]:
    stats = _load_run_stats(run_root)
    aggregate = load_json(eval_dir / "aggregate_formal.json")
    metrics = aggregate.get("metrics") or {}
    judge_cache = aggregate.get("judge_cache") or {}
    n_users = int(aggregate.get("n_users") or len(stats))
    judge_misses = judge_cache.get("misses")
    judge_calls_per_user = float(judge_misses) / n_users if isinstance(judge_misses, (int, float)) and n_users else None
    prompt_total = _sum_numeric(stats, "prompt_tokens")
    completion_total = _sum_numeric(stats, "completion_tokens")
    total_tokens = _sum_numeric(stats, "total_tokens")
    return {
        "method": method,
        "label": label,
        "n_users": n_users,
        "OFR": metrics.get("OFR"),
        "PIR": metrics.get("PIR"),
        "PIR-hard": metrics.get("PIR-hard"),
        "EFS": metrics.get("EFS"),
        "nLLM": metrics.get("nLLM"),
        "prompt_tokens_per_user": (prompt_total / n_users) if prompt_total is not None and n_users else None,
        "completion_tokens_per_user": (completion_total / n_users) if completion_total is not None and n_users else None,
        "total_tokens_per_user": (total_tokens / n_users) if total_tokens is not None and n_users else None,
        "sec_per_user": _numeric_mean(stats, "total_time_s") or _numeric_mean(stats, "elapsed_sec"),
        "parse_fail_rate": _parse_fail_rate(stats),
        "judge_calls_per_user": judge_calls_per_user,
        "judge_cache_items": judge_cache.get("n_items"),
        "judge_cache_hits": judge_cache.get("hits"),
        "judge_cache_misses": judge_cache.get("misses"),
        "token_logging_coverage": sum(1 for row in stats if isinstance(row.get("total_tokens"), (int, float))) / len(stats) if stats else None,
        "run_root": str(run_root),
        "eval_dir": str(eval_dir),
        "cost_user_usd": None,
        "cost_note": "not estimated: provider pricing is not part of this artifact",
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Cost / Runtime / Token Diagnostics",
        "",
        "| Method | OFR | PIR | PIR-hard | EFS | nLLM/user | Prompt tok/user | Completion tok/user | Total tok/user | sec/user | ParseFailRate | Judge calls/user |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {_fmt(row['OFR'])} | {_fmt(row['PIR'])} | {_fmt(row['PIR-hard'])} | {_fmt(row['EFS'])} | "
            f"{_fmt(row['nLLM'])} | {_fmt(row['prompt_tokens_per_user'], 1)} | {_fmt(row['completion_tokens_per_user'], 1)} | "
            f"{_fmt(row['total_tokens_per_user'], 1)} | {_fmt(row['sec_per_user'], 2)} | {_fmt(row['parse_fail_rate'])} | "
            f"{_fmt(row['judge_calls_per_user'], 2)} |"
        )
    lines.extend(
        [
            "",
            "Dollar cost is intentionally left unestimated because provider pricing is not part of this artifact.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cost/runtime diagnostics")
    parser.add_argument("--method", action="append", required=True, help="method=label=run_root=eval_dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_summarize(*_parse_method(item)) for item in args.method]
    save_json(
        {
            "schema_version": "cost_runtime_diagnostics.v1",
            "rows": rows,
        },
        output_dir / "cost_runtime.json",
    )
    _write_csv(rows, output_dir / "cost_runtime.csv")
    _write_markdown(rows, output_dir / "cost_equivalent.md")
    print(f"[cost-runtime] -> {output_dir / 'cost_equivalent.md'}")


if __name__ == "__main__":
    main()

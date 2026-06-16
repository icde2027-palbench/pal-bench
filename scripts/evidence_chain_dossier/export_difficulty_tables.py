#!/usr/bin/env python3
"""Export difficulty-stratified official PAL-Bench results."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import load_json, save_json

DIFFICULTIES = ["easy", "medium", "hard"]


def _parse_method(value: str) -> tuple[str, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise ValueError(f"Expected method=label=eval_dir, got {value}")
    return parts[0], parts[1], Path(parts[2])


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _target_meta(users_root: Path, user_id: str) -> dict[str, dict[str, Any]]:
    gt = load_json(users_root / user_id / f"{user_id}_eval_gt.json")
    return {
        str(row.get("target_id") or ""): row
        for row in gt.get("evaluation_targets") or []
        if str(row.get("target_id") or "")
    }


def _difficulty(row: dict[str, Any], gt_row: dict[str, Any]) -> str:
    if row.get("target_type") == "owner_fact_atom":
        return str(gt_row.get("difficulty") or row.get("difficulty") or "unknown").lower()
    return str(gt_row.get("alignment_difficulty") or row.get("alignment_difficulty") or "unknown").lower()


def _collect_rows(eval_dir: Path, users_root: Path) -> list[dict[str, Any]]:
    rows = []
    for report_path in sorted((eval_dir / "per_user").glob("*/formal_eval_report.json")):
        user_id = report_path.parent.name
        meta = _target_meta(users_root, user_id)
        report = load_json(report_path)
        for row in report.get("targets") or []:
            target_id = str(row.get("target_id") or "")
            gt_row = meta.get(target_id, {})
            copied = dict(row)
            copied["user_id"] = user_id
            copied["difficulty"] = _difficulty(row, gt_row)
            copied["inference_type"] = row.get("inference_type") or gt_row.get("inference_type")
            rows.append(copied)
    return rows


def _summarize(method: str, label: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for difficulty in DIFFICULTIES:
        group = [r for r in rows if r.get("difficulty") == difficulty]
        owner = [float(r.get("correctness")) for r in group if r.get("target_type") == "owner_fact_atom" and isinstance(r.get("correctness"), (int, float))]
        names = [float(r.get("correctness")) for r in group if r.get("target_type") == "person_name" and isinstance(r.get("correctness"), (int, float))]
        rel_id_scores = []
        efs = []
        for row in group:
            efs_value = row.get("EFS", row.get("efs"))
            if isinstance(efs_value, (int, float)):
                efs.append(float(efs_value))
            details = row.get("judge_details") or {}
            if row.get("target_type") == "person_relation" and details.get("identity_bound") is True:
                raw = details.get("raw_relation_score")
                if isinstance(raw, (int, float)):
                    rel_id_scores.append(float(raw))
        out.append(
            {
                "method": method,
                "label": label,
                "difficulty": difficulty,
                "target_count": len(group),
                "owner_target_count": len(owner),
                "person_name_target_count": len(names),
                "relation_id_support_count": len(rel_id_scores),
                "OFR": _mean(owner),
                "PIR": _mean(names),
                "PIR-hard": _mean(names) if difficulty == "hard" else None,
                "PRR-ID": _mean(rel_id_scores),
                "EFS": _mean(efs),
            }
        )
    return out


def _inference_summary(method: str, label: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("target_type") == "owner_fact_atom":
            groups[str(row.get("inference_type") or "unknown")].append(row)
    out = []
    for inference_type, group in sorted(groups.items()):
        scores = [float(r.get("correctness")) for r in group if isinstance(r.get("correctness"), (int, float))]
        efs = [float(r.get("EFS")) for r in group if isinstance(r.get("EFS"), (int, float))]
        out.append(
            {
                "method": method,
                "label": label,
                "inference_type": inference_type,
                "target_count": len(group),
                "OFR": _mean(scores),
                "EFS-owner": _mean(efs),
            }
        )
    return out


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
        "# Difficulty-Stratified Official Results",
        "",
        "| Method | Difficulty | Target count | OFR | PIR | PIR-hard | PRR-ID | EFS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['difficulty'].title()} | {row['target_count']} | "
            f"{_fmt(row['OFR'])} | {_fmt(row['PIR'])} | {_fmt(row['PIR-hard'])} | "
            f"{_fmt(row['PRR-ID'])} | {_fmt(row['EFS'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_inference_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Owner-Fact Inference Type Breakdown",
        "",
        "| Method | Inference type | Target count | OFR | EFS-owner |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['inference_type']} | {row['target_count']} | "
            f"{_fmt(row['OFR'])} | {_fmt(row['EFS-owner'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export difficulty-stratified tables")
    parser.add_argument("--users-root", default="data/full/users")
    parser.add_argument("--method", action="append", required=True, help="method=label=eval_dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    users_root = Path(args.users_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    difficulty_rows = []
    inference_rows = []
    sources = {}
    for method, label, eval_dir in [_parse_method(item) for item in args.method]:
        rows = _collect_rows(eval_dir, users_root)
        difficulty_rows.extend(_summarize(method, label, rows))
        inference_rows.extend(_inference_summary(method, label, rows))
        sources[method] = str(eval_dir)

    save_json(
        {
            "schema_version": "difficulty_stratified_results.v1",
            "users_root": str(users_root),
            "sources": sources,
            "difficulty_rows": difficulty_rows,
            "owner_inference_rows": inference_rows,
        },
        output_dir / "difficulty_results.json",
    )
    _write_csv(difficulty_rows, output_dir / "difficulty_results.csv")
    _write_csv(inference_rows, output_dir / "owner_inference_type_results.csv")
    _write_markdown(difficulty_rows, output_dir / "difficulty_results.md")
    _write_inference_markdown(inference_rows, output_dir / "owner_inference_type_results.md")
    print(f"[difficulty] -> {output_dir / 'difficulty_results.md'}")


if __name__ == "__main__":
    main()

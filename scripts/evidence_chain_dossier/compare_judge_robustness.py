#!/usr/bin/env python3
"""Compare official and second-judge formal evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmark.eval.formal import aggregate_formal_reports
from src.utils.io import load_json, save_json


MAIN_KEYS = ["OFR", "PIR", "PIR-hard", "PRR-ID", "EFS", "ECE", "nLLM"]
CORR_KEYS = ["OFR", "PIR", "PIR-hard", "EFS"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Qwen and GPT-OSS judge robustness on a subset.")
    parser.add_argument("--phase-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    phase_root = Path(args.phase_root)
    manifest_path = Path(args.manifest) if args.manifest else phase_root / "phase1c_manifest.json"
    output_dir = Path(args.output_dir) if args.output_dir else phase_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path)
    users = list(manifest.get("subset_users") or [])

    method_rows: list[dict[str, Any]] = []
    per_method: dict[str, dict[str, Any]] = {}
    for method in manifest.get("methods") or []:
        method_id = str(method.get("method_id") or "")
        qwen_aggregate_path = Path(method.get("official_qwen_aggregate") or "")
        gpt_root = phase_root / "eval_gpt_oss_120b" / method_id / "official_llm_semantic_subset10"
        qwen_root = qwen_aggregate_path.parent
        qwen_full_agg = load_json(qwen_aggregate_path)
        qwen_reports = _load_reports(qwen_root, users)
        gpt_reports = _load_reports(gpt_root, users)
        qwen_subset = aggregate_formal_reports(
            qwen_reports,
            runs_root=Path(qwen_full_agg.get("runs_root") or method.get("runs_root") or ""),
            users_root=Path(qwen_full_agg.get("users_root") or "data/full/users"),
            judge_mode=str(qwen_full_agg.get("judge_mode") or "llm_semantic"),
            evidence_judge_mode=str(qwen_full_agg.get("evidence_judge_mode") or "llm_semantic"),
        )
        gpt_aggregate_path = gpt_root / "aggregate_formal.json"
        if gpt_aggregate_path.exists():
            gpt_subset = load_json(gpt_aggregate_path)
        else:
            gpt_subset = aggregate_formal_reports(
                gpt_reports,
                runs_root=Path(method.get("runs_root") or ""),
                users_root=Path("data/full/users"),
                judge_mode="llm_semantic",
                evidence_judge_mode="llm_semantic",
            )

        correlations = {
            key: _metric_correlation(qwen_reports, gpt_reports, key)
            for key in CORR_KEYS
        }
        target_agreement = _target_agreement(qwen_reports, gpt_reports)
        per_method[method_id] = {
            "method_id": method_id,
            "qwen_subset_metrics": qwen_subset.get("metrics") or {},
            "gpt_oss_subset_metrics": gpt_subset.get("metrics") or {},
            "per_user_metric_correlation": correlations,
            "target_level_agreement": target_agreement,
            "qwen_root": str(qwen_root),
            "gpt_oss_root": str(gpt_root),
        }
        row = {"method_id": method_id}
        for key in MAIN_KEYS:
            row[f"qwen_{_key_name(key)}"] = _fmt_num((qwen_subset.get("metrics") or {}).get(key))
            row[f"gpt_oss_{_key_name(key)}"] = _fmt_num((gpt_subset.get("metrics") or {}).get(key))
            row[f"delta_{_key_name(key)}"] = _fmt_delta(
                (gpt_subset.get("metrics") or {}).get(key),
                (qwen_subset.get("metrics") or {}).get(key),
            )
        method_rows.append(row)

    ranking = _ranking_stability(per_method)
    pal_trace_deltas = _pal_trace_deltas(per_method)
    summary = {
        "schema_version": "phase1c_judge_robustness_summary.v1",
        "phase_root": str(phase_root),
        "manifest": manifest,
        "n_methods": len(per_method),
        "n_users": len(users),
        "per_method": per_method,
        "ranking_stability": ranking,
        "pal_trace_deltas": pal_trace_deltas,
    }
    save_json(summary, output_dir / "judge_robustness_summary.json")
    _write_csv(method_rows, output_dir / "judge_robustness_metrics.csv")
    _write_markdown(summary, method_rows, output_dir / "judge_robustness_summary.md")
    print(output_dir / "judge_robustness_summary.json")
    print(output_dir / "judge_robustness_summary.md")
    print(output_dir / "judge_robustness_metrics.csv")


def _load_reports(root: Path, users: list[str]) -> list[dict[str, Any]]:
    reports = []
    for user_id in users:
        path = root / "per_user" / user_id / "formal_eval_report.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing per-user report: {path}")
        reports.append(load_json(path))
    return reports


def _metric_correlation(qwen_reports: list[dict[str, Any]], gpt_reports: list[dict[str, Any]], key: str) -> dict[str, Any]:
    by_user = {str(report.get("user_id") or ""): report for report in gpt_reports}
    xs: list[float] = []
    ys: list[float] = []
    for q_report in qwen_reports:
        user_id = str(q_report.get("user_id") or "")
        g_report = by_user.get(user_id)
        if not g_report:
            continue
        q_value = (q_report.get("metrics") or {}).get(key)
        g_value = (g_report.get("metrics") or {}).get(key)
        if isinstance(q_value, (int, float)) and isinstance(g_value, (int, float)):
            xs.append(float(q_value))
            ys.append(float(g_value))
    return {
        "n": len(xs),
        "spearman": _spearman(xs, ys),
        "pearson": _pearson(xs, ys),
        "mae": _mae(xs, ys),
    }


def _target_agreement(qwen_reports: list[dict[str, Any]], gpt_reports: list[dict[str, Any]]) -> dict[str, Any]:
    gpt_by_user = {str(report.get("user_id") or ""): report for report in gpt_reports}
    correctness_pairs: list[tuple[float, float]] = []
    efs_pairs: list[tuple[float, float]] = []
    by_type: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for q_report in qwen_reports:
        user_id = str(q_report.get("user_id") or "")
        g_report = gpt_by_user.get(user_id) or {}
        g_targets = {
            str(row.get("target_id") or ""): row
            for row in g_report.get("targets") or []
        }
        for q_row in q_report.get("targets") or []:
            target_id = str(q_row.get("target_id") or "")
            g_row = g_targets.get(target_id)
            if not g_row:
                continue
            target_type = str(q_row.get("target_type") or "unknown")
            bucket = by_type.setdefault(target_type, {"correctness": [], "efs": []})
            q_correct = q_row.get("correctness")
            g_correct = g_row.get("correctness")
            if isinstance(q_correct, (int, float)) and isinstance(g_correct, (int, float)):
                pair = (float(q_correct), float(g_correct))
                correctness_pairs.append(pair)
                bucket["correctness"].append(pair)
            q_efs = q_row.get("EFS")
            g_efs = g_row.get("EFS")
            if isinstance(q_efs, (int, float)) and isinstance(g_efs, (int, float)):
                pair = (float(q_efs), float(g_efs))
                efs_pairs.append(pair)
                bucket["efs"].append(pair)

    return {
        "overall": {
            "correctness": _agreement_stats(correctness_pairs),
            "EFS": _agreement_stats(efs_pairs),
        },
        "by_target_type": {
            target_type: {
                "correctness": _agreement_stats(values["correctness"]),
                "EFS": _agreement_stats(values["efs"]),
            }
            for target_type, values in sorted(by_type.items())
        },
    }


def _agreement_stats(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    exact = sum(1 for x, y in pairs if abs(x - y) < 1e-9)
    return {
        "n": len(pairs),
        "exact_agreement": round(exact / len(pairs), 4) if pairs else None,
        "mae": _mae(xs, ys),
        "weighted_kappa": _weighted_kappa(pairs),
        "spearman": _spearman(xs, ys),
    }


def _ranking_stability(per_method: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in CORR_KEYS:
        methods = []
        q_values = []
        g_values = []
        for method_id, row in sorted(per_method.items()):
            q = (row.get("qwen_subset_metrics") or {}).get(key)
            g = (row.get("gpt_oss_subset_metrics") or {}).get(key)
            if isinstance(q, (int, float)) and isinstance(g, (int, float)):
                methods.append(method_id)
                q_values.append(float(q))
                g_values.append(float(g))
        out[key] = {
            "methods": methods,
            "spearman": _spearman(q_values, g_values),
            "kendall_tau": _kendall_tau(q_values, g_values),
        }
    return out


def _pal_trace_deltas(per_method: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if "pal_trace" not in per_method:
        return {}
    baselines = [
        "no_llm_heuristic",
        "text_only_profile",
        "multimodal_rag",
        "long_context_mm_llm",
        "generic_tool_agent",
        "adapted_prior_lifelog",
        "long_context_iso_compute",
    ]
    result: dict[str, Any] = {}
    pal = per_method["pal_trace"]
    for baseline in baselines:
        if baseline not in per_method:
            continue
        row: dict[str, Any] = {}
        base = per_method[baseline]
        for key in CORR_KEYS:
            q_delta = _num_delta((pal["qwen_subset_metrics"] or {}).get(key), (base["qwen_subset_metrics"] or {}).get(key))
            g_delta = _num_delta((pal["gpt_oss_subset_metrics"] or {}).get(key), (base["gpt_oss_subset_metrics"] or {}).get(key))
            row[key] = {
                "qwen_delta": q_delta,
                "gpt_oss_delta": g_delta,
                "same_direction": (
                    None
                    if q_delta is None or g_delta is None
                    else (q_delta >= 0 and g_delta >= 0) or (q_delta <= 0 and g_delta <= 0)
                ),
            }
        result[baseline] = row
    return result


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(summary: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Phase 1C Judge Robustness Summary",
        "",
        f"- Users: {', '.join(summary.get('manifest', {}).get('subset_users') or [])}",
        f"- Second judge: {summary.get('manifest', {}).get('judge_model')}",
        f"- Methods: {summary.get('n_methods')}",
        "",
        "## Subset Metrics",
        "",
        "| Method | Qwen OFR | GPT-OSS OFR | Qwen PIR | GPT-OSS PIR | Qwen EFS | GPT-OSS EFS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method_id} | {qwen_OFR} | {gpt_oss_OFR} | {qwen_PIR} | {gpt_oss_PIR} | {qwen_EFS} | {gpt_oss_EFS} |".format(
                **row
            )
        )
    lines.extend(["", "## Per-Method Agreement", ""])
    lines.append("| Method | OFR rho | EFS rho | Correctness kappa | EFS kappa | Correctness MAE | EFS MAE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method_id, row in sorted((summary.get("per_method") or {}).items()):
        corr = row.get("per_user_metric_correlation") or {}
        agree = ((row.get("target_level_agreement") or {}).get("overall") or {})
        correctness = agree.get("correctness") or {}
        efs = agree.get("EFS") or {}
        lines.append(
            "| {method} | {ofr_rho} | {efs_rho} | {ck} | {ek} | {cmae} | {emae} |".format(
                method=method_id,
                ofr_rho=_fmt_num((corr.get("OFR") or {}).get("spearman")),
                efs_rho=_fmt_num((corr.get("EFS") or {}).get("spearman")),
                ck=_fmt_num(correctness.get("weighted_kappa")),
                ek=_fmt_num(efs.get("weighted_kappa")),
                cmae=_fmt_num(correctness.get("mae")),
                emae=_fmt_num(efs.get("mae")),
            )
        )
    lines.extend(["", "## Ranking Stability", ""])
    lines.append("| Metric | Method-rank Spearman | Method-rank Kendall tau |")
    lines.append("|---|---:|---:|")
    for key, row in (summary.get("ranking_stability") or {}).items():
        lines.append(f"| {key} | {_fmt_num(row.get('spearman'))} | {_fmt_num(row.get('kendall_tau'))} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy), 4)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _kendall_tau(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    concordant = 0
    discordant = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = _sign(xs[i] - xs[j])
            dy = _sign(ys[i] - ys[j])
            if dx == 0 or dy == 0:
                continue
            if dx == dy:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return None
    return round((concordant - discordant) / total, 4)


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _mae(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or not xs:
        return None
    return round(sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs), 4)


def _weighted_kappa(pairs: list[tuple[float, float]]) -> float | None:
    if not pairs:
        return None
    cats = [0.0, 0.5, 1.0]
    matrix = [[0.0 for _ in cats] for _ in cats]
    for x, y in pairs:
        i = _nearest_cat(x, cats)
        j = _nearest_cat(y, cats)
        matrix[i][j] += 1.0
    n = sum(sum(row) for row in matrix)
    if n <= 0:
        return None
    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(matrix[i][j] for i in range(len(cats))) for j in range(len(cats))]
    observed = 0.0
    expected = 0.0
    denom = (len(cats) - 1) ** 2
    for i in range(len(cats)):
        for j in range(len(cats)):
            weight = ((i - j) ** 2) / denom
            observed += weight * matrix[i][j] / n
            expected += weight * (row_sums[i] * col_sums[j]) / (n * n)
    if expected <= 0:
        return None
    return round(1.0 - observed / expected, 4)


def _nearest_cat(value: float, cats: list[float]) -> int:
    return min(range(len(cats)), key=lambda i: abs(float(value) - cats[i]))


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _fmt_num(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "NA"


def _num_delta(left: Any, right: Any) -> float | None:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return round(float(left) - float(right), 4)
    return None


def _fmt_delta(left: Any, right: Any) -> str:
    value = _num_delta(left, right)
    return "NA" if value is None else f"{value:.4f}"


def _key_name(key: str) -> str:
    return key.replace("-", "_")


if __name__ == "__main__":
    main()

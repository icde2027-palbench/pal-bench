#!/usr/bin/env python3
"""Evaluate one Evidence-Chain Dossier run with formal album metrics v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmark.eval.formal import JsonJudgeCache, MAIN_METRICS, evaluate_formal_paths
from src.llm import create_llm_for_role
from src.utils.io import save_json


def _needs_llm(mode: str) -> bool:
    value = str(mode or "").strip().lower()
    return value in {"llm", "llm_semantic", "semantic_llm", "qwen36", "qwen3.6", "qwen3_6"} or value.startswith("llm_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ECD predictions on one user")
    parser.add_argument("--user", required=True)
    parser.add_argument("--users-root", default="data/full/users")
    parser.add_argument("--runs-root", default="outputs/evidence_chain_dossier_runs")
    parser.add_argument("--judge-mode", default="lexical_dev")
    parser.add_argument("--evidence-judge-mode", default="heuristic_dev")
    parser.add_argument("--judge-role", default="agent_llm")
    parser.add_argument("--judge-cache", default=None)
    args = parser.parse_args()

    user_id = args.user
    users_root = Path(args.users_root)
    runs_root = Path(args.runs_root)
    predicted = runs_root / user_id / "predicted_profile.json"
    gt = users_root / user_id / f"{user_id}_eval_gt.json"
    album = users_root / user_id / f"{user_id}_agent_album.json"
    stats = runs_root / user_id / "run_stats.json"
    if not predicted.exists():
        print(f"ERROR: Missing prediction: {predicted}", file=sys.stderr)
        sys.exit(1)
    if not gt.exists():
        print(f"ERROR: Missing ground truth: {gt}", file=sys.stderr)
        sys.exit(1)

    use_llm = _needs_llm(args.judge_mode) or _needs_llm(args.evidence_judge_mode)
    llm = create_llm_for_role(args.judge_role) if use_llm else None
    judge_cache = JsonJudgeCache(args.judge_cache or (runs_root / "formal_judge_cache.json")) if use_llm else None
    report = evaluate_formal_paths(
        predicted_profile_path=predicted,
        eval_gt_path=gt,
        agent_album_path=album,
        run_stats_path=stats if stats.exists() else None,
        judge_mode=args.judge_mode,
        evidence_judge_mode=args.evidence_judge_mode,
        llm=llm,
        judge_cache=judge_cache,
    )
    if judge_cache is not None:
        judge_cache.save()
    report_path = runs_root / user_id / "eval_report.json"
    save_json(report, report_path)
    save_json(report, runs_root / user_id / "formal_eval_report.json")
    metrics = report.get("metrics") or {}
    print(f"[ECD Eval] {user_id}")
    for key in MAIN_METRICS:
        if key in metrics:
            value = metrics[key]
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()

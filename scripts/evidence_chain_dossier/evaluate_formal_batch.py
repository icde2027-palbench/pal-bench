#!/usr/bin/env python3
"""Evaluate Evidence-Chain Dossier runs with formal album metrics v2."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmark.eval.formal import (
    JsonJudgeCache,
    MAIN_METRICS,
    aggregate_formal_reports,
    evaluate_formal_paths,
    write_formal_markdown,
)
from src.llm import create_llm_for_role
from src.utils.io import save_json


def _needs_llm(mode: str) -> bool:
    value = str(mode or "").strip().lower()
    return value in {"llm", "llm_semantic", "semantic_llm", "qwen36", "qwen3.6", "qwen3_6"} or value.startswith("llm_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal v2 evaluation on an ECD batch output root.")
    parser.add_argument("--runs-root", required=True, help="Run root containing user_*/predicted_profile.json")
    parser.add_argument("--users-root", default="data/full/users")
    parser.add_argument("--users", default=None, help="Optional comma-separated user ids")
    parser.add_argument("--output-dir", default=None, help="Defaults to <runs-root>/formal_eval")
    parser.add_argument("--judge-mode", default="lexical_dev", help="Reported value judge mode")
    parser.add_argument("--evidence-judge-mode", default="heuristic_dev", help="Reported EFS judge mode")
    parser.add_argument("--judge-role", default="agent_llm", help="LLM role for llm_semantic judge modes")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent users for LLM judge evaluation")
    parser.add_argument("--judge-cache", default=None, help="JSON cache path for LLM judge calls")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    users_root = Path(args.users_root)
    output_dir = Path(args.output_dir) if args.output_dir else runs_root / "formal_eval"
    wanted = {u.strip() for u in args.users.split(",") if u.strip()} if args.users else None

    use_llm = _needs_llm(args.judge_mode) or _needs_llm(args.evidence_judge_mode)
    llm = create_llm_for_role(args.judge_role) if use_llm else None
    judge_cache = JsonJudgeCache(args.judge_cache or (output_dir / "judge_cache.json")) if use_llm else None

    jobs = []
    for profile_path in sorted(runs_root.glob("user_*/predicted_profile.json")):
        user_id = profile_path.parent.name
        if wanted and user_id not in wanted:
            continue
        gt_path = users_root / user_id / f"{user_id}_eval_gt.json"
        album_path = users_root / user_id / f"{user_id}_agent_album.json"
        stats_path = profile_path.parent / "run_stats.json"
        if not gt_path.exists():
            print(f"[skip] missing GT for {user_id}: {gt_path}", file=sys.stderr)
            continue
        if not album_path.exists():
            print(f"[warn] missing agent album for {user_id}: {album_path}", file=sys.stderr)
        jobs.append((user_id, profile_path, gt_path, album_path, stats_path))

    def evaluate_job(job):
        user_id, profile_path, gt_path, album_path, stats_path = job
        report_path = output_dir / "per_user" / user_id / "formal_eval_report.json"
        return evaluate_formal_paths(
            predicted_profile_path=profile_path,
            eval_gt_path=gt_path,
            agent_album_path=album_path if album_path.exists() else None,
            run_stats_path=stats_path if stats_path.exists() else None,
            output_path=report_path,
            judge_mode=args.judge_mode,
            evidence_judge_mode=args.evidence_judge_mode,
            llm=llm,
            judge_cache=judge_cache,
        )

    reports = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for job in jobs:
            report = evaluate_job(job)
            reports.append(report)
            if judge_cache is not None:
                judge_cache.save()
            print(f"[eval] {report.get('user_id')} done", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_job, job): job[0] for job in jobs}
            for future in as_completed(futures):
                user_id = futures[future]
                report = future.result()
                reports.append(report)
                if judge_cache is not None:
                    judge_cache.save()
                print(f"[eval] {user_id} done", file=sys.stderr)

    if not reports:
        raise SystemExit("No formal reports were generated.")
    reports.sort(key=lambda report: str(report.get("user_id") or ""))

    aggregate = aggregate_formal_reports(
        reports,
        runs_root=runs_root,
        users_root=users_root,
        judge_mode=args.judge_mode,
        evidence_judge_mode=args.evidence_judge_mode,
    )
    if judge_cache is not None:
        judge_cache.save()
        aggregate["judge_cache"] = judge_cache.stats()
    save_json(aggregate, output_dir / "aggregate_formal.json")
    write_formal_markdown(aggregate, output_dir / "formal_metrics_summary.md")

    print(f"[Formal Eval v2] users={len(reports)}")
    for key in MAIN_METRICS:
        value = aggregate["metrics"].get(key)
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"  JSON: {output_dir / 'aggregate_formal.json'}")
    print(f"  MD:   {output_dir / 'formal_metrics_summary.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit formal ECD failures across users.

This script is deliberately diagnostic-only.  It reads existing formal eval
reports and optional ECD intermediate files, then separates metric loss into
owner recall, identity discovery/binding, relation/category, and evidence
faithfulness buckets.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit formal ECD failures.")
    parser.add_argument("--formal-root", required=True, help="Formal eval output root containing per_user/")
    parser.add_argument("--runs-root", required=True, help="Run root containing user_*/predicted_profile.json")
    parser.add_argument("--users-root", default="data/full/users")
    parser.add_argument("--diagnostics-root", default=None, help="Optional run root with identity_hypotheses/name_clusters")
    parser.add_argument("--users", default=None, help="Optional comma-separated user ids")
    parser.add_argument("--output-dir", default=None, help="Defaults to <formal-root>/failure_audit")
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    formal_root = Path(args.formal_root)
    runs_root = Path(args.runs_root)
    users_root = Path(args.users_root)
    diagnostics_root = Path(args.diagnostics_root) if args.diagnostics_root else runs_root
    output_dir = Path(args.output_dir) if args.output_dir else formal_root / "failure_audit"
    wanted = {u.strip() for u in args.users.split(",") if u.strip()} if args.users else None

    reports = []
    for report_path in sorted((formal_root / "per_user").glob("user_*/formal_eval_report.json")):
        user_id = report_path.parent.name
        if wanted and user_id not in wanted:
            continue
        reports.append(
            _audit_user(
                user_id=user_id,
                report_path=report_path,
                run_dir=runs_root / user_id,
                user_dir=users_root / user_id,
                diagnostics_dir=diagnostics_root / user_id,
            )
        )

    if not reports:
        raise SystemExit(f"No formal reports found under {formal_root / 'per_user'}")

    aggregate = _aggregate_reports(reports, max_examples=args.max_examples)
    result = {
        "schema_version": "ecd_failure_audit.v1",
        "formal_root": str(formal_root),
        "runs_root": str(runs_root),
        "users_root": str(users_root),
        "diagnostics_root": str(diagnostics_root),
        "n_users": len(reports),
        "aggregate": aggregate,
        "users": reports,
    }
    save_json(result, output_dir / "failure_audit.json")
    _write_markdown(result, output_dir / "failure_audit.md")
    print(f"[audit] users={len(reports)}")
    print(f"  JSON: {output_dir / 'failure_audit.json'}")
    print(f"  MD:   {output_dir / 'failure_audit.md'}")


def _audit_user(
    *,
    user_id: str,
    report_path: Path,
    run_dir: Path,
    user_dir: Path,
    diagnostics_dir: Path,
) -> dict[str, Any]:
    report = load_json(report_path)
    profile = load_json(run_dir / "predicted_profile.json") if (run_dir / "predicted_profile.json").exists() else {}
    gt = load_json(user_dir / f"{user_id}_eval_gt.json") if (user_dir / f"{user_id}_eval_gt.json").exists() else {}
    hypotheses = _load_list(diagnostics_dir / "identity_hypotheses.json")
    clusters = _load_list(diagnostics_dir / "name_clusters.json")
    diagnostics = load_json(diagnostics_dir / "diagnostics.json") if (diagnostics_dir / "diagnostics.json").exists() else {}

    target_rows = list(report.get("targets") or [])
    owner_rows = [r for r in target_rows if r.get("target_type") == "owner_fact_atom"]
    person_name_rows = [r for r in target_rows if r.get("target_type") == "person_name"]
    person_relation_rows = [r for r in target_rows if r.get("target_type") == "person_relation"]
    person_category_rows = [r for r in target_rows if r.get("target_type") == "person_category"]

    owner_audit = _audit_owner_rows(owner_rows)
    identity_audit = _audit_identity_rows(
        person_name_rows=person_name_rows,
        profile=profile,
        gt=gt,
        hypotheses=hypotheses,
        clusters=clusters,
        diagnostics=diagnostics,
    )
    person_value_audit = _audit_person_value_rows(person_relation_rows, person_category_rows)

    return {
        "user_id": user_id,
        "metrics": report.get("metrics") or {},
        "diagnostics": report.get("diagnostics") or {},
        "owner": owner_audit,
        "identity": identity_audit,
        "person_values": person_value_audit,
    }


def _audit_owner_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_examples = []
    for row in rows:
        by_type[str(row.get("inference_type") or "unknown")].append(row)
        correctness = _float(row.get("correctness"))
        if correctness < 1.0:
            failure_examples.append(
                {
                    "target_id": row.get("target_id"),
                    "inference_type": row.get("inference_type") or "unknown",
                    "gt_value": row.get("gt_value"),
                    "prediction": row.get("prediction"),
                    "correctness": correctness,
                    "EFS": _float(row.get("EFS")),
                    "reason": _owner_failure_reason(row),
                    "key_photo_ids_public": row.get("key_photo_ids_public") or [],
                }
            )
    by_type_summary = {}
    for key, items in sorted(by_type.items()):
        by_type_summary[key] = {
            "n": len(items),
            "correctness": _mean(_float(r.get("correctness")) for r in items),
            "EFS": _mean(_float(r.get("EFS")) for r in items),
            "missing_rate": _mean(1.0 if not str(r.get("prediction") or "").strip() else 0.0 for r in items),
            "partial_rate": _mean(1.0 if 0.0 < _float(r.get("correctness")) < 1.0 else 0.0 for r in items),
        }
    return {
        "n_targets": len(rows),
        "correctness": _mean(_float(r.get("correctness")) for r in rows),
        "EFS": _mean(_float(r.get("EFS")) for r in rows),
        "by_inference_type": by_type_summary,
        "failure_reason_counts": dict(Counter(ex["reason"] for ex in failure_examples)),
        "failure_examples": failure_examples,
    }


def _owner_failure_reason(row: dict[str, Any]) -> str:
    prediction = str(row.get("prediction") or "").strip()
    correctness = _float(row.get("correctness"))
    efs = _float(row.get("EFS"))
    if not prediction:
        return "missing_prediction"
    if correctness <= 0.0:
        return "semantic_mismatch"
    if correctness < 1.0:
        return "partial_semantic_match"
    if efs < 1.0:
        return "evidence_weak"
    return "other"


def _audit_identity_rows(
    *,
    person_name_rows: list[dict[str, Any]],
    profile: dict[str, Any],
    gt: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    pred_by_face = {
        str(row.get("face_id") or ""): row
        for row in profile.get("persons") or []
        if str(row.get("face_id") or "")
    }
    gt_name_targets = {
        str(t.get("target_id") or ""): t
        for t in gt.get("evaluation_targets") or []
        if str(t.get("target_type") or "") == "person_name"
    }
    cluster_by_id = {str(c.get("cluster_id") or ""): c for c in clusters}
    hyps_by_face: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hyp in hypotheses:
        hyps_by_face[str(hyp.get("face_id") or "")].append(hyp)
    for rows in hyps_by_face.values():
        rows.sort(key=lambda h: (-_float(h.get("score")), str(h.get("name_cluster_id") or "")))
    rejected = _resolver_rejections(diagnostics)

    rows = []
    reason_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    for row in person_name_rows:
        target_id = str(row.get("target_id") or "")
        target = gt_name_targets.get(target_id, {})
        face_id = str(row.get("public_face_id") or target.get("public_face_id") or "")
        gt_name = str(row.get("gt_value") or target.get("gt_value") or target.get("canonical_name") or "")
        aliases = [gt_name, *(str(a) for a in target.get("aliases") or [])]
        pred = pred_by_face.get(face_id) or {}
        face_hyps = hyps_by_face.get(face_id, [])
        candidate = _matching_hypothesis(face_hyps, aliases)
        cluster_present = _matching_cluster(clusters, aliases)
        discovered_face = str((row.get("judge_details") or {}).get("discovered_face_id") or "")
        candidate_state = _candidate_state(candidate, cluster_present)
        candidate_counts[candidate_state] += 1
        reason = _identity_failure_reason(row, pred, discovered_face, candidate, cluster_present)
        reason_counts[reason] += 1
        rows.append(
            {
                "target_id": target_id,
                "person_id": row.get("person_id"),
                "face_id": face_id,
                "gt_name": gt_name,
                "alignment_difficulty": row.get("alignment_difficulty"),
                "prediction": row.get("prediction") or "",
                "predicted_at_target_face": pred.get("canonical_name") or "",
                "correctness": _float(row.get("correctness")),
                "EFS": _float(row.get("EFS")),
                "discovered_face_id": discovered_face,
                "match_mode": (row.get("judge_details") or {}).get("match_mode") or "",
                "failure_reason": reason,
                "candidate_state": candidate_state,
                "candidate_rank": _hyp_rank(face_hyps, candidate),
                "candidate_score": _float(candidate.get("score")) if candidate else None,
                "candidate_margin": _float(candidate.get("margin")) if candidate else None,
                "candidate_status": candidate.get("status") if candidate else "",
                "top_candidate_name": _hyp_name(face_hyps[0], cluster_by_id) if face_hyps else "",
                "top_candidate_score": _float(face_hyps[0].get("score")) if face_hyps else None,
                "cluster_id": candidate.get("name_cluster_id") if candidate else "",
                "resolver_reject_reason": rejected.get((face_id, str(candidate.get("name_cluster_id") if candidate else "")), ""),
            }
        )

    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_difficulty[str(row.get("alignment_difficulty") or "unknown")].append(row)
    return {
        "n_targets": len(rows),
        "correctness": _mean(row["correctness"] for row in rows),
        "EFS": _mean(row["EFS"] for row in rows),
        "failure_reason_counts": dict(reason_counts),
        "candidate_state_counts": dict(candidate_counts),
        "by_alignment_difficulty": {
            key: {
                "n": len(items),
                "correctness": _mean(item["correctness"] for item in items),
                "candidate_present_rate": _mean(1.0 if item["candidate_state"] == "face_name_candidate_present" else 0.0 for item in items),
                "wrong_face_rate": _mean(1.0 if item["failure_reason"] == "wrong_face_assignment" else 0.0 for item in items),
            }
            for key, items in sorted(by_difficulty.items())
        },
        "failure_examples": [row for row in rows if row["correctness"] < 1.0],
    }


def _audit_person_value_rows(relation_rows: list[dict[str, Any]], category_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(rows),
            "correctness": _mean(_float(row.get("correctness")) for row in rows),
            "EFS": _mean(_float(row.get("EFS")) for row in rows),
            "identity_bound_false_rate": _mean(
                1.0 if not bool((row.get("judge_details") or {}).get("identity_bound", False)) else 0.0
                for row in rows
            ),
            "value_correct_given_id": _mean_optional(
                _float((row.get("judge_details") or {}).get("raw_relation_score", (row.get("judge_details") or {}).get("raw_category_score", 0.0)))
                for row in rows
                if bool((row.get("judge_details") or {}).get("identity_bound", False))
            ),
        }

    examples = []
    for row in [*relation_rows, *category_rows]:
        if _float(row.get("correctness")) < 1.0:
            examples.append(
                {
                    "target_id": row.get("target_id"),
                    "target_type": row.get("target_type"),
                    "person_id": row.get("person_id"),
                    "public_face_id": row.get("public_face_id"),
                    "gt_value": row.get("gt_value"),
                    "prediction": row.get("prediction"),
                    "correctness": _float(row.get("correctness")),
                    "EFS": _float(row.get("EFS")),
                    "identity_bound": bool((row.get("judge_details") or {}).get("identity_bound", False)),
                }
            )
    return {
        "relation": summarize(relation_rows),
        "category": summarize(category_rows),
        "failure_examples": examples,
    }


def _aggregate_reports(reports: list[dict[str, Any]], *, max_examples: int) -> dict[str, Any]:
    metrics = defaultdict(list)
    diagnostics = defaultdict(list)
    for report in reports:
        for key, value in (report.get("metrics") or {}).items():
            if isinstance(value, (int, float)):
                metrics[key].append(float(value))
        for key, value in (report.get("diagnostics") or {}).items():
            if isinstance(value, (int, float)):
                diagnostics[key].append(float(value))

    owner_by_type: dict[str, list[dict[str, float]]] = defaultdict(list)
    owner_reasons: Counter[str] = Counter()
    identity_reasons: Counter[str] = Counter()
    candidate_states: Counter[str] = Counter()
    identity_by_diff: dict[str, list[dict[str, float]]] = defaultdict(list)
    user_rows = []
    owner_examples = []
    identity_examples = []
    person_value_examples = []

    for report in reports:
        user_rows.append(
            {
                "user_id": report["user_id"],
                "OFR": report["metrics"].get("OFR"),
                "PIR": report["metrics"].get("PIR"),
                "EFS": report["metrics"].get("EFS"),
                "IDR": report["diagnostics"].get("IDR"),
                "IBR": report["diagnostics"].get("IBR"),
                "WrongFaceRate": report["diagnostics"].get("WrongFaceRate"),
                "BlankNameRate": report["diagnostics"].get("BlankNameRate"),
            }
        )
        for key, value in report["owner"]["by_inference_type"].items():
            owner_by_type[key].append(value)
        owner_reasons.update(report["owner"]["failure_reason_counts"])
        identity_reasons.update(report["identity"]["failure_reason_counts"])
        candidate_states.update(report["identity"]["candidate_state_counts"])
        for key, value in report["identity"]["by_alignment_difficulty"].items():
            identity_by_diff[key].append(value)
        owner_examples.extend({**ex, "user_id": report["user_id"]} for ex in report["owner"]["failure_examples"])
        identity_examples.extend({**ex, "user_id": report["user_id"]} for ex in report["identity"]["failure_examples"])
        person_value_examples.extend({**ex, "user_id": report["user_id"]} for ex in report["person_values"]["failure_examples"])

    owner_type_summary = {
        key: {
            "n": int(sum(item["n"] for item in items)),
            "correctness": _mean(item["correctness"] for item in items),
            "EFS": _mean(item["EFS"] for item in items),
            "missing_rate": _mean(item["missing_rate"] for item in items),
            "partial_rate": _mean(item["partial_rate"] for item in items),
        }
        for key, items in sorted(owner_by_type.items())
    }
    identity_diff_summary = {
        key: {
            "n": int(sum(item["n"] for item in items)),
            "correctness": _mean(item["correctness"] for item in items),
            "candidate_present_rate": _mean(item["candidate_present_rate"] for item in items),
            "wrong_face_rate": _mean(item["wrong_face_rate"] for item in items),
        }
        for key, items in sorted(identity_by_diff.items())
    }

    return {
        "macro_metrics": {key: _mean(values) for key, values in sorted(metrics.items())},
        "macro_diagnostics": {key: _mean(values) for key, values in sorted(diagnostics.items())},
        "owner_by_inference_type": owner_type_summary,
        "owner_failure_reason_counts": dict(owner_reasons),
        "identity_failure_reason_counts": dict(identity_reasons),
        "identity_candidate_state_counts": dict(candidate_states),
        "identity_by_alignment_difficulty": identity_diff_summary,
        "lowest_pir_users": sorted(user_rows, key=lambda r: _float(r.get("PIR")))[:max_examples],
        "lowest_ofr_users": sorted(user_rows, key=lambda r: _float(r.get("OFR")))[:max_examples],
        "owner_failure_examples": _rank_owner_examples(owner_examples)[:max_examples],
        "identity_failure_examples": _rank_identity_examples(identity_examples)[:max_examples],
        "person_value_failure_examples": person_value_examples[:max_examples],
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    agg = result["aggregate"]
    lines = [
        "# ECD Failure Audit",
        "",
        f"- Formal root: `{result['formal_root']}`",
        f"- Runs root: `{result['runs_root']}`",
        f"- Users: `{result['n_users']}`",
        "",
        "## Main Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ["OFR", "PIR", "PIR-hard", "PRR-ID", "EFS", "ECE", "nLLM"]:
        lines.append(f"| {key} | {_fmt((agg['macro_metrics'] or {}).get(key))} |")

    lines.extend(["", "## Owner By Inference Type", "", "| Type | n | Correctness | EFS | Missing | Partial |", "|---|---:|---:|---:|---:|---:|"])
    for key, row in agg["owner_by_inference_type"].items():
        lines.append(
            f"| {key} | {row['n']} | {_fmt(row['correctness'])} | {_fmt(row['EFS'])} | "
            f"{_fmt(row['missing_rate'])} | {_fmt(row['partial_rate'])} |"
        )

    lines.extend(["", "## Identity Bottlenecks", "", "| Bucket | Count |", "|---|---:|"])
    for key, value in sorted(agg["identity_failure_reason_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "| Candidate State | Count |", "|---|---:|"])
    for key, value in sorted(agg["identity_candidate_state_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "| Difficulty | n | Correctness | Candidate Present | Wrong Face |", "|---|---:|---:|---:|---:|"])
    for key, row in agg["identity_by_alignment_difficulty"].items():
        lines.append(
            f"| {key} | {row['n']} | {_fmt(row['correctness'])} | "
            f"{_fmt(row['candidate_present_rate'])} | {_fmt(row['wrong_face_rate'])} |"
        )

    lines.extend(["", "## Lowest PIR Users", "", "| User | PIR | OFR | EFS | IDR | IBR | WrongFace |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in agg["lowest_pir_users"][:12]:
        lines.append(
            f"| {row['user_id']} | {_fmt(row.get('PIR'))} | {_fmt(row.get('OFR'))} | {_fmt(row.get('EFS'))} | "
            f"{_fmt(row.get('IDR'))} | {_fmt(row.get('IBR'))} | {_fmt(row.get('WrongFaceRate'))} |"
        )

    lines.extend(["", "## Representative Identity Failures", "", "| User | Face | GT | Pred | Reason | Candidate | Rank | Score |", "|---|---|---|---|---|---|---:|---:|"])
    for row in agg["identity_failure_examples"][:15]:
        lines.append(
            f"| {row.get('user_id')} | {row.get('face_id')} | {_md(row.get('gt_name'))} | "
            f"{_md(row.get('prediction') or row.get('predicted_at_target_face'))} | {row.get('failure_reason')} | "
            f"{row.get('candidate_state')} | {_fmt(row.get('candidate_rank'))} | {_fmt(row.get('candidate_score'))} |"
        )

    lines.extend(["", "## Representative Owner Failures", "", "| User | Type | GT | Prediction | Reason |", "|---|---|---|---|---|"])
    for row in agg["owner_failure_examples"][:15]:
        lines.append(
            f"| {row.get('user_id')} | {row.get('inference_type')} | {_md(row.get('gt_value'))} | "
            f"{_md(row.get('prediction'))} | {row.get('reason')} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _matching_hypothesis(rows: list[dict[str, Any]], aliases: list[str]) -> dict[str, Any] | None:
    for hyp in rows:
        names = [str(hyp.get("canonical_name_candidate") or ""), str(hyp.get("observed_surface") or "")]
        for name in names:
            if _name_matches_any(name, aliases):
                return hyp
    return None


def _matching_cluster(clusters: list[dict[str, Any]], aliases: list[str]) -> dict[str, Any] | None:
    for cluster in clusters:
        names = [
            str(cluster.get("primary_surface") or ""),
            *(str(surface) for surface in cluster.get("surfaces") or []),
        ]
        for name in names:
            if _name_matches_any(name, aliases):
                return cluster
    return None


def _name_matches_any(name: str, aliases: list[str]) -> bool:
    norm = _norm_name(name)
    if not norm:
        return False
    for alias in aliases:
        alias_norm = _norm_name(alias)
        if not alias_norm:
            continue
        if norm == alias_norm:
            return True
        name_tokens = norm.split()
        alias_tokens = alias_norm.split()
        if len(name_tokens) == 1 and name_tokens[0] == alias_tokens[0]:
            return True
    return False


def _identity_failure_reason(
    row: dict[str, Any],
    pred_at_face: dict[str, Any],
    discovered_face: str,
    candidate: dict[str, Any] | None,
    cluster_present: dict[str, Any] | None,
) -> str:
    if _float(row.get("correctness")) >= 1.0:
        return "correct"
    target_face = str(row.get("public_face_id") or "")
    pred_name = str(pred_at_face.get("canonical_name") or row.get("prediction") or "").strip()
    if discovered_face and discovered_face != target_face:
        return "wrong_face_assignment"
    if not pred_name:
        return "blank_target_face"
    if candidate is not None:
        return "candidate_present_not_selected"
    if cluster_present is not None:
        return "name_cluster_present_no_face_edge"
    return "name_cluster_missing"


def _candidate_state(candidate: dict[str, Any] | None, cluster_present: dict[str, Any] | None) -> str:
    if candidate is not None:
        return "face_name_candidate_present"
    if cluster_present is not None:
        return "name_cluster_present_no_face_edge"
    return "name_cluster_missing"


def _resolver_rejections(diagnostics: dict[str, Any]) -> dict[tuple[str, str], str]:
    out = {}
    for row in ((diagnostics.get("resolver") or {}).get("rejected") or []):
        out[(str(row.get("face_id") or ""), str(row.get("name_cluster_id") or ""))] = str(row.get("reject_reason") or "")
    return out


def _hyp_rank(rows: list[dict[str, Any]], hyp: dict[str, Any] | None) -> int | None:
    if not hyp:
        return None
    target_cluster = str(hyp.get("name_cluster_id") or "")
    for idx, row in enumerate(rows, start=1):
        if str(row.get("name_cluster_id") or "") == target_cluster:
            return idx
    return None


def _hyp_name(hyp: dict[str, Any], cluster_by_id: dict[str, dict[str, Any]]) -> str:
    cluster = cluster_by_id.get(str(hyp.get("name_cluster_id") or "")) or {}
    return str(hyp.get("canonical_name_candidate") or hyp.get("observed_surface") or cluster.get("primary_surface") or "")


def _rank_owner_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {"missing_prediction": 0, "semantic_mismatch": 1, "partial_semantic_match": 2, "evidence_weak": 3}
    return sorted(rows, key=lambda r: (priority.get(str(r.get("reason")), 9), str(r.get("inference_type")), str(r.get("user_id"))))


def _rank_identity_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        "wrong_face_assignment": 0,
        "candidate_present_not_selected": 1,
        "name_cluster_present_no_face_edge": 2,
        "name_cluster_missing": 3,
        "blank_target_face": 4,
    }
    return sorted(rows, key=lambda r: (priority.get(str(r.get("failure_reason")), 9), _float(r.get("candidate_rank") or 999), str(r.get("user_id"))))


def _load_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    value = load_json(path)
    return value if isinstance(value, list) else []


def _norm_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z]+", str(value or "").lower()))


def _mean(values: Any) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return round(mean(vals), 4) if vals else 0.0


def _mean_optional(values: Any) -> float | None:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return round(mean(vals), 4) if vals else None


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _md(value: Any, limit: int = 80) -> str:
    text = " ".join(str(value or "").split())
    text = text.replace("|", "\\|")
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


if __name__ == "__main__":
    main()

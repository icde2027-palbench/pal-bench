#!/usr/bin/env python3
"""Corrected metric audit for Evidence-Chain Dossier batch runs.

This script intentionally does not replace the legacy evaluator.  It produces a
diagnostic metric suite that separates name discovery, face-name binding,
identity-conditioned relation recovery, hierarchical owner fact recovery, and
pooled calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmark.eval.judge import lexical_judge
from src.utils.io import load_json, save_json


WORD_RE = re.compile(r"[a-z0-9]+")

PRIMARY_METRICS = [
    "OFR_parent_lexical",
    "OFR_atom_lexical",
    "PIR_strict",
    "PIR_soft",
    "PIR_hard_strict",
    "NameDiscovery_full",
    "NameDiscovery_first",
    "BindingAcc_given_discovered_full",
    "WrongFaceNameRate",
    "BlankNameRate",
    "PRR_soft",
    "PRR_identity_conditioned",
    "PRR_given_identity",
    "PCR_strict",
    "PCR_identity_conditioned",
    "PCR_given_identity",
    "EvidenceGroundedRate",
    "ReasoningPresenceRate",
    "KeyPhotoF1_overlap",
    "ECE_observed_strict",
    "ECE_observed_soft",
    "ECE_coverage_strict",
    "ECE_coverage_soft",
    "n_users",
    "n_person_targets",
    "n_owner_atom_targets",
]


RELATION_SYNONYMS = {
    "mom": "mother",
    "mum": "mother",
    "mother": "mother",
    "dad": "father",
    "father": "father",
    "sister": "sibling",
    "brother": "sibling",
    "sibling": "sibling",
    "daughter": "child",
    "son": "child",
    "child": "child",
    "parent": "parent",
    "wife": "partner",
    "husband": "partner",
    "spouse": "partner",
    "partner": "partner",
    "boyfriend": "partner",
    "girlfriend": "partner",
    "fiance": "partner",
    "fiancee": "partner",
    "significant other": "partner",
    "coworker": "colleague",
    "co worker": "colleague",
    "colleague": "colleague",
    "work colleague": "colleague",
    "professional contact": "colleague",
    "classmate": "classmate",
    "school friend": "classmate",
    "neighbor": "neighbor",
    "neighbour": "neighbor",
    "friend": "friend",
    "close friend": "friend",
    "best friend": "friend",
    "old friend": "friend",
    "aunt": "extended family",
    "uncle": "extended family",
    "cousin": "extended family",
    "grandmother": "extended family",
    "grandfather": "extended family",
    "grandparent": "extended family",
}


RELATION_TO_CATEGORY = {
    "mother": "family",
    "father": "family",
    "parent": "family",
    "sibling": "family",
    "child": "family",
    "partner": "family",
    "extended family": "family",
    "colleague": "colleague",
    "classmate": "classmate",
    "neighbor": "neighbor",
    "friend": "friend",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run corrected ECD metric audit on a batch output root.")
    parser.add_argument("--runs-root", required=True, help="ECD run root containing user_*/predicted_profile.json")
    parser.add_argument("--users-root", default="data/full/users")
    parser.add_argument("--users", default=None, help="Optional comma-separated user ids")
    parser.add_argument("--output-dir", default=None, help="Defaults to <runs-root>/corrected_eval")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    users_root = Path(args.users_root)
    output_dir = Path(args.output_dir) if args.output_dir else runs_root / "corrected_eval"
    wanted = {u.strip() for u in args.users.split(",") if u.strip()} if args.users else None

    rows = []
    target_rows = []
    for profile_path in sorted(runs_root.glob("user_*/predicted_profile.json")):
        user_id = profile_path.parent.name
        if wanted and user_id not in wanted:
            continue
        gt_path = users_root / user_id / f"{user_id}_eval_gt.json"
        stats_path = profile_path.parent / "run_stats.json"
        if not gt_path.exists():
            print(f"[skip] missing GT for {user_id}: {gt_path}", file=sys.stderr)
            continue
        profile = load_json(profile_path)
        gt = load_json(gt_path)
        run_stats = load_json(stats_path) if stats_path.exists() else profile.get("run_stats", {})
        user_report = evaluate_user(user_id, profile, gt, run_stats)
        rows.append(user_report["metrics"])
        target_rows.extend(user_report["targets"])
        public_report = {
            **user_report,
            "metrics": {
                key: value
                for key, value in user_report["metrics"].items()
                if not key.startswith("_")
            },
        }
        save_json(public_report, output_dir / "per_user" / user_id / "corrected_eval_report.json")

    if not rows:
        raise SystemExit("No user reports were generated.")

    aggregate = aggregate_reports(rows, target_rows)
    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    result = {
        "schema_version": "ecd_metric_audit.v1",
        "runs_root": str(runs_root),
        "users_root": str(users_root),
        "n_users": len(rows),
        "metrics": aggregate,
        "per_user": public_rows,
        "metric_columns": PRIMARY_METRICS,
    }
    save_json(result, output_dir / "aggregate_corrected.json")
    write_markdown(result, output_dir / "corrected_metrics_summary.md")

    print(f"[Corrected Eval] users={len(rows)}")
    for key in PRIMARY_METRICS:
        if key in aggregate:
            value = aggregate[key]
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"  JSON: {output_dir / 'aggregate_corrected.json'}")
    print(f"  MD:   {output_dir / 'corrected_metrics_summary.md'}")


def evaluate_user(user_id: str, profile: dict[str, Any], gt: dict[str, Any], run_stats: dict[str, Any]) -> dict[str, Any]:
    persons = {str(row.get("face_id") or ""): row for row in profile.get("persons") or []}
    predicted_names = [
        str(row.get("canonical_name") or "").strip()
        for row in profile.get("persons") or []
        if str(row.get("canonical_name") or "").strip()
    ]
    predicted_name_norms = {_norm_name(name) for name in predicted_names if _norm_name(name)}
    predicted_firsts = {_first_name(name) for name in predicted_names if _first_name(name)}
    owner_facts = list((profile.get("owner") or {}).get("facts") or [])

    targets_out: list[dict[str, Any]] = []
    owner_atom_scores: list[float] = []
    owner_parent_scores: dict[str, list[float]] = defaultdict(list)
    owner_by_inference: dict[str, list[float]] = defaultdict(list)
    name_scores_strict: list[float] = []
    name_scores_soft: list[float] = []
    name_hard_scores: list[float] = []
    name_by_difficulty: dict[str, list[float]] = defaultdict(list)
    discovery_full: list[float] = []
    discovery_first: list[float] = []
    binding_given_discovered: list[float] = []
    wrong_face_name: list[float] = []
    blank_name: list[float] = []
    rel_scores: list[float] = []
    rel_id_scores: list[float] = []
    rel_given_id: list[float] = []
    cat_scores: list[float] = []
    cat_id_scores: list[float] = []
    cat_given_id: list[float] = []
    grounded: list[float] = []
    reasoning_present: list[float] = []
    evidence_f1: list[float] = []
    target_conf_rows: list[dict[str, float | str | None]] = []
    exact_identity_by_person: dict[str, float] = {}

    targets = gt.get("evaluation_targets") or []
    for target in targets:
        target_type = str(target.get("target_type") or "")
        target_id = str(target.get("target_id") or "")
        target_row: dict[str, Any] = {
            "user_id": user_id,
            "target_id": target_id,
            "target_type": target_type,
            "person_id": target.get("person_id"),
            "alignment_difficulty": target.get("alignment_difficulty"),
            "score_strict": None,
            "score_soft": None,
            "confidence": None,
        }

        if target_type == "owner_fact_atom":
            gt_text = str(target.get("gt_text") or "")
            texts = [str(f.get("text") or "") for f in owner_facts]
            judge = lexical_judge(gt_text, texts)
            score = float(judge.score)
            owner_atom_scores.append(score)
            owner_parent_scores[str(target.get("parent_fact_id") or target_id)].append(score)
            owner_by_inference[str(target.get("inference_type") or "unknown")].append(score)
            matched = owner_facts[judge.matched_index] if judge.matched_index is not None and judge.matched_index < len(owner_facts) else {}
            conf = _confidence(matched)
            evidence = [str(pid) for pid in matched.get("evidence_photo_ids") or []]
            reasoning = str(matched.get("reasoning_path") or "")
            grounded.append(1.0 if evidence else 0.0)
            reasoning_present.append(1.0 if reasoning.strip() else 0.0)
            evidence_f1.append(_evidence_f1(evidence, target.get("key_photo_ids_public") or []))
            _append_conf(target_conf_rows, user_id, target_id, target_type, conf, score, score)
            target_row.update(score_strict=score, score_soft=score, confidence=conf)

        elif target_type == "person_name":
            face_id = str(target.get("public_face_id") or "")
            pred = persons.get(face_id) or {}
            pred_name = str(pred.get("canonical_name") or "").strip()
            aliases = [str(a) for a in target.get("aliases") or [] if str(a).strip()]
            gt_name = str(target.get("gt_value") or "")
            if gt_name and gt_name not in aliases:
                aliases.insert(0, gt_name)
            strict = 1.0 if _name_exact(pred_name, aliases) else 0.0
            partial = 0.5 if strict == 0.0 and _name_partial(pred_name, aliases) else 0.0
            soft = strict or partial
            diff = str(target.get("alignment_difficulty") or "unknown")
            discovered_full = 1.0 if any(alias and _norm_name(alias) in predicted_name_norms for alias in aliases) else 0.0
            discovered_first = 1.0 if any(_first_name(alias) in predicted_firsts for alias in aliases if _first_name(alias)) else 0.0
            wrong_face = 1.0 if discovered_full and not strict else 0.0
            blank = 1.0 if not pred_name else 0.0
            name_scores_strict.append(strict)
            name_scores_soft.append(float(soft))
            name_by_difficulty[diff].append(strict)
            if diff == "hard":
                name_hard_scores.append(strict)
            discovery_full.append(discovered_full)
            discovery_first.append(discovered_first)
            if discovered_full:
                binding_given_discovered.append(strict)
            wrong_face_name.append(wrong_face)
            blank_name.append(blank)
            exact_identity_by_person[str(target.get("person_id") or "")] = strict
            conf = _confidence(pred)
            evidence = [str(pid) for pid in pred.get("evidence_photo_ids") or []]
            reasoning = str(pred.get("reasoning_path") or "")
            grounded.append(1.0 if evidence else 0.0)
            reasoning_present.append(1.0 if reasoning.strip() else 0.0)
            evidence_f1.append(_evidence_f1(evidence, target.get("key_photo_ids_public") or []))
            _append_conf(target_conf_rows, user_id, target_id, target_type, conf, strict, float(soft))
            target_row.update(score_strict=strict, score_soft=float(soft), confidence=conf)

        elif target_type == "person_relation":
            face_id = str(target.get("public_face_id") or "")
            pred = persons.get(face_id) or {}
            gt_relation = str(target.get("gt_value") or "")
            pred_relation = str(pred.get("relation_to_owner") or "")
            gt_category = str(target.get("relation_category") or "")
            pred_category = str(pred.get("relation_category") or "")
            score = _relation_score(gt_relation, pred_relation, gt_category, pred_category)
            identity_ok = exact_identity_by_person.get(str(target.get("person_id") or ""), 0.0)
            rel_scores.append(score)
            rel_id_scores.append(score if identity_ok == 1.0 else 0.0)
            if identity_ok == 1.0:
                rel_given_id.append(score)
            conf = _confidence(pred)
            evidence = [str(pid) for pid in pred.get("evidence_photo_ids") or []]
            reasoning = str(pred.get("reasoning_path") or "")
            grounded.append(1.0 if evidence else 0.0)
            reasoning_present.append(1.0 if reasoning.strip() else 0.0)
            evidence_f1.append(_evidence_f1(evidence, target.get("key_photo_ids_public") or []))
            _append_conf(target_conf_rows, user_id, target_id, target_type, conf, score, score)
            target_row.update(score_strict=score, score_soft=score, confidence=conf)

        elif target_type == "person_category":
            face_id = str(target.get("public_face_id") or "")
            pred = persons.get(face_id) or {}
            gt_category = _norm_category(str(target.get("gt_value") or target.get("relation_category") or ""))
            pred_category = _norm_category(str(pred.get("relation_category") or ""))
            score = 1.0 if gt_category and pred_category == gt_category else 0.0
            identity_ok = exact_identity_by_person.get(str(target.get("person_id") or ""), 0.0)
            cat_scores.append(score)
            cat_id_scores.append(score if identity_ok == 1.0 else 0.0)
            if identity_ok == 1.0:
                cat_given_id.append(score)
            conf = _confidence(pred)
            evidence = [str(pid) for pid in pred.get("evidence_photo_ids") or []]
            reasoning = str(pred.get("reasoning_path") or "")
            grounded.append(1.0 if evidence else 0.0)
            reasoning_present.append(1.0 if reasoning.strip() else 0.0)
            evidence_f1.append(_evidence_f1(evidence, target.get("key_photo_ids_public") or []))
            _append_conf(target_conf_rows, user_id, target_id, target_type, conf, score, score)
            target_row.update(score_strict=score, score_soft=score, confidence=conf)

        targets_out.append(target_row)

    metrics = {
        "user_id": user_id,
        "OFR_parent_lexical": _mean(_mean(v) for v in owner_parent_scores.values()),
        "OFR_atom_lexical": _mean(owner_atom_scores),
        "PIR_strict": _mean(name_scores_strict),
        "PIR_soft": _mean(name_scores_soft),
        "PIR_hard_strict": _mean(name_hard_scores) if name_hard_scores else None,
        "PIR_easy_strict": _mean(name_by_difficulty.get("easy", [])) if name_by_difficulty.get("easy") else None,
        "PIR_medium_strict": _mean(name_by_difficulty.get("medium", [])) if name_by_difficulty.get("medium") else None,
        "NameDiscovery_full": _mean(discovery_full),
        "NameDiscovery_first": _mean(discovery_first),
        "BindingAcc_given_discovered_full": _mean(binding_given_discovered) if binding_given_discovered else None,
        "WrongFaceNameRate": _mean(wrong_face_name),
        "BlankNameRate": _mean(blank_name),
        "PRR_soft": _mean(rel_scores),
        "PRR_identity_conditioned": _mean(rel_id_scores),
        "PRR_given_identity": _mean(rel_given_id) if rel_given_id else None,
        "PCR_strict": _mean(cat_scores),
        "PCR_identity_conditioned": _mean(cat_id_scores),
        "PCR_given_identity": _mean(cat_given_id) if cat_given_id else None,
        "EvidenceGroundedRate": _mean(grounded),
        "ReasoningPresenceRate": _mean(reasoning_present),
        "KeyPhotoF1_overlap": _mean(evidence_f1),
        "n_person_targets": len(name_scores_strict),
        "n_owner_atom_targets": len(owner_atom_scores),
        "n_llm": int(run_stats.get("n_llm_calls", 0) or 0),
    }
    for key, values in owner_by_inference.items():
        metrics[f"OFR_atom_{key}"] = _mean(values)
    metrics["_calibration_rows"] = target_conf_rows
    return {
        "schema_version": "ecd_metric_audit_user.v1",
        "user_id": user_id,
        "metrics": _round_metrics(metrics),
        "targets": targets_out,
    }


def aggregate_reports(rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted({k for row in rows for k in row if not k.startswith("_") and k != "user_id"}):
        values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        if key.startswith("n_"):
            out[key] = int(sum(values)) if key in {"n_person_targets", "n_owner_atom_targets", "n_llm"} else round(mean(values), 4)
        else:
            out[key] = round(mean(float(v) for v in values), 4)
    out["n_users"] = len(rows)

    conf_rows = [item for row in rows for item in row.get("_calibration_rows", [])]
    out["ECE_observed_strict"] = round(_ece(conf_rows, "strict", fill_missing=None), 4)
    out["ECE_observed_soft"] = round(_ece(conf_rows, "soft", fill_missing=None), 4)
    out["ECE_coverage_strict"] = round(_ece(conf_rows, "strict", fill_missing=0.0), 4)
    out["ECE_coverage_soft"] = round(_ece(conf_rows, "soft", fill_missing=0.0), 4)
    out["score_counts_person_name_strict"] = dict(Counter(
        row.get("score_strict")
        for row in target_rows
        if row.get("target_type") == "person_name"
    ))
    return out


def write_markdown(result: dict[str, Any], path: Path) -> None:
    metrics = result["metrics"]
    rows = [
        ("Users", metrics.get("n_users")),
        ("Owner fact recovery, parent macro lexical", metrics.get("OFR_parent_lexical")),
        ("Owner fact recovery, atom macro lexical", metrics.get("OFR_atom_lexical")),
        ("PIR strict, face-conditioned exact alias", metrics.get("PIR_strict")),
        ("PIR soft, exact plus first-name partial", metrics.get("PIR_soft")),
        ("PIR hard strict, skip users without hard targets", metrics.get("PIR_hard_strict")),
        ("Name discovery, full name anywhere in user profile", metrics.get("NameDiscovery_full")),
        ("Name discovery, first name anywhere in user profile", metrics.get("NameDiscovery_first")),
        ("Binding accuracy given full-name discovery", metrics.get("BindingAcc_given_discovered_full")),
        ("Wrong-face name rate", metrics.get("WrongFaceNameRate")),
        ("Blank target-face name rate", metrics.get("BlankNameRate")),
        ("PRR soft", metrics.get("PRR_soft")),
        ("PRR identity-conditioned", metrics.get("PRR_identity_conditioned")),
        ("PRR given identity correct", metrics.get("PRR_given_identity")),
        ("PCR strict", metrics.get("PCR_strict")),
        ("PCR identity-conditioned", metrics.get("PCR_identity_conditioned")),
        ("PCR given identity correct", metrics.get("PCR_given_identity")),
        ("Evidence grounded rate", metrics.get("EvidenceGroundedRate")),
        ("Reasoning presence rate", metrics.get("ReasoningPresenceRate")),
        ("Key-photo F1 strict overlap", metrics.get("KeyPhotoF1_overlap")),
        ("ECE observed strict", metrics.get("ECE_observed_strict")),
        ("ECE coverage strict", metrics.get("ECE_coverage_strict")),
    ]
    lines = [
        "# Corrected Metric Audit Summary",
        "",
        f"- Runs root: `{result['runs_root']}`",
        f"- Users root: `{result['users_root']}`",
        f"- Users evaluated: `{result['n_users']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for label, value in rows:
        lines.append(f"| {label} | {_fmt(value)} |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- These metrics are diagnostic and do not overwrite legacy `eval_report.json` files.",
        "- Owner fact scores still use a lexical lower-bound judge; semantic LLM judging should be run separately for final paper tables.",
        "- `PRR_identity_conditioned` and `PCR_identity_conditioned` set relation/category credit to zero unless the target face identity is exactly correct.",
        "- `KeyPhotoF1_overlap` is a strict public key-photo overlap diagnostic, not a semantic evidence sufficiency judge.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_conf(
    rows: list[dict[str, float | str | None]],
    user_id: str,
    target_id: str,
    target_type: str,
    confidence: float | None,
    strict_score: float,
    soft_score: float,
) -> None:
    rows.append({
        "user_id": user_id,
        "target_id": target_id,
        "target_type": target_type,
        "confidence": confidence,
        "strict": strict_score,
        "soft": soft_score,
    })


def _ece(rows: list[dict[str, Any]], score_key: str, *, fill_missing: float | None, bins: int = 10) -> float:
    pairs = []
    for row in rows:
        confidence = row.get("confidence")
        if confidence is None:
            if fill_missing is None:
                continue
            confidence = fill_missing
        try:
            conf = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            continue
        score = float(row.get(score_key) or 0.0)
        correct = 1.0 if score == 1.0 else 0.0 if score_key == "strict" else score
        pairs.append((conf, correct))
    if not pairs:
        return 0.0
    err = 0.0
    total = len(pairs)
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        bucket = [(c, y) for c, y in pairs if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        avg_acc = sum(y for _, y in bucket) / len(bucket)
        err += len(bucket) / total * abs(avg_conf - avg_acc)
    return err


def _name_exact(pred: str, aliases: list[str]) -> bool:
    pred_norm = _norm_name(pred)
    return bool(pred_norm) and any(pred_norm == _norm_name(alias) for alias in aliases if _norm_name(alias))


def _name_partial(pred: str, aliases: list[str]) -> bool:
    pred_first = _first_name(pred)
    if not pred_first:
        return False
    for alias in aliases:
        alias_first = _first_name(alias)
        if alias_first and pred_first == alias_first:
            return True
    return False


def _norm_name(text: str) -> str:
    return " ".join(WORD_RE.findall(str(text).lower()))


def _first_name(text: str) -> str:
    parts = _norm_name(text).split()
    return parts[0] if parts else ""


def _norm_relation(text: str) -> str:
    norm = _norm_name(text)
    if not norm:
        return ""
    if norm in RELATION_SYNONYMS:
        return RELATION_SYNONYMS[norm]
    for phrase, canonical in sorted(RELATION_SYNONYMS.items(), key=lambda kv: len(kv[0]), reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", norm):
            return canonical
    return norm


def _norm_category(text: str) -> str:
    norm = _norm_name(text)
    if norm in {"co worker", "coworker", "work", "professional"}:
        return "colleague"
    if norm in {"relative", "relatives"}:
        return "family"
    return norm


def _relation_score(gt_relation: str, pred_relation: str, gt_category: str, pred_category: str) -> float:
    gt_rel = _norm_relation(gt_relation)
    pred_rel = _norm_relation(pred_relation)
    if gt_rel and pred_rel and gt_rel == pred_rel:
        return 1.0
    gt_cat = _norm_category(gt_category) or RELATION_TO_CATEGORY.get(gt_rel, "")
    pred_cat = _norm_category(pred_category) or RELATION_TO_CATEGORY.get(pred_rel, "")
    if gt_cat and pred_cat and gt_cat == pred_cat:
        return 0.5
    return 0.0


def _evidence_f1(predicted: list[str], gt_ids: list[str]) -> float:
    pred_set = {str(pid) for pid in predicted if str(pid).strip()}
    gt_set = {str(pid) for pid in gt_ids if str(pid).strip()}
    if not pred_set and not gt_set:
        return 1.0
    if not pred_set or not gt_set:
        return 0.0
    overlap = len(pred_set & gt_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_set)
    recall = overlap / len(gt_set)
    return 2 * precision * recall / (precision + recall)


def _confidence(row: dict[str, Any]) -> float | None:
    if not row or row.get("confidence") is None:
        return None
    try:
        return max(0.0, min(1.0, float(row.get("confidence"))))
    except (TypeError, ValueError):
        return None


def _mean(values) -> float:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else 0.0


def _round_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            out[key] = round(value, 4)
        else:
            out[key] = value
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()

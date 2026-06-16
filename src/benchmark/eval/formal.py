"""Formal album benchmark evaluator (v2).

Implements the official metric suite:

    OFR | PIR | PIR-hard | PRR-ID | EFS | ECE | nLLM

The default implementation uses deterministic lexical/value matching and a
public-evidence heuristic for EFS.  `llm_semantic` modes use cached deterministic
LLM judges while preserving the same aggregation semantics.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from src.benchmark.eval.judge import lexical_judge, llm_owner_fact_judge
from src.utils.io import load_json, save_json


MAIN_METRICS = ["OFR", "PIR", "PIR-hard", "PRR-ID", "EFS", "ECE", "nLLM"]


class JsonJudgeCache:
    """Small thread-safe JSON cache for expensive LLM-as-Judge calls."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        if self.path and self.path.exists():
            raw = load_json(self.path)
            items = raw.get("items") if isinstance(raw, dict) else None
            if isinstance(items, dict):
                self._items = {
                    str(key): value
                    for key, value in items.items()
                    if isinstance(value, dict)
                }

    def get_or_compute(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        compute: Any,
    ) -> dict[str, Any]:
        key = _cache_key(kind, payload)
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                self.hits += 1
                return {**cached, "cache_hit": True}
        result = dict(compute() or {})
        result.setdefault("cache_key", key)
        result.setdefault("cache_kind", kind)
        with self._lock:
            existing = self._items.get(key)
            if existing is not None:
                self.hits += 1
                return {**existing, "cache_hit": True}
            self._items[key] = result
            self.misses += 1
            return {**result, "cache_hit": False}

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            items = dict(self._items)
            hits = self.hits
            misses = self.misses
        save_json(
            {
                "schema_version": "formal_judge_cache.v1",
                "n_items": len(items),
                "hits": hits,
                "misses": misses,
                "items": items,
            },
            self.path,
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            n_items = len(self._items)
            hits = self.hits
            misses = self.misses
        return {
            "path": str(self.path) if self.path else None,
            "n_items": n_items,
            "hits": hits,
            "misses": misses,
        }

RELATION_SYNONYMS = {
    "mom": "mother",
    "mum": "mother",
    "mama": "mother",
    "mother": "mother",
    "dad": "father",
    "papa": "father",
    "father": "father",
    "sister": "sibling",
    "brother": "sibling",
    "sibling": "sibling",
    "daughter": "child",
    "son": "child",
    "child": "child",
    "wife": "partner",
    "husband": "partner",
    "spouse": "partner",
    "partner": "partner",
    "boyfriend": "partner",
    "girlfriend": "partner",
    "fiance": "partner",
    "fiancee": "partner",
    "coworker": "colleague",
    "co worker": "colleague",
    "colleague": "colleague",
    "work colleague": "colleague",
    "professional contact": "colleague",
    "classmate": "classmate",
    "schoolmate": "classmate",
    "school friend": "classmate",
    "neighbor": "neighbor",
    "neighbour": "neighbor",
    "friend": "friend",
    "close friend": "friend",
    "best friend": "friend",
    "old friend": "friend",
    "aunt": "extended_family",
    "uncle": "extended_family",
    "cousin": "extended_family",
    "grandmother": "extended_family",
    "grandfather": "extended_family",
    "grandparent": "extended_family",
    "niece": "extended_family",
    "nephew": "extended_family",
    "great niece": "extended_family",
    "great nephew": "extended_family",
    "granddaughter": "extended_family",
    "grandson": "extended_family",
}

RELATION_TO_CATEGORY = {
    "mother": "family",
    "father": "family",
    "sibling": "family",
    "child": "family",
    "partner": "family",
    "extended_family": "family",
    "colleague": "colleague",
    "classmate": "classmate",
    "neighbor": "neighbor",
    "friend": "friend",
}

CATEGORY_SYNONYMS = {
    "co worker": "colleague",
    "coworker": "colleague",
    "work": "colleague",
    "professional": "colleague",
    "relative": "family",
    "relatives": "family",
    "extended family": "family",
    "extended_family": "family",
}

UNICODE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


@dataclass(frozen=True)
class TargetPerson:
    person_id: str
    face_id: str
    canonical_name: str
    accepted_aliases: tuple[str, ...]
    relation: str
    relation_category: str
    alignment_difficulty: str
    name_target_id: str
    name_target: dict[str, Any]


def evaluate_formal(
    *,
    predicted_profile: dict[str, Any],
    eval_gt: dict[str, Any],
    agent_album: dict[str, Any] | None = None,
    run_stats: dict[str, Any] | None = None,
    judge_mode: str = "lexical_dev",
    evidence_judge_mode: str = "heuristic_dev",
    llm: Any = None,
    judge_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a single user with the formal v2 metric suite."""

    user_id = str(eval_gt.get("user_id") or predicted_profile.get("user_id") or "")
    run_stats = run_stats or (predicted_profile.get("run_stats") or {})
    targets = list(eval_gt.get("evaluation_targets") or [])
    owner_facts = list((predicted_profile.get("owner") or {}).get("facts") or [])
    persons = list(predicted_profile.get("persons") or [])
    pred_by_face = {
        str(row.get("face_id") or ""): row
        for row in persons
        if str(row.get("face_id") or "")
    }
    target_people = _build_target_people(targets)
    relation_target_by_person = {
        str(t.get("person_id") or ""): t
        for t in targets
        if str(t.get("target_type") or "") == "person_relation"
    }
    category_target_by_person = {
        str(t.get("person_id") or ""): t
        for t in targets
        if str(t.get("target_type") or "") == "person_category"
    }
    photo_index = _photo_index(agent_album or {})

    target_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    efs_scores: list[float] = []
    efs_by_target_type: dict[str, list[float]] = defaultdict(list)
    efs_given_recovered_scores: list[float] = []
    efs_given_correct_scores: list[float] = []
    key_photo_f1_scores: list[float] = []
    owner_scores: list[float] = []
    owner_by_inference: dict[str, list[float]] = defaultdict(list)
    owner_by_parent: dict[str, list[float]] = defaultdict(list)

    owner_match_by_target: dict[str, dict[str, Any]] = {}
    for target in targets:
        if str(target.get("target_type") or "") != "owner_fact_atom":
            continue
        score_info = _score_owner_atom(
            target,
            owner_facts,
            judge_mode=judge_mode,
            llm=llm,
            judge_cache=judge_cache,
        )
        score = score_info["score"]
        owner_scores.append(score)
        owner_by_inference[str(target.get("inference_type") or "unknown")].append(score)
        owner_by_parent[str(target.get("parent_fact_id") or target.get("target_id") or "")].append(score)
        owner_match_by_target[str(target.get("target_id") or "")] = score_info

        matched_fact = score_info.get("matched_prediction") or {}
        pred_text = str(matched_fact.get("text") or "")
        evidence_ids = _evidence_ids(matched_fact)
        confidence = _confidence_or_zero(matched_fact)
        efs_info = _score_efs(
            target_type="owner_fact_atom",
            target=target,
            correctness=score,
            raw_value_score=score,
            identity_bound=True,
            prediction=pred_text,
            evidence_ids=evidence_ids,
            reasoning_path=str(matched_fact.get("reasoning_path") or ""),
            public_face_id="",
            photo_index=photo_index,
            evidence_judge_mode=evidence_judge_mode,
            llm=llm,
            judge_cache=judge_cache,
        )
        efs = float(efs_info["score"])
        efs_scores.append(efs)
        _append_efs_diagnostics(
            efs_by_target_type,
            efs_given_recovered_scores,
            efs_given_correct_scores,
            target_type="owner_fact_atom",
            correctness=score,
            efs=efs,
        )
        key_photo_f1_scores.append(_evidence_f1(evidence_ids, target.get("key_photo_ids_public") or []))
        _append_calibration(
            calibration_rows,
            target=target,
            confidence=confidence,
            correctness=score,
            strict_correctness=1.0 if score == 1.0 else 0.0,
        )
        target_rows.append(
            _target_row(
                user_id=user_id,
                target=target,
                prediction=pred_text,
                predicted_face_id=None,
                confidence=confidence,
                correctness=score,
                efs=efs,
                evidence_ids=evidence_ids,
                matched_index=score_info.get("matched_index"),
                judge_details={
                    **(score_info.get("judge_details") or {}),
                    "evidence_judge": efs_info.get("judge_details"),
                },
            )
        )

    identity_rows = _evaluate_identity(target_people, persons, pred_by_face)
    identity_by_person = {row["person_id"]: row for row in identity_rows}
    pir_components = _pir_components(identity_rows)
    hard_identity_rows = [
        row for row in identity_rows
        if row.get("alignment_difficulty") == "hard"
    ]
    hard_pir_components = _pir_components(hard_identity_rows) if hard_identity_rows else None

    relation_scores_for_prr_id: list[float] = []
    relation_scores_e2e: list[float] = []
    category_scores: list[float] = []
    category_scores_e2e: list[float] = []
    category_scores_given_id: list[float] = []

    for person in target_people:
        id_row = identity_by_person.get(person.person_id, {})
        pred = pred_by_face.get(person.face_id) or {}
        pred_name = str(pred.get("canonical_name") or "")
        evidence_ids = _evidence_ids(pred)
        confidence = _confidence_or_zero(pred)
        name_correctness = float(id_row.get("B", 0.0))
        name_target = id_row.get("target") or {}
        name_efs_info = _score_efs(
            target_type="person_name",
            target=name_target,
            correctness=name_correctness,
            raw_value_score=name_correctness,
            identity_bound=bool(id_row.get("B")),
            prediction=pred_name,
            evidence_ids=evidence_ids,
            reasoning_path=str(pred.get("reasoning_path") or ""),
            public_face_id=person.face_id,
            photo_index=photo_index,
            evidence_judge_mode=evidence_judge_mode,
            llm=llm,
            judge_cache=judge_cache,
        )
        name_efs = float(name_efs_info["score"])
        efs_scores.append(name_efs)
        _append_efs_diagnostics(
            efs_by_target_type,
            efs_given_recovered_scores,
            efs_given_correct_scores,
            target_type="person_name",
            correctness=name_correctness,
            efs=name_efs,
        )
        key_photo_f1_scores.append(_evidence_f1(evidence_ids, name_target.get("key_photo_ids_public") or []))
        _append_calibration(
            calibration_rows,
            target=name_target,
            confidence=confidence,
            correctness=name_correctness,
            strict_correctness=name_correctness,
        )
        target_rows.append(
            _target_row(
                user_id=user_id,
                target=name_target,
                prediction=pred_name,
                predicted_face_id=person.face_id,
                confidence=confidence,
                correctness=name_correctness,
                efs=name_efs,
                evidence_ids=evidence_ids,
                matched_index=None,
                judge_details={
                    "discovered": bool(id_row.get("D")),
                    "bound": bool(id_row.get("B")),
                    "discovered_face_id": id_row.get("discovered_face_id"),
                    "match_mode": id_row.get("match_mode"),
                    "evidence_judge": name_efs_info.get("judge_details"),
                },
            )
        )

        rel_target = relation_target_by_person.get(person.person_id)
        if rel_target is not None:
            pred_relation = str(pred.get("relation_to_owner") or "")
            rel_score_raw = _relation_score(
                str(rel_target.get("gt_value") or person.relation),
                pred_relation,
                str(rel_target.get("relation_category") or person.relation_category),
                str(pred.get("relation_category") or ""),
            )
            rel_score_e2e = rel_score_raw if id_row.get("B") == 1.0 else 0.0
            relation_scores_e2e.append(rel_score_e2e)
            if id_row.get("B") == 1.0:
                relation_scores_for_prr_id.append(rel_score_raw)
            rel_efs_info = _score_efs(
                target_type="person_relation",
                target=rel_target,
                correctness=rel_score_e2e,
                raw_value_score=rel_score_raw,
                identity_bound=bool(id_row.get("B")),
                prediction=pred_relation,
                evidence_ids=evidence_ids,
                reasoning_path=str(pred.get("reasoning_path") or ""),
                public_face_id=person.face_id,
                photo_index=photo_index,
                evidence_judge_mode=evidence_judge_mode,
                llm=llm,
                judge_cache=judge_cache,
            )
            rel_efs = float(rel_efs_info["score"])
            efs_scores.append(rel_efs)
            _append_efs_diagnostics(
                efs_by_target_type,
                efs_given_recovered_scores,
                efs_given_correct_scores,
                target_type="person_relation",
                correctness=rel_score_e2e,
                efs=rel_efs,
            )
            key_photo_f1_scores.append(_evidence_f1(evidence_ids, rel_target.get("key_photo_ids_public") or []))
            _append_calibration(
                calibration_rows,
                target=rel_target,
                confidence=confidence,
                correctness=rel_score_e2e,
                strict_correctness=1.0 if rel_score_e2e == 1.0 else 0.0,
            )
            target_rows.append(
                _target_row(
                    user_id=user_id,
                    target=rel_target,
                    prediction=pred_relation,
                    predicted_face_id=person.face_id,
                    confidence=confidence,
                    correctness=rel_score_e2e,
                    efs=rel_efs,
                    evidence_ids=evidence_ids,
                    matched_index=None,
                    judge_details={
                        "identity_bound": bool(id_row.get("B")),
                        "raw_relation_score": rel_score_raw,
                        "evidence_judge": rel_efs_info.get("judge_details"),
                    },
                )
            )

        cat_target = category_target_by_person.get(person.person_id)
        if cat_target is not None:
            gt_category = str(cat_target.get("gt_value") or cat_target.get("relation_category") or person.relation_category)
            pred_category = str(pred.get("relation_category") or "")
            cat_score_raw = _category_score(gt_category, pred_category)
            category_scores.append(cat_score_raw)
            cat_score_e2e = cat_score_raw if id_row.get("B") == 1.0 else 0.0
            category_scores_e2e.append(cat_score_e2e)
            if id_row.get("B") == 1.0:
                category_scores_given_id.append(cat_score_raw)
            cat_efs_info = _score_efs(
                target_type="person_category",
                target=cat_target,
                correctness=cat_score_e2e,
                raw_value_score=cat_score_raw,
                identity_bound=bool(id_row.get("B")),
                prediction=pred_category,
                evidence_ids=evidence_ids,
                reasoning_path=str(pred.get("reasoning_path") or ""),
                public_face_id=person.face_id,
                photo_index=photo_index,
                evidence_judge_mode=evidence_judge_mode,
                llm=llm,
                judge_cache=judge_cache,
            )
            cat_efs = float(cat_efs_info["score"])
            efs_scores.append(cat_efs)
            _append_efs_diagnostics(
                efs_by_target_type,
                efs_given_recovered_scores,
                efs_given_correct_scores,
                target_type="person_category",
                correctness=cat_score_e2e,
                efs=cat_efs,
            )
            key_photo_f1_scores.append(_evidence_f1(evidence_ids, cat_target.get("key_photo_ids_public") or []))
            _append_calibration(
                calibration_rows,
                target=cat_target,
                confidence=confidence,
                correctness=cat_score_e2e,
                strict_correctness=1.0 if cat_score_e2e == 1.0 else 0.0,
            )
            target_rows.append(
                _target_row(
                    user_id=user_id,
                    target=cat_target,
                    prediction=pred_category,
                    predicted_face_id=person.face_id,
                    confidence=confidence,
                    correctness=cat_score_e2e,
                    efs=cat_efs,
                    evidence_ids=evidence_ids,
                    matched_index=None,
                    judge_details={
                        "identity_bound": bool(id_row.get("B")),
                        "raw_category_score": cat_score_raw,
                        "evidence_judge": cat_efs_info.get("judge_details"),
                    },
                )
            )

    n_llm = _n_llm(run_stats, predicted_profile)
    metrics = {
        "OFR": _mean(owner_scores),
        "PIR": pir_components["PIR"],
        "PIR-hard": hard_pir_components["PIR"] if hard_pir_components else None,
        "PRR-ID": _mean_optional(relation_scores_for_prr_id),
        "EFS": _mean(efs_scores),
        "ECE": _ece(calibration_rows, score_key="correctness", fill_missing=0.0),
        "nLLM": float(n_llm),
    }
    diagnostics: dict[str, Any] = {
        "IDR": pir_components["IDR"],
        "IBR": pir_components["IBR"],
        "IDR-hard": hard_pir_components["IDR"] if hard_pir_components else None,
        "IBR-hard": hard_pir_components["IBR"] if hard_pir_components else None,
        "PIR-strict": _mean(row["strict_bound_score"] for row in identity_rows),
        "PIR-soft": _mean(row["soft_bound_score"] for row in identity_rows),
        "PRR-e2e": _mean(relation_scores_e2e),
        "PCR": _mean(category_scores),
        "PCR-e2e": _mean(category_scores_e2e),
        "PCR-ID": _mean_optional(category_scores_given_id),
        "WrongFaceRate": _mean(row["wrong_face"] for row in identity_rows),
        "BlankNameRate": _mean(row["blank_name"] for row in identity_rows),
        "KeyPhotoF1": _mean(key_photo_f1_scores),
        "EFS-owner": _mean(efs_by_target_type.get("owner_fact_atom", [])),
        "EFS-name": _mean(efs_by_target_type.get("person_name", [])),
        "EFS-relation": _mean(efs_by_target_type.get("person_relation", [])),
        "EFS-category": _mean(efs_by_target_type.get("person_category", [])),
        "EFS-given-recovered": _mean(efs_given_recovered_scores),
        "EFS-given-correct": _mean(efs_given_correct_scores),
        "ECE-strict": _ece(calibration_rows, score_key="strict_correctness", fill_missing=0.0),
        "n_targets": len(target_rows),
        "n_owner_atom_targets": len(owner_scores),
        "n_person_targets": len(identity_rows),
        "n_prr_id_support": len(relation_scores_for_prr_id),
    }
    diagnostics["OFR-by-type"] = {
        key: round(_mean(values), 4)
        for key, values in sorted(owner_by_inference.items())
    }
    diagnostics["OFR-parent"] = _mean(_mean(values) for values in owner_by_parent.values())

    return {
        "schema_version": "album_eval_user.v2",
        "user_id": user_id,
        "framework": predicted_profile.get("framework"),
        "variant": predicted_profile.get("variant"),
        "judge_mode": judge_mode,
        "evidence_judge_mode": evidence_judge_mode,
        "metrics": _round_mapping(metrics),
        "diagnostics": _round_mapping(diagnostics),
        "targets": target_rows,
        "calibration_rows": calibration_rows,
    }


def evaluate_formal_paths(
    *,
    predicted_profile_path: str | Path,
    eval_gt_path: str | Path,
    agent_album_path: str | Path | None = None,
    run_stats_path: str | Path | None = None,
    output_path: str | Path | None = None,
    judge_mode: str = "lexical_dev",
    evidence_judge_mode: str = "heuristic_dev",
    llm: Any = None,
    judge_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = load_json(predicted_profile_path)
    gt = load_json(eval_gt_path)
    album = load_json(agent_album_path) if agent_album_path and Path(agent_album_path).exists() else None
    run_stats = load_json(run_stats_path) if run_stats_path and Path(run_stats_path).exists() else profile.get("run_stats", {})
    report = evaluate_formal(
        predicted_profile=profile,
        eval_gt=gt,
        agent_album=album,
        run_stats=run_stats,
        judge_mode=judge_mode,
        evidence_judge_mode=evidence_judge_mode,
        llm=llm,
        judge_cache=judge_cache,
    )
    if output_path:
        save_json(report, output_path)
    return report


def aggregate_formal_reports(
    reports: list[dict[str, Any]],
    *,
    runs_root: str | Path | None = None,
    users_root: str | Path | None = None,
    judge_mode: str = "lexical_dev",
    evidence_judge_mode: str = "heuristic_dev",
) -> dict[str, Any]:
    if not reports:
        raise ValueError("No formal reports to aggregate.")

    metric_rows = [r.get("metrics") or {} for r in reports]
    diagnostic_rows = [r.get("diagnostics") or {} for r in reports]
    calibration_rows = [row for report in reports for row in report.get("calibration_rows", [])]
    aggregate_metrics = {
        "OFR": _macro_metric(metric_rows, "OFR"),
        "PIR": _macro_metric(metric_rows, "PIR"),
        "PIR-hard": _macro_metric(metric_rows, "PIR-hard"),
        "PRR-ID": _macro_metric(metric_rows, "PRR-ID"),
        "EFS": _macro_metric(metric_rows, "EFS"),
        "ECE": _ece(calibration_rows, score_key="correctness", fill_missing=0.0),
        "nLLM": _macro_metric(metric_rows, "nLLM"),
    }

    aggregate_diagnostics: dict[str, Any] = {
        "IDR": _macro_metric(diagnostic_rows, "IDR"),
        "IBR": _macro_metric(diagnostic_rows, "IBR"),
        "IDR-hard": _macro_metric(diagnostic_rows, "IDR-hard"),
        "IBR-hard": _macro_metric(diagnostic_rows, "IBR-hard"),
        "PIR-strict": _macro_metric(diagnostic_rows, "PIR-strict"),
        "PIR-soft": _macro_metric(diagnostic_rows, "PIR-soft"),
        "PRR-e2e": _macro_metric(diagnostic_rows, "PRR-e2e"),
        "PCR": _macro_metric(diagnostic_rows, "PCR"),
        "PCR-e2e": _macro_metric(diagnostic_rows, "PCR-e2e"),
        "PCR-ID": _macro_metric(diagnostic_rows, "PCR-ID"),
        "WrongFaceRate": _macro_metric(diagnostic_rows, "WrongFaceRate"),
        "BlankNameRate": _macro_metric(diagnostic_rows, "BlankNameRate"),
        "KeyPhotoF1": _macro_metric(diagnostic_rows, "KeyPhotoF1"),
        "EFS-owner": _macro_metric(diagnostic_rows, "EFS-owner"),
        "EFS-name": _macro_metric(diagnostic_rows, "EFS-name"),
        "EFS-relation": _macro_metric(diagnostic_rows, "EFS-relation"),
        "EFS-category": _macro_metric(diagnostic_rows, "EFS-category"),
        "EFS-given-recovered": _macro_metric(diagnostic_rows, "EFS-given-recovered"),
        "EFS-given-correct": _macro_metric(diagnostic_rows, "EFS-given-correct"),
        "ECE-strict": _ece(calibration_rows, score_key="strict_correctness", fill_missing=0.0),
        "n_targets": int(sum(int((row.get("n_targets") or 0)) for row in diagnostic_rows)),
        "n_owner_atom_targets": int(sum(int((row.get("n_owner_atom_targets") or 0)) for row in diagnostic_rows)),
        "n_person_targets": int(sum(int((row.get("n_person_targets") or 0)) for row in diagnostic_rows)),
        "n_prr_id_support": int(sum(int((row.get("n_prr_id_support") or 0)) for row in diagnostic_rows)),
    }
    aggregate_diagnostics["OFR-by-type"] = _aggregate_nested_mean(diagnostic_rows, "OFR-by-type")
    aggregate_diagnostics["score_counts_person_name"] = dict(Counter(
        row.get("correctness")
        for report in reports
        for row in report.get("targets", [])
        if row.get("target_type") == "person_name"
    ))

    per_user = [
        {
            "user_id": report.get("user_id"),
            "metrics": report.get("metrics"),
            "diagnostics": report.get("diagnostics"),
        }
        for report in reports
    ]
    return {
        "schema_version": "album_eval_aggregate.v2",
        "runs_root": str(runs_root) if runs_root is not None else None,
        "users_root": str(users_root) if users_root is not None else None,
        "n_users": len(reports),
        "judge_mode": judge_mode,
        "evidence_judge_mode": evidence_judge_mode,
        "metric_columns": MAIN_METRICS,
        "metrics": _round_mapping(aggregate_metrics),
        "diagnostics": _round_mapping(aggregate_diagnostics),
        "per_user": per_user,
    }


def write_formal_markdown(result: dict[str, Any], path: str | Path) -> None:
    metrics = result.get("metrics") or {}
    diagnostics = result.get("diagnostics") or {}
    lines = [
        "# Formal Album Eval Summary",
        "",
        f"- Schema: `{result.get('schema_version')}`",
        f"- Runs root: `{result.get('runs_root')}`",
        f"- Users root: `{result.get('users_root')}`",
        f"- Users evaluated: `{result.get('n_users')}`",
        f"- Judge mode: `{result.get('judge_mode')}`",
        f"- Evidence judge mode: `{result.get('evidence_judge_mode')}`",
        "",
        "## Main Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in MAIN_METRICS:
        lines.append(f"| {key} | {_fmt(metrics.get(key))} |")
    lines.extend([
        "",
        "## Diagnostics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ])
    for key in [
        "IDR",
        "IBR",
        "IDR-hard",
        "IBR-hard",
        "PIR-strict",
        "PIR-soft",
        "PRR-e2e",
        "PCR",
        "PCR-e2e",
        "PCR-ID",
        "WrongFaceRate",
        "BlankNameRate",
        "KeyPhotoF1",
        "EFS-owner",
        "EFS-name",
        "EFS-relation",
        "EFS-category",
        "EFS-given-recovered",
        "EFS-given-correct",
        "ECE-strict",
        "n_targets",
        "n_owner_atom_targets",
        "n_person_targets",
        "n_prr_id_support",
    ]:
        lines.append(f"| {key} | {_fmt(diagnostics.get(key))} |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Recovery metrics are macro-averaged over users.",
        "- `ECE` is pooled, coverage-aware, and uses soft target correctness.",
        "- `PRR-ID` skips users with no correctly bound identities.",
        "- `PIR-hard` skips users with no hard identity targets.",
        "- Current `lexical_dev` and `heuristic_dev` scores are development diagnostics; final paper scores should use cached deterministic semantic judges.",
        "",
    ])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_target_people(targets: list[dict[str, Any]]) -> list[TargetPerson]:
    relation_by_person = {
        str(t.get("person_id") or ""): str(t.get("gt_value") or "")
        for t in targets
        if str(t.get("target_type") or "") == "person_relation"
    }
    category_by_person = {
        str(t.get("person_id") or ""): str(t.get("gt_value") or t.get("relation_category") or "")
        for t in targets
        if str(t.get("target_type") or "") == "person_category"
    }
    people = []
    for target in targets:
        if str(target.get("target_type") or "") != "person_name":
            continue
        canonical = str(target.get("gt_value") or "")
        aliases = _unique([canonical, *(str(a) for a in target.get("aliases") or [])])
        person_id = str(target.get("person_id") or "")
        people.append(
            TargetPerson(
                person_id=person_id,
                face_id=str(target.get("public_face_id") or ""),
                canonical_name=canonical,
                accepted_aliases=tuple(aliases),
                relation=relation_by_person.get(person_id, ""),
                relation_category=str(target.get("relation_category") or category_by_person.get(person_id, "")),
                alignment_difficulty=str(target.get("alignment_difficulty") or ""),
                name_target_id=str(target.get("target_id") or ""),
                name_target=dict(target),
            )
        )
    return people


def _score_owner_atom(
    target: dict[str, Any],
    owner_facts: list[dict[str, Any]],
    *,
    judge_mode: str = "lexical_dev",
    llm: Any = None,
    judge_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    texts = [str(fact.get("text") or "") for fact in owner_facts]
    gt_text = str(target.get("gt_text") or "")
    if _uses_llm_judge(judge_mode) and llm is not None:
        payload = {
            "version": "formal_owner_value_v1",
            "judge_mode": judge_mode,
            "target_id": str(target.get("target_id") or ""),
            "gt_text": gt_text,
            "candidate_texts": texts,
        }

        def compute() -> dict[str, Any]:
            result = llm_owner_fact_judge(gt_text, texts, llm, allow_fallback=False)
            out = result.to_dict()
            out["mode"] = judge_mode
            return out

        judge_details = _cached_judge_result(
            judge_cache,
            kind="owner_value",
            payload=payload,
            compute=compute,
        )
        matched_index = _valid_index(judge_details.get("matched_index"), len(owner_facts))
        matched = owner_facts[matched_index] if matched_index is not None else {}
        return {
            "score": float(judge_details.get("score") or 0.0),
            "matched_index": matched_index,
            "matched_prediction": matched,
            "judge_details": judge_details,
        }

    judge = lexical_judge(gt_text, texts)
    matched = {}
    if judge.matched_index is not None and 0 <= judge.matched_index < len(owner_facts):
        matched = owner_facts[judge.matched_index]
    return {
        "score": float(judge.score),
        "matched_index": judge.matched_index,
        "matched_prediction": matched,
        "judge_details": judge.to_dict(),
    }


def _score_efs(
    *,
    target_type: str,
    target: dict[str, Any],
    correctness: float,
    raw_value_score: float,
    identity_bound: bool,
    prediction: str,
    evidence_ids: list[str],
    reasoning_path: str,
    public_face_id: str,
    photo_index: dict[str, dict[str, Any]],
    evidence_judge_mode: str = "heuristic_dev",
    llm: Any = None,
    judge_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not _uses_llm_judge(evidence_judge_mode) or llm is None:
        score = _heuristic_efs(
            target_type=target_type,
            correctness=correctness,
            raw_value_score=raw_value_score,
            identity_bound=identity_bound,
            prediction=prediction,
            evidence_ids=evidence_ids,
            reasoning_path=reasoning_path,
            public_face_id=public_face_id,
            photo_index=photo_index,
        )
        return {
            "score": score,
            "judge_details": {
                "mode": "heuristic_dev",
                "verdict": _efs_verdict(score),
                "score": score,
            },
        }

    precheck = _efs_precheck(
        target_type=target_type,
        correctness=correctness,
        raw_value_score=raw_value_score,
        identity_bound=identity_bound,
        prediction=prediction,
        evidence_ids=evidence_ids,
        photo_index=photo_index,
    )
    if precheck.get("stop"):
        score = float(precheck["score"])
        return {
            "score": score,
            "judge_details": {
                "mode": evidence_judge_mode,
                "verdict": _efs_verdict(score),
                "score": score,
                "grounded": False,
                "rationale": precheck.get("reason", ""),
                "precheck": True,
            },
        }

    max_score = float(precheck.get("max_score", 1.0))
    gt_value = str(target.get("gt_value") if target.get("gt_value") is not None else target.get("gt_text") or "")
    query_terms = _efs_query_terms(gt_value, prediction, public_face_id)
    evidence_payload = [
        _public_photo_excerpt(photo_index[photo_id], query_terms=query_terms)
        for photo_id in evidence_ids[:6]
        if photo_id in photo_index
    ]
    payload = {
        "version": "formal_efs_v2_bridge_aware",
        "evidence_judge_mode": evidence_judge_mode,
        "target_id": str(target.get("target_id") or ""),
        "target_type": target_type,
        "ground_truth_value": gt_value,
        "value_correctness": round(float(correctness), 4),
        "raw_value_score": round(float(raw_value_score), 4),
        "identity_bound": bool(identity_bound),
        "max_allowed_score": max_score,
        "public_face_id": public_face_id,
        "agent_prediction": str(prediction or ""),
        "agent_reasoning_path": _truncate(str(reasoning_path or ""), 1600),
        "cited_evidence": evidence_payload,
    }

    def compute() -> dict[str, Any]:
        result = _call_efs_judge_llm(llm, payload)
        result["mode"] = evidence_judge_mode
        return result

    judge_details = _cached_judge_result(
        judge_cache,
        kind="evidence_faithfulness",
        payload=payload,
        compute=compute,
    )
    score = _coerce_efs_score(judge_details, max_score=max_score)
    judge_details["score"] = score
    judge_details["verdict"] = _efs_verdict(score, fallback=str(judge_details.get("verdict") or ""))
    return {"score": score, "judge_details": judge_details}


_EFS_JUDGE_SYSTEM = """You are an evidence faithfulness judge for a personal album benchmark.

Given a target, an agent prediction, the agent's reasoning_path, and public album evidence cited by the agent, decide whether the cited evidence faithfully supports the agent's actual prediction.

Scoring:
- faithful (1.0): cited public evidence is concrete, non-fabricated, and sufficient for the prediction.
- partial (0.5): cited evidence is relevant but incomplete, indirect, or supports only part of the prediction.
- unfaithful (0.0): evidence is missing, fabricated, irrelevant, contradictory, or too weak.

Rules:
- Do not require exact overlap with ground-truth key photos.
- Do not use private generation metadata.
- Respect max_allowed_score: never output a score above it.
- This benchmark intentionally includes cross-photo public evidence chains. For person_name, a text/OCR anchor naming the person plus a face photo can be faithful when the reasoning cites a concrete bridge such as same event, month/location, activity, cofaces, owner co-presence, or recurring scene context. Do not require the name and target face to appear in the same photo.
- If value_correctness is 1.0 and identity_bound is true, judge whether the cited evidence chain faithfully supports the agent's prediction; do not re-try the whole identity task from scratch under a stricter same-photo-only standard.
- For person_relation and person_category, social/family/professional categories may be supported by repeated co-presence and event context. Exact roles such as father/mother/spouse need direct relation text or very strong family-context evidence.
- If identity_bound is false, score person_relation/person_category at most 0.5.
- For owner_fact_atom, evidence should support the predicted owner fact, not merely mention generic album context.
- `relevant_text` is public OCR/entity text selected from the same cited photo for the target/prediction; treat it as part of the cited evidence.

Output ONLY JSON:
{"verdict":"faithful|partial|unfaithful","score":1.0|0.5|0.0,"grounded":true|false,"rationale":"brief reason"}"""


def _call_efs_judge_llm(llm: Any, payload: dict[str, Any]) -> dict[str, Any]:
    user_prompt = "Evaluate this target:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    parsed = _call_json_judge_llm(
        llm,
        prompt=user_prompt,
        system=_EFS_JUDGE_SYSTEM,
        max_tokens=512,
        error_label="LLM evidence judge",
    )
    out = {
        "verdict": str(parsed.get("verdict") or "unfaithful"),
        "score": parsed.get("score", 0.0),
        "grounded": bool(parsed.get("grounded", False)),
        "rationale": str(parsed.get("rationale") or ""),
    }
    if parsed.get("json_repaired"):
        out["json_repaired"] = True
    return out


def _call_json_judge_llm(
    llm: Any,
    *,
    prompt: str,
    system: str,
    max_tokens: int,
    error_label: str,
) -> dict[str, Any]:
    last_error = ""
    for attempt in range(2):
        actual_prompt = prompt
        if attempt:
            actual_prompt = (
                prompt
                + "\n\nIMPORTANT: Return ONLY one valid JSON object matching the requested schema. "
                "No markdown, no prose outside JSON."
            )
        try:
            raw = _llm_json_text(
                llm,
                prompt=actual_prompt,
                system=system,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            last_error = str(exc)
            continue
        parsed = _extract_json_object(raw)
        if isinstance(parsed, dict):
            return parsed
        if attempt:
            repaired = _repair_judge_json_from_text(raw)
            if repaired:
                return repaired
        last_error = "invalid JSON"
    raise RuntimeError(f"{error_label} did not return valid JSON: {last_error}")


def _llm_json_text(llm: Any, *, prompt: str, system: str, max_tokens: int) -> str:
    from src.llm.base import LLMMessage

    messages = [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=prompt),
    ]
    try:
        response = llm.chat(
            messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception:
        response = llm.chat(
            messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
    return str(getattr(response, "content", "") or "")


def _repair_judge_json_from_text(raw: Any) -> dict[str, Any] | None:
    text = " ".join(str(raw or "").split())
    if not text:
        return None
    lower = text.lower()
    verdict = ""
    if "unfaithful" in lower:
        verdict = "unfaithful"
    elif "partial" in lower:
        verdict = "partial"
    elif "faithful" in lower:
        verdict = "faithful"
    if not verdict:
        return None
    score_match = re.search(r"\bscore\b[^0-9]*(1(?:\.0)?|0(?:\.5|\.0)?|0?\.\d+)", lower)
    if score_match:
        try:
            score = float(score_match.group(1))
        except ValueError:
            score = _score_for_verdict(verdict)
    else:
        score = _score_for_verdict(verdict)
    return {
        "verdict": verdict,
        "score": score,
        "grounded": verdict != "unfaithful",
        "rationale": _truncate(text, 500),
        "json_repaired": True,
    }


def _score_for_verdict(verdict: str) -> float:
    if verdict == "faithful":
        return 1.0
    if verdict == "partial":
        return 0.5
    return 0.0


def _efs_precheck(
    *,
    target_type: str,
    correctness: float,
    raw_value_score: float,
    identity_bound: bool,
    prediction: str,
    evidence_ids: list[str],
    photo_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not str(prediction or "").strip():
        return {"stop": True, "score": 0.0, "reason": "empty prediction"}
    if not evidence_ids:
        return {"stop": True, "score": 0.0, "reason": "missing cited evidence"}
    invalid = _invalid_photo_ids(evidence_ids, photo_index)
    if invalid:
        return {"stop": True, "score": 0.0, "reason": f"invalid cited photo ids: {invalid[:5]}"}
    if target_type == "person_name" and correctness < 1.0:
        return {"stop": True, "score": 0.0, "reason": "identity target is not correctly bound"}
    if target_type in {"person_relation", "person_category"} and not identity_bound:
        if raw_value_score <= 0.0:
            return {"stop": True, "score": 0.0, "reason": "wrong identity and wrong relation/category value"}
        return {"stop": False, "max_score": 0.5}
    if correctness <= 0.0:
        return {"stop": True, "score": 0.0, "reason": "target value is not recovered"}
    if correctness < 1.0:
        return {"stop": False, "max_score": 0.5}
    return {"stop": False, "max_score": 1.0}


def _coerce_efs_score(details: dict[str, Any], *, max_score: float) -> float:
    verdict = str(details.get("verdict") or "").strip().lower()
    if verdict in {"faithful", "match", "equivalent"}:
        score = 1.0
    elif verdict in {"partial", "weak"}:
        score = 0.5
    elif verdict in {"unfaithful", "mismatch", "none"}:
        score = 0.0
    else:
        try:
            raw = float(details.get("score", 0.0))
        except (TypeError, ValueError):
            raw = 0.0
        if raw >= 0.75:
            score = 1.0
        elif raw >= 0.25:
            score = 0.5
        else:
            score = 0.0
    return max(0.0, min(float(max_score), score))


def _efs_verdict(score: float, fallback: str = "") -> str:
    if score >= 1.0:
        return "faithful"
    if score >= 0.5:
        return "partial"
    return fallback if fallback in {"faithful", "partial", "unfaithful"} else "unfaithful"


def _evaluate_identity(
    target_people: list[TargetPerson],
    persons: list[dict[str, Any]],
    pred_by_face: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for person in target_people:
        discovered_face_id = None
        discovered_mode = ""
        for row in persons:
            matched, mode = _identity_match(str(row.get("canonical_name") or ""), person, target_people)
            if matched:
                discovered_face_id = str(row.get("face_id") or "")
                discovered_mode = mode
                break
        target_pred = pred_by_face.get(person.face_id) or {}
        pred_name = str(target_pred.get("canonical_name") or "")
        bound, bound_mode = _identity_match(pred_name, person, target_people)
        strict = 1.0 if _identity_exact_alias(pred_name, person) else 0.0
        partial = 0.5 if strict == 0.0 and _identity_first_name_partial(pred_name, person) else 0.0
        soft = strict if strict else partial
        rows.append(
            {
                "person_id": person.person_id,
                "target_face_id": person.face_id,
                "target_name": person.canonical_name,
                "alignment_difficulty": person.alignment_difficulty,
                "D": 1.0 if discovered_face_id else 0.0,
                "B": 1.0 if bound else 0.0,
                "discovered_face_id": discovered_face_id,
                "match_mode": bound_mode or discovered_mode,
                "strict_bound_score": strict,
                "soft_bound_score": float(soft),
                "wrong_face": 1.0 if discovered_face_id and not bound else 0.0,
                "blank_name": 1.0 if not pred_name.strip() else 0.0,
                "target": {
                    **person.name_target,
                    "target_id": person.name_target_id,
                    "target_type": "person_name",
                    "person_id": person.person_id,
                    "public_face_id": person.face_id,
                    "gt_value": person.canonical_name,
                    "aliases": list(person.accepted_aliases),
                    "relation_category": person.relation_category,
                    "alignment_difficulty": person.alignment_difficulty,
                },
            }
        )
    return rows


def _pir_components(identity_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not identity_rows:
        return {"IDR": 0.0, "IBR": 0.0, "PIR": 0.0}
    discovered = [float(row.get("D") or 0.0) for row in identity_rows]
    bound = [float(row.get("B") or 0.0) for row in identity_rows]
    idr = _mean(discovered)
    denom = sum(discovered)
    ibr = sum(bound) / denom if denom > 0 else 0.0
    pir = 2 * idr * ibr / (idr + ibr) if (idr + ibr) > 0 else 0.0
    return {"IDR": idr, "IBR": ibr, "PIR": pir}


def _identity_match(pred_name: str, person: TargetPerson, all_targets: list[TargetPerson]) -> tuple[bool, str]:
    pred_norm = norm_text(pred_name)
    if not pred_norm:
        return False, ""
    aliases = {norm_text(alias) for alias in person.accepted_aliases if norm_text(alias)}
    if pred_norm in aliases:
        return True, "exact_alias"
    if len(name_tokens(pred_name)) >= 2:
        return False, ""
    pred_first = first_name(pred_name)
    if not pred_first:
        return False, ""
    matched_person_ids = []
    for other in all_targets:
        surfaces = [other.canonical_name, *other.accepted_aliases]
        firsts = {first_name(surface) for surface in surfaces if first_name(surface)}
        if pred_first in firsts:
            matched_person_ids.append(other.person_id)
    unique_ids = set(matched_person_ids)
    if len(unique_ids) == 1 and person.person_id in unique_ids:
        return True, "unique_first_name"
    return False, ""


def _identity_exact_alias(pred_name: str, person: TargetPerson) -> bool:
    pred_norm = norm_text(pred_name)
    return bool(pred_norm) and any(pred_norm == norm_text(alias) for alias in person.accepted_aliases)


def _identity_first_name_partial(pred_name: str, person: TargetPerson) -> bool:
    pred_first = first_name(pred_name)
    if not pred_first:
        return False
    return any(pred_first == first_name(alias) for alias in person.accepted_aliases if first_name(alias))


def norm_text(text: str) -> str:
    text = str(text or "").translate(UNICODE_TRANSLATION).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def name_tokens(text: str) -> list[str]:
    return norm_text(text).split()


def first_name(text: str) -> str:
    tokens = name_tokens(text)
    return tokens[0] if tokens else ""


def _canonicalize_relation(text: str) -> str:
    norm = norm_text(text)
    if not norm:
        return ""
    if norm in RELATION_SYNONYMS:
        return RELATION_SYNONYMS[norm]
    for phrase, canonical in sorted(RELATION_SYNONYMS.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", norm):
            return canonical
    return norm


def _canonicalize_category(text: str) -> str:
    norm = norm_text(text)
    if not norm:
        return ""
    return CATEGORY_SYNONYMS.get(norm, norm)


def _relation_score(gt_relation: str, pred_relation: str, gt_category: str, pred_category: str) -> float:
    gt_rel = _canonicalize_relation(gt_relation)
    pred_rel = _canonicalize_relation(pred_relation)
    if gt_rel and pred_rel and gt_rel == pred_rel:
        return 1.0
    gt_cat = _canonicalize_category(gt_category) or RELATION_TO_CATEGORY.get(gt_rel, "")
    pred_cat = _canonicalize_category(pred_category) or RELATION_TO_CATEGORY.get(pred_rel, "")
    if gt_cat and pred_cat and gt_cat == pred_cat:
        return 0.5
    return 0.0


def _category_score(gt_category: str, pred_category: str) -> float:
    gt = _canonicalize_category(gt_category)
    pred = _canonicalize_category(pred_category)
    return 1.0 if gt and pred and gt == pred else 0.0


def _heuristic_efs(
    *,
    target_type: str,
    correctness: float,
    raw_value_score: float,
    identity_bound: bool,
    prediction: str,
    evidence_ids: list[str],
    reasoning_path: str,
    public_face_id: str,
    photo_index: dict[str, dict[str, Any]],
) -> float:
    if not str(prediction or "").strip():
        return 0.0
    if not evidence_ids:
        return 0.0
    if _invalid_photo_ids(evidence_ids, photo_index):
        return 0.0

    has_reasoning = bool(str(reasoning_path or "").strip())
    has_face = bool(public_face_id) and _evidence_has_face(evidence_ids, public_face_id, photo_index)
    has_text_anchor = _evidence_mentions_prediction(evidence_ids, prediction, photo_index)

    if target_type == "owner_fact_atom":
        if correctness >= 1.0:
            return 1.0 if has_reasoning else 0.5
        if correctness >= 0.5:
            return 0.5
        return 0.0

    if target_type == "person_name":
        if correctness < 1.0:
            return 0.0
        if has_face and (has_text_anchor or has_reasoning):
            return 1.0
        return 0.5

    if target_type in {"person_relation", "person_category"}:
        if identity_bound:
            if correctness >= 1.0:
                return 1.0 if (has_face or has_reasoning) else 0.5
            if correctness >= 0.5:
                return 0.5
            return 0.0
        if raw_value_score > 0.0:
            return 0.5
        return 0.0

    return 0.0


def _photo_index(agent_album: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(photo.get("photo_id") or ""): photo
        for photo in agent_album.get("photos") or []
        if str(photo.get("photo_id") or "")
    }


def _invalid_photo_ids(evidence_ids: list[str], photo_index: dict[str, dict[str, Any]]) -> list[str]:
    if not photo_index:
        return list(evidence_ids)
    return [photo_id for photo_id in evidence_ids if photo_id not in photo_index]


def _evidence_has_face(evidence_ids: list[str], face_id: str, photo_index: dict[str, dict[str, Any]]) -> bool:
    for photo_id in evidence_ids:
        photo = photo_index.get(photo_id) or {}
        faces = {str(face) for face in photo.get("visible_face_ids") or []}
        if face_id in faces:
            return True
    return False


def _evidence_mentions_prediction(evidence_ids: list[str], prediction: str, photo_index: dict[str, dict[str, Any]]) -> bool:
    pred_tokens = {tok for tok in name_tokens(prediction) if len(tok) >= 3}
    if not pred_tokens:
        return False
    text_parts = []
    for photo_id in evidence_ids:
        photo = photo_index.get(photo_id) or {}
        text_parts.append(str(photo.get("caption") or ""))
        text_parts.extend(str(x) for x in photo.get("visible_text") or [])
        for entity in photo.get("text_entities") or []:
            if isinstance(entity, dict):
                text_parts.append(str(entity.get("surface") or entity.get("text") or ""))
            else:
                text_parts.append(str(entity))
    blob_tokens = set(name_tokens(" ".join(text_parts)))
    return bool(pred_tokens & blob_tokens)


def _append_calibration(
    rows: list[dict[str, Any]],
    *,
    target: dict[str, Any],
    confidence: float,
    correctness: float,
    strict_correctness: float,
) -> None:
    rows.append(
        {
            "target_id": str(target.get("target_id") or ""),
            "target_type": str(target.get("target_type") or ""),
            "confidence": confidence,
            "correctness": max(0.0, min(1.0, float(correctness))),
            "strict_correctness": max(0.0, min(1.0, float(strict_correctness))),
        }
    )


def _append_efs_diagnostics(
    efs_by_target_type: dict[str, list[float]],
    efs_given_recovered_scores: list[float],
    efs_given_correct_scores: list[float],
    *,
    target_type: str,
    correctness: float,
    efs: float,
) -> None:
    score = max(0.0, min(1.0, float(efs)))
    corr = max(0.0, min(1.0, float(correctness)))
    efs_by_target_type[target_type].append(score)
    if corr > 0.0:
        efs_given_recovered_scores.append(score)
    if corr >= 1.0:
        efs_given_correct_scores.append(score)


def _target_row(
    *,
    user_id: str,
    target: dict[str, Any],
    prediction: str,
    predicted_face_id: str | None,
    confidence: float,
    correctness: float,
    efs: float,
    evidence_ids: list[str],
    matched_index: int | None,
    judge_details: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "target_id": str(target.get("target_id") or ""),
        "target_type": str(target.get("target_type") or ""),
        "person_id": target.get("person_id"),
        "public_face_id": target.get("public_face_id"),
        "predicted_face_id": predicted_face_id,
        "gt_value": target.get("gt_value") if target.get("gt_value") is not None else target.get("gt_text"),
        "prediction": prediction,
        "confidence": confidence,
        "correctness": round(float(correctness), 4),
        "EFS": round(float(efs), 4),
        "evidence_photo_ids": evidence_ids,
        "key_photo_ids_public": [str(pid) for pid in target.get("key_photo_ids_public") or []],
        "alignment_difficulty": target.get("alignment_difficulty"),
        "inference_type": target.get("inference_type"),
        "matched_index": matched_index,
        "judge_details": judge_details or {},
    }


def _evidence_ids(row: dict[str, Any]) -> list[str]:
    return [str(pid) for pid in row.get("evidence_photo_ids") or [] if str(pid).strip()]


def _confidence_or_zero(row: dict[str, Any]) -> float:
    if not row or row.get("confidence") is None:
        return 0.0
    try:
        return max(0.0, min(1.0, float(row.get("confidence"))))
    except (TypeError, ValueError):
        return 0.0


def _n_llm(run_stats: dict[str, Any], predicted_profile: dict[str, Any]) -> int:
    stats = run_stats or (predicted_profile.get("run_stats") or {})
    try:
        return int(stats.get("n_llm_calls", 0) or 0)
    except (TypeError, ValueError):
        return 0


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


def _ece(rows: list[dict[str, Any]], *, score_key: str, fill_missing: float | None, bins: int = 10) -> float:
    pairs = []
    for row in rows:
        confidence = row.get("confidence")
        if confidence is None:
            if fill_missing is None:
                continue
            confidence = fill_missing
        try:
            conf = max(0.0, min(1.0, float(confidence)))
            correct = max(0.0, min(1.0, float(row.get(score_key) or 0.0)))
        except (TypeError, ValueError):
            continue
        pairs.append((conf, correct))
    if not pairs:
        return 0.0
    err = 0.0
    total = len(pairs)
    for bucket_id in range(bins):
        lo = bucket_id / bins
        hi = (bucket_id + 1) / bins
        bucket = [
            (conf, correct)
            for conf, correct in pairs
            if (lo <= conf < hi) or (bucket_id == bins - 1 and conf == 1.0)
        ]
        if not bucket:
            continue
        avg_conf = sum(conf for conf, _ in bucket) / len(bucket)
        avg_acc = sum(correct for _, correct in bucket) / len(bucket)
        err += len(bucket) / total * abs(avg_conf - avg_acc)
    return err


def _mean(values) -> float:
    vals = [
        float(value)
        for value in values
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    ]
    return sum(vals) / len(vals) if vals else 0.0


def _mean_optional(values) -> float | None:
    vals = [
        float(value)
        for value in values
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    ]
    return sum(vals) / len(vals) if vals else None


def _macro_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    return _mean_optional(row.get(key) for row in rows)


def _aggregate_nested_mean(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values_by_name: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        nested = row.get(key) or {}
        if not isinstance(nested, dict):
            continue
        for nested_key, value in nested.items():
            if isinstance(value, (int, float)):
                values_by_name[str(nested_key)].append(float(value))
    return {
        nested_key: round(_mean(values), 4)
        for nested_key, values in sorted(values_by_name.items())
    }


def _round_mapping(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, float):
            out[key] = round(value, 4)
        elif isinstance(value, dict):
            out[key] = _round_mapping(value)
        else:
            out[key] = value
    return out


def _unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        norm = norm_text(value)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(str(value))
    return out


def _uses_llm_judge(mode: str) -> bool:
    normalized = str(mode or "").strip().lower()
    return normalized in {
        "llm",
        "llm_semantic",
        "semantic_llm",
        "qwen36",
        "qwen3.6",
        "qwen3_6",
    } or normalized.startswith("llm_")


def _cached_judge_result(
    judge_cache: Any,
    *,
    kind: str,
    payload: dict[str, Any],
    compute: Any,
) -> dict[str, Any]:
    if judge_cache is None:
        return dict(compute() or {})
    if hasattr(judge_cache, "get_or_compute"):
        return dict(judge_cache.get_or_compute(kind=kind, payload=payload, compute=compute))

    key = _cache_key(kind, payload)
    if isinstance(judge_cache, dict):
        existing = judge_cache.get(key)
        if isinstance(existing, dict):
            return {**existing, "cache_hit": True}
        result = dict(compute() or {})
        result.setdefault("cache_key", key)
        result.setdefault("cache_kind", kind)
        judge_cache[key] = result
        return {**result, "cache_hit": False}
    return dict(compute() or {})


def _cache_key(kind: str, payload: dict[str, Any]) -> str:
    text = json.dumps(
        {"kind": kind, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _valid_index(value: Any, length: int) -> int | None:
    if value is None:
        return None
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return None
    return idx if 0 <= idx < length else None


def _public_photo_excerpt(photo: dict[str, Any], *, query_terms: list[str] | None = None) -> dict[str, Any]:
    metadata = photo.get("metadata") if isinstance(photo.get("metadata"), dict) else {}
    text_entities = []
    for entity in photo.get("text_entities") or []:
        if isinstance(entity, dict):
            text_entities.append(
                {
                    "surface": _truncate(str(entity.get("surface") or entity.get("text") or ""), 80),
                    "entity_type": str(entity.get("entity_type") or ""),
                    "source": str(entity.get("source") or ""),
                }
            )
        else:
            text_entities.append({"surface": _truncate(str(entity), 80)})
        if len(text_entities) >= 16:
            break
    relevant_text = _relevant_public_text(photo, text_entities, query_terms or [])
    return {
        "photo_id": str(photo.get("photo_id") or ""),
        "year_month": str(photo.get("year_month") or ""),
        "caption": _truncate(str(photo.get("caption") or ""), 700),
        "visible_text": [
            _truncate(str(item), 140)
            for item in (photo.get("visible_text") or [])[:14]
        ],
        "relevant_text": relevant_text,
        "text_entities": text_entities,
        "visible_face_ids": [str(face) for face in (photo.get("visible_face_ids") or [])[:30]],
        "metadata": {
            "gps_city": metadata.get("gps_city"),
            "gps_location": metadata.get("gps_location"),
            "device": metadata.get("device"),
        },
    }


def _efs_query_terms(*values: str) -> list[str]:
    stop = {
        "album",
        "owner",
        "person",
        "photo",
        "face",
        "friend",
        "family",
        "member",
        "other",
        "relation",
        "category",
        "colleague",
        "classmate",
        "neighbor",
        "the",
        "and",
        "with",
    }
    terms: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        norm = norm_text(text)
        if len(norm) >= 4 and norm not in stop and not norm.startswith("face "):
            terms.append(norm)
        for token in name_tokens(text):
            if len(token) >= 4 and token not in stop:
                terms.append(token)
    return _unique(terms)[:12]


def _relevant_public_text(
    photo: dict[str, Any],
    text_entities: list[dict[str, Any]],
    query_terms: list[str],
) -> list[str]:
    if not query_terms:
        return []
    items = [
        str(photo.get("caption") or ""),
        *(str(item) for item in (photo.get("visible_text") or [])),
        *(str(entity.get("surface") or "") for entity in text_entities),
    ]
    relevant: list[str] = []
    for item in items:
        text = " ".join(str(item or "").split())
        if not text:
            continue
        normalized = norm_text(text)
        if any(term and term in normalized for term in query_terms):
            relevant.append(_truncate(text, 220))
        if len(relevant) >= 10:
            break
    return _unique(relevant)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _extract_json_object(raw: Any) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1).strip())
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

"""Batched LLM identity adjudication for ERB-expanded candidates."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from typing import Any

from .config import ECDConfig
from .llm_context import photo_brief, truncate
from .llm_reranker import (
    _face_review_priority,
    _is_overbroad_recall_candidate,
    _owner_like_candidate,
    _select_review_candidates,
)
from .schemas import EvidenceInventory, IdentityHypothesis, NameCluster
from .text import dedupe, normalize_name

logger = logging.getLogger(__name__)

_SYSTEM = """You are the batched identity adjudicator of Evidence-Chain Dossier.
You receive several face_ids and compact candidate cards.

Rules:
- Assign at most one candidate to each face_id.
- A name_cluster_id should be assigned to at most one face unless duplicate-name evidence is explicit.
- Use only candidate_index/name_cluster_id values provided for that face.
- Do not assign owner names to non-owner faces.
- overbroad_recall=true means the candidate is useful for recall but weak for identity; select it only with direct or very specific supporting evidence.
- Prefer specific evidence: same-photo face/name labels, role words, unique event bridges, repeated co-face patterns.
- Abstain when candidates are indistinguishable.
- Keep each reasoning_path under 30 words, one sentence, no newline.
- Output ONLY one JSON object."""


def adjudicate_identity_batches(
    *,
    inventory: EvidenceInventory,
    clusters: list[NameCluster],
    hypotheses: list[IdentityHypothesis],
    config: ECDConfig,
    llm: Any,
) -> tuple[list[IdentityHypothesis], dict[str, Any]]:
    """Use fewer LLM calls to review multiple high-value faces per call."""
    if llm is None:
        return hypotheses, {"enabled": False, "reason": "no_llm"}
    if config.llm_batch_max_calls <= 0 or config.llm_batch_max_faces <= 0:
        return hypotheses, {"enabled": False, "reason": "batch_budget_zero"}

    cluster_lookup = {c.cluster_id: c for c in clusters}
    by_face: dict[str, list[IdentityHypothesis]] = defaultdict(list)
    for hyp in hypotheses:
        by_face[hyp.face_id].append(hyp)
    for rows in by_face.values():
        rows.sort(key=lambda h: (-h.score, h.name_cluster_id))

    face_ids = [
        fid for fid in sorted(
            by_face,
            key=lambda face_id: (-_batch_face_priority(inventory, cluster_lookup, by_face[face_id]), face_id),
        )
        if fid != inventory.owner_face_id
    ][: config.llm_batch_max_faces]

    batches = [
        face_ids[i : i + max(1, config.llm_batch_faces_per_call)]
        for i in range(0, len(face_ids), max(1, config.llm_batch_faces_per_call))
    ][: config.llm_batch_max_calls]

    diagnostics: dict[str, Any] = {
        "enabled": True,
        "n_batches": len(batches),
        "batches": {},
        "errors": [],
    }

    used_clusters: set[str] = set()
    for batch_idx, batch_faces in enumerate(batches, start=1):
        batch_id = f"identity_batch_{batch_idx:02d}"
        candidate_rows = {
            face_id: _batch_candidates(inventory, cluster_lookup, by_face.get(face_id, []), config)
            for face_id in batch_faces
        }
        candidate_rows = {face_id: rows for face_id, rows in candidate_rows.items() if rows}
        if not candidate_rows:
            diagnostics["batches"][batch_id] = {"faces": batch_faces, "skipped": "no_candidates"}
            continue

        payload = _build_batch_payload(inventory, cluster_lookup, candidate_rows, config)
        result = llm.call_json(
            prompt=json.dumps(payload, ensure_ascii=False),
            system=_SYSTEM,
            temperature=0.1,
            max_tokens=4096,
            retries=1,
            stage="llm_batch_adjudicator",
        )
        if not isinstance(result, dict):
            diagnostics["errors"].append({"batch_id": batch_id, "error": "no_json"})
            diagnostics["batches"][batch_id] = {"faces": list(candidate_rows), "error": "no_json"}
            continue

        applied = _apply_batch_result(
            candidate_rows=candidate_rows,
            cluster_lookup=cluster_lookup,
            result=result,
            config=config,
            used_clusters=used_clusters,
        )
        diagnostics["batches"][batch_id] = {
            "faces": list(candidate_rows),
            "raw_result": result,
            "applied": applied,
        }

    _recompute_margins(by_face)
    return hypotheses, diagnostics


def _batch_candidates(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    rows: list[IdentityHypothesis],
    config: ECDConfig,
) -> list[IdentityHypothesis]:
    limit = max(1, config.llm_batch_top_candidates_per_face)
    selected = _select_review_candidates(inventory, cluster_lookup, rows, config)
    return selected[:limit]


def _build_batch_payload(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    candidate_rows: dict[str, list[IdentityHypothesis]],
    config: ECDConfig,
) -> dict[str, Any]:
    chars = min(config.llm_photo_snippet_chars, 140)
    return {
        "user_id": inventory.album.get("user_id"),
        "owner": {
            "face_id": inventory.owner_face_id,
            "name_candidate": inventory.owner_name,
            "last_name": inventory.owner_last_name,
        },
        "policy": {
            "goal": "Select identity bindings only when evidence separates candidates.",
            "overbroad_recall": "Do not let broad home/city/event similarity override cleaner direct or one-to-one evidence.",
            "confidence": "Use 0.55-0.75 for indirect but specific bridges; >0.8 only for direct/repeated evidence.",
        },
        "faces": [
            _compact_face_card(inventory, face_id, chars=chars)
            for face_id in candidate_rows
        ],
        "candidate_options_by_face": {
            face_id: [
                _compact_candidate_card(cluster_lookup, hyp, index=i)
                for i, hyp in enumerate(rows)
            ]
            for face_id, rows in candidate_rows.items()
        },
        "output_schema": {
            "assignments": [
                {
                    "face_id": "face id",
                    "selected_candidate_index": "integer index from candidate_options_by_face[face_id], or null",
                    "name_cluster_id": "selected cluster id",
                    "canonical_name": "candidate name",
                    "confidence": "0.0-1.0",
                    "relation_to_owner": "specific relation if inferable, else empty",
                    "relation_category": "family|friend|colleague|classmate|neighbor|other|empty",
                    "evidence_photo_ids": ["photo ids"],
                    "reasoning_path": "one short sentence, <=30 words, no newline",
                }
            ],
            "abstentions": [
                {"face_id": "face id", "failure_mode": "short reason"}
            ],
        },
    }


def _compact_face_card(inventory: EvidenceInventory, face_id: str, *, chars: int) -> dict[str, Any]:
    units = inventory.face_units_by_face.get(face_id, [])
    owner_count = sum(1 for unit in units if unit.owner_present)
    co_faces = Counter(cf for unit in units for cf in unit.co_faces)
    relation_terms = Counter(term.lower() for unit in units for term in unit.relation_terms)
    activity_terms = Counter(term.lower() for unit in units for term in unit.activity_terms)
    venue_terms = Counter(term.lower() for unit in units for term in unit.venue_terms)
    photos = _representative_face_photos(units, max_photos=3)
    return {
        "face_id": face_id,
        "n_appearances": len(units),
        "owner_coappearances": owner_count,
        "top_co_faces": [f"{fid}:{n}" for fid, n in co_faces.most_common(5)],
        "relation_terms": [term for term, _ in relation_terms.most_common(5)],
        "activity_terms": [term for term, _ in activity_terms.most_common(6)],
        "venue_terms": [term for term, _ in venue_terms.most_common(6)],
        "photos": [photo_brief(inventory, pid, chars=chars) for pid in photos],
    }


def _compact_candidate_card(
    cluster_lookup: dict[str, NameCluster],
    hyp: IdentityHypothesis,
    *,
    index: int,
) -> dict[str, Any]:
    cluster = cluster_lookup.get(hyp.name_cluster_id)
    signals = hyp.signal_breakdown or {}
    return {
        "candidate_index": index,
        "name_cluster_id": hyp.name_cluster_id,
        "canonical_name_candidate": hyp.canonical_name_candidate,
        "observed_surface": hyp.observed_surface,
        "score": round(float(hyp.score), 4),
        "margin": round(float(hyp.margin), 4),
        "surfaces": (cluster.surfaces[:5] if cluster else [hyp.observed_surface]),
        "quality_flags": cluster.quality_flags if cluster else [],
        "overbroad_recall": _is_overbroad_recall_candidate(hyp),
        "signals": {
            key: round(float(signals.get(key) or 0.0), 4)
            for key in [
                "same_photo_weight",
                "same_photo_count",
                "strict_bridge_count",
                "near_bridge_count",
                "event_bridge",
                "text_only_event_bridge",
                "role_channel_bridge",
                "event_bridge_count",
            ]
        },
        "same_photo_ids": hyp.evidence_packet.same_photo_ids[:4],
        "bridge_pairs": _compact_bridge_pairs(hyp)[:5],
        "relation_clues": hyp.evidence_packet.relation_clues[:6],
        "activity_clues": hyp.evidence_packet.activity_clues[:6],
        "venue_clues": hyp.evidence_packet.venue_clues[:6],
        "narrative_summary": truncate(hyp.evidence_packet.narrative_summary, 260),
    }


def _compact_bridge_pairs(hyp: IdentityHypothesis) -> list[dict[str, Any]]:
    rows = []
    for pair in hyp.evidence_packet.bridge_photo_pairs:
        rows.append(
            {
                "text_photo_id": str(pair.get("text_photo_id") or ""),
                "face_photo_id": str(pair.get("face_photo_id") or ""),
                "bridge": str(pair.get("bridge") or ""),
                "event_score": _float(pair.get("event_score")),
                "event_channels": [str(k) for k in (pair.get("event_channels") or [])[:4]],
                "shared_keywords": [str(k) for k in (pair.get("shared_keywords") or [])[:4]],
                "strong_signal": bool(pair.get("strong_signal")),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["event_score"]),
            row["bridge"],
            row["text_photo_id"],
            row["face_photo_id"],
        )
    )
    return rows


def _apply_batch_result(
    *,
    candidate_rows: dict[str, list[IdentityHypothesis]],
    cluster_lookup: dict[str, NameCluster],
    result: dict[str, Any],
    config: ECDConfig,
    used_clusters: set[str],
) -> list[dict[str, Any]]:
    assignments = result.get("assignments") or []
    if not isinstance(assignments, list):
        return []

    applied = []
    used_faces: set[str] = set()
    parsed = []
    for assignment in assignments:
        if isinstance(assignment, dict):
            parsed.append((_bounded_float(assignment.get("confidence"), 0.0), assignment))
    parsed.sort(key=lambda row: -row[0])

    for confidence, assignment in parsed:
        face_id = str(assignment.get("face_id") or "")
        if face_id in used_faces or face_id not in candidate_rows:
            continue
        if confidence < config.llm_accept_min_confidence:
            applied.append({"face_id": face_id, "status": "skipped_low_confidence", "confidence": round(confidence, 4)})
            continue
        selected = _match_assignment(assignment, candidate_rows[face_id], cluster_lookup)
        if selected is None:
            applied.append({"face_id": face_id, "status": "skipped_no_candidate_match", "confidence": round(confidence, 4)})
            continue
        if selected.name_cluster_id in used_clusters:
            applied.append(
                {
                    "face_id": face_id,
                    "name_cluster_id": selected.name_cluster_id,
                    "status": "skipped_name_used_by_prior_batch",
                    "confidence": round(confidence, 4),
                }
            )
            continue
        if _is_overbroad_recall_candidate(selected):
            selected.llm_decision = "batch_guarded_overbroad_select"
            selected.llm_confidence = confidence
            selected.llm_failure_mode = "overbroad_recall_guard"
            selected.llm_reasoning_path = str(assignment.get("reasoning_path") or "")
            applied.append(
                {
                    "face_id": face_id,
                    "name_cluster_id": selected.name_cluster_id,
                    "status": "skipped_overbroad_recall_candidate",
                    "confidence": round(confidence, 4),
                }
            )
            continue
        if not _safe_to_apply_identity(selected, candidate_rows[face_id]):
            selected.llm_decision = "batch_guarded_non_top_select"
            selected.llm_confidence = confidence
            selected.llm_failure_mode = "non_top_without_strong_anchor"
            selected.llm_reasoning_path = str(assignment.get("reasoning_path") or "")
            applied.append(
                {
                    "face_id": face_id,
                    "name_cluster_id": selected.name_cluster_id,
                    "status": "skipped_non_top_without_strong_anchor",
                    "confidence": round(confidence, 4),
                }
            )
            continue

        _apply_selected(selected, assignment, confidence, config)
        if config.llm_batch_identity_mode != "readout_only":
            _demote_competitors(selected, candidate_rows[face_id], confidence)
        used_faces.add(face_id)
        if config.llm_batch_identity_mode != "readout_only":
            used_clusters.add(selected.name_cluster_id)
        applied.append(
            {
                "face_id": face_id,
                "name_cluster_id": selected.name_cluster_id,
                "canonical_name": selected.canonical_name_candidate,
                "status": "applied",
                "confidence": round(confidence, 4),
                "score": round(selected.score, 4),
            }
        )
    return applied


def _apply_selected(
    hyp: IdentityHypothesis,
    assignment: dict[str, Any],
    confidence: float,
    config: ECDConfig,
) -> None:
    readout_only = config.llm_batch_identity_mode == "readout_only"
    if not readout_only:
        hyp.status = "llm_batch_selected"
    hyp.llm_decision = "batch_readout" if readout_only else "batch_selected"
    hyp.llm_confidence = confidence
    hyp.llm_relation_to_owner = str(assignment.get("relation_to_owner") or "")
    hyp.llm_relation_category = str(assignment.get("relation_category") or "")
    hyp.llm_reasoning_path = str(assignment.get("reasoning_path") or "")
    hyp.llm_evidence_photo_ids = [
        str(pid) for pid in (assignment.get("evidence_photo_ids") or []) if str(pid).startswith("photo_")
    ]
    hyp.signal_breakdown["llm_batch_confidence"] = confidence
    if not readout_only:
        hyp.llm_canonical_name = normalize_name(str(assignment.get("canonical_name") or hyp.canonical_name_candidate))
    if hyp.llm_canonical_name and not readout_only:
        hyp.canonical_name_candidate = hyp.llm_canonical_name
    if hyp.llm_reasoning_path:
        hyp.evidence_packet.narrative_summary = hyp.llm_reasoning_path
    if readout_only:
        return
    if _has_strong_direct_anchor(hyp):
        hyp.score = max(hyp.score, min(0.98, 0.42 + 0.52 * confidence))
    else:
        hyp.score = max(hyp.score, min(0.62, hyp.score + 0.08))


def _demote_competitors(
    selected: IdentityHypothesis,
    rows: list[IdentityHypothesis],
    confidence: float,
) -> None:
    if not _has_strong_direct_anchor(selected):
        return
    margin = 0.14 if confidence < 0.82 else 0.18
    for hyp in rows:
        if hyp is selected:
            continue
        hyp.llm_decision = hyp.llm_decision or "batch_not_selected"
        hyp.llm_confidence = max(float(hyp.llm_confidence or 0.0), confidence)
        hyp.status = "llm_batch_rejected"
        hyp.score = min(hyp.score, max(0.0, selected.score - margin))


def _match_assignment(
    assignment: dict[str, Any],
    rows: list[IdentityHypothesis],
    cluster_lookup: dict[str, NameCluster],
) -> IdentityHypothesis | None:
    try:
        idx = int(assignment.get("selected_candidate_index"))
    except (TypeError, ValueError):
        idx = -1
    if 0 <= idx < len(rows):
        candidate = rows[idx]
        requested_cluster = str(assignment.get("name_cluster_id") or "")
        if not requested_cluster or requested_cluster == candidate.name_cluster_id:
            return candidate

    requested_cluster = str(assignment.get("name_cluster_id") or "")
    if requested_cluster:
        for hyp in rows:
            if hyp.name_cluster_id == requested_cluster:
                return hyp

    requested_name = normalize_name(str(assignment.get("canonical_name") or "")).lower()
    if not requested_name:
        return None
    for hyp in rows:
        names = {
            normalize_name(hyp.canonical_name_candidate).lower(),
            normalize_name(hyp.observed_surface).lower(),
            normalize_name(hyp.llm_canonical_name).lower(),
        }
        cluster = cluster_lookup.get(hyp.name_cluster_id)
        if cluster:
            names.update(normalize_name(surface).lower() for surface in cluster.surfaces)
        if requested_name in names:
            return hyp
    return None


def _safe_to_apply_identity(
    selected: IdentityHypothesis,
    rows: list[IdentityHypothesis],
) -> bool:
    if not rows:
        return False
    if selected is rows[0]:
        return True
    return _has_strong_direct_anchor(selected)


def _batch_face_priority(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    rows: list[IdentityHypothesis],
) -> float:
    base = _face_review_priority(inventory, cluster_lookup, rows)
    recall_bonus = 0.0
    for hyp in rows[:8]:
        try:
            if float(hyp.signal_breakdown.get("event_bridge") or 0.0) >= 0.5:
                recall_bonus += 0.05
            if _is_overbroad_recall_candidate(hyp):
                recall_bonus -= 0.02
        except (TypeError, ValueError):
            continue
    return base + min(0.25, recall_bonus)


def _representative_face_photos(units, *, max_photos: int) -> list[str]:
    scored = []
    for unit in units:
        score = 0
        if unit.owner_present:
            score += 3
        if unit.visible_text:
            score += 2
        score += min(3, len(unit.relation_terms) + len(unit.activity_terms))
        score += 1 if len(unit.co_faces) <= 2 else 0
        scored.append((score, unit.timestamp, unit.photo_id))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    return dedupe(pid for _, _, pid in scored)[:max_photos]


def _has_strong_direct_anchor(hyp: IdentityHypothesis) -> bool:
    try:
        same_weight = float(hyp.signal_breakdown.get("same_photo_weight") or 0.0)
    except (TypeError, ValueError):
        same_weight = 0.0
    return bool(hyp.evidence_packet.same_photo_ids and same_weight >= 0.45)


def _recompute_margins(by_face: dict[str, list[IdentityHypothesis]]) -> None:
    for rows in by_face.values():
        rows.sort(key=lambda h: (-h.score, h.name_cluster_id))
        for idx, hyp in enumerate(rows):
            next_score = rows[idx + 1].score if idx + 1 < len(rows) else 0.0
            hyp.margin = max(0.0, hyp.score - next_score)


def _bounded_float(value: Any, fallback: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return fallback


def _float(value: Any) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return 0.0

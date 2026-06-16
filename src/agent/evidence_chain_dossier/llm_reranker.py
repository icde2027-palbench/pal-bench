"""LLM-backed face-name evidence-chain reranking."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from .config import ECDConfig
from .llm_context import candidate_context, face_context
from .schemas import EvidenceInventory, IdentityHypothesis, NameCluster
from .text import first_name, is_full_name, last_name, normalize_name

logger = logging.getLogger(__name__)

_SYSTEM = """You are the LLM core of Evidence-Chain Dossier, an album reasoning agent.
You receive public album evidence only. Your task is to decide which name candidate, if any, best binds to a face_id.

Important rules:
- Do not invent names outside the candidate list.
- Prefer explicit relation/role evidence over raw bridge counts when they conflict.
- A same-photo name is strong only when the photo plausibly labels the visible person; menus, posters, books, software labels, and random OCR can be noise.
- In multi-person family scenes, use role words (mom, dad, partner), age/gender cues in captions, co-face pattern, and repeated context to avoid swaps.
- Treat event_bridge_pairs as recall evidence: text-only names can bind to faces in different photos when time, place, event keywords, activity, or role channel line up coherently.
- If evidence is ambiguous, abstain instead of forcing a name.
- Keep reasoning_path under 35 words and do not write per-candidate analysis.
- Output ONLY one JSON object."""


def rerank_identity_hypotheses(
    *,
    inventory: EvidenceInventory,
    clusters: list[NameCluster],
    hypotheses: list[IdentityHypothesis],
    config: ECDConfig,
    llm: Any,
) -> tuple[list[IdentityHypothesis], dict[str, Any]]:
    """Use an LLM to review top-k candidates per face and adjust scores."""
    if llm is None:
        return hypotheses, {"enabled": False, "reason": "no_llm"}

    cluster_lookup = {c.cluster_id: c for c in clusters}
    by_face: dict[str, list[IdentityHypothesis]] = defaultdict(list)
    for hyp in hypotheses:
        by_face[hyp.face_id].append(hyp)
    for rows in by_face.values():
        rows.sort(key=lambda h: (-h.score, h.name_cluster_id))

    diagnostics: dict[str, Any] = {"enabled": True, "faces": {}, "errors": []}
    face_ids = sorted(
        by_face,
        key=lambda fid: (-_face_review_priority(inventory, cluster_lookup, by_face[fid]), fid),
    )[: config.llm_max_face_reviews]

    for face_id in face_ids:
        candidates = _select_review_candidates(
            inventory,
            cluster_lookup,
            by_face.get(face_id, []),
            config,
        )
        if not candidates:
            continue
        payload = {
            "user_id": inventory.album.get("user_id"),
            "owner": {
                "face_id": inventory.owner_face_id,
                "name_candidate": inventory.owner_name,
                "last_name": inventory.owner_last_name,
            },
            "face": face_context(
                inventory,
                face_id,
                max_photos=6 if config.use_evidence_recall_broadener else 8,
                chars=config.llm_photo_snippet_chars,
            ),
            "candidates": [
                candidate_context(
                    inventory,
                    cluster_lookup,
                    hyp,
                    index=i,
                    chars=config.llm_photo_snippet_chars,
                )
                for i, hyp in enumerate(candidates)
            ],
            "output_schema": {
                "decision": "select|abstain",
                "selected_candidate_index": "integer or null",
                "confidence": "0.0-1.0",
                "canonical_name": "selected candidate name or empty",
                "relation_to_owner": "specific relation if inferable, else empty",
                "relation_category": "family|friend|colleague|classmate|neighbor|other|empty",
                "evidence_photo_ids": ["photo ids supporting the decision"],
                "failure_mode": "short label if abstaining or if evidence is weak",
                "reasoning_path": "one short sentence, <=35 words, cite photo_ids, no newline",
            },
        }
        result = llm.call_json(
            prompt=json.dumps(payload, ensure_ascii=False),
            system=_SYSTEM,
            temperature=0.1,
            max_tokens=4096,
            retries=1,
            stage="llm_reranker",
        )
        if not isinstance(result, dict):
            diagnostics["errors"].append({"face_id": face_id, "error": "no_json"})
            continue
        diagnostics["faces"][face_id] = result
        _apply_face_decision(candidates, result, config)

    _recompute_margins(by_face)
    return hypotheses, diagnostics


def _apply_face_decision(
    candidates: list[IdentityHypothesis],
    result: dict[str, Any],
    config: ECDConfig,
) -> None:
    decision = str(result.get("decision") or "").lower()
    try:
        selected_idx = int(result.get("selected_candidate_index"))
    except (TypeError, ValueError):
        selected_idx = -1
    try:
        confidence = float(result.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    if decision != "select" or selected_idx < 0 or selected_idx >= len(candidates):
        for hyp in candidates:
            hyp.llm_decision = decision or "abstain"
            hyp.llm_failure_mode = str(result.get("failure_mode") or "")
        return

    selected = candidates[selected_idx]
    if confidence < config.llm_accept_min_confidence:
        selected.llm_decision = "low_confidence_select"
        selected.llm_confidence = confidence
        selected.llm_failure_mode = str(result.get("failure_mode") or "below_accept_threshold")
        return
    if _is_overbroad_recall_candidate(selected):
        selected.llm_decision = "guarded_overbroad_select"
        selected.llm_confidence = confidence
        selected.llm_failure_mode = "overbroad_recall_guard"
        selected.llm_reasoning_path = str(result.get("reasoning_path") or "")
        return

    selected.llm_decision = "selected"
    selected.llm_confidence = confidence
    selected.llm_canonical_name = str(result.get("canonical_name") or selected.canonical_name_candidate)
    selected.llm_relation_to_owner = str(result.get("relation_to_owner") or "")
    selected.llm_relation_category = str(result.get("relation_category") or "")
    selected.llm_reasoning_path = str(result.get("reasoning_path") or "")
    selected.llm_failure_mode = str(result.get("failure_mode") or "")
    selected.llm_evidence_photo_ids = [
        str(pid) for pid in (result.get("evidence_photo_ids") or []) if str(pid).startswith("photo_")
    ]
    selected.status = "llm_selected"
    selected.signal_breakdown["llm_confidence"] = confidence
    if _has_strong_direct_anchor(selected):
        selected.score = max(selected.score, min(0.98, 0.42 + 0.52 * confidence))
    else:
        selected.score = max(selected.score, min(selected.score + 0.08, 0.62))
    if selected.llm_canonical_name:
        selected.canonical_name_candidate = selected.llm_canonical_name
    if selected.llm_reasoning_path:
        selected.evidence_packet.narrative_summary = selected.llm_reasoning_path
    for idx, hyp in enumerate(candidates):
        if idx == selected_idx:
            continue
        hyp.llm_decision = "not_selected"
        hyp.llm_confidence = confidence
        if _has_strong_direct_anchor(selected):
            hyp.score = min(hyp.score, max(0.0, selected.score - 0.14))
        hyp.status = "llm_rejected"


def _recompute_margins(by_face: dict[str, list[IdentityHypothesis]]) -> None:
    for rows in by_face.values():
        rows.sort(key=lambda h: (-h.score, h.name_cluster_id))
        for idx, hyp in enumerate(rows):
            next_score = rows[idx + 1].score if idx + 1 < len(rows) else 0.0
            hyp.margin = max(0.0, hyp.score - next_score)


def _select_review_candidates(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    rows: list[IdentityHypothesis],
    config: ECDConfig,
) -> list[IdentityHypothesis]:
    filtered = [
        hyp for hyp in rows
        if not _owner_like_candidate(inventory, cluster_lookup.get(hyp.name_cluster_id), hyp)
    ]
    if not filtered:
        return []

    base_limit = max(1, int(config.llm_top_candidates_per_face))
    limit = base_limit
    if config.use_evidence_recall_broadener:
        limit = max(base_limit, int(config.llm_recall_top_candidates_per_face))

    selected: list[IdentityHypothesis] = []
    for hyp in filtered[:base_limit]:
        _append_unique(selected, hyp)

    if config.use_evidence_recall_broadener:
        recall_rows = sorted(
            [hyp for hyp in filtered if _recall_priority(hyp) > 0.0],
            key=lambda h: (-_recall_priority(h), -h.score, h.name_cluster_id),
        )
        for hyp in recall_rows:
            if len(selected) >= limit:
                break
            _append_unique(selected, hyp)

    for hyp in filtered:
        if len(selected) >= limit:
            break
        _append_unique(selected, hyp)
    selected.sort(key=lambda h: (-h.score, -_recall_priority(h), h.name_cluster_id))
    return selected[:limit]


def _append_unique(rows: list[IdentityHypothesis], hyp: IdentityHypothesis) -> None:
    if all(existing.name_cluster_id != hyp.name_cluster_id for existing in rows):
        rows.append(hyp)


def _recall_priority(hyp: IdentityHypothesis) -> float:
    signals = hyp.signal_breakdown or {}
    try:
        event_bridge = float(signals.get("event_bridge") or 0.0)
        text_only = float(signals.get("text_only_event_bridge") or 0.0)
        role = float(signals.get("role_channel_bridge") or 0.0)
        count = float(signals.get("event_bridge_count") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    priority = max(event_bridge, text_only + 0.04, role + 0.06)
    if count >= 2:
        priority += 0.04
    if hyp.status == "recall_broadened_candidate":
        priority += 0.04
    return priority if priority >= 0.28 else 0.0


def _has_direct_anchor(hyp: IdentityHypothesis) -> bool:
    try:
        same_weight = float(hyp.signal_breakdown.get("same_photo_weight") or 0.0)
        same_count = float(hyp.signal_breakdown.get("same_photo_count") or 0.0)
    except (TypeError, ValueError):
        same_weight = 0.0
        same_count = 0.0
    return bool(hyp.evidence_packet.same_photo_ids or same_weight > 0.0 or same_count > 0.0)


def _has_strong_direct_anchor(hyp: IdentityHypothesis) -> bool:
    try:
        same_weight = float(hyp.signal_breakdown.get("same_photo_weight") or 0.0)
    except (TypeError, ValueError):
        same_weight = 0.0
    return bool(hyp.evidence_packet.same_photo_ids and same_weight >= 0.45)


def _is_overbroad_recall_candidate(hyp: IdentityHypothesis) -> bool:
    try:
        event_count = float(hyp.signal_breakdown.get("event_bridge_count") or 0.0)
        same_weight = float(hyp.signal_breakdown.get("same_photo_weight") or 0.0)
    except (TypeError, ValueError):
        return False
    return event_count >= 60.0 and same_weight < 0.45


def _face_review_priority(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    rows: list[IdentityHypothesis],
) -> float:
    if not rows:
        return 0.0
    rows = sorted(rows, key=lambda h: (-h.score, h.name_cluster_id))
    top = rows[0]
    second_score = rows[1].score if len(rows) > 1 else 0.0
    margin = max(0.0, top.score - second_score)
    ambiguity = max(0.0, 0.22 - margin) * 3.0
    competing = sum(1 for h in rows[:5] if h.score >= top.score - 0.16) * 0.10
    owner_pressure = 0.35 if _owner_like_candidate(inventory, cluster_lookup.get(top.name_cluster_id), top) else 0.0
    relation_terms = {
        term.lower()
        for h in rows[:4]
        for term in h.evidence_packet.relation_clues
    }
    family_pressure = 0.18 if {"mom", "mother", "dad", "father", "partner"} & relation_terms else 0.0
    first_only = 0.16 if not is_full_name(top.observed_surface) else 0.0
    return top.score + ambiguity + competing + owner_pressure + family_pressure + first_only


def _owner_like_candidate(
    inventory: EvidenceInventory,
    cluster: NameCluster | None,
    hyp: IdentityHypothesis,
) -> bool:
    owner_name = inventory.owner_name
    candidate = hyp.canonical_name_candidate or hyp.observed_surface
    if not owner_name or not candidate:
        return False
    candidate_norm = normalize_name(candidate).lower()
    owner_norm = normalize_name(owner_name).lower()
    if candidate_norm == owner_norm:
        return True
    owner_first = first_name(owner_name).lower()
    owner_last = last_name(owner_name).lower()
    cand_first = first_name(candidate).lower()
    cand_last = last_name(candidate).lower()
    if cand_first == owner_first and not is_full_name(candidate):
        return True
    if cand_last == owner_last and len(cand_first) <= 2 and owner_first.startswith(cand_first.replace(".", "")):
        return True
    if cluster and cluster.first_name.lower() == owner_first and "first_name_only" in cluster.quality_flags:
        return True
    return False

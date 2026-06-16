"""Bridge-aware event assignment for ECD identities.

BEA is a conservative LLM layer over the deterministic EventGraph.  It does not
replace candidate generation; it reviews high-risk faces with event-graph name
edges and marks accepted candidates as joint selections for the existing global
resolver.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .config import ECDConfig
from .event_graph import EventIdentityEdge, build_event_graph
from .llm_context import face_context, photo_brief, truncate
from .schemas import EvidenceInventory, EvidencePacket, IdentityHypothesis, NameCluster
from .text import dedupe, looks_like_person_name, normalize_name

logger = logging.getLogger(__name__)

_HONORIFIC_ONLY_PREFIXES = {"mr", "mrs", "ms", "miss", "sir", "madam", "prof", "professor", "coach"}
_NON_PERSON_NAME_TOKENS = {
    "kitty",
    "sparky",
    "poster",
    "posters",
    "movie",
    "movies",
    "cafe",
    "coffee",
    "restaurant",
    "school",
    "clinic",
    "church",
    "museum",
    "gallery",
    "market",
    "supermarket",
    "tesco",
    "aldi",
    "sainsbury",
    "amazon",
    "royal",
    "mail",
    "iphone",
    "android",
    "canon",
    "nikon",
    "dyson",
    "toyota",
    "honda",
}

_SYSTEM = """You are the Bridge-aware Event Assignment layer of Evidence-Chain Dossier.
You receive several face_ids and candidate name bindings derived from public event evidence.

Rules:
- Assign at most one candidate to each face_id.
- A name_cluster_id may be assigned to at most one face in this batch.
- Use only candidate_index/name_cluster_id values provided for that face.
- Do not invent names, relations, or photo ids.
- This benchmark intentionally contains cross-photo identity chains: a text-only name anchor can identify a face in another photo when the bridge is a concrete same event, same month/location, shared activity, co-face pattern, owner co-presence, or recurring scene context.
- Prefer event-specific bridges over generic same-city similarity.
- Do not assume the order of names in a message equals the order of faces.
- Abstain if candidates are not separable.
- Keep reasoning_path under 35 words and cite 1-3 photo ids.

Output ONLY JSON:
{"assignments":[{"face_id":"face_001","selected_candidate_index":0,"name_cluster_id":"name_0001","canonical_name":"...","confidence":0.0,"relation_to_owner":"","relation_category":"","evidence_photo_ids":["photo_..."],"reasoning_path":"..."}],"abstentions":[{"face_id":"face_002","failure_mode":"..."}]}"""


@dataclass(slots=True)
class _BEACandidate:
    face_id: str
    cluster_id: str
    name: str
    edge: EventIdentityEdge | None
    hypothesis: IdentityHypothesis
    source: str
    event_score: float
    local_rank: int | None


def assign_bridge_events(
    *,
    inventory: EvidenceInventory,
    clusters: list[NameCluster],
    hypotheses: list[IdentityHypothesis],
    config: ECDConfig,
    llm: Any,
) -> tuple[list[IdentityHypothesis], dict[str, Any]]:
    """Review EventGraph face-name candidates and mark accepted hypotheses."""
    if llm is None:
        return hypotheses, {"enabled": False, "reason": "no_llm"}
    if config.bea_max_calls <= 0 or config.bea_review_max_faces <= 0:
        return hypotheses, {"enabled": False, "reason": "bea_budget_zero"}

    cluster_lookup = {cluster.cluster_id: cluster for cluster in clusters}
    cluster_index = _build_cluster_name_index(clusters)
    by_face: dict[str, list[IdentityHypothesis]] = defaultdict(list)
    for hyp in hypotheses:
        by_face[hyp.face_id].append(hyp)
    for rows in by_face.values():
        rows.sort(key=lambda h: (-h.score, h.name_cluster_id))
    reserved_clusters = _strong_direct_cluster_reservations(by_face, config)

    graph = build_event_graph(
        inventory,
        max_cross_edges_per_name=config.bea_max_cross_edges_per_name,
        min_cross_bridge_score=config.bea_min_edge_score,
    )
    candidates_by_face = _build_bea_candidates(
        inventory=inventory,
        clusters=clusters,
        cluster_lookup=cluster_lookup,
        cluster_index=cluster_index,
        hypotheses=hypotheses,
        by_face=by_face,
        edges=graph.edges,
        config=config,
    )
    reviewed_faces = _select_faces_to_review(inventory, by_face, candidates_by_face, config)
    batches = [
        reviewed_faces[i : i + max(1, config.bea_faces_per_call)]
        for i in range(0, len(reviewed_faces), max(1, config.bea_faces_per_call))
    ][: config.bea_max_calls]

    diagnostics: dict[str, Any] = {
        "enabled": True,
        "event_graph": {
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges),
            "n_candidate_faces": len(candidates_by_face),
            "n_reviewed_faces": len(reviewed_faces),
        },
        "batches": {},
        "errors": [],
    }

    used_clusters: set[str] = set()
    for batch_idx, face_ids in enumerate(batches, start=1):
        batch_id = f"bea_batch_{batch_idx:02d}"
        candidate_rows = {
            face_id: candidates_by_face.get(face_id, [])[: config.bea_max_candidate_options_per_face]
            for face_id in face_ids
        }
        candidate_rows = {face_id: rows for face_id, rows in candidate_rows.items() if rows}
        if not candidate_rows:
            diagnostics["batches"][batch_id] = {"faces": face_ids, "skipped": "no_candidates"}
            continue
        payload = _build_payload(inventory, candidate_rows, config)
        result = llm.call_json(
            prompt=json.dumps(payload, ensure_ascii=False),
            system=_SYSTEM,
            temperature=0.0,
            max_tokens=4096,
            retries=1,
            stage="bridge_event_assignment",
        )
        if not isinstance(result, dict):
            diagnostics["errors"].append({"batch_id": batch_id, "error": "no_json"})
            diagnostics["batches"][batch_id] = {"faces": list(candidate_rows), "error": "no_json"}
            continue
        applied = _apply_bea_result(
            candidate_rows=candidate_rows,
            result=result,
            config=config,
            used_clusters=used_clusters,
            reserved_clusters=reserved_clusters,
        )
        diagnostics["batches"][batch_id] = {
            "faces": list(candidate_rows),
            "raw_result": result,
            "applied": applied,
        }

    _recompute_margins(by_face)
    return hypotheses, diagnostics


def _build_bea_candidates(
    *,
    inventory: EvidenceInventory,
    clusters: list[NameCluster],
    cluster_lookup: dict[str, NameCluster],
    cluster_index: dict[str, str],
    hypotheses: list[IdentityHypothesis],
    by_face: dict[str, list[IdentityHypothesis]],
    edges: list[EventIdentityEdge],
    config: ECDConfig,
) -> dict[str, list[_BEACandidate]]:
    out: dict[str, list[_BEACandidate]] = defaultdict(list)
    hyp_by_face_cluster = {
        (hyp.face_id, hyp.name_cluster_id): hyp
        for hyp in hypotheses
    }
    local_rank = {
        (face_id, hyp.name_cluster_id): idx
        for face_id, rows in by_face.items()
        for idx, hyp in enumerate(rows, start=1)
    }
    for edge in edges:
        if edge.face_id == inventory.owner_face_id:
            continue
        if edge.score < config.bea_min_edge_score:
            continue
        if not _usable_name(edge.name_surface):
            continue
        cluster_id = _cluster_id_for_name(edge.name_surface, cluster_index)
        if not cluster_id or cluster_id not in cluster_lookup:
            continue
        hyp = hyp_by_face_cluster.get((edge.face_id, cluster_id))
        source = "event_existing_hypothesis"
        if hyp is None:
            hyp = _create_event_hypothesis(
                edge=edge,
                cluster=cluster_lookup[cluster_id],
                config=config,
            )
            hypotheses.append(hyp)
            by_face[edge.face_id].append(hyp)
            hyp_by_face_cluster[(edge.face_id, cluster_id)] = hyp
            source = "event_synthetic_hypothesis"
        out[edge.face_id].append(
            _BEACandidate(
                face_id=edge.face_id,
                cluster_id=cluster_id,
                name=edge.name_surface,
                edge=edge,
                hypothesis=hyp,
                source=source,
                event_score=float(edge.score),
                local_rank=local_rank.get((edge.face_id, cluster_id)),
            )
        )

    for face_id, rows in by_face.items():
        for rank, hyp in enumerate(rows[: config.bea_local_candidates_per_face], start=1):
            if hyp.face_id == inventory.owner_face_id:
                continue
            if any(c.cluster_id == hyp.name_cluster_id for c in out.get(face_id, [])):
                continue
            out[face_id].append(
                _BEACandidate(
                    face_id=face_id,
                    cluster_id=hyp.name_cluster_id,
                    name=hyp.canonical_name_candidate,
                    edge=None,
                    hypothesis=hyp,
                    source="local_hypothesis",
                    event_score=0.0,
                    local_rank=rank,
                )
            )

    for face_id, rows in out.items():
        rows.sort(
            key=lambda c: (
                0 if c.edge is not None else 1,
                -c.event_score,
                c.local_rank or 999,
                -c.hypothesis.score,
                c.cluster_id,
            )
        )
        out[face_id] = rows[: max(config.bea_max_candidate_options_per_face, config.bea_top_event_edges_per_face)]
    return dict(out)


def _create_event_hypothesis(edge: EventIdentityEdge, cluster: NameCluster, config: ECDConfig) -> IdentityHypothesis:
    text_ids = dedupe(edge.text_photo_ids)[:10]
    face_ids = dedupe(edge.face_photo_ids)[:10]
    bridge_pairs = []
    if text_ids and face_ids:
        bridge_pairs.append(
            {
                "text_photo_id": text_ids[0],
                "face_photo_id": face_ids[0],
                "bridge": edge.bridge_type,
                "event_score": round(float(edge.score), 4),
                "shared_keywords": edge.shared_terms[:8],
                "event_channels": edge.shared_channels[:8],
                "text_photo_has_faces": False,
                "strong_signal": float(edge.score) >= 0.58,
            }
        )
    packet = EvidencePacket(
        text_photo_ids=text_ids,
        face_photo_ids=face_ids,
        same_photo_ids=[pid for pid in text_ids if pid in face_ids][:5],
        bridge_photo_pairs=bridge_pairs,
        relation_clues=cluster.relation_terms[:10],
        activity_clues=cluster.activity_terms[:10],
        venue_clues=cluster.venue_terms[:10],
        narrative_summary=(
            f"BEA event edge links {cluster.primary_surface} to {edge.face_id} via "
            f"{edge.bridge_type} evidence ({', '.join(edge.evidence_photo_ids[:4])})."
        ),
    )
    # Keep synthetic event candidates below the resolver threshold until BEA
    # explicitly accepts them. This prevents high-recall EventGraph edges from
    # perturbing the baseline resolver by mere presence.
    score = max(0.01, min(config.min_identity_score - 0.04, 0.24 + 0.20 * float(edge.score)))
    return IdentityHypothesis(
        face_id=edge.face_id,
        name_cluster_id=cluster.cluster_id,
        observed_surface=cluster.primary_surface,
        canonical_name_candidate=cluster.primary_surface,
        score=score,
        margin=0.0,
        signal_breakdown={
            "bea_event_score": round(float(edge.score), 4),
            "bea_synthetic": 1.0,
        },
        evidence_packet=packet,
        status="bea_event_candidate",
    )


def _select_faces_to_review(
    inventory: EvidenceInventory,
    by_face: dict[str, list[IdentityHypothesis]],
    candidates_by_face: dict[str, list[_BEACandidate]],
    config: ECDConfig,
) -> list[str]:
    scored = []
    for face_id, candidates in candidates_by_face.items():
        if face_id == inventory.owner_face_id or not candidates:
            continue
        rows = sorted(by_face.get(face_id, []), key=lambda h: (-h.score, h.name_cluster_id))
        top_local = rows[0].name_cluster_id if rows else ""
        top_event = next((c for c in candidates if c.edge is not None), None)
        if top_event is None:
            continue
        margin = rows[0].margin if rows else 0.0
        if (
            rows
            and _has_strong_direct_anchor(rows[0])
            and margin >= config.bea_skip_direct_margin
            and top_event.cluster_id != top_local
        ):
            continue
        priority = top_event.event_score
        if top_event.cluster_id != top_local:
            priority += 0.28
        if margin < 0.12:
            priority += 0.18
        if rows and not _has_strong_direct_anchor(rows[0]):
            priority += 0.10
        if top_event.source == "event_synthetic_hypothesis":
            priority += 0.20
        if len(candidates) >= 4:
            priority += 0.05
        scored.append((priority, face_id))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [face_id for _, face_id in scored[: config.bea_review_max_faces]]


def _build_payload(
    inventory: EvidenceInventory,
    candidate_rows: dict[str, list[_BEACandidate]],
    config: ECDConfig,
) -> dict[str, Any]:
    chars = min(config.llm_photo_snippet_chars, 220)
    return {
        "user_id": inventory.album.get("user_id"),
        "owner": {
            "face_id": inventory.owner_face_id,
            "name_candidate": inventory.owner_name,
            "last_name": inventory.owner_last_name,
        },
        "policy": {
            "goal": "Choose only separable face-name assignments from event bridges.",
            "acceptance": "Prefer high-confidence direct/specific event bridges; abstain on generic same-city ties or when a face already has strong direct same-photo evidence.",
            "confidence": f"Use >={config.bea_accept_min_confidence:.2f} only for a concrete bridge or direct evidence.",
        },
        "faces": [
            face_context(inventory, face_id, max_photos=4, chars=chars)
            for face_id in candidate_rows
        ],
        "candidate_options_by_face": {
            face_id: [
                _candidate_card(inventory, candidate, index=idx, chars=chars)
                for idx, candidate in enumerate(rows)
            ]
            for face_id, rows in candidate_rows.items()
        },
    }


def _candidate_card(
    inventory: EvidenceInventory,
    candidate: _BEACandidate,
    *,
    index: int,
    chars: int,
) -> dict[str, Any]:
    hyp = candidate.hypothesis
    edge = candidate.edge
    evidence_ids = dedupe(
        (edge.evidence_photo_ids if edge else [])
        + hyp.evidence_packet.same_photo_ids
        + hyp.evidence_packet.text_photo_ids[:2]
        + hyp.evidence_packet.face_photo_ids[:2]
    )[:6]
    return {
        "candidate_index": index,
        "name_cluster_id": candidate.cluster_id,
        "canonical_name_candidate": hyp.canonical_name_candidate,
        "source": candidate.source,
        "local_rank": candidate.local_rank,
        "local_score": round(float(hyp.score), 4),
        "local_margin": round(float(hyp.margin), 4),
        "event_score": round(float(candidate.event_score), 4),
        "bridge_type": edge.bridge_type if edge else "",
        "shared_channels": edge.shared_channels[:6] if edge else [],
        "shared_terms": edge.shared_terms[:8] if edge else [],
        "same_photo_ids": hyp.evidence_packet.same_photo_ids[:5],
        "text_photo_ids": (edge.text_photo_ids[:5] if edge else hyp.evidence_packet.text_photo_ids[:5]),
        "face_photo_ids": (edge.face_photo_ids[:5] if edge else hyp.evidence_packet.face_photo_ids[:5]),
        "relation_clues": hyp.evidence_packet.relation_clues[:8],
        "activity_clues": hyp.evidence_packet.activity_clues[:8],
        "venue_clues": hyp.evidence_packet.venue_clues[:8],
        "reasoning_seed": truncate(hyp.evidence_packet.narrative_summary, 320),
        "evidence_photos": [photo_brief(inventory, pid, chars=chars) for pid in evidence_ids],
    }


def _apply_bea_result(
    *,
    candidate_rows: dict[str, list[_BEACandidate]],
    result: dict[str, Any],
    config: ECDConfig,
    used_clusters: set[str],
    reserved_clusters: dict[str, str],
) -> list[dict[str, Any]]:
    assignments = result.get("assignments") or []
    if not isinstance(assignments, list):
        return []
    parsed = []
    for assignment in assignments:
        if isinstance(assignment, dict):
            parsed.append((_bounded_float(assignment.get("confidence"), 0.0), assignment))
    parsed.sort(key=lambda row: -row[0])

    applied = []
    used_faces: set[str] = set()
    for confidence, assignment in parsed:
        face_id = str(assignment.get("face_id") or "")
        if face_id in used_faces or face_id not in candidate_rows:
            continue
        if confidence < config.bea_accept_min_confidence:
            applied.append({"face_id": face_id, "status": "skipped_low_confidence", "confidence": round(confidence, 4)})
            continue
        selected = _match_candidate(assignment, candidate_rows[face_id])
        if selected is None:
            applied.append({"face_id": face_id, "status": "skipped_no_candidate_match", "confidence": round(confidence, 4)})
            continue
        if selected.cluster_id in used_clusters:
            applied.append(
                {
                    "face_id": face_id,
                    "name_cluster_id": selected.cluster_id,
                    "status": "skipped_cluster_used",
                    "confidence": round(confidence, 4),
                }
            )
            continue
        reserved_face = reserved_clusters.get(selected.cluster_id)
        if reserved_face and reserved_face != face_id:
            selected.hypothesis.llm_decision = "bea_guarded_select"
            selected.hypothesis.llm_confidence = confidence
            selected.hypothesis.llm_failure_mode = "strong_direct_cluster_reserved"
            selected.hypothesis.llm_reasoning_path = str(assignment.get("reasoning_path") or "")
            applied.append(
                {
                    "face_id": face_id,
                    "name_cluster_id": selected.cluster_id,
                    "reserved_face_id": reserved_face,
                    "status": "skipped_cluster_reserved",
                    "confidence": round(confidence, 4),
                    "event_score": round(selected.event_score, 4),
                }
            )
            continue
        if not _safe_to_apply(selected, confidence, config, candidate_rows[face_id]):
            selected.hypothesis.llm_decision = "bea_guarded_select"
            selected.hypothesis.llm_confidence = confidence
            selected.hypothesis.llm_failure_mode = "weak_event_bridge_guard"
            selected.hypothesis.llm_reasoning_path = str(assignment.get("reasoning_path") or "")
            applied.append(
                {
                    "face_id": face_id,
                    "name_cluster_id": selected.cluster_id,
                    "status": "skipped_weak_event_bridge",
                    "confidence": round(confidence, 4),
                    "event_score": round(selected.event_score, 4),
                }
            )
            continue
        _apply_selected(selected, assignment, confidence, candidate_rows[face_id], config)
        used_faces.add(face_id)
        used_clusters.add(selected.cluster_id)
        applied.append(
            {
                "face_id": face_id,
                "name_cluster_id": selected.cluster_id,
                "canonical_name": selected.hypothesis.canonical_name_candidate,
                "status": "applied",
                "confidence": round(confidence, 4),
                "event_score": round(selected.event_score, 4),
                "score": round(selected.hypothesis.score, 4),
            }
        )
    return applied


def _apply_selected(
    selected: _BEACandidate,
    assignment: dict[str, Any],
    confidence: float,
    face_rows: list[_BEACandidate],
    config: ECDConfig,
) -> None:
    hyp = selected.hypothesis
    hyp.status = "llm_joint_selected"
    hyp.llm_decision = "bea_selected"
    hyp.llm_confidence = confidence
    hyp.llm_canonical_name = normalize_name(str(assignment.get("canonical_name") or hyp.canonical_name_candidate))
    hyp.llm_relation_to_owner = str(assignment.get("relation_to_owner") or "")
    hyp.llm_relation_category = str(assignment.get("relation_category") or "")
    hyp.llm_reasoning_path = str(assignment.get("reasoning_path") or "")
    hyp.llm_evidence_photo_ids = [
        str(pid) for pid in (assignment.get("evidence_photo_ids") or []) if str(pid).startswith("photo_")
    ]
    if selected.edge:
        hyp.llm_evidence_photo_ids = dedupe(selected.edge.evidence_photo_ids + hyp.llm_evidence_photo_ids)[:8]
    hyp.signal_breakdown["bea_llm_confidence"] = round(confidence, 4)
    hyp.signal_breakdown["bea_event_score"] = round(selected.event_score, 4)
    if hyp.llm_canonical_name:
        hyp.canonical_name_candidate = hyp.llm_canonical_name
    if hyp.llm_reasoning_path:
        hyp.evidence_packet.narrative_summary = hyp.llm_reasoning_path
    boost = min(0.96, 0.38 + confidence * 0.42 + selected.event_score * 0.20)
    hyp.score = max(hyp.score, boost, config.min_identity_score + 0.05)
    for candidate in face_rows:
        other = candidate.hypothesis
        if other is hyp:
            continue
        other.llm_decision = other.llm_decision or "bea_not_selected"
        other.llm_confidence = max(float(other.llm_confidence or 0.0), confidence)
        other.status = "llm_joint_rejected"
        other.score = min(other.score, max(0.0, hyp.score - 0.12))


def _match_candidate(assignment: dict[str, Any], rows: list[_BEACandidate]) -> _BEACandidate | None:
    try:
        idx = int(assignment.get("selected_candidate_index"))
    except (TypeError, ValueError):
        idx = -1
    if 0 <= idx < len(rows):
        candidate = rows[idx]
        requested_cluster = str(assignment.get("name_cluster_id") or "")
        if not requested_cluster or requested_cluster == candidate.cluster_id:
            return candidate
    requested_cluster = str(assignment.get("name_cluster_id") or "")
    if requested_cluster:
        for candidate in rows:
            if candidate.cluster_id == requested_cluster:
                return candidate
    requested_name = normalize_name(str(assignment.get("canonical_name") or "")).lower()
    if requested_name:
        for candidate in rows:
            names = {
                normalize_name(candidate.name).lower(),
                normalize_name(candidate.hypothesis.canonical_name_candidate).lower(),
                normalize_name(candidate.hypothesis.observed_surface).lower(),
            }
            if requested_name in names:
                return candidate
    return None


def _safe_to_apply(
    candidate: _BEACandidate,
    confidence: float,
    config: ECDConfig,
    face_rows: list[_BEACandidate],
) -> bool:
    direct_top = _strong_direct_top_candidate(face_rows)
    if direct_top and direct_top.cluster_id != candidate.cluster_id:
        return bool(
            candidate.edge is not None
            and confidence >= config.bea_override_direct_min_confidence
            and candidate.event_score >= config.bea_override_direct_min_edge_score
        )
    if _has_strong_direct_anchor(candidate.hypothesis):
        return True
    if candidate.event_score >= config.bea_min_apply_edge_score and confidence >= config.bea_accept_min_confidence:
        return True
    return bool(
        candidate.source == "local_hypothesis"
        and candidate.local_rank == 1
        and confidence >= max(config.bea_accept_min_confidence, 0.74)
    )


def _build_cluster_name_index(clusters: list[NameCluster]) -> dict[str, str]:
    exact: dict[str, list[str]] = defaultdict(list)
    first: dict[str, list[str]] = defaultdict(list)
    for cluster in clusters:
        names = dedupe([cluster.primary_surface, *cluster.surfaces])
        for name in names:
            key = _name_key(name)
            if key:
                exact[key].append(cluster.cluster_id)
            first_key = _first_key(name)
            if first_key:
                first[first_key].append(cluster.cluster_id)
    out = {}
    for key, values in exact.items():
        ids = dedupe(values)
        if len(ids) == 1:
            out[f"exact::{key}"] = ids[0]
    for key, values in first.items():
        ids = dedupe(values)
        if len(ids) == 1:
            out[f"first::{key}"] = ids[0]
    return out


def _cluster_id_for_name(name: str, index: dict[str, str]) -> str:
    key = _name_key(name)
    if key and f"exact::{key}" in index:
        return index[f"exact::{key}"]
    first = _first_key(name)
    if first and f"first::{first}" in index:
        return index[f"first::{first}"]
    return ""


def _usable_name(name: str) -> bool:
    value = normalize_name(name)
    if not looks_like_person_name(value):
        return False
    lower = value.lower()
    if lower.startswith(("and ", "with ", "from ", "dear ")):
        return False
    if lower in {"mom", "dad", "friends", "friend"}:
        return False
    key = _name_key(value)
    parts = key.split()
    if not parts:
        return False
    if parts[0] in _HONORIFIC_ONLY_PREFIXES and len(parts) <= 2:
        return False
    if any(part in _NON_PERSON_NAME_TOKENS for part in parts):
        return False
    if len(parts) == 2 and parts[0] in {"hello", "dear", "happy"}:
        return False
    return True


def _strong_direct_cluster_reservations(
    by_face: dict[str, list[IdentityHypothesis]],
    config: ECDConfig,
) -> dict[str, str]:
    reserved: dict[str, str] = {}
    for face_id, rows in by_face.items():
        if not rows:
            continue
        top = rows[0]
        if (
            top.score >= config.min_identity_score
            and top.margin >= config.bea_skip_direct_margin
            and _has_strong_direct_anchor(top)
        ):
            reserved.setdefault(top.name_cluster_id, face_id)
    return reserved


def _strong_direct_top_candidate(face_rows: list[_BEACandidate]) -> _BEACandidate | None:
    top = next((row for row in face_rows if row.local_rank == 1), None)
    if top and _has_strong_direct_anchor(top.hypothesis):
        return top
    return None


def _name_key(name: str) -> str:
    return " ".join(re.findall(r"[a-z]+", normalize_name(name).lower()))


def _first_key(name: str) -> str:
    parts = _name_key(name).split()
    return parts[0] if parts else ""


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

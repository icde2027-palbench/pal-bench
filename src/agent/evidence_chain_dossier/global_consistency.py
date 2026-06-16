"""LLM global consistency pass for resolved ECD identities."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .config import ECDConfig
from .llm_context import candidate_context, face_context, resolved_context
from .resolver import resolve_identities
from .schemas import EvidenceInventory, IdentityHypothesis, NameCluster, ResolvedIdentity
from .text import dedupe, first_name, is_full_name, last_name, normalize_name

_SYSTEM = """You are the global consistency resolver for Evidence-Chain Dossier.
You receive public evidence summaries, the current identity assignments, and top competing candidates.

Your job:
- Fix likely swaps and conflicts across faces.
- Keep one person name assigned to at most one visible face unless the evidence explicitly says otherwise.
- Never assign the owner name to a non-owner face.
- Prefer grounded relation/role evidence over raw bridge counts.
- Do not invent names that are absent from the candidate lists.
- Do not clear a non-empty current canonical_name merely because evidence is weak. Keep plausible partial/first-name predictions; they are useful uncertain hypotheses.
- Clear a current name only when it is obvious OCR/noise, an owner-name leak, a duplicate conflict, or contradicted by strong role evidence.

Output ONLY one JSON object with:
{"updates":[{"face_id":"face_001","canonical_name":"...","relation_to_owner":"...","relation_category":"...","confidence":0.0,"evidence_photo_ids":["photo_..."],"reasoning_path":"..."}],"notes":["..."]}"""


def apply_global_consistency(
    *,
    inventory: EvidenceInventory,
    clusters: list[NameCluster],
    hypotheses: list[IdentityHypothesis],
    resolved: list[ResolvedIdentity],
    config: ECDConfig,
    llm: Any,
) -> tuple[list[ResolvedIdentity], dict[str, Any]]:
    """Ask the LLM to repair cross-face conflicts in the resolved profile."""
    if llm is None:
        return resolved, {"enabled": False, "reason": "no_llm"}

    cluster_lookup = {c.cluster_id: c for c in clusters}
    by_face: dict[str, list[IdentityHypothesis]] = defaultdict(list)
    for hyp in hypotheses:
        by_face[hyp.face_id].append(hyp)
    for rows in by_face.values():
        rows.sort(key=lambda h: (-h.score, h.name_cluster_id))

    payload = {
        "user_id": inventory.album.get("user_id"),
        "owner": {
            "face_id": inventory.owner_face_id,
            "name_candidate": inventory.owner_name,
            "last_name": inventory.owner_last_name,
        },
        "current_assignments": [
            resolved_context(inventory, row, chars=config.llm_photo_snippet_chars)
            for row in resolved
        ],
        "top_candidates_by_face": {
            face_id: [
                candidate_context(
                    inventory,
                    cluster_lookup,
                    hyp,
                    index=i,
                    chars=config.llm_photo_snippet_chars,
                )
                for i, hyp in enumerate(_filtered_candidates(inventory, cluster_lookup, rows, config))
            ]
            for face_id, rows in sorted(by_face.items())
        },
        "face_summaries": {
            face_id: face_context(
                inventory,
                face_id,
                max_photos=5,
                chars=config.llm_photo_snippet_chars,
            )
            for face_id in sorted(by_face)
        },
    }
    result = llm.call_json(
        prompt=json.dumps(payload, ensure_ascii=False),
        system=_SYSTEM,
        temperature=0.1,
        max_tokens=8192,
        retries=1,
        stage="llm_global_consistency",
    )
    if not isinstance(result, dict):
        return resolved, {"enabled": True, "error": "no_json"}

    updated = _apply_updates(
        inventory=inventory,
        hypotheses=hypotheses,
        resolved=resolved,
        updates=result.get("updates") or [],
    )
    diagnostics = {
        "enabled": True,
        "raw_result": result,
        "n_updates": len(result.get("updates") or []),
    }
    return updated, diagnostics


def _apply_updates(
    *,
    inventory: EvidenceInventory,
    hypotheses: list[IdentityHypothesis],
    resolved: list[ResolvedIdentity],
    updates: list[dict[str, Any]],
) -> list[ResolvedIdentity]:
    by_face = {row.face_id: row for row in resolved}
    hyp_by_face_name: dict[tuple[str, str], IdentityHypothesis] = {}
    for hyp in hypotheses:
        keys = {
            normalize_name(hyp.observed_surface).lower(),
            normalize_name(hyp.canonical_name_candidate).lower(),
            normalize_name(hyp.llm_canonical_name).lower(),
        }
        for key in keys:
            if key:
                hyp_by_face_name[(hyp.face_id, key)] = hyp

    used_names = {
        normalize_name(row.canonical_name).lower(): row.face_id
        for row in resolved
        if row.canonical_name
    }
    owner_name = normalize_name(inventory.owner_name).lower()

    for update in updates:
        face_id = str(update.get("face_id") or "")
        if face_id not in by_face:
            continue
        row = by_face[face_id]
        new_name = normalize_name(str(update.get("canonical_name") or ""))
        new_key = new_name.lower()
        if new_key and new_key == owner_name:
            continue
        prior_face = used_names.get(new_key)
        if new_key and prior_face and prior_face != face_id:
            continue
        if not new_key and _should_keep_current_name(row, inventory):
            new_name = normalize_name(row.canonical_name)
            new_key = new_name.lower()

        if row.canonical_name:
            used_names.pop(normalize_name(row.canonical_name).lower(), None)
        row.canonical_name = new_name
        if new_key:
            used_names[new_key] = face_id

        row.relation_to_owner = str(update.get("relation_to_owner") or row.relation_to_owner or "")
        row.relation_category = str(update.get("relation_category") or row.relation_category or "")
        row.confidence = _bounded_float(update.get("confidence"), row.confidence)
        reasoning = _best_reasoning(update)
        if reasoning:
            row.reasoning_path = reasoning
        evidence_ids = [str(pid) for pid in (update.get("evidence_photo_ids") or []) if str(pid).startswith("photo_")]
        if evidence_ids:
            row.evidence_packet.text_photo_ids = dedupe(evidence_ids + row.evidence_packet.text_photo_ids)
        row.name_source = "llm_global_consistency"

        hyp = hyp_by_face_name.get((face_id, new_key))
        if hyp:
            row.evidence_packet = hyp.evidence_packet
            if reasoning:
                row.evidence_packet.narrative_summary = reasoning
            row.score = max(row.score, hyp.score)
    return sorted(resolved, key=lambda row: (-row.score, row.face_id))


def _should_keep_current_name(row: ResolvedIdentity, inventory: EvidenceInventory) -> bool:
    """Avoid over-applying LLM abstentions to plausible name hypotheses."""
    if not row.canonical_name:
        return False
    if _owner_like_name(row.canonical_name, inventory):
        return False
    if _obvious_noise_name(row.canonical_name):
        return False
    if row.score < 0.25 and row.confidence < 0.45:
        return False
    return True


def _filtered_candidates(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    rows: list[IdentityHypothesis],
    config: ECDConfig,
) -> list[IdentityHypothesis]:
    out = []
    for hyp in rows:
        if _owner_like_name(hyp.canonical_name_candidate or hyp.observed_surface, inventory):
            continue
        cluster = cluster_lookup.get(hyp.name_cluster_id)
        if cluster and _owner_like_name(cluster.primary_surface, inventory):
            continue
        out.append(hyp)
        if len(out) >= config.llm_top_candidates_per_face:
            break
    return out


def _owner_like_name(name: str, inventory: EvidenceInventory) -> bool:
    owner_name = inventory.owner_name
    if not name or not owner_name:
        return False
    name_norm = normalize_name(name).lower()
    owner_norm = normalize_name(owner_name).lower()
    if name_norm == owner_norm:
        return True
    owner_first = first_name(owner_name).lower()
    owner_last = last_name(owner_name).lower()
    cand_first = first_name(name).lower()
    cand_last = last_name(name).lower()
    if cand_first == owner_first and not is_full_name(name):
        return True
    if cand_last == owner_last and len(cand_first) <= 2 and owner_first.startswith(cand_first.replace(".", "")):
        return True
    return False


def _obvious_noise_name(name: str) -> bool:
    norm = normalize_name(name)
    lower = norm.lower()
    if any(bad in lower for bad in {"trump", "shakespeare", "billie idol"}):
        return True
    if any(ch.isdigit() for ch in norm):
        return True
    if any(ch in norm for ch in {"_", "@", "#", "$"}):
        return True
    alpha = "".join(ch for ch in norm if ch.isalpha())
    if len(alpha) >= 10 and sum(1 for ch in alpha.lower() if ch not in "aeiou") / len(alpha) > 0.78:
        return True
    return False


def _bounded_float(value: Any, fallback: float) -> float:
    try:
        return max(0.05, min(0.95, float(value)))
    except (TypeError, ValueError):
        return fallback


def _best_reasoning(update: dict[str, Any]) -> str:
    candidates = [
        str(update.get("reasoning_path") or ""),
        str(update.get("reasoning") or ""),
        str(update.get("rationale") or ""),
    ]
    candidates = [c.strip() for c in candidates if c.strip()]
    if not candidates:
        return ""
    return max(candidates, key=len)

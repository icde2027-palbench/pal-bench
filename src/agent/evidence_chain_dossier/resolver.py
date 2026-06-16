"""Delayed global resolver for Evidence-Chain Dossier."""

from __future__ import annotations

from collections import defaultdict

from .config import ECDConfig
from .schemas import EvidenceInventory, EvidencePacket, IdentityHypothesis, NameCluster, ResolvedIdentity
from .text import (
    dedupe,
    first_name,
    is_full_name,
    last_name,
    normalize_name,
    relation_value,
)


def resolve_identities(
    inventory: EvidenceInventory,
    clusters: list[NameCluster],
    hypotheses: list[IdentityHypothesis],
    config: ECDConfig,
) -> tuple[list[ResolvedIdentity], dict]:
    """Resolve all hypotheses after candidate generation.

    The resolver is intentionally delayed: every face-name pair is scored first,
    then a conservative greedy pass chooses one cluster per face and one face per
    cluster. This avoids DOSSIER's early local commitments.
    """
    cluster_lookup = {c.cluster_id: c for c in clusters}
    by_face: dict[str, list[IdentityHypothesis]] = defaultdict(list)
    for hyp in hypotheses:
        by_face[hyp.face_id].append(hyp)
    for rows in by_face.values():
        rows.sort(key=lambda h: (-h.score, -h.margin, h.name_cluster_id))

    used_clusters: set[str] = set()
    resolved: list[ResolvedIdentity] = []
    rejected: list[dict] = []

    face_ids = sorted(fid for fid in inventory.face_units_by_face if fid != inventory.owner_face_id)
    has_joint_assignments = any(hyp.status == "llm_joint_selected" for hyp in hypotheses)
    if has_joint_assignments:
        joint_faces = {
            hyp.face_id
            for hyp in hypotheses
            if hyp.status == "llm_joint_selected"
        }
        face_order = sorted(
            face_ids,
            key=lambda fid: (
                0 if fid in joint_faces else 1,
                -(
                    _best_resolvable_score(by_face.get(fid, []), cluster_lookup, inventory)
                    if fid in joint_faces
                    else _original_top_score(by_face.get(fid, []))
                ),
                fid,
            ),
        )
    else:
        face_order = sorted(
            face_ids,
            key=lambda fid: (-(by_face.get(fid, [None])[0].score if by_face.get(fid) else 0.0), fid),
        )

    for face_id in face_order:
        chosen = None
        joint_selected_clusters = {
            hyp.name_cluster_id
            for hyp in by_face.get(face_id, [])
            if hyp.status == "llm_joint_selected"
        }
        for hyp in by_face.get(face_id, []):
            if hyp.name_cluster_id in used_clusters:
                continue
            cluster = cluster_lookup[hyp.name_cluster_id]
            if joint_selected_clusters and hyp.status == "llm_joint_rejected":
                rejected.append({**hyp.to_dict(), "reject_reason": "joint_rejected"})
                continue
            if _is_owner_like(hyp, cluster, inventory):
                rejected.append({**hyp.to_dict(), "reject_reason": "owner_name_candidate"})
                continue
            if hyp.score < config.min_identity_score:
                rejected.append({**hyp.to_dict(), "reject_reason": "below_threshold"})
                continue
            if hyp.margin < config.min_identity_margin and hyp.score < (config.min_identity_score + 0.08):
                rejected.append({**hyp.to_dict(), "reject_reason": "low_margin"})
                continue
            chosen = hyp
            break

        if chosen is None:
            resolved.append(_unresolved_identity(inventory, face_id))
            continue

        used_clusters.add(chosen.name_cluster_id)
        cluster = cluster_lookup[chosen.name_cluster_id]
        canonical, source = _canonicalize(cluster, inventory, config)
        if chosen.llm_canonical_name:
            canonical = normalize_name(chosen.llm_canonical_name)
            source = "llm_reranker"
        relation_to_owner, relation_category = _infer_relation(canonical, chosen.evidence_packet, inventory)
        if chosen.llm_relation_to_owner:
            relation_to_owner = chosen.llm_relation_to_owner
        if chosen.llm_relation_category:
            relation_category = chosen.llm_relation_category
        confidence = _calibrate_confidence(chosen.score, chosen.margin)
        if chosen.llm_confidence is not None:
            confidence = max(confidence, min(0.95, 0.35 + 0.55 * chosen.llm_confidence))
        if chosen.llm_evidence_photo_ids:
            chosen.evidence_packet.text_photo_ids = dedupe(
                chosen.llm_evidence_photo_ids + chosen.evidence_packet.text_photo_ids
            )
        reasoning = chosen.llm_reasoning_path or _reasoning_path(
            chosen,
            canonical,
            relation_to_owner,
            relation_category,
        )
        resolved.append(
            ResolvedIdentity(
                face_id=face_id,
                canonical_name=canonical,
                observed_surface=chosen.observed_surface,
                confidence=confidence,
                evidence_packet=chosen.evidence_packet,
                name_source=source,
                relation_to_owner=relation_to_owner,
                relation_category=relation_category,
                reasoning_path=reasoning,
                score=chosen.score,
            )
        )

    resolved.sort(key=lambda row: (-row.score, row.face_id))
    diagnostics = {
        "n_hypotheses": len(hypotheses),
        "n_resolved": sum(1 for row in resolved if row.canonical_name),
        "n_unresolved": sum(1 for row in resolved if not row.canonical_name),
        "rejected": rejected[:200],
        "used_clusters": sorted(used_clusters),
    }
    return resolved, diagnostics


def _best_resolvable_score(
    rows: list[IdentityHypothesis],
    cluster_lookup: dict[str, NameCluster],
    inventory: EvidenceInventory,
) -> float:
    for hyp in rows:
        cluster = cluster_lookup[hyp.name_cluster_id]
        if _is_owner_like(hyp, cluster, inventory):
            continue
        bonus = 0.08 if hyp.status == "llm_joint_selected" else 0.0
        return hyp.score + bonus
    return 0.0


def _original_top_score(rows: list[IdentityHypothesis]) -> float:
    return rows[0].score if rows else 0.0


def _is_owner_like(hyp: IdentityHypothesis, cluster: NameCluster, inventory: EvidenceInventory) -> bool:
    candidate = hyp.canonical_name_candidate
    owner_name = inventory.owner_name
    if not candidate or not owner_name:
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
    if cluster.first_name.lower() == owner_first and "first_name_only" in cluster.quality_flags:
        return True
    return False


def _canonicalize(
    cluster: NameCluster,
    inventory: EvidenceInventory,
    config: ECDConfig,
) -> tuple[str, str]:
    primary = normalize_name(cluster.primary_surface)
    if is_full_name(primary):
        return primary, "observed_full_name"
    if cluster.last_name_candidates:
        last = max(cluster.last_name_candidates.items(), key=lambda kv: (kv[1], kv[0]))[0]
        return normalize_name(f"{cluster.first_name} {last}"), "observed_first_plus_text_last"
    if (
        config.infer_family_last_name
        and inventory.owner_last_name
        and any(_relation_category(term) == "family" for term in cluster.relation_terms)
    ):
        return normalize_name(f"{cluster.first_name} {inventory.owner_last_name}"), "first_name_plus_owner_family_surname"
    if config.allow_first_name_predictions:
        return primary, "observed_first_name"
    return "", "unresolved_first_name_only"


def _infer_relation(
    canonical_name: str,
    packet: EvidencePacket,
    inventory: EvidenceInventory,
) -> tuple[str, str]:
    clues = [c.lower() for c in packet.relation_clues]
    priority = [
        "mother",
        "mom",
        "mother's day",
        "father",
        "dad",
        "father's day",
        "wife",
        "husband",
        "spouse",
        "partner",
        "boyfriend",
        "girlfriend",
        "sister",
        "brother",
        "daughter",
        "son",
        "aunt",
        "auntie",
        "uncle",
        "cousin",
        "coworker",
        "co-worker",
        "colleague",
        "classmate",
        "neighbor",
        "close friend",
        "best friend",
        "friend",
        "friends",
    ]
    for key in priority:
        if key in clues:
            relation, category = relation_value(key)
            if relation and category:
                return relation, category

    activity = {c.lower() for c in packet.activity_clues}
    venues = {c.lower() for c in packet.venue_clues}
    if {"law", "legal", "attorney", "bar association", "networking", "office", "court"} & activity:
        return "colleague", "colleague"
    if {"hospital", "clinic", "doctor", "nurse", "office"} & activity:
        return "colleague", "colleague"
    if {"school", "teacher", "class"} & activity:
        return "classmate", "classmate"
    if "neighbor" in activity or "neighbor" in venues:
        return "neighbor", "neighbor"
    if inventory.owner_last_name and last_name(canonical_name).lower() == inventory.owner_last_name.lower():
        return "family member", "family"
    if {"thanksgiving", "christmas", "holiday"} & activity and {"home", "house", "living room", "patio"} & venues:
        return "family member", "family"
    if {"brunch", "dinner", "birthday", "cycling", "ride", "bike"} & activity:
        return "friend", "friend"
    return "", ""


def _relation_category(term: str) -> str:
    return relation_value(term)[1]


def _calibrate_confidence(score: float, margin: float) -> float:
    return max(0.05, min(0.95, 0.36 + score * 0.52 + margin * 0.30))


def _reasoning_path(
    hyp: IdentityHypothesis,
    canonical_name: str,
    relation_to_owner: str,
    relation_category: str,
) -> str:
    packet = hyp.evidence_packet
    parts = [
        f"Candidate name '{hyp.observed_surface}' resolves to '{canonical_name}' for {hyp.face_id}.",
        packet.narrative_summary,
    ]
    if packet.same_photo_ids:
        parts.append(f"Direct same-photo anchors: {', '.join(packet.same_photo_ids[:5])}.")
    if packet.bridge_photo_pairs:
        rendered = [
            f"{p['text_photo_id']}->{p['face_photo_id']} ({p['bridge']})"
            for p in packet.bridge_photo_pairs[:4]
        ]
        parts.append(f"Text-to-face bridge examples: {', '.join(rendered)}.")
    if relation_to_owner or relation_category:
        parts.append(
            f"Relation readout uses clues {dedupe(packet.relation_clues)[:5]} and predicts "
            f"{relation_to_owner or relation_category}."
        )
    parts.append(
        "The prediction is made after global face-name resolution, so this name cluster is not reused for another face."
    )
    return " ".join(p for p in parts if p)


def _unresolved_identity(inventory: EvidenceInventory, face_id: str) -> ResolvedIdentity:
    face_units = inventory.face_units_by_face.get(face_id, [])
    face_photo_ids = [u.photo_id for u in face_units[:10]]
    relation_terms = dedupe(term for u in face_units for term in u.relation_terms)
    activity_terms = dedupe(term for u in face_units for term in u.activity_terms)
    venue_terms = dedupe(term for u in face_units for term in u.venue_terms)
    packet = EvidencePacket(
        face_photo_ids=face_photo_ids,
        relation_clues=relation_terms,
        activity_clues=activity_terms,
        venue_clues=venue_terms,
        coface_clues=[f"{inventory.owner_face_id}:owner"] if any(u.owner_present for u in face_units) else [],
        narrative_summary=f"{face_id} has visible face evidence but no name cluster passed the ECD threshold.",
    )
    relation_to_owner, relation_category = _infer_relation("", packet, inventory)
    return ResolvedIdentity(
        face_id=face_id,
        canonical_name="",
        observed_surface="",
        confidence=0.12,
        evidence_packet=packet,
        name_source="unresolved",
        relation_to_owner=relation_to_owner,
        relation_category=relation_category,
        reasoning_path=packet.narrative_summary,
        score=0.0,
    )

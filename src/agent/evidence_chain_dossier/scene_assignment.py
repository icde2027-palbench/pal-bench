"""LLM-backed joint face-name assignment for ambiguous album scenes."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .config import ECDConfig
from .event_bridge import photo_event_features, social_channels_from_terms
from .llm_context import candidate_context, face_context, photo_brief
from .schemas import EvidenceInventory, IdentityHypothesis, NameCluster
from .text import dedupe, first_name, is_full_name, last_name, normalize_name, tokens

logger = logging.getLogger(__name__)

_SYSTEM = """You are the joint assignment layer of Evidence-Chain Dossier.
You receive one ambiguous album group with multiple face_ids and multiple candidate name clusters.

Your job:
- Treat each face_id as a variable and assign at most one candidate name cluster to it.
- A name_cluster_id may be assigned to at most one face in this group unless the evidence explicitly shows duplicate names.
- Use ONLY candidate_index/name_cluster_id values present in the candidate options.
- Never assign the owner name to a non-owner face.
- Do not invent names, surnames, relations, or photo ids.
- This benchmark contains hard identity alignments: a name can be text-only in one photo and the matching face can appear in a different photo linked by time, place, event, co-face pattern, activity, or relation.
- Your goal is comparative assignment, not direct labeling only. When two mappings are not equally plausible, choose the best-supported one-to-one mapping and use confidence 0.55-0.75 for indirect but coherent evidence.
- Same-photo text is strong only when it plausibly labels the visible person. Menus, posters, books, forms, software labels, and other OCR can be noise.
- If same-photo text is noisy, downweight it but still use bridge_evidence_examples and recurring context to resolve the group.
- Use comparative_bridge_matrix as the compact evidence scorecard. Event-specific bridges such as birthday text -> birthday cake scene or brunch text -> brunch table scene are stronger than many generic same-cafe bridges.
- In multi-person scenes, do not assume the order of names in text matches the order of faces. Use repeated context, role words, co-face patterns, photo metadata, bridge examples, and exclusivity.
- Abstain only when the candidate mappings are genuinely indistinguishable after comparing all bridge examples and context.
- Keep each reasoning_path under 35 words and do not write per-candidate analysis.

Output ONLY one JSON object."""


@dataclass(slots=True)
class _JointGroup:
    group_id: str
    face_ids: list[str]
    name_cluster_ids: list[str]
    evidence_photo_ids: list[str]
    reason: str
    priority: float
    target_face_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "group_id": self.group_id,
            "face_ids": self.face_ids,
            "name_cluster_ids": self.name_cluster_ids,
            "evidence_photo_ids": self.evidence_photo_ids,
            "reason": self.reason,
            "priority": round(self.priority, 4),
        }
        if self.target_face_ids:
            data["target_face_ids"] = self.target_face_ids
        return data


def assign_identity_jointly(
    *,
    inventory: EvidenceInventory,
    clusters: list[NameCluster],
    hypotheses: list[IdentityHypothesis],
    config: ECDConfig,
    llm: Any,
) -> tuple[list[IdentityHypothesis], dict[str, Any]]:
    """Ask an LLM to solve high-value multi-face/multi-name conflicts jointly."""
    if llm is None:
        return hypotheses, {"enabled": False, "reason": "no_llm"}
    if config.llm_max_joint_groups <= 0:
        return hypotheses, {"enabled": False, "reason": "max_joint_groups_zero"}

    cluster_lookup = {c.cluster_id: c for c in clusters}
    by_face: dict[str, list[IdentityHypothesis]] = defaultdict(list)
    for hyp in hypotheses:
        by_face[hyp.face_id].append(hyp)
    for rows in by_face.values():
        rows.sort(key=lambda h: (-h.score, h.name_cluster_id))

    groups = _build_joint_groups(inventory, cluster_lookup, by_face, config)
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "n_groups_built": len(groups),
        "n_groups_reviewed": 0,
        "n_repair_groups_built": 0,
        "n_repair_groups_reviewed": 0,
        "groups": {},
        "errors": [],
    }
    locked_faces: set[str] = set()
    locked_clusters: set[str] = set()
    reviewed_group_ids: set[str] = set()

    def _review_groups(review_groups: list[_JointGroup], *, limit: int, phase: str) -> None:
        for group in review_groups[: max(0, limit)]:
            reviewed_group_ids.add(group.group_id)
            if all(face_id in locked_faces for face_id in group.face_ids):
                diagnostics["groups"][group.group_id] = {
                    "phase": phase,
                    "group": group.to_dict(),
                    "skipped": "all_faces_locked",
                }
                continue
            candidate_rows = _candidate_rows_for_group(inventory, cluster_lookup, by_face, group, config)
            if not _has_enough_joint_options(candidate_rows):
                diagnostics["groups"][group.group_id] = {
                    "phase": phase,
                    "group": group.to_dict(),
                    "skipped": "insufficient_joint_options",
                }
                continue
            payload = _build_payload(inventory, cluster_lookup, group, candidate_rows, config)
            result = llm.call_json(
                prompt=json.dumps(payload, ensure_ascii=False),
                system=_SYSTEM,
                temperature=0.0,
                max_tokens=4096,
                retries=1,
                stage=f"llm_joint_assignment.{phase}",
            )
            diagnostics["n_groups_reviewed"] += 1
            if phase == "adaptive_repair":
                diagnostics["n_repair_groups_reviewed"] += 1
            if not isinstance(result, dict):
                diagnostics["errors"].append({"group_id": group.group_id, "phase": phase, "error": "no_json"})
                diagnostics["groups"][group.group_id] = {
                    "phase": phase,
                    "group": group.to_dict(),
                    "error": "no_json",
                }
                continue
            applied = _apply_joint_result(
                inventory=inventory,
                cluster_lookup=cluster_lookup,
                by_face=by_face,
                group=group,
                candidate_rows=candidate_rows,
                result=result,
                config=config,
                locked_faces=locked_faces,
                locked_clusters=locked_clusters,
            )
            diagnostics["groups"][group.group_id] = {
                "phase": phase,
                "group": group.to_dict(),
                "raw_result": result,
                "applied": applied,
            }

    _review_groups(groups, limit=config.llm_max_joint_groups, phase="primary")

    if config.use_llm_adaptive_identity_repair and config.llm_adaptive_repair_max_groups > 0:
        repair_groups = _build_adaptive_repair_groups(
            inventory=inventory,
            cluster_lookup=cluster_lookup,
            by_face=by_face,
            config=config,
            locked_faces=locked_faces,
            locked_clusters=locked_clusters,
            reviewed_group_ids=reviewed_group_ids,
        )
        diagnostics["n_repair_groups_built"] = len(repair_groups)
        _review_groups(
            repair_groups,
            limit=config.llm_adaptive_repair_max_groups,
            phase="adaptive_repair",
        )

    _recompute_margins(by_face)
    return hypotheses, diagnostics


def _build_joint_groups(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    by_face: dict[str, list[IdentityHypothesis]],
    config: ECDConfig,
) -> list[_JointGroup]:
    raw_groups: list[_JointGroup] = []
    if config.use_llm_joint_photo_scenes:
        raw_groups.extend(_photo_scene_groups(inventory, cluster_lookup, by_face, config))
    if config.use_llm_joint_pair_swaps:
        raw_groups.extend(_pair_swap_groups(inventory, cluster_lookup, by_face, config))
    if config.use_llm_joint_name_conflicts:
        raw_groups.extend(_name_conflict_groups(inventory, cluster_lookup, by_face, config))
    if config.use_llm_joint_coface_ambiguity:
        raw_groups.extend(_coface_ambiguity_groups(inventory, cluster_lookup, by_face, config))
    if config.use_hard_identity_blocks:
        raw_groups.extend(_typed_social_block_groups(inventory, cluster_lookup, by_face, config))

    deduped: list[_JointGroup] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str]] = set()
    dedupe_target = max(config.llm_max_joint_groups * 2, config.llm_max_joint_groups)
    if config.use_hard_identity_blocks:
        dedupe_target += max(0, config.llm_hard_max_blocks * 2)
    for group in sorted(raw_groups, key=lambda g: (-g.priority, g.group_id)):
        if len(group.face_ids) < 2 or len(group.name_cluster_ids) < 2:
            continue
        face_limit = _group_face_limit(group, config)
        name_limit = _group_name_limit(group, config)
        if len(group.face_ids) > face_limit:
            group = _trim_group_faces(group, by_face, face_limit)
        if len(group.name_cluster_ids) > name_limit:
            group = _trim_group_names(group, by_face, name_limit)
        key = (tuple(sorted(group.face_ids)), tuple(sorted(group.name_cluster_ids)), group.reason)
        if key in seen:
            continue
        overlap_pool = (
            [existing for existing in deduped if _is_hard_identity_group(existing)]
            if _is_hard_identity_group(group)
            else deduped
        )
        if _overlaps_existing_group(group, overlap_pool):
            continue
        seen.add(key)
        deduped.append(group)
        hard_count = sum(1 for existing in deduped if _is_hard_identity_group(existing))
        if len(deduped) >= dedupe_target and (
            not config.use_hard_identity_blocks or hard_count >= config.llm_hard_max_blocks
        ):
            break
    return _reserve_hard_block_review_slots(deduped, config)


def _group_face_limit(group: _JointGroup, config: ECDConfig) -> int:
    if _is_hard_identity_group(group):
        return max(2, min(config.llm_joint_max_faces_per_group, config.llm_hard_max_faces_per_block))
    return max(2, config.llm_joint_max_faces_per_group)


def _group_name_limit(group: _JointGroup, config: ECDConfig) -> int:
    if _is_hard_identity_group(group):
        return max(2, config.llm_hard_max_names_per_block)
    return max(2, config.llm_joint_max_names_per_group)


def _is_hard_identity_group(group: _JointGroup) -> bool:
    return group.reason == "typed_social_hard_identity_block"


def _is_adaptive_repair_group(group: _JointGroup) -> bool:
    return group.reason.startswith("adaptive_identity_repair:")


def _is_adaptive_window_group(group: _JointGroup) -> bool:
    return group.reason == "adaptive_identity_repair:risk_coface_repair_window"


def _reserve_hard_block_review_slots(groups: list[_JointGroup], config: ECDConfig) -> list[_JointGroup]:
    if not config.use_hard_identity_blocks or config.llm_max_joint_groups <= 0:
        return groups
    review_limit = min(len(groups), config.llm_max_joint_groups)
    if review_limit <= 0:
        return groups
    visible = list(groups[:review_limit])
    hard_visible = sum(1 for group in visible if _is_hard_identity_group(group))
    target_hard = min(config.llm_hard_max_blocks, review_limit)
    if hard_visible >= target_hard:
        return groups
    reserve = [
        group
        for group in groups[review_limit:]
        if _is_hard_identity_group(group)
    ][: target_hard - hard_visible]
    if not reserve:
        return groups
    for hard_group in reserve:
        replace_idx = next(
            (
                idx
                for idx in range(len(visible) - 1, -1, -1)
                if not _is_hard_identity_group(visible[idx])
            ),
            -1,
        )
        if replace_idx < 0:
            break
        visible[replace_idx] = hard_group
    visible_ids = {group.group_id for group in visible}
    tail = [group for group in groups if group.group_id not in visible_ids]
    return visible + tail


def _build_adaptive_repair_groups(
    *,
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    by_face: dict[str, list[IdentityHypothesis]],
    config: ECDConfig,
    locked_faces: set[str],
    locked_clusters: set[str],
    reviewed_group_ids: set[str],
) -> list[_JointGroup]:
    """Build a small second-pass queue for unresolved or high-risk identities."""
    cluster_competition = _candidate_cluster_competition(inventory, cluster_lookup, by_face, config)
    risk_by_face = {
        face_id: _face_repair_risk(inventory, cluster_lookup, rows, config, cluster_competition)
        for face_id, rows in by_face.items()
        if face_id != inventory.owner_face_id and face_id not in locked_faces
    }
    risk_faces = {
        face_id
        for face_id, risk in risk_by_face.items()
        if risk >= float(config.llm_adaptive_repair_min_risk)
    }
    if not risk_faces:
        return []

    raw_groups: list[_JointGroup] = []
    raw_groups.extend(_photo_scene_groups(inventory, cluster_lookup, by_face, config))
    raw_groups.extend(_pair_swap_groups(inventory, cluster_lookup, by_face, config))
    raw_groups.extend(_name_conflict_groups(inventory, cluster_lookup, by_face, config))
    raw_groups.extend(_coface_ambiguity_groups(inventory, cluster_lookup, by_face, config))
    raw_groups.extend(_risk_window_repair_groups(inventory, cluster_lookup, by_face, config, risk_by_face))

    selected: list[_JointGroup] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str]] = set()
    top_faces = {
        face_id
        for face_id, _ in sorted(risk_by_face.items(), key=lambda row: (-row[1], row[0]))[
            : max(2, int(config.llm_adaptive_repair_top_faces))
        ]
    }
    for group in sorted(
        raw_groups,
        key=lambda g: (
            -sum(risk_by_face.get(face_id, 0.0) for face_id in g.face_ids),
            -g.priority,
            g.group_id,
        ),
    ):
        if group.group_id in reviewed_group_ids:
            continue
        faces = [face_id for face_id in group.face_ids if face_id not in locked_faces]
        if len(faces) < 2:
            continue
        if not (set(faces) & risk_faces & top_faces):
            continue
        name_ids = [name_id for name_id in group.name_cluster_ids if name_id not in locked_clusters]
        if len(name_ids) < 2:
            continue
        if len(faces) > config.llm_joint_max_faces_per_group:
            faces = sorted(faces, key=lambda fid: (-risk_by_face.get(fid, 0.0), fid))[
                : config.llm_joint_max_faces_per_group
            ]
        if len(name_ids) > config.llm_joint_max_names_per_group:
            name_ids = _repair_group_name_ids(
                group,
                by_face,
                faces,
                name_ids,
                limit=config.llm_joint_max_names_per_group,
            )
        base_targets = group.target_face_ids or sorted(set(faces) & risk_faces & top_faces)
        target_face_ids = [face_id for face_id in base_targets if face_id in faces and face_id in risk_faces]
        if not target_face_ids:
            continue
        priority = group.priority + 0.58 * sum(risk_by_face.get(face_id, 0.0) for face_id in faces)
        adaptive = _JointGroup(
            group_id=f"adaptive:{group.group_id}",
            face_ids=faces,
            name_cluster_ids=name_ids,
            evidence_photo_ids=group.evidence_photo_ids,
            reason=f"adaptive_identity_repair:{group.reason}",
            priority=priority,
            target_face_ids=target_face_ids,
        )
        key = (tuple(sorted(adaptive.face_ids)), tuple(sorted(adaptive.name_cluster_ids)), adaptive.reason)
        if key in seen or _overlaps_existing_group(adaptive, selected):
            continue
        seen.add(key)
        selected.append(adaptive)
        if len(selected) >= max(config.llm_adaptive_repair_max_groups * 3, config.llm_adaptive_repair_max_groups):
            break
    return selected


def _candidate_cluster_competition(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    by_face: dict[str, list[IdentityHypothesis]],
    config: ECDConfig,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for rows in by_face.values():
        filtered = _filtered_top_rows(inventory, cluster_lookup, rows, config, limit=4)
        if not filtered:
            continue
        top_score = filtered[0].score
        for rank, hyp in enumerate(filtered[:4]):
            if rank <= 1 or hyp.score >= top_score - 0.12 or _has_direct_anchor(hyp):
                counts[hyp.name_cluster_id] += 1
    return counts


def _face_repair_risk(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    rows: list[IdentityHypothesis],
    config: ECDConfig,
    cluster_competition: Counter[str],
) -> float:
    filtered = _filtered_top_rows(inventory, cluster_lookup, rows, config, limit=6)
    if len(filtered) < 2:
        return 0.0
    top = filtered[0]
    second_score = filtered[1].score
    margin = max(0.0, top.score - second_score)
    risk = 0.0
    risk += max(0.0, 0.58 - top.score) * 0.90
    risk += max(0.0, 0.18 - margin) * 1.25
    if _is_ambiguous_face(filtered):
        risk += 0.12
    if not _has_strong_direct_anchor(top):
        risk += 0.12
    if not _has_direct_anchor(top):
        risk += 0.10
    if any(_recall_priority(hyp) >= 0.28 for hyp in filtered[:6]):
        risk += 0.16
    if any(cluster_competition.get(hyp.name_cluster_id, 0) >= 2 for hyp in filtered[:3]):
        risk += 0.14
    if top.status == "llm_joint_selected" and float(top.llm_confidence or 0.0) >= 0.70:
        risk -= 0.20
    if _has_strong_direct_anchor(top) and margin >= 0.20:
        risk -= 0.20
    return round(max(0.0, risk), 4)


def _risk_window_repair_groups(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    by_face: dict[str, list[IdentityHypothesis]],
    config: ECDConfig,
    risk_by_face: dict[str, float],
) -> list[_JointGroup]:
    groups: list[_JointGroup] = []
    risk_faces = [
        face_id
        for face_id, risk in sorted(risk_by_face.items(), key=lambda row: (-row[1], row[0]))
        if risk >= float(config.llm_adaptive_repair_min_risk)
    ][: max(2, int(config.llm_adaptive_repair_top_faces))]
    for face_id in risk_faces:
        partner_counts: Counter[str] = Counter()
        evidence_photo_ids: list[str] = []
        for unit in inventory.face_units_by_face.get(face_id, []):
            photo = inventory.photo_lookup.get(unit.photo_id) or {}
            visible_faces = [
                str(fid)
                for fid in (photo.get("visible_face_ids") or [])
                if str(fid) in by_face and str(fid) != inventory.owner_face_id and str(fid) != face_id
            ]
            for other in visible_faces:
                partner_counts[other] += 1
            evidence_photo_ids.append(unit.photo_id)
        partners = [
            other
            for other, _ in sorted(
                partner_counts.items(),
                key=lambda row: (-row[1], -risk_by_face.get(row[0], 0.0), row[0]),
            )
            if other in by_face
        ][: max(1, config.llm_joint_max_faces_per_group - 1)]
        if not partners:
            continue
        faces = [face_id] + partners
        name_ids: list[str] = []
        for fid in faces:
            rows = _filtered_top_rows(inventory, cluster_lookup, by_face.get(fid, []), config, limit=4)
            name_ids.extend(hyp.name_cluster_id for hyp in rows[:3])
        name_ids = dedupe(name_ids)
        if len(name_ids) < 2:
            continue
        priority = 1.05 + risk_by_face.get(face_id, 0.0) + min(0.48, 0.08 * sum(partner_counts.values()))
        groups.append(
            _JointGroup(
                group_id=f"repair_window:{face_id}",
                face_ids=faces,
                name_cluster_ids=name_ids,
                evidence_photo_ids=dedupe(evidence_photo_ids)[:12],
                reason="risk_coface_repair_window",
                priority=priority,
            )
        )
    return groups


def _repair_group_name_ids(
    group: _JointGroup,
    by_face: dict[str, list[IdentityHypothesis]],
    faces: list[str],
    name_ids: list[str],
    *,
    limit: int,
) -> list[str]:
    counts: Counter[str] = Counter()
    scores: dict[str, float] = defaultdict(float)
    allowed = set(name_ids)
    for face_id in faces:
        for hyp in by_face.get(face_id, [])[:8]:
            if hyp.name_cluster_id not in allowed:
                continue
            counts[hyp.name_cluster_id] += 1
            scores[hyp.name_cluster_id] = max(scores[hyp.name_cluster_id], hyp.score + _recall_priority(hyp))
    return sorted(name_ids, key=lambda name_id: (-counts[name_id], -scores[name_id], name_id))[: max(2, limit)]


def _photo_scene_groups(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    by_face: dict[str, list[IdentityHypothesis]],
    config: ECDConfig,
) -> list[_JointGroup]:
    groups: list[_JointGroup] = []
    for photo in inventory.photos:
        photo_id = str(photo.get("photo_id") or "")
        faces = [
            str(fid)
            for fid in (photo.get("visible_face_ids") or [])
            if str(fid) != inventory.owner_face_id and str(fid) in by_face
        ]
        faces = dedupe(faces)
        if len(faces) < 2:
            continue
        if len(faces) > config.llm_joint_max_scene_faces:
            continue
        direct_name_ids: list[str] = []
        fallback_name_ids: list[str] = []
        direct_faces = 0
        for face_id in faces:
            rows = _filtered_top_rows(inventory, cluster_lookup, by_face.get(face_id, []), config, limit=4)
            face_has_direct = False
            for hyp in rows:
                if photo_id in hyp.evidence_packet.same_photo_ids:
                    direct_name_ids.append(hyp.name_cluster_id)
                    face_has_direct = True
            if face_has_direct:
                direct_faces += 1
            fallback_name_ids.extend(h.name_cluster_id for h in rows[:2])
        name_ids = dedupe(direct_name_ids or fallback_name_ids)
        if len(name_ids) < 2:
            continue
        n_person_entities = _person_entity_count(photo)
        priority = (
            1.0
            + 0.22 * len(faces)
            + 0.12 * len(name_ids)
            + 0.34 * direct_faces
            + (0.28 if n_person_entities >= 2 else 0.0)
            + _scene_ambiguity_bonus(by_face, faces)
            + _relation_bonus(cluster_lookup, name_ids, by_face, faces)
        )
        groups.append(
            _JointGroup(
                group_id=f"scene:{photo_id}",
                face_ids=faces,
                name_cluster_ids=name_ids,
                evidence_photo_ids=[photo_id],
                reason="same_photo_multi_face_scene",
                priority=priority,
            )
        )
    return groups


def _pair_swap_groups(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    by_face: dict[str, list[IdentityHypothesis]],
    config: ECDConfig,
) -> list[_JointGroup]:
    """Build compact two-face groups for reciprocal candidate ambiguity."""
    face_ids = sorted(by_face)
    groups: list[_JointGroup] = []
    for i, left_face in enumerate(face_ids):
        left_rows = _filtered_top_rows(
            inventory,
            cluster_lookup,
            by_face.get(left_face, []),
            config,
            limit=5,
        )
        if len(left_rows) < 2:
            continue
        left_names = [h.name_cluster_id for h in left_rows]
        left_photos = set(inventory.face_photo_ids.get(left_face, []))
        for right_face in face_ids[i + 1 :]:
            right_rows = _filtered_top_rows(
                inventory,
                cluster_lookup,
                by_face.get(right_face, []),
                config,
                limit=5,
            )
            if len(right_rows) < 2:
                continue
            right_names = [h.name_cluster_id for h in right_rows]
            overlapping_names = set(left_names) & set(right_names)
            shared_face_photos = sorted(left_photos & set(inventory.face_photo_ids.get(right_face, [])))
            if not overlapping_names or not shared_face_photos:
                continue
            name_ids = dedupe(left_names[:4] + right_names[:4])
            if len(name_ids) < 2:
                continue
            shared_anchor = _shared_anchor_count(left_rows, right_rows)
            if shared_anchor == 0 and len(overlapping_names) < 2 and not (
                _is_ambiguous_face(left_rows) or _is_ambiguous_face(right_rows)
            ):
                continue
            priority = (
                2.55
                + 0.26 * len(overlapping_names)
                + min(0.45, 0.08 * len(shared_face_photos))
                + 0.36 * shared_anchor
                + _scene_ambiguity_bonus(by_face, [left_face, right_face])
                + _relation_bonus(cluster_lookup, name_ids, by_face, [left_face, right_face])
            )
            groups.append(
                _JointGroup(
                    group_id=f"pair_swap:{left_face}:{right_face}",
                    face_ids=[left_face, right_face],
                    name_cluster_ids=name_ids,
                    evidence_photo_ids=dedupe(shared_face_photos)[:10],
                    reason="reciprocal_pair_swap_candidates",
                    priority=priority,
                )
            )
    return groups


def _name_conflict_groups(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    by_face: dict[str, list[IdentityHypothesis]],
    config: ECDConfig,
) -> list[_JointGroup]:
    name_to_face_scores: dict[str, list[tuple[str, float, int, bool]]] = defaultdict(list)
    for face_id, rows in by_face.items():
        filtered = _filtered_top_rows(inventory, cluster_lookup, rows, config, limit=5)
        top_score = filtered[0].score if filtered else 0.0
        for rank, hyp in enumerate(filtered):
            if hyp.score >= 0.18:
                near_top = hyp.score >= top_score - 0.10
                if rank <= 2 or near_top or _has_direct_anchor(hyp):
                    name_to_face_scores[hyp.name_cluster_id].append(
                        (face_id, hyp.score, rank, _has_direct_anchor(hyp))
                    )

    groups: list[_JointGroup] = []
    for name_id, face_scores in name_to_face_scores.items():
        if len(face_scores) < 2:
            continue
        face_scores.sort(key=lambda row: (row[2], -row[1], row[0]))
        max_faces = min(config.llm_joint_max_faces_per_group, 4)
        faces = [face_id for face_id, _, _, _ in face_scores[:max_faces]]
        name_ids = [name_id]
        evidence_photo_ids: list[str] = []
        for face_id in faces:
            rows = _filtered_top_rows(inventory, cluster_lookup, by_face.get(face_id, []), config, limit=4)
            name_ids.extend(h.name_cluster_id for h in rows[:3])
            for hyp in rows[:3]:
                evidence_photo_ids.extend(hyp.evidence_packet.same_photo_ids[:3])
                evidence_photo_ids.extend(hyp.evidence_packet.text_photo_ids[:2])
        name_ids = dedupe(name_ids)
        if len(name_ids) < 2:
            continue
        score_spread = max(score for _, score, _, _ in face_scores) - min(
            score for _, score, _, _ in face_scores[: len(faces)]
        )
        direct_count = sum(1 for _, _, _, direct in face_scores[: len(faces)] if direct)
        priority = (
            0.58
            + 0.16 * len(faces)
            + 0.10 * len(name_ids)
            + 0.24 * direct_count
            + max(0.0, 0.25 - score_spread)
            + _scene_ambiguity_bonus(by_face, faces)
            + _relation_bonus(cluster_lookup, name_ids, by_face, faces)
        )
        groups.append(
            _JointGroup(
                group_id=f"name_conflict:{name_id}",
                face_ids=faces,
                name_cluster_ids=name_ids,
                evidence_photo_ids=dedupe(evidence_photo_ids)[:12],
                reason="same_name_cluster_competes_across_faces",
                priority=priority,
            )
        )
    return groups


def _coface_ambiguity_groups(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    by_face: dict[str, list[IdentityHypothesis]],
    config: ECDConfig,
) -> list[_JointGroup]:
    photo_counts: Counter[tuple[str, ...]] = Counter()
    evidence_by_faces: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for photo in inventory.photos:
        photo_id = str(photo.get("photo_id") or "")
        faces = tuple(
            sorted(
                str(fid)
                for fid in (photo.get("visible_face_ids") or [])
                if str(fid) != inventory.owner_face_id and str(fid) in by_face
            )
        )
        if len(faces) < 2:
            continue
        photo_counts[faces] += 1
        evidence_by_faces[faces].append(photo_id)

    groups: list[_JointGroup] = []
    for idx, (faces_tuple, count) in enumerate(photo_counts.most_common(80)):
        faces = list(faces_tuple)
        if len(faces) < 2:
            continue
        if not any(_is_ambiguous_face(by_face.get(face_id, [])) for face_id in faces):
            continue
        name_ids: list[str] = []
        for face_id in faces:
            rows = _filtered_top_rows(inventory, cluster_lookup, by_face.get(face_id, []), config, limit=3)
            name_ids.extend(h.name_cluster_id for h in rows[:2])
        name_ids = dedupe(name_ids)
        if len(name_ids) < 2:
            continue
        priority = (
            0.72
            + min(0.5, count * 0.08)
            + 0.15 * len(faces)
            + _scene_ambiguity_bonus(by_face, faces)
            + _relation_bonus(cluster_lookup, name_ids, by_face, faces)
        )
        groups.append(
            _JointGroup(
                group_id=f"coface:{idx}",
                face_ids=faces,
                name_cluster_ids=name_ids,
                evidence_photo_ids=dedupe(evidence_by_faces[faces_tuple])[:10],
                reason="repeated_coface_ambiguous_candidates",
                priority=priority,
            )
        )
    return groups


def _typed_social_block_groups(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    by_face: dict[str, list[IdentityHypothesis]],
    config: ECDConfig,
) -> list[_JointGroup]:
    """Build channel-specific hard identity blocks for indirect social evidence."""
    if config.llm_hard_max_blocks <= 0:
        return []

    channel_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_limit = max(3, int(config.llm_hard_top_candidates_per_face))
    for face_id, rows in by_face.items():
        filtered = _filtered_top_rows(inventory, cluster_lookup, rows, config, limit=row_limit)
        if not filtered:
            continue
        face_scores = _face_social_channel_scores(inventory, face_id)
        hard_face = _is_hard_identity_face(filtered)
        hyp_scores: Counter[str] = Counter()
        for rank, hyp in enumerate(filtered[:4]):
            recall_priority = _recall_priority(hyp)
            if rank >= 3 and recall_priority <= 0.0:
                continue
            rank_weight = 0.45 if rank == 0 else 0.28 if rank == 1 else 0.18
            for channel, channel_score in _hypothesis_social_channel_scores(cluster_lookup, hyp).items():
                if channel_score < 1.0 and recall_priority <= 0.0:
                    continue
                hyp_scores[channel] += min(
                    0.85,
                    rank_weight + 0.12 * channel_score + (0.15 if recall_priority > 0.0 else 0.0),
                )
        combined = Counter(face_scores)
        for channel, score in hyp_scores.items():
            if face_scores.get(channel, 0.0) > 0.0 or hard_face:
                combined[channel] += score
        for channel, score in combined.items():
            if channel not in _HARD_BLOCK_CHANNELS:
                continue
            if score < float(config.llm_hard_min_face_channel_score):
                continue
            channel_records[channel].append(
                {
                    "face_id": face_id,
                    "score": float(score),
                    "hard": hard_face,
                    "priority": _face_priority(filtered),
                }
            )

    groups: list[_JointGroup] = []
    for channel, records in channel_records.items():
        records = _dedupe_channel_records(records)
        hard_count = sum(1 for record in records if record["hard"])
        if hard_count == 0 or len(records) < 2:
            continue
        if channel in _BROAD_HARD_CHANNELS and hard_count < 2:
            continue
        records.sort(
            key=lambda record: (
                -int(bool(record["hard"])),
                -float(record["score"]),
                -float(record["priority"]),
                str(record["face_id"]),
            )
        )
        face_limit = max(2, min(config.llm_hard_max_faces_per_block, config.llm_joint_max_faces_per_group))
        selected_records = records[:face_limit]
        if len(selected_records) < 2:
            continue
        faces = [str(record["face_id"]) for record in selected_records]
        name_ids = _hard_block_name_ids(
            channel,
            inventory,
            cluster_lookup,
            by_face,
            faces,
            config,
        )
        if len(name_ids) < 2:
            continue
        evidence_photo_ids = _hard_block_evidence_photo_ids(
            channel,
            inventory,
            cluster_lookup,
            by_face,
            faces,
            config,
        )
        channel_specificity = 0.22 if channel in _SPECIFIC_HARD_CHANNELS else -0.18
        if channel in _BROAD_HARD_CHANNELS:
            channel_specificity -= 0.28
        ambiguity_bonus = min(0.85, _scene_ambiguity_bonus(by_face, faces))
        relation_bonus = min(0.42, _relation_bonus(cluster_lookup, name_ids, by_face, faces))
        priority = (
            1.92
            + 0.18 * min(4, hard_count)
            + 0.10 * len(faces)
            + 0.035 * min(10, len(name_ids))
            + min(0.45, 0.08 * sum(float(record["score"]) for record in selected_records))
            + ambiguity_bonus
            + relation_bonus
            + channel_specificity
        )
        groups.append(
            _JointGroup(
                group_id=f"hard_block:{channel}:{len(groups)}",
                face_ids=faces,
                name_cluster_ids=name_ids,
                evidence_photo_ids=evidence_photo_ids,
                reason="typed_social_hard_identity_block",
                priority=priority,
            )
        )

    groups.sort(key=lambda group: (-group.priority, group.group_id))
    return groups[: max(0, int(config.llm_hard_max_blocks))]


def _dedupe_channel_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        face_id = str(record.get("face_id") or "")
        if not face_id:
            continue
        current = best.get(face_id)
        if current is None or float(record.get("score") or 0.0) > float(current.get("score") or 0.0):
            best[face_id] = record
    return list(best.values())


def _hard_block_name_ids(
    channel: str,
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    by_face: dict[str, list[IdentityHypothesis]],
    faces: list[str],
    config: ECDConfig,
) -> list[str]:
    name_scores: Counter[str] = Counter()
    row_limit = max(3, int(config.llm_hard_top_candidates_per_face))
    for face_id in faces:
        rows = _filtered_top_rows(inventory, cluster_lookup, by_face.get(face_id, []), config, limit=row_limit)
        top_score = rows[0].score if rows else 0.0
        for rank, hyp in enumerate(rows):
            channel_score = _hypothesis_social_channel_scores(cluster_lookup, hyp).get(channel, 0.0)
            near_top = hyp.score >= top_score - 0.16
            include = (
                rank < 3
                or channel_score > 0.0
                or _recall_priority(hyp) > 0.0
                or (_has_direct_anchor(hyp) and near_top)
            )
            if not include:
                continue
            name_scores[hyp.name_cluster_id] += 1.0 + 0.35 * channel_score + 0.12 * max(0.0, hyp.score)
    name_ids = sorted(name_scores, key=lambda name_id: (-name_scores[name_id], name_id))
    return name_ids[: max(2, int(config.llm_hard_max_names_per_block))]


def _hard_block_evidence_photo_ids(
    channel: str,
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    by_face: dict[str, list[IdentityHypothesis]],
    faces: list[str],
    config: ECDConfig,
) -> list[str]:
    ids: list[str] = []
    row_limit = max(3, int(config.llm_hard_top_candidates_per_face))
    for face_id in faces:
        ids.extend(_channel_photo_ids_for_face(inventory, face_id, channel, limit=4))
        rows = _filtered_top_rows(inventory, cluster_lookup, by_face.get(face_id, []), config, limit=row_limit)
        for hyp in rows[:row_limit]:
            ids.extend(_channel_photo_ids_for_hypothesis(channel, hyp, limit=4))
    return dedupe(ids)[:16]


def _candidate_rows_for_group(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    by_face: dict[str, list[IdentityHypothesis]],
    group: _JointGroup,
    config: ECDConfig,
) -> dict[str, list[IdentityHypothesis]]:
    out: dict[str, list[IdentityHypothesis]] = {}
    group_names = set(group.name_cluster_ids)
    for face_id in group.face_ids:
        rows = [
            hyp for hyp in by_face.get(face_id, [])
            if not _owner_like_candidate(inventory, cluster_lookup.get(hyp.name_cluster_id), hyp)
        ]
        selected: list[IdentityHypothesis] = []
        base_limit = max(1, int(_group_candidate_base_limit(group, config)))
        row_limit = base_limit
        if config.use_evidence_recall_broadener:
            row_limit = max(base_limit, int(_group_candidate_recall_limit(group, config)))
        for hyp in rows:
            if len(selected) < base_limit:
                selected.append(hyp)
            elif hyp.name_cluster_id in group_names:
                selected.append(hyp)
        if config.use_evidence_recall_broadener:
            recall_rows = sorted(
                [hyp for hyp in rows if _recall_priority(hyp) > 0.0],
                key=lambda h: (-_recall_priority(h), -h.score, h.name_cluster_id),
            )
            for hyp in recall_rows:
                if len(selected) >= row_limit:
                    break
                if all(existing.name_cluster_id != hyp.name_cluster_id for existing in selected):
                    selected.append(hyp)
        selected.sort(key=lambda h: (-h.score, h.name_cluster_id))
        out[face_id] = selected[: max(row_limit + 3, 1)]
    return out


def _group_candidate_base_limit(group: _JointGroup, config: ECDConfig) -> int:
    if _is_hard_identity_group(group):
        return max(config.llm_joint_top_candidates_per_face, config.llm_hard_top_candidates_per_face)
    return config.llm_joint_top_candidates_per_face


def _group_candidate_recall_limit(group: _JointGroup, config: ECDConfig) -> int:
    if _is_hard_identity_group(group):
        return max(
            config.llm_joint_recall_top_candidates_per_face,
            config.llm_hard_top_candidates_per_face,
        )
    return config.llm_joint_recall_top_candidates_per_face


def _build_payload(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    group: _JointGroup,
    candidate_rows: dict[str, list[IdentityHypothesis]],
    config: ECDConfig,
) -> dict[str, Any]:
    shared_photo_ids = _shared_photo_ids(inventory, group, candidate_rows)
    joint_chars = min(config.llm_photo_snippet_chars, 180)
    name_menu = [
        _name_cluster_brief(cluster_lookup[name_id])
        for name_id in group.name_cluster_ids
        if name_id in cluster_lookup
    ]
    assignment_policy = {
        "why_called": "This group was selected because local scores suggest a reciprocal swap or multi-face conflict.",
        "preferred_behavior": "Return the most likely one-to-one assignment when bridge evidence separates candidates, even if no direct face label exists.",
        "indirect_confidence_band": "Use 0.55-0.75 for coherent text-only-to-face bridge assignments; reserve >0.80 for direct/repeated evidence.",
        "abstain_rule": "Abstain only if both candidate mappings remain equally plausible after comparing bridge_evidence_examples.",
    }
    if _is_hard_identity_group(group):
        channel = _group_channel(group)
        assignment_policy.update(
            {
                "why_called": "This typed hard block was selected because several ambiguous faces and candidate names share a social/event channel.",
                "preferred_behavior": "Use the typed channel only as a scaffold; still require concrete photo bridges, recurring co-face patterns, direct labels, or role-compatible evidence for each assignment.",
                "hard_block_channel": channel,
                "hard_block_channel_meaning": _channel_description(channel),
            }
        )
    if _is_adaptive_repair_group(group):
        source_reason = group.reason.split(":", 1)[1] if ":" in group.reason else "unknown"
        assignment_policy.update(
            {
                "why_called": "This second-pass repair group was selected because local resolution remains risky after the primary hard-block pass.",
                "preferred_behavior": "Repair only identities with concrete evidence separation. Prefer direct labels, specific text-to-face bridges, repeated co-face patterns, and one-to-one exclusivity.",
                "repair_source_reason": source_reason,
                "repair_target_face_ids": group.target_face_ids or group.face_ids,
                "context_face_rule": "Only assign target faces. Non-target faces are included to compare co-face context and should be abstained unless explicitly listed as a repair target.",
                "abstain_rule": "Abstain when the only support is a generic venue, generic meal/social context, or a broad name mention that could fit multiple faces.",
            }
        )
    payload = {
        "user_id": inventory.album.get("user_id"),
        "owner": {
            "face_id": inventory.owner_face_id,
            "name_candidate": inventory.owner_name,
            "last_name": inventory.owner_last_name,
        },
        "group": group.to_dict(),
        "assignment_policy": assignment_policy,
        "name_cluster_menu": name_menu,
        "faces": [
            face_context(
                inventory,
                face_id,
                max_photos=3 if config.use_evidence_recall_broadener else 4,
                chars=joint_chars,
            )
            for face_id in group.face_ids
        ],
        "candidate_options_by_face": {
            face_id: [
                candidate_context(
                    inventory,
                    cluster_lookup,
                    hyp,
                    index=i,
                    chars=joint_chars,
                )
                for i, hyp in enumerate(rows)
            ]
            for face_id, rows in candidate_rows.items()
        },
        "comparative_bridge_matrix": _comparative_bridge_matrix(
            inventory,
            candidate_rows,
            config,
        ),
        "shared_evidence_photos": [
            photo_brief(inventory, pid, chars=joint_chars)
            for pid in shared_photo_ids[:8]
        ],
        "output_schema": {
            "assignments": [
                {
                    "face_id": "face id from group",
                    "selected_candidate_index": "integer index from candidate_options_by_face[face_id], or null",
                    "name_cluster_id": "selected cluster id",
                    "canonical_name": "selected candidate name",
                    "confidence": "0.0-1.0",
                    "relation_to_owner": "specific relation if inferable, else empty",
                    "relation_category": "family|friend|colleague|classmate|neighbor|other|empty",
                    "evidence_photo_ids": ["supporting photo ids from the provided evidence"],
                    "reasoning_path": "one short sentence, <=35 words, cite photo_ids, no newline",
                }
            ],
            "abstentions": [
                {
                    "face_id": "face id",
                    "failure_mode": "why the evidence is not sufficient",
                }
            ],
            "notes": ["optional global notes"],
        },
    }
    if _is_hard_identity_group(group):
        payload["typed_social_block_context"] = _typed_social_block_context(
            inventory,
            cluster_lookup,
            group,
            candidate_rows,
            config,
        )
    if _is_adaptive_repair_group(group):
        payload["adaptive_repair_context"] = _adaptive_repair_context(
            inventory,
            cluster_lookup,
            group,
            candidate_rows,
            config,
        )
    return payload


def _apply_joint_result(
    *,
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    by_face: dict[str, list[IdentityHypothesis]],
    group: _JointGroup,
    candidate_rows: dict[str, list[IdentityHypothesis]],
    result: dict[str, Any],
    config: ECDConfig,
    locked_faces: set[str],
    locked_clusters: set[str],
) -> list[dict[str, Any]]:
    assignments = result.get("assignments") or []
    if not isinstance(assignments, list):
        return []

    parsed = []
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        confidence = _bounded_float(assignment.get("confidence"), 0.0)
        parsed.append((confidence, assignment))
    parsed.sort(key=lambda row: -row[0])
    requested_cluster_counts = _requested_cluster_counts(assignments) if _is_hard_identity_group(group) else Counter()

    used_faces: set[str] = set()
    used_clusters: set[str] = set()
    applied: list[dict[str, Any]] = []

    for confidence, assignment in parsed:
        face_id = str(assignment.get("face_id") or "")
        if face_id in used_faces or face_id not in candidate_rows:
            continue
        if face_id in locked_faces:
            applied.append(
                {
                    "face_id": face_id,
                    "status": "skipped_face_locked_by_prior_group",
                    "confidence": round(confidence, 4),
                }
            )
            continue
        if _is_adaptive_repair_group(group) and group.target_face_ids and face_id not in set(group.target_face_ids):
            applied.append(
                {
                    "face_id": face_id,
                    "status": "skipped_adaptive_context_face",
                    "confidence": round(confidence, 4),
                }
            )
            continue
        accept_min_confidence = max(config.llm_accept_min_confidence, 0.45)
        if _is_hard_identity_group(group):
            accept_min_confidence = max(accept_min_confidence, config.llm_hard_accept_min_confidence)
        if _is_adaptive_repair_group(group):
            accept_min_confidence = max(
                accept_min_confidence,
                config.llm_adaptive_repair_accept_min_confidence,
            )
        if confidence < accept_min_confidence:
            applied.append(
                {
                    "face_id": face_id,
                    "status": "skipped_low_confidence",
                    "confidence": round(confidence, 4),
                }
            )
            continue
        selected = _match_assignment(assignment, candidate_rows.get(face_id, []), cluster_lookup)
        if not selected:
            applied.append(
                {
                    "face_id": face_id,
                    "status": "skipped_no_candidate_match",
                    "confidence": round(confidence, 4),
                }
            )
            continue
        selected_rank = _candidate_rank(selected, candidate_rows.get(face_id, []))
        if (
            _is_hard_identity_group(group)
            and config.llm_hard_guard_duplicate_clusters
            and requested_cluster_counts.get(selected.name_cluster_id, 0) > 1
            and not _has_strong_direct_anchor(selected)
        ):
            selected.llm_decision = selected.llm_decision or "hard_block_guarded_duplicate_cluster"
            selected.llm_confidence = max(float(selected.llm_confidence or 0.0), confidence)
            selected.llm_failure_mode = "hard_block_duplicate_cluster_guard"
            applied.append(
                {
                    "face_id": face_id,
                    "name_cluster_id": selected.name_cluster_id,
                    "status": "skipped_hard_block_duplicate_cluster",
                    "confidence": round(confidence, 4),
                }
            )
            continue
        if _is_hard_identity_group(group) and not _hard_assignment_has_support(
            inventory,
            cluster_lookup,
            group,
            selected,
            confidence,
            config,
        ):
            selected.llm_decision = selected.llm_decision or "hard_block_guarded_weak_support"
            selected.llm_confidence = max(float(selected.llm_confidence or 0.0), confidence)
            selected.llm_failure_mode = "hard_block_weak_evidence_guard"
            applied.append(
                {
                    "face_id": face_id,
                    "name_cluster_id": selected.name_cluster_id,
                    "status": "skipped_hard_block_weak_evidence",
                    "confidence": round(confidence, 4),
                }
            )
            continue
        if _is_adaptive_window_group(group) and not _adaptive_window_assignment_allowed(
            inventory,
            selected,
            selected_rank,
            confidence,
            config,
        ):
            selected.llm_decision = selected.llm_decision or "adaptive_window_guarded_weak_or_deep"
            selected.llm_confidence = max(float(selected.llm_confidence or 0.0), confidence)
            selected.llm_failure_mode = "adaptive_window_guard"
            applied.append(
                {
                    "face_id": face_id,
                    "name_cluster_id": selected.name_cluster_id,
                    "status": "skipped_adaptive_window_guard",
                    "confidence": round(confidence, 4),
                    "candidate_rank": selected_rank,
                }
            )
            continue
        if _is_adaptive_repair_group(group) and not _adaptive_assignment_has_support(
            inventory,
            selected,
            confidence,
            config,
        ):
            selected.llm_decision = selected.llm_decision or "adaptive_repair_guarded_weak_support"
            selected.llm_confidence = max(float(selected.llm_confidence or 0.0), confidence)
            selected.llm_failure_mode = "adaptive_repair_weak_evidence_guard"
            applied.append(
                {
                    "face_id": face_id,
                    "name_cluster_id": selected.name_cluster_id,
                    "status": "skipped_adaptive_repair_weak_evidence",
                    "confidence": round(confidence, 4),
                }
            )
            continue
        if selected.name_cluster_id in locked_clusters:
            applied.append(
                {
                    "face_id": face_id,
                    "name_cluster_id": selected.name_cluster_id,
                    "status": "skipped_name_locked_by_prior_group",
                    "confidence": round(confidence, 4),
                }
            )
            continue
        if selected.name_cluster_id in used_clusters:
            applied.append(
                {
                    "face_id": face_id,
                    "name_cluster_id": selected.name_cluster_id,
                    "status": "skipped_duplicate_name_cluster",
                    "confidence": round(confidence, 4),
                }
            )
            continue
        if _owner_like_candidate(inventory, cluster_lookup.get(selected.name_cluster_id), selected):
            applied.append(
                {
                    "face_id": face_id,
                    "name_cluster_id": selected.name_cluster_id,
                    "status": "skipped_owner_like",
                    "confidence": round(confidence, 4),
                }
            )
            continue
        if _is_overbroad_recall_candidate(selected):
            selected.llm_decision = selected.llm_decision or "joint_guarded_overbroad_select"
            selected.llm_confidence = max(float(selected.llm_confidence or 0.0), confidence)
            selected.llm_failure_mode = "overbroad_recall_guard"
            applied.append(
                {
                    "face_id": face_id,
                    "name_cluster_id": selected.name_cluster_id,
                    "status": "skipped_overbroad_recall_candidate",
                    "confidence": round(confidence, 4),
                }
            )
            continue

        _apply_assignment_to_hypothesis(selected, assignment, confidence)
        _boost_selected(selected, confidence)
        if confidence >= 0.65:
            _demote_face_competitors(selected, by_face.get(face_id, []), confidence)
            _demote_name_competitors(selected, by_face, group.face_ids, confidence)

        used_faces.add(face_id)
        used_clusters.add(selected.name_cluster_id)
        if confidence >= 0.65:
            locked_faces.add(face_id)
            locked_clusters.add(selected.name_cluster_id)
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


def _apply_assignment_to_hypothesis(
    hyp: IdentityHypothesis,
    assignment: dict[str, Any],
    confidence: float,
) -> None:
    hyp.status = "llm_joint_selected"
    hyp.llm_decision = "joint_selected"
    hyp.llm_confidence = confidence
    hyp.llm_canonical_name = normalize_name(
        str(assignment.get("canonical_name") or hyp.canonical_name_candidate)
    )
    hyp.llm_relation_to_owner = str(assignment.get("relation_to_owner") or "")
    hyp.llm_relation_category = str(assignment.get("relation_category") or "")
    hyp.llm_reasoning_path = str(assignment.get("reasoning_path") or "")
    hyp.llm_evidence_photo_ids = [
        str(pid) for pid in (assignment.get("evidence_photo_ids") or []) if str(pid).startswith("photo_")
    ]
    hyp.signal_breakdown["llm_joint_confidence"] = confidence
    if hyp.llm_canonical_name:
        hyp.canonical_name_candidate = hyp.llm_canonical_name
    if hyp.llm_reasoning_path:
        hyp.evidence_packet.narrative_summary = hyp.llm_reasoning_path


def _boost_selected(hyp: IdentityHypothesis, confidence: float) -> None:
    if _has_strong_direct_anchor(hyp):
        hyp.score = max(hyp.score, min(0.99, 0.46 + 0.50 * confidence))
    else:
        hyp.score = max(hyp.score, min(0.72, 0.40 + 0.36 * confidence))


def _hard_assignment_has_support(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    group: _JointGroup,
    hyp: IdentityHypothesis,
    confidence: float,
    config: ECDConfig,
) -> bool:
    if _has_strong_direct_anchor(hyp):
        return True
    channel = _group_channel(group)
    channel_score = _hypothesis_social_channel_scores(cluster_lookup, hyp).get(channel, 0.0) if channel else 0.0
    if config.llm_hard_require_channel_support and channel and channel_score < _hard_channel_support_threshold(channel):
        return False
    bridge_examples = _ranked_bridge_matches(inventory, hyp, max_examples=3)
    best_bridge = max((float(ex.get("match_score") or 0.0) for ex in bridge_examples), default=0.0)
    recall_priority = _recall_priority(hyp)
    try:
        event_count = float(hyp.signal_breakdown.get("event_bridge_count") or 0.0)
    except (TypeError, ValueError):
        event_count = 0.0
    if event_count >= 40.0 and best_bridge < 0.58 and not _has_direct_anchor(hyp):
        return False
    if best_bridge >= 0.46 or recall_priority >= 0.36:
        return True
    return bool(confidence >= 0.82 and (best_bridge >= 0.34 or recall_priority >= 0.28))


def _hard_channel_support_threshold(channel: str) -> float:
    if channel in _SPECIFIC_HARD_CHANNELS:
        return 0.75
    return 1.0


def _adaptive_assignment_has_support(
    inventory: EvidenceInventory,
    hyp: IdentityHypothesis,
    confidence: float,
    config: ECDConfig,
) -> bool:
    if _has_strong_direct_anchor(hyp):
        return True
    bridge_examples = _ranked_bridge_matches(inventory, hyp, max_examples=3)
    best_bridge = max((float(ex.get("match_score") or 0.0) for ex in bridge_examples), default=0.0)
    recall_priority = _recall_priority(hyp)
    if best_bridge >= float(config.llm_adaptive_repair_min_bridge_score):
        return True
    if recall_priority >= float(config.llm_adaptive_repair_min_recall_priority):
        return True
    if _has_direct_anchor(hyp) and confidence >= 0.68:
        return True
    bridge_floor = max(0.0, float(config.llm_adaptive_repair_min_bridge_score) - 0.08)
    recall_floor = max(0.0, float(config.llm_adaptive_repair_min_recall_priority) - 0.06)
    return bool(confidence >= 0.82 and (best_bridge >= bridge_floor or recall_priority >= recall_floor))


def _adaptive_window_assignment_allowed(
    inventory: EvidenceInventory,
    hyp: IdentityHypothesis,
    selected_rank: int,
    confidence: float,
    config: ECDConfig,
) -> bool:
    if confidence < float(config.llm_adaptive_repair_window_accept_min_confidence) and not _has_strong_direct_anchor(hyp):
        return False
    max_rank = max(0, int(config.llm_adaptive_repair_window_max_candidate_rank))
    if selected_rank <= max_rank:
        return True
    return bool(confidence >= 0.84 and _has_unambiguous_direct_anchor(inventory, hyp))


def _demote_face_competitors(
    selected: IdentityHypothesis,
    rows: list[IdentityHypothesis],
    confidence: float,
) -> None:
    margin = 0.16 if _has_strong_direct_anchor(selected) else 0.10
    if confidence >= 0.82:
        margin += 0.04
    for hyp in rows:
        if hyp is selected:
            continue
        hyp.llm_decision = hyp.llm_decision or "joint_not_selected"
        hyp.status = "llm_joint_rejected"
        hyp.llm_confidence = max(float(hyp.llm_confidence or 0.0), confidence)
        hyp.score = min(hyp.score, max(0.0, selected.score - margin))


def _demote_name_competitors(
    selected: IdentityHypothesis,
    by_face: dict[str, list[IdentityHypothesis]],
    group_faces: list[str],
    confidence: float,
) -> None:
    margin = 0.14 if confidence < 0.82 else 0.18
    for face_id in group_faces:
        if face_id == selected.face_id:
            continue
        for hyp in by_face.get(face_id, []):
            if hyp.name_cluster_id != selected.name_cluster_id:
                continue
            hyp.llm_decision = hyp.llm_decision or "joint_name_taken_elsewhere"
            hyp.status = "llm_joint_rejected"
            hyp.llm_confidence = max(float(hyp.llm_confidence or 0.0), confidence)
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


def _candidate_rank(selected: IdentityHypothesis, rows: list[IdentityHypothesis]) -> int:
    for idx, hyp in enumerate(rows):
        if hyp is selected:
            return idx
    return 999


def _requested_cluster_counts(assignments: list[Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        name_cluster_id = str(assignment.get("name_cluster_id") or "")
        if name_cluster_id:
            counts[name_cluster_id] += 1
    return counts


def _filtered_top_rows(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    rows: list[IdentityHypothesis],
    config: ECDConfig,
    *,
    limit: int,
) -> list[IdentityHypothesis]:
    out = []
    filtered = []
    for hyp in rows:
        if _owner_like_candidate(inventory, cluster_lookup.get(hyp.name_cluster_id), hyp):
            continue
        filtered.append(hyp)
    for hyp in filtered[:limit]:
        out.append(hyp)
    if config.use_evidence_recall_broadener:
        recall_rows = sorted(
            [hyp for hyp in filtered if _recall_priority(hyp) > 0.0],
            key=lambda h: (-_recall_priority(h), -h.score, h.name_cluster_id),
        )
        expanded_limit = max(limit, min(len(filtered), limit + 3))
        for hyp in recall_rows:
            if len(out) >= expanded_limit:
                break
            if all(existing.name_cluster_id != hyp.name_cluster_id for existing in out):
                out.append(hyp)
    return out


def _has_enough_joint_options(candidate_rows: dict[str, list[IdentityHypothesis]]) -> bool:
    faces_with_options = [face_id for face_id, rows in candidate_rows.items() if rows]
    name_ids = {hyp.name_cluster_id for rows in candidate_rows.values() for hyp in rows}
    return len(faces_with_options) >= 2 and len(name_ids) >= 2


def _shared_photo_ids(
    inventory: EvidenceInventory,
    group: _JointGroup,
    candidate_rows: dict[str, list[IdentityHypothesis]],
) -> list[str]:
    ids: list[str] = list(group.evidence_photo_ids)
    for face_id in group.face_ids:
        ids.extend(inventory.face_photo_ids.get(face_id, [])[:4])
    for rows in candidate_rows.values():
        for hyp in rows[:4]:
            ids.extend(hyp.evidence_packet.same_photo_ids[:4])
            ids.extend(hyp.evidence_packet.text_photo_ids[:3])
            ids.extend(hyp.evidence_packet.face_photo_ids[:3])
    return dedupe(ids)


def _comparative_bridge_matrix(
    inventory: EvidenceInventory,
    candidate_rows: dict[str, list[IdentityHypothesis]],
    config: ECDConfig,
) -> dict[str, list[dict[str, Any]]]:
    matrix: dict[str, list[dict[str, Any]]] = {}
    for face_id, rows in candidate_rows.items():
        entries = []
        for idx, hyp in enumerate(rows):
            bridge_examples = _ranked_bridge_matches(inventory, hyp, max_examples=3)
            semantic_score = max((ex["match_score"] for ex in bridge_examples), default=0.0)
            text_only_score = max(
                (ex["match_score"] for ex in bridge_examples if not ex["text_photo_has_faces"]),
                default=0.0,
            )
            direct = _same_photo_quality(inventory, hyp)
            entries.append(
                {
                    "candidate_index": idx,
                    "name_cluster_id": hyp.name_cluster_id,
                    "canonical_name_candidate": hyp.canonical_name_candidate,
                    "local_score": round(hyp.score, 4),
                    "local_margin": round(hyp.margin, 4),
                    "same_photo_quality": direct,
                    "semantic_bridge_score": round(semantic_score, 4),
                    "text_only_bridge_score": round(text_only_score, 4),
                    "top_event_bridge_examples": bridge_examples,
                }
            )
        matrix[face_id] = entries
    return matrix


def _ranked_bridge_matches(
    inventory: EvidenceInventory,
    hyp: IdentityHypothesis,
    *,
    max_examples: int,
) -> list[dict[str, Any]]:
    examples = []
    for pair in hyp.evidence_packet.bridge_photo_pairs:
        text_photo_id = str(pair.get("text_photo_id") or "")
        face_photo_id = str(pair.get("face_photo_id") or "")
        text_photo = inventory.photo_lookup.get(text_photo_id) or {}
        face_photo = inventory.photo_lookup.get(face_photo_id) or {}
        shared = _shared_event_keywords(text_photo, face_photo)
        match_score = _bridge_match_score(pair, text_photo, face_photo, shared)
        examples.append(
            {
                "bridge": str(pair.get("bridge") or ""),
                "text_photo_id": text_photo_id,
                "face_photo_id": face_photo_id,
                "match_score": round(match_score, 4),
                "shared_keywords": shared[:8],
                "text_photo_has_faces": bool(text_photo.get("visible_face_ids")),
                "face_photo_faces": [str(fid) for fid in (face_photo.get("visible_face_ids") or [])],
                "text_caption": _short_caption(text_photo, limit=150),
                "face_caption": _short_caption(face_photo, limit=150),
            }
        )
    examples.sort(
        key=lambda ex: (
            -float(ex["match_score"]),
            str(ex["bridge"]),
            str(ex["text_photo_id"]),
            str(ex["face_photo_id"]),
        )
    )
    return examples[:max_examples]


_EVENT_KEYWORDS = {
    "anniversary",
    "association",
    "bake",
    "baked",
    "bakery",
    "bar",
    "birthday",
    "breakfast",
    "brunch",
    "cafe",
    "cake",
    "candles",
    "card",
    "christmas",
    "clinic",
    "coffee",
    "condo",
    "conference",
    "cookies",
    "court",
    "courthouse",
    "cycling",
    "dessert",
    "dinner",
    "doctor",
    "downtown",
    "father",
    "friend",
    "garden",
    "holiday",
    "home",
    "hospital",
    "kitchen",
    "law",
    "legal",
    "lobby",
    "marina",
    "meeting",
    "mother",
    "networking",
    "office",
    "parent",
    "park",
    "party",
    "reception",
    "reservation",
    "restaurant",
    "school",
    "thanksgiving",
    "volunteer",
}

_GENERIC_EVENT_KEYWORDS = {
    "bar",
    "cafe",
    "friend",
    "restaurant",
}

_STOPWORDS = {
    "about",
    "after",
    "again",
    "around",
    "being",
    "camera",
    "clear",
    "conversation",
    "displayed",
    "during",
    "faces",
    "front",
    "group",
    "image",
    "inside",
    "message",
    "people",
    "photo",
    "scene",
    "seated",
    "shows",
    "smile",
    "smiling",
    "standing",
    "table",
    "their",
    "there",
    "three",
    "visible",
    "while",
    "with",
    "woman",
    "women",
}

_HARD_BLOCK_CHANNELS = {
    "birthday_social",
    "clinic_health",
    "family_home",
    "hobby_activity",
    "meal_social",
    "neighbor_residential",
    "professional_legal",
    "school_class",
    "volunteer_community",
}

_SPECIFIC_HARD_CHANNELS = {
    "birthday_social",
    "clinic_health",
    "neighbor_residential",
    "professional_legal",
    "school_class",
    "volunteer_community",
}

_BROAD_HARD_CHANNELS = {
    "family_home",
    "meal_social",
    "hobby_activity",
}

_CHANNEL_DESCRIPTIONS = {
    "birthday_social": "birthday, party, cake, or celebration context",
    "clinic_health": "clinic, doctor, hospital, or health-admin context",
    "family_home": "family, holiday, kitchen, home, parent, or relative context",
    "hobby_activity": "music, choir, cycling, garden, baking, or hobby context",
    "meal_social": "brunch, dinner, cafe, restaurant, coffee, or casual social meal context",
    "neighbor_residential": "neighbor, apartment, condo, lobby, or community-building context",
    "professional_legal": "office, meeting, legal, conference, badge, or professional context",
    "school_class": "school, class, campus, study, or classmate context",
    "volunteer_community": "church, volunteer, neighborhood, or community-service context",
}


def _shared_event_keywords(text_photo: dict[str, Any], face_photo: dict[str, Any]) -> list[str]:
    text_terms = _photo_event_keywords(text_photo)
    face_terms = _photo_event_keywords(face_photo)
    return sorted(text_terms & face_terms)


def _photo_event_keywords(photo: dict[str, Any]) -> set[str]:
    parts = [str(photo.get("caption") or "")]
    parts.extend(str(t) for t in (photo.get("visible_text") or []))
    parts.extend(str(e.get("surface") or "") for e in (photo.get("text_entities") or []))
    metadata = photo.get("metadata") or {}
    parts.append(str(metadata.get("gps_location") or ""))
    parts.append(str(metadata.get("gps_city") or ""))
    found = set()
    text_tokens = tokens(" ".join(parts))
    for token in text_tokens:
        if len(token) < 4 or token in _STOPWORDS:
            continue
        if token in _EVENT_KEYWORDS:
            found.add(token)
    joined = " ".join(parts).lower()
    for phrase in {
        "birthday cake",
        "coffee cups",
        "legal association",
        "mother's day",
        "open table",
        "opentable",
        "text message",
    }:
        if phrase in joined:
            found.update(token for token in tokens(phrase) if len(token) >= 4)
    return _expand_event_keywords(found)


def _expand_event_keywords(found: set[str]) -> set[str]:
    expanded = set(found)
    if {"brunch", "breakfast", "coffee", "cafe"} & found:
        expanded.add("__brunch_meal__")
    if {"birthday", "cake", "candles", "party"} & found:
        expanded.add("__birthday_event__")
    if {"law", "legal", "court", "courthouse", "association", "networking", "reception"} & found:
        expanded.add("__professional_event__")
    if {"mother", "father", "parent", "card", "holiday", "thanksgiving", "christmas"} & found:
        expanded.add("__family_event__")
    if {"condo", "home", "kitchen", "cookies", "lobby"} & found:
        expanded.add("__home_visit__")
    return expanded


def _face_social_channel_scores(inventory: EvidenceInventory, face_id: str) -> Counter[str]:
    scores: Counter[str] = Counter()
    for unit in inventory.face_units_by_face.get(face_id, []):
        term_channels = social_channels_from_terms(
            list(unit.relation_terms) + list(unit.activity_terms) + list(unit.venue_terms)
        )
        for channel in term_channels:
            if channel in _HARD_BLOCK_CHANNELS:
                scores[channel] += 1.0
        photo = inventory.photo_lookup.get(unit.photo_id) or {}
        features = photo_event_features(photo)
        for channel in features.get("channels", []):
            if channel in _HARD_BLOCK_CHANNELS:
                scores[str(channel)] += 0.65
    return scores


def _hypothesis_social_channel_scores(
    cluster_lookup: dict[str, NameCluster],
    hyp: IdentityHypothesis,
) -> Counter[str]:
    scores: Counter[str] = Counter()
    cluster = cluster_lookup.get(hyp.name_cluster_id)
    terms: list[str] = []
    if cluster:
        terms.extend(cluster.relation_terms)
        terms.extend(cluster.activity_terms)
        terms.extend(cluster.venue_terms)
    terms.extend(hyp.evidence_packet.relation_clues)
    terms.extend(hyp.evidence_packet.activity_clues)
    terms.extend(hyp.evidence_packet.venue_clues)
    for channel in social_channels_from_terms(terms):
        if channel in _HARD_BLOCK_CHANNELS:
            scores[channel] += 1.0
    for pair in hyp.evidence_packet.bridge_photo_pairs:
        for channel in pair.get("event_channels") or []:
            channel = str(channel)
            if channel in _HARD_BLOCK_CHANNELS:
                scores[channel] += 0.90 if pair.get("strong_signal") else 0.45
        for channel in social_channels_from_terms(pair.get("shared_keywords") or []):
            if channel in _HARD_BLOCK_CHANNELS:
                scores[channel] += 0.45
    return scores


def _is_hard_identity_face(rows: list[IdentityHypothesis]) -> bool:
    if len(rows) < 2:
        return False
    top = rows[0]
    margin = top.score - rows[1].score
    if _is_ambiguous_face(rows):
        return True
    if top.score < 0.52:
        return True
    if not _has_strong_direct_anchor(top) and margin < 0.25:
        return True
    if _recall_priority(top) > 0.0:
        return True
    if any(_recall_priority(hyp) >= 0.34 for hyp in rows[:5]):
        return True
    return bool(not _has_direct_anchor(top) and top.score < 0.65)


def _channel_photo_ids_for_face(
    inventory: EvidenceInventory,
    face_id: str,
    channel: str,
    *,
    limit: int,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for unit in inventory.face_units_by_face.get(face_id, []):
        score = 0.0
        if channel in social_channels_from_terms(
            list(unit.relation_terms) + list(unit.activity_terms) + list(unit.venue_terms)
        ):
            score += 1.0
        photo = inventory.photo_lookup.get(unit.photo_id) or {}
        features = photo_event_features(photo)
        if channel in set(features.get("channels") or []):
            score += 0.75
        if score > 0:
            scored.append((score, unit.photo_id))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return dedupe(pid for _, pid in scored)[:limit]


def _channel_photo_ids_for_hypothesis(
    channel: str,
    hyp: IdentityHypothesis,
    *,
    limit: int,
) -> list[str]:
    ids: list[str] = []
    for pair in hyp.evidence_packet.bridge_photo_pairs:
        channels = {str(ch) for ch in (pair.get("event_channels") or [])}
        if channel not in channels and channel not in social_channels_from_terms(pair.get("shared_keywords") or []):
            continue
        ids.append(str(pair.get("face_photo_id") or ""))
        ids.append(str(pair.get("text_photo_id") or ""))
    ids.extend(hyp.evidence_packet.same_photo_ids[:2])
    if not ids:
        ids.extend(hyp.evidence_packet.text_photo_ids[:1])
        ids.extend(hyp.evidence_packet.face_photo_ids[:1])
    return [pid for pid in dedupe(ids) if pid.startswith("photo_")][:limit]


def _typed_social_block_context(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    group: _JointGroup,
    candidate_rows: dict[str, list[IdentityHypothesis]],
    config: ECDConfig,
) -> dict[str, Any]:
    channel = _group_channel(group)
    return {
        "channel": channel,
        "channel_meaning": _channel_description(channel),
        "warning": "The channel is a grouping prior, not proof of identity. Prefer direct labels, event-specific bridges, and one-to-one exclusivity.",
        "face_channel_evidence": {
            face_id: {
                "matched_terms": _face_channel_terms(inventory, face_id, channel, limit=10),
                "channel_photos": [
                    photo_brief(inventory, pid, chars=config.llm_photo_snippet_chars)
                    for pid in _channel_photo_ids_for_face(inventory, face_id, channel, limit=3)
                ],
                "candidate_channel_scores": [
                    {
                        "candidate_index": idx,
                        "name_cluster_id": hyp.name_cluster_id,
                        "canonical_name_candidate": hyp.canonical_name_candidate,
                        "channel_score": round(
                            _hypothesis_social_channel_scores(cluster_lookup, hyp).get(channel, 0.0),
                            4,
                        ),
                        "recall_priority": round(_recall_priority(hyp), 4),
                    }
                    for idx, hyp in enumerate(candidate_rows.get(face_id, [])[: max(3, config.llm_hard_top_candidates_per_face)])
                ],
            }
            for face_id in group.face_ids
        },
    }


def _adaptive_repair_context(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    group: _JointGroup,
    candidate_rows: dict[str, list[IdentityHypothesis]],
    config: ECDConfig,
) -> dict[str, Any]:
    cluster_competition = _candidate_cluster_competition(inventory, cluster_lookup, candidate_rows, config)
    return {
        "warning": "This is a precision-oriented repair pass. Do not replace uncertainty with a guess.",
        "repair_target_face_ids": group.target_face_ids or group.face_ids,
        "faces": {
            face_id: {
                "is_repair_target": face_id in set(group.target_face_ids or group.face_ids),
                "repair_risk": _face_repair_risk(
                    inventory,
                    cluster_lookup,
                    candidate_rows.get(face_id, []),
                    config,
                    cluster_competition,
                ),
                "top_candidates": [
                    {
                        "candidate_index": idx,
                        "name_cluster_id": hyp.name_cluster_id,
                        "canonical_name_candidate": hyp.canonical_name_candidate,
                        "score": round(hyp.score, 4),
                        "margin": round(hyp.margin, 4),
                        "has_direct_anchor": _has_direct_anchor(hyp),
                        "recall_priority": round(_recall_priority(hyp), 4),
                        "best_bridge_score": round(
                            max(
                                (
                                    float(example.get("match_score") or 0.0)
                                    for example in _ranked_bridge_matches(inventory, hyp, max_examples=2)
                                ),
                                default=0.0,
                            ),
                            4,
                        ),
                    }
                    for idx, hyp in enumerate(candidate_rows.get(face_id, [])[:5])
                ],
            }
            for face_id in group.face_ids
        },
    }


def _face_channel_terms(
    inventory: EvidenceInventory,
    face_id: str,
    channel: str,
    *,
    limit: int,
) -> list[str]:
    terms: list[str] = []
    for unit in inventory.face_units_by_face.get(face_id, []):
        unit_terms = list(unit.relation_terms) + list(unit.activity_terms) + list(unit.venue_terms)
        if channel in social_channels_from_terms(unit_terms):
            terms.extend(unit_terms)
        photo = inventory.photo_lookup.get(unit.photo_id) or {}
        features = photo_event_features(photo)
        if channel in set(features.get("channels") or []):
            terms.extend(str(term) for term in (features.get("terms") or []))
    return dedupe(term.lower() for term in terms if str(term).strip())[:limit]


def _group_channel(group: _JointGroup) -> str:
    parts = group.group_id.split(":")
    if len(parts) >= 3 and parts[0] == "hard_block":
        return parts[1]
    return ""


def _channel_description(channel: str) -> str:
    return _CHANNEL_DESCRIPTIONS.get(channel, channel or "unknown social/event channel")


def _bridge_match_score(
    pair: dict[str, str],
    text_photo: dict[str, Any],
    face_photo: dict[str, Any],
    shared_keywords: list[str],
) -> float:
    bridge = str(pair.get("bridge") or "")
    base = {"location_month": 0.34, "near_date_city": 0.24, "city_month": 0.16}.get(bridge, 0.08)
    strong = [kw for kw in shared_keywords if kw not in _GENERIC_EVENT_KEYWORDS]
    generic = [kw for kw in shared_keywords if kw in _GENERIC_EVENT_KEYWORDS]
    score = base + min(0.42, 0.14 * len(strong) + 0.04 * len(generic))
    if not text_photo.get("visible_face_ids"):
        score += 0.12
    else:
        score -= 0.06
    if _person_entity_count(text_photo) >= 2 and len(text_photo.get("visible_face_ids") or []) >= 2:
        score -= 0.08
    if len(face_photo.get("visible_face_ids") or []) <= 2:
        score += 0.04
    return max(0.0, min(1.0, score))


def _same_photo_quality(inventory: EvidenceInventory, hyp: IdentityHypothesis) -> list[dict[str, Any]]:
    out = []
    for photo_id in hyp.evidence_packet.same_photo_ids[:4]:
        photo = inventory.photo_lookup.get(photo_id) or {}
        faces = [str(fid) for fid in (photo.get("visible_face_ids") or [])]
        n_person_entities = _person_entity_count(photo)
        ambiguous = len(faces) >= 2 and n_person_entities >= 1
        out.append(
            {
                "photo_id": photo_id,
                "n_faces": len(faces),
                "n_person_entities": n_person_entities,
                "ambiguous_multi_face_text": ambiguous,
                "caption": _short_caption(photo, limit=120),
            }
        )
    return out


def _short_caption(photo: dict[str, Any], *, limit: int) -> str:
    text = " ".join(str(photo.get("caption") or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _name_cluster_brief(cluster: NameCluster) -> dict[str, Any]:
    return {
        "name_cluster_id": cluster.cluster_id,
        "primary_surface": cluster.primary_surface,
        "surfaces": cluster.surfaces[:8],
        "first_name": cluster.first_name,
        "last_name_candidates": cluster.last_name_candidates,
        "mention_photo_ids": cluster.mention_photo_ids[:8],
        "relation_terms": cluster.relation_terms[:8],
        "activity_terms": cluster.activity_terms[:8],
        "venue_terms": cluster.venue_terms[:8],
        "quality_flags": cluster.quality_flags,
    }


def _scene_ambiguity_bonus(
    by_face: dict[str, list[IdentityHypothesis]],
    faces: list[str],
) -> float:
    bonus = 0.0
    for face_id in faces:
        rows = by_face.get(face_id, [])
        if not rows:
            continue
        top = rows[0]
        second = rows[1].score if len(rows) > 1 else 0.0
        bonus += max(0.0, 0.22 - max(0.0, top.score - second)) * 1.8
        bonus += min(0.24, sum(1 for hyp in rows[:5] if hyp.score >= top.score - 0.12) * 0.04)
    return bonus


def _relation_bonus(
    cluster_lookup: dict[str, NameCluster],
    name_ids: list[str],
    by_face: dict[str, list[IdentityHypothesis]],
    faces: list[str],
) -> float:
    terms = {
        term.lower()
        for name_id in name_ids
        for term in (cluster_lookup.get(name_id).relation_terms if cluster_lookup.get(name_id) else [])
    }
    for face_id in faces:
        for hyp in by_face.get(face_id, [])[:4]:
            terms.update(term.lower() for term in hyp.evidence_packet.relation_clues)
    family_terms = {"mom", "mother", "dad", "father", "partner", "wife", "husband", "sister", "brother"}
    social_terms = {"friend", "friends", "colleague", "coworker", "classmate", "neighbor"}
    return (0.28 if terms & family_terms else 0.0) + (0.12 if terms & social_terms else 0.0)


def _is_ambiguous_face(rows: list[IdentityHypothesis]) -> bool:
    if len(rows) < 2:
        return False
    margin = max(0.0, rows[0].score - rows[1].score)
    return margin < 0.16 or sum(1 for hyp in rows[:5] if hyp.score >= rows[0].score - 0.12) >= 3


def _trim_group_faces(
    group: _JointGroup,
    by_face: dict[str, list[IdentityHypothesis]],
    limit: int,
) -> _JointGroup:
    face_ids = sorted(
        group.face_ids,
        key=lambda fid: (-_face_priority(by_face.get(fid, [])), fid),
    )[:limit]
    return _JointGroup(
        group_id=group.group_id,
        face_ids=face_ids,
        name_cluster_ids=group.name_cluster_ids,
        evidence_photo_ids=group.evidence_photo_ids,
        reason=group.reason,
        priority=group.priority,
        target_face_ids=[fid for fid in (group.target_face_ids or []) if fid in set(face_ids)] or None,
    )


def _trim_group_names(
    group: _JointGroup,
    by_face: dict[str, list[IdentityHypothesis]],
    limit: int,
) -> _JointGroup:
    counts: Counter[str] = Counter()
    scores: dict[str, float] = defaultdict(float)
    for face_id in group.face_ids:
        for hyp in by_face.get(face_id, []):
            if hyp.name_cluster_id in group.name_cluster_ids:
                counts[hyp.name_cluster_id] += 1
                scores[hyp.name_cluster_id] = max(scores[hyp.name_cluster_id], hyp.score)
    name_ids = sorted(group.name_cluster_ids, key=lambda nid: (-counts[nid], -scores[nid], nid))[:limit]
    return _JointGroup(
        group_id=group.group_id,
        face_ids=group.face_ids,
        name_cluster_ids=name_ids,
        evidence_photo_ids=group.evidence_photo_ids,
        reason=group.reason,
        priority=group.priority,
        target_face_ids=group.target_face_ids,
    )


def _face_priority(rows: list[IdentityHypothesis]) -> float:
    if not rows:
        return 0.0
    top = rows[0]
    margin = top.score - (rows[1].score if len(rows) > 1 else 0.0)
    return top.score + max(0.0, 0.25 - margin)


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


def _overlaps_existing_group(group: _JointGroup, selected: list[_JointGroup]) -> bool:
    faces = set(group.face_ids)
    names = set(group.name_cluster_ids)
    for other in selected:
        other_faces = set(other.face_ids)
        other_names = set(other.name_cluster_ids)
        face_overlap = len(faces & other_faces) / max(1, min(len(faces), len(other_faces)))
        name_overlap = len(names & other_names) / max(1, min(len(names), len(other_names)))
        if face_overlap >= 0.7 and name_overlap >= 0.45:
            return True
    return False


def _person_entity_count(photo: dict[str, Any]) -> int:
    return sum(
        1
        for ent in (photo.get("text_entities") or [])
        if str(ent.get("entity_type") or "").lower() == "person"
    )


def _recompute_margins(by_face: dict[str, list[IdentityHypothesis]]) -> None:
    for rows in by_face.values():
        rows.sort(key=lambda h: (-h.score, h.name_cluster_id))
        for idx, hyp in enumerate(rows):
            next_score = rows[idx + 1].score if idx + 1 < len(rows) else 0.0
            hyp.margin = max(0.0, hyp.score - next_score)


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


def _has_unambiguous_direct_anchor(inventory: EvidenceInventory, hyp: IdentityHypothesis) -> bool:
    for quality in _same_photo_quality(inventory, hyp):
        if not quality.get("ambiguous_multi_face_text"):
            return True
    return False


def _is_overbroad_recall_candidate(hyp: IdentityHypothesis) -> bool:
    try:
        event_count = float(hyp.signal_breakdown.get("event_bridge_count") or 0.0)
        same_weight = float(hyp.signal_breakdown.get("same_photo_weight") or 0.0)
    except (TypeError, ValueError):
        return False
    return event_count >= 60.0 and same_weight < 0.45


def _shared_anchor_count(
    left_rows: list[IdentityHypothesis],
    right_rows: list[IdentityHypothesis],
) -> int:
    left_by_name = {hyp.name_cluster_id: hyp for hyp in left_rows}
    count = 0
    for right in right_rows:
        left = left_by_name.get(right.name_cluster_id)
        if not left:
            continue
        if set(left.evidence_packet.same_photo_ids) & set(right.evidence_packet.same_photo_ids):
            count += 1
    return count


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


def _bounded_float(value: Any, fallback: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return fallback

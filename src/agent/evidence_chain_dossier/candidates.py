"""Generate face-name identity hypotheses from evidence chains."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from math import log

from .event_bridge import (
    photo_event_features,
    score_event_bridge_from_features,
    social_channels_from_terms,
)
from .schemas import EvidenceInventory, EvidencePacket, IdentityHypothesis, NameCluster
from .text import dedupe, is_full_name, norm_key


def build_identity_hypotheses(
    inventory: EvidenceInventory,
    clusters: list[NameCluster],
    *,
    max_per_face: int = 12,
    config=None,
) -> tuple[list[IdentityHypothesis], dict[str, list[dict]]]:
    """Build scored face-name candidates for every non-owner face."""
    use_recall = bool(getattr(config, "use_evidence_recall_broadener", False))
    effective_max_per_face = max_per_face
    if use_recall:
        effective_max_per_face = max(
            max_per_face,
            int(getattr(config, "recall_max_candidates_per_face", max_per_face) or max_per_face),
        )
    cluster_units = {
        c.cluster_id: [inventory.text_units_by_id[uid] for uid in c.text_unit_ids if uid in inventory.text_units_by_id]
        for c in clusters
    }
    hypotheses: list[IdentityHypothesis] = []
    event_cache: dict[tuple[str, str], dict] = {}
    event_feature_cache: dict[str, dict] = {}

    for face_id, face_units in inventory.face_units_by_face.items():
        if face_id == inventory.owner_face_id:
            continue
        face_hyps: list[IdentityHypothesis] = []
        for cluster in clusters:
            units = cluster_units.get(cluster.cluster_id) or []
            if not units:
                continue
            hyp = _score_pair(
                inventory,
                face_id,
                face_units,
                cluster,
                units,
                config=config,
                event_cache=event_cache,
                event_feature_cache=event_feature_cache,
            )
            if hyp.score > 0:
                face_hyps.append(hyp)
        face_hyps.sort(key=lambda h: (-h.score, h.name_cluster_id))
        for idx, hyp in enumerate(face_hyps):
            next_score = face_hyps[idx + 1].score if idx + 1 < len(face_hyps) else 0.0
            hyp.margin = max(0.0, hyp.score - next_score)
        hypotheses.extend(face_hyps[:effective_max_per_face])

    diagnostics = _candidate_diagnostics(hypotheses)
    return hypotheses, diagnostics


def _score_pair(
    inventory: EvidenceInventory,
    face_id: str,
    face_units,
    cluster: NameCluster,
    text_units,
    *,
    config=None,
    event_cache: dict[tuple[str, str], dict] | None = None,
    event_feature_cache: dict[str, dict] | None = None,
) -> IdentityHypothesis:
    use_recall = bool(getattr(config, "use_evidence_recall_broadener", False))
    min_event_score = float(getattr(config, "recall_min_event_bridge_score", 0.30) or 0.30)
    event_bonus_weight = float(getattr(config, "recall_event_bonus_weight", 0.18) or 0.18)
    text_only_bonus_weight = float(getattr(config, "recall_text_only_bonus_weight", 0.08) or 0.08)
    role_bonus_weight = float(getattr(config, "recall_role_bonus_weight", 0.08) or 0.08)

    face_photo_ids = {u.photo_id for u in face_units}
    text_photo_ids = {u.photo_id for u in text_units}
    same_photo_ids = sorted(face_photo_ids & text_photo_ids)
    same_photo_weight = _same_photo_weight(inventory, same_photo_ids)
    bridge_pairs = []
    strict_bridge = 0
    city_bridge = 0
    near_bridge = 0
    event_bridge_count = 0
    event_bridge_total = 0.0
    event_bridge_max = 0.0
    text_only_event_bridge_max = 0.0
    role_channel_bridge_max = 0.0
    cluster_channels = social_channels_from_terms(
        list(cluster.relation_terms) + list(cluster.activity_terms) + list(cluster.venue_terms)
    )
    for f_unit in face_units:
        for t_unit in text_units:
            if f_unit.photo_id == t_unit.photo_id:
                continue
            relation = _bridge_relation(f_unit, t_unit)
            event_match = None
            event_score = 0.0
            if use_recall:
                cache_key = (t_unit.photo_id, f_unit.photo_id)
                if event_cache is not None and cache_key in event_cache:
                    event_match = event_cache[cache_key]
                else:
                    text_photo = inventory.photo_lookup.get(t_unit.photo_id) or {}
                    face_photo = inventory.photo_lookup.get(f_unit.photo_id) or {}
                    event_match = score_event_bridge_from_features(
                        _photo_features(t_unit.photo_id, text_photo, event_feature_cache),
                        _photo_features(f_unit.photo_id, face_photo, event_feature_cache),
                        text_photo_has_faces=bool(text_photo.get("visible_face_ids")),
                        face_photo_face_count=len(face_photo.get("visible_face_ids") or []),
                        text_photo_person_entities=_person_entity_count(text_photo),
                    )
                    if event_cache is not None:
                        event_cache[cache_key] = event_match
                event_score = float(event_match.get("score") or 0.0)
                if relation == "location_month":
                    event_score = min(1.0, event_score + 0.12)
                elif relation == "near_date_city":
                    event_score = min(1.0, event_score + 0.08)
                elif relation == "city_month" and event_match.get("shared_channels"):
                    event_score = min(1.0, event_score + 0.04)
            if not relation and event_score < min_event_score:
                continue
            pair = {
                "face_photo_id": f_unit.photo_id,
                "text_photo_id": t_unit.photo_id,
                "bridge": relation or "event_semantic",
            }
            if event_match and event_score >= min_event_score and bool(event_match.get("strong_signal")):
                shared_channels = [str(c) for c in (event_match.get("shared_channels") or [])]
                pair.update(
                    {
                        "event_score": round(event_score, 4),
                        "shared_keywords": [str(k) for k in (event_match.get("shared_keywords") or [])[:8]],
                        "event_channels": shared_channels[:8],
                        "text_photo_has_faces": bool(event_match.get("text_photo_has_faces")),
                        "strong_signal": bool(event_match.get("strong_signal")),
                    }
                )
                event_bridge_count += 1
                event_bridge_total += event_score
                event_bridge_max = max(event_bridge_max, event_score)
                if not event_match.get("text_photo_has_faces"):
                    text_only_event_bridge_max = max(text_only_event_bridge_max, event_score)
                face_channels = social_channels_from_terms(
                    list(f_unit.relation_terms) + list(f_unit.activity_terms) + list(f_unit.venue_terms)
                )
                bridge_channels = set(shared_channels) | face_channels
                if cluster_channels and (cluster_channels & bridge_channels):
                    role_channel_bridge_max = max(role_channel_bridge_max, min(1.0, event_score + 0.12))
            if relation == "location_month":
                strict_bridge += 1
            elif relation == "city_month":
                city_bridge += 1
            elif relation == "near_date_city":
                near_bridge += 1
            if len(bridge_pairs) < 10:
                bridge_pairs.append(pair)

    face_activity = Counter(term.lower() for u in face_units for term in u.activity_terms)
    face_venues = Counter(term.lower() for u in face_units for term in u.venue_terms)
    cluster_activity = Counter(term.lower() for term in cluster.activity_terms)
    cluster_venues = Counter(term.lower() for term in cluster.venue_terms)
    activity_overlap = sum(min(face_activity[k], cluster_activity[k]) for k in face_activity.keys() & cluster_activity.keys())
    venue_overlap = sum(min(face_venues[k], cluster_venues[k]) for k in face_venues.keys() & cluster_venues.keys())

    relation_terms = dedupe(list(cluster.relation_terms) + [term for u in face_units for term in u.relation_terms])
    owner_cooccur = sum(1 for u in face_units if u.owner_present)
    n_face_photos = max(1, len(face_units))
    owner_cooccur_rate = owner_cooccur / n_face_photos

    name_quality = 1.0 if is_full_name(cluster.primary_surface) else 0.45
    same_strength = min(1.0, same_photo_weight)
    strict_strength = _count_strength(strict_bridge, scale=5)
    city_strength = _count_strength(city_bridge, scale=12)
    near_strength = _count_strength(near_bridge, scale=8)
    activity_strength = min(1.0, activity_overlap / 3.0)
    venue_strength = min(1.0, venue_overlap / 3.0)
    relation_strength = min(1.0, len(relation_terms) / 2.0)
    coowner_strength = min(1.0, owner_cooccur_rate * 1.4)
    event_bridge_strength = 0.0
    text_only_event_bridge_strength = 0.0
    role_channel_bridge_strength = 0.0
    if use_recall:
        event_bridge_strength = min(
            1.0,
            event_bridge_max + min(0.30, 0.04 * max(0, event_bridge_count - 1)),
        )
        if event_bridge_total >= 1.2:
            event_bridge_strength = max(event_bridge_strength, min(1.0, event_bridge_total / 2.4))
        text_only_event_bridge_strength = min(1.0, text_only_event_bridge_max)
        role_channel_bridge_strength = min(1.0, role_channel_bridge_max)

    overbroad_penalty = 0.0
    if city_bridge > 30 and strict_bridge == 0 and not same_photo_ids:
        overbroad_penalty += 0.10
    if not is_full_name(cluster.primary_surface) and not same_photo_ids:
        overbroad_penalty += 0.06

    score = (
        0.50 * same_strength
        + 0.25 * strict_strength
        + 0.08 * city_strength
        + 0.08 * near_strength
        + 0.06 * activity_strength
        + 0.04 * venue_strength
        + 0.08 * relation_strength
        + 0.06 * coowner_strength
        + 0.06 * name_quality
        + event_bonus_weight * event_bridge_strength
        + text_only_bonus_weight * text_only_event_bridge_strength
        + role_bonus_weight * role_channel_bridge_strength
        - overbroad_penalty
    )
    score = max(0.0, min(1.0, score))

    text_evidence = sorted(text_photo_ids, key=lambda pid: (0 if pid in same_photo_ids else 1, pid))[:10]
    face_evidence = sorted(face_photo_ids, key=lambda pid: (0 if pid in same_photo_ids else 1, pid))[:10]
    packet = EvidencePacket(
        text_photo_ids=text_evidence,
        face_photo_ids=face_evidence,
        same_photo_ids=same_photo_ids[:10],
        bridge_photo_pairs=bridge_pairs,
        relation_clues=relation_terms[:10],
        activity_clues=dedupe(list(cluster.activity_terms) + [term for u in face_units for term in u.activity_terms])[:10],
        venue_clues=dedupe(list(cluster.venue_terms) + [term for u in face_units for term in u.venue_terms])[:10],
        coface_clues=_coface_clues(inventory, face_id, face_units),
        negative_clues=[],
        narrative_summary=_narrative(
            face_id,
            cluster,
            same_photo_ids,
            strict_bridge,
            city_bridge,
            near_bridge,
            event_bridge_count,
        ),
    )

    signal_breakdown = {
        "same_photo": same_strength,
        "strict_bridge": strict_strength,
        "city_bridge": city_strength,
        "near_date_city": near_strength,
        "activity_overlap": activity_strength,
        "venue_overlap": venue_strength,
        "relation_clue": relation_strength,
        "owner_cooccur": coowner_strength,
        "name_quality": name_quality,
        "overbroad_penalty": overbroad_penalty,
        "same_photo_count": float(len(same_photo_ids)),
        "same_photo_weight": float(same_photo_weight),
        "strict_bridge_count": float(strict_bridge),
        "city_bridge_count": float(city_bridge),
        "near_bridge_count": float(near_bridge),
    }
    if use_recall:
        signal_breakdown.update(
            {
                "event_bridge": event_bridge_strength,
                "text_only_event_bridge": text_only_event_bridge_strength,
                "role_channel_bridge": role_channel_bridge_strength,
                "event_bridge_count": float(event_bridge_count),
                "event_bridge_total": float(event_bridge_total),
            }
        )

    status = "candidate"
    if use_recall and event_bridge_strength >= min_event_score:
        status = "recall_broadened_candidate"

    return IdentityHypothesis(
        face_id=face_id,
        name_cluster_id=cluster.cluster_id,
        observed_surface=cluster.primary_surface,
        canonical_name_candidate=cluster.primary_surface,
        score=score,
        margin=0.0,
        signal_breakdown=signal_breakdown,
        evidence_packet=packet,
        status=status,
    )


def _bridge_relation(face_unit, text_unit) -> str:
    if face_unit.year_month and face_unit.year_month == text_unit.year_month:
        f_loc = norm_key(face_unit.gps_location)
        t_loc = norm_key(text_unit.gps_location)
        if f_loc and t_loc and (f_loc == t_loc or f_loc in t_loc or t_loc in f_loc):
            return "location_month"
        if face_unit.gps_city and text_unit.gps_city and norm_key(face_unit.gps_city) == norm_key(text_unit.gps_city):
            return "city_month"
    if face_unit.gps_city and text_unit.gps_city and norm_key(face_unit.gps_city) == norm_key(text_unit.gps_city):
        if _days_apart(face_unit.timestamp, text_unit.timestamp) <= 14:
            return "near_date_city"
    return ""


def _same_photo_weight(inventory: EvidenceInventory, photo_ids: list[str]) -> float:
    weight = 0.0
    for photo_id in photo_ids:
        photo = inventory.photo_lookup.get(photo_id) or {}
        n_faces = max(1, len(photo.get("visible_face_ids") or []))
        n_names = 0
        for ent in photo.get("text_entities") or []:
            if str(ent.get("entity_type") or "").lower() == "person":
                n_names += 1
        n_names = max(1, n_names)
        weight += 1.0 / (n_faces * n_names)
    return weight


def _photo_features(
    photo_id: str,
    photo: dict,
    event_feature_cache: dict[str, dict] | None,
) -> dict:
    if event_feature_cache is None:
        return photo_event_features(photo)
    if photo_id not in event_feature_cache:
        event_feature_cache[photo_id] = photo_event_features(photo)
    return event_feature_cache[photo_id]


def _person_entity_count(photo: dict) -> int:
    return sum(
        1
        for ent in (photo.get("text_entities") or [])
        if str(ent.get("entity_type") or "").lower() == "person"
    )


def _days_apart(a: str, b: str) -> int:
    try:
        da = datetime.fromisoformat(a.replace("Z", "+00:00"))
        db = datetime.fromisoformat(b.replace("Z", "+00:00"))
        return abs((da - db).days)
    except Exception:
        return 99999


def _count_strength(count: int, scale: int) -> float:
    if count <= 0:
        return 0.0
    return min(1.0, log(count + 1) / log(scale + 1))


def _coface_clues(inventory: EvidenceInventory, face_id: str, face_units) -> list[str]:
    counts = Counter(cf for u in face_units for cf in u.co_faces if cf != face_id)
    return [f"{fid}:{n}" for fid, n in counts.most_common(8)]


def _narrative(
    face_id: str,
    cluster: NameCluster,
    same_photo_ids: list[str],
    strict_bridge: int,
    city_bridge: int,
    near_bridge: int,
    event_bridge: int = 0,
) -> str:
    pieces = [f"{face_id} is compared with the name cluster '{cluster.primary_surface}'."]
    if same_photo_ids:
        pieces.append(f"Same-photo name/face anchors appear in {', '.join(same_photo_ids[:4])}.")
    if strict_bridge:
        pieces.append(f"{strict_bridge} same-location same-month text-to-face bridges support the binding.")
    if city_bridge:
        pieces.append(f"{city_bridge} weaker same-city same-month bridges are present.")
    if near_bridge:
        pieces.append(f"{near_bridge} near-date same-city bridges are present.")
    if event_bridge:
        pieces.append(f"{event_bridge} event-conditioned text-to-face bridges are present.")
    if cluster.relation_terms:
        pieces.append(f"Relation terms near the name include {', '.join(cluster.relation_terms[:4])}.")
    return " ".join(pieces)


def _candidate_diagnostics(hypotheses: list[IdentityHypothesis]) -> dict[str, list[dict]]:
    by_face: dict[str, list[IdentityHypothesis]] = defaultdict(list)
    for hyp in hypotheses:
        by_face[hyp.face_id].append(hyp)
    return {
        face_id: [h.to_dict() for h in sorted(rows, key=lambda h: -h.score)[:8]]
        for face_id, rows in sorted(by_face.items())
    }

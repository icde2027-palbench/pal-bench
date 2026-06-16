"""Compact context builders for ECD LLM calls."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .schemas import EvidenceInventory, IdentityHypothesis, NameCluster, ResolvedIdentity
from .text import combined_text, dedupe


def truncate(text: str, limit: int = 360) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def photo_brief(inventory: EvidenceInventory, photo_id: str, *, chars: int = 360) -> dict[str, Any]:
    photo = inventory.photo_lookup.get(photo_id) or {}
    metadata = photo.get("metadata") or {}
    visible_text = [str(t) for t in (photo.get("visible_text") or []) if str(t).strip()]
    text_limit = 5 if chars <= 220 else 8 if chars <= 300 else 10
    entity_limit = 6 if chars <= 220 else 8 if chars <= 300 else 12
    entities = [
        {
            "surface": str(e.get("surface") or ""),
            "type": str(e.get("entity_type") or ""),
            "source": str(e.get("source") or ""),
        }
        for e in (photo.get("text_entities") or [])[:entity_limit]
    ]
    return {
        "photo_id": photo_id,
        "year_month": str(photo.get("year_month") or ""),
        "timestamp": str(photo.get("timestamp") or ""),
        "gps_city": str(metadata.get("gps_city") or ""),
        "gps_location": truncate(str(metadata.get("gps_location") or ""), 120),
        "faces": [str(f) for f in (photo.get("visible_face_ids") or [])],
        "caption": truncate(str(photo.get("caption") or ""), chars),
        "visible_text": [truncate(t, 110 if chars <= 220 else 140) for t in visible_text[:text_limit]],
        "entities": entities,
    }


def face_context(
    inventory: EvidenceInventory,
    face_id: str,
    *,
    max_photos: int = 8,
    chars: int = 300,
) -> dict[str, Any]:
    units = inventory.face_units_by_face.get(face_id, [])
    owner_count = sum(1 for u in units if u.owner_present)
    relation_terms = Counter(term.lower() for u in units for term in u.relation_terms)
    activity_terms = Counter(term.lower() for u in units for term in u.activity_terms)
    venue_terms = Counter(term.lower() for u in units for term in u.venue_terms)
    co_faces = Counter(cf for u in units for cf in u.co_faces)
    selected = _representative_face_photos(units, max_photos=max_photos)
    return {
        "face_id": face_id,
        "n_appearances": len(units),
        "owner_coappearances": owner_count,
        "top_co_faces": [f"{fid}:{n}" for fid, n in co_faces.most_common(8)],
        "relation_terms": [term for term, _ in relation_terms.most_common(8)],
        "activity_terms": [term for term, _ in activity_terms.most_common(10)],
        "venue_terms": [term for term, _ in venue_terms.most_common(10)],
        "photos": [photo_brief(inventory, pid, chars=chars) for pid in selected],
    }


def candidate_context(
    inventory: EvidenceInventory,
    cluster_lookup: dict[str, NameCluster],
    hyp: IdentityHypothesis,
    *,
    index: int,
    chars: int = 300,
) -> dict[str, Any]:
    cluster = cluster_lookup.get(hyp.name_cluster_id)
    packet = hyp.evidence_packet
    photo_limit = _candidate_photo_limit(chars)
    bridge_limit = _bridge_pair_limit(chars)
    raw_bridge_limit = 4 if chars <= 220 else 8
    text_photos = dedupe(packet.same_photo_ids + packet.text_photo_ids)[:photo_limit]
    face_photos = dedupe(packet.same_photo_ids + packet.face_photo_ids)[:photo_limit]
    bridge_examples = _bridge_evidence_examples(
        inventory,
        packet.bridge_photo_pairs,
        chars=chars,
        max_pairs=bridge_limit,
    )
    return {
        "candidate_index": index,
        "name_cluster_id": hyp.name_cluster_id,
        "observed_surface": hyp.observed_surface,
        "canonical_name_candidate": hyp.canonical_name_candidate,
        "score": round(hyp.score, 4),
        "margin": round(hyp.margin, 4),
        "surfaces": cluster.surfaces if cluster else [hyp.observed_surface],
        "first_name": cluster.first_name if cluster else "",
        "last_name_candidates": cluster.last_name_candidates if cluster else {},
        "quality_flags": cluster.quality_flags if cluster else [],
        "signals": hyp.signal_breakdown,
        "same_photo_ids": packet.same_photo_ids[:6],
        "bridge_photo_pairs": _compact_bridge_pairs(packet.bridge_photo_pairs, limit=raw_bridge_limit),
        "event_bridge_pairs": _event_bridge_pairs(packet.bridge_photo_pairs, limit=raw_bridge_limit),
        "relation_clues": packet.relation_clues[:10],
        "activity_clues": packet.activity_clues[:10],
        "venue_clues": packet.venue_clues[:10],
        "coface_clues": packet.coface_clues[:8],
        "narrative_summary": packet.narrative_summary,
        "name_text_photos": [photo_brief(inventory, pid, chars=chars) for pid in text_photos],
        "face_anchor_photos": [photo_brief(inventory, pid, chars=chars) for pid in face_photos],
        "bridge_evidence_examples": bridge_examples,
    }


def resolved_context(
    inventory: EvidenceInventory,
    resolved: ResolvedIdentity,
    *,
    chars: int = 300,
) -> dict[str, Any]:
    packet = resolved.evidence_packet
    evidence = dedupe(packet.same_photo_ids + packet.text_photo_ids + packet.face_photo_ids)[:8]
    return {
        "face_id": resolved.face_id,
        "canonical_name": resolved.canonical_name,
        "observed_surface": resolved.observed_surface,
        "confidence": round(resolved.confidence, 3),
        "score": round(resolved.score, 4),
        "relation_to_owner": resolved.relation_to_owner,
        "relation_category": resolved.relation_category,
        "name_source": resolved.name_source,
        "relation_clues": packet.relation_clues[:8],
        "activity_clues": packet.activity_clues[:8],
        "venue_clues": packet.venue_clues[:8],
        "coface_clues": packet.coface_clues[:8],
        "reasoning_path": truncate(resolved.reasoning_path, 600),
        "evidence_photos": [photo_brief(inventory, pid, chars=chars) for pid in evidence],
    }


def owner_context(inventory: EvidenceInventory, *, max_units: int = 28, chars: int = 320) -> dict[str, Any]:
    units = sorted(
        inventory.text_units,
        key=lambda u: (
            -u.owner_reference_score,
            0 if _photo_has_owner(inventory, u.photo_id) else 1,
            u.photo_id,
        ),
    )[:max_units]
    owner_face_units = inventory.face_units_by_face.get(inventory.owner_face_id, [])[:max_units]
    photo_ids = dedupe([u.photo_id for u in units] + [u.photo_id for u in owner_face_units])[:max_units]
    return {
        "owner_face_id": inventory.owner_face_id,
        "owner_name_candidate": inventory.owner_name,
        "owner_last_name": inventory.owner_last_name,
        "owner_name_candidates": inventory.owner_name_candidates[:10],
        "album_time": (inventory.album.get("album_summary") or {}),
        "evidence_photos": [photo_brief(inventory, pid, chars=chars) for pid in photo_ids],
    }


def _representative_face_photos(units, *, max_photos: int) -> list[str]:
    if not units:
        return []
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


def _photo_has_owner(inventory: EvidenceInventory, photo_id: str) -> bool:
    photo = inventory.photo_lookup.get(photo_id) or {}
    return inventory.owner_face_id in [str(f) for f in (photo.get("visible_face_ids") or [])]


def _bridge_evidence_examples(
    inventory: EvidenceInventory,
    bridge_pairs: list[dict[str, Any]],
    *,
    chars: int,
    max_pairs: int = 5,
) -> list[dict[str, Any]]:
    priority = {"location_month": 0, "near_date_city": 1, "city_month": 2}
    ranked = sorted(
        bridge_pairs,
        key=lambda pair: (
            priority.get(str(pair.get("bridge") or ""), 9),
            -_float(pair.get("event_score")),
            0 if _photo_has_owner(inventory, str(pair.get("face_photo_id") or "")) else 1,
            str(pair.get("face_photo_id") or ""),
            str(pair.get("text_photo_id") or ""),
        ),
    )
    examples = []
    for pair in ranked[:max_pairs]:
        face_photo_id = str(pair.get("face_photo_id") or "")
        text_photo_id = str(pair.get("text_photo_id") or "")
        if not face_photo_id or not text_photo_id:
            continue
        examples.append(
            {
                "bridge": str(pair.get("bridge") or ""),
                "event_score": _float(pair.get("event_score")),
                "shared_keywords": [str(k) for k in (pair.get("shared_keywords") or [])[:8]],
                "event_channels": [str(k) for k in (pair.get("event_channels") or [])[:8]],
                "text_photo_has_faces": bool(pair.get("text_photo_has_faces")),
                "strong_signal": bool(pair.get("strong_signal")),
                "face_photo_id": face_photo_id,
                "text_photo_id": text_photo_id,
                "face_photo": photo_brief(inventory, face_photo_id, chars=chars),
                "text_photo": photo_brief(inventory, text_photo_id, chars=chars),
            }
        )
    return examples


def _candidate_photo_limit(chars: int) -> int:
    if chars <= 220:
        return 2
    if chars <= 300:
        return 4
    return 6


def _bridge_pair_limit(chars: int) -> int:
    if chars <= 220:
        return 1
    if chars <= 300:
        return 3
    return 5


def _compact_bridge_pairs(bridge_pairs: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return [
        {
            "face_photo_id": str(pair.get("face_photo_id") or ""),
            "text_photo_id": str(pair.get("text_photo_id") or ""),
            "bridge": str(pair.get("bridge") or ""),
            "event_score": _float(pair.get("event_score")),
            "shared_keywords": [str(k) for k in (pair.get("shared_keywords") or [])[:5]],
            "event_channels": [str(k) for k in (pair.get("event_channels") or [])[:5]],
            "strong_signal": bool(pair.get("strong_signal")),
        }
        for pair in bridge_pairs[:limit]
    ]


def _event_bridge_pairs(bridge_pairs: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    pairs = [
        {
            "face_photo_id": str(pair.get("face_photo_id") or ""),
            "text_photo_id": str(pair.get("text_photo_id") or ""),
            "bridge": str(pair.get("bridge") or ""),
            "event_score": _float(pair.get("event_score")),
            "shared_keywords": [str(k) for k in (pair.get("shared_keywords") or [])[:8]],
            "event_channels": [str(k) for k in (pair.get("event_channels") or [])[:8]],
            "text_photo_has_faces": bool(pair.get("text_photo_has_faces")),
            "strong_signal": bool(pair.get("strong_signal")),
        }
        for pair in bridge_pairs
        if _float(pair.get("event_score")) > 0.0
    ]
    pairs.sort(key=lambda pair: (-float(pair["event_score"]), pair["face_photo_id"], pair["text_photo_id"]))
    return pairs[:limit]


def _float(value: Any) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return 0.0


def compact_profile_snapshot(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "owner": profile.get("owner") or {},
        "persons": [
            {
                "face_id": row.get("face_id"),
                "canonical_name": row.get("canonical_name"),
                "relation_to_owner": row.get("relation_to_owner"),
                "relation_category": row.get("relation_category"),
                "confidence": row.get("confidence"),
                "evidence_photo_ids": row.get("evidence_photo_ids"),
                "reasoning_path": truncate(str(row.get("reasoning_path") or ""), 500),
            }
            for row in (profile.get("persons") or [])
        ],
    }


def evidence_text_from_photo(photo: dict[str, Any]) -> str:
    return combined_text([
        photo.get("caption") or "",
        *(photo.get("visible_text") or []),
        *(str(e.get("surface") or "") for e in (photo.get("text_entities") or [])),
    ])

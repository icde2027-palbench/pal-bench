"""Build the public evidence inventory used by ECD."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .schemas import EvidenceInventory, FaceEvidenceUnit, TextEvidenceUnit
from .text import (
    combined_text,
    dedupe,
    extract_activity_terms,
    extract_relation_terms,
    extract_venue_terms,
    is_full_name,
    last_name,
    looks_like_person_name,
    normalize_name,
    owner_reference_score,
    strip_relation_prefix,
)


TEXT_ARTIFACT_TERMS = {
    "screen",
    "phone",
    "screenshot",
    "card",
    "badge",
    "sign",
    "directory",
    "invitation",
    "message",
    "thread",
    "profile",
    "account",
    "document",
    "certificate",
    "reservation",
    "confirmation",
    "label",
    "envelope",
    "letter",
}


def build_inventory(album: dict[str, Any]) -> EvidenceInventory:
    """Build public-only text and face evidence units from an agent album."""
    photos = list(album.get("photos") or [])
    photo_lookup = {str(p.get("photo_id") or ""): p for p in photos}
    face_photo_ids = _build_face_photo_ids(photos)
    face_counts = {fid: len(pids) for fid, pids in face_photo_ids.items()}
    owner_face_id = _estimate_owner_face(album, face_counts)

    text_units = _build_text_units(photos)
    text_units_by_id = {unit.unit_id: unit for unit in text_units}
    owner_name, owner_candidates = _estimate_owner_name(text_units, owner_face_id)
    owner_last_name = last_name(owner_name)

    face_units = _build_face_units(photos, owner_face_id)
    face_units_by_face: dict[str, list[FaceEvidenceUnit]] = defaultdict(list)
    for unit in face_units:
        face_units_by_face[unit.face_id].append(unit)

    return EvidenceInventory(
        album=album,
        photos=photos,
        photo_lookup=photo_lookup,
        text_units=text_units,
        text_units_by_id=text_units_by_id,
        face_units=face_units,
        face_units_by_face=dict(face_units_by_face),
        face_photo_ids={k: list(v) for k, v in face_photo_ids.items()},
        owner_face_id=owner_face_id,
        owner_name=owner_name,
        owner_name_candidates=owner_candidates,
        owner_last_name=owner_last_name,
        face_appearance_counts=face_counts,
    )


def _build_face_photo_ids(photos: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for photo in photos:
        photo_id = str(photo.get("photo_id") or "")
        for face_id in photo.get("visible_face_ids") or []:
            if photo_id:
                out[str(face_id)].append(photo_id)
    return dict(out)


def _estimate_owner_face(album: dict[str, Any], face_counts: dict[str, int]) -> str:
    registry = album.get("faces") or []
    candidates = []
    for row in registry:
        fid = str(row.get("face_id") or "")
        if not fid:
            continue
        n = int(row.get("n_appearances") or face_counts.get(fid, 0) or 0)
        candidates.append((n, fid))
    if not candidates:
        candidates = [(n, fid) for fid, n in face_counts.items()]
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def _build_text_units(photos: list[dict[str, Any]]) -> list[TextEvidenceUnit]:
    units: list[TextEvidenceUnit] = []
    for photo in photos:
        photo_id = str(photo.get("photo_id") or "")
        visible_text = [str(t) for t in (photo.get("visible_text") or []) if str(t).strip()]
        caption = str(photo.get("caption") or "")
        entities = list(photo.get("text_entities") or [])
        metadata = photo.get("metadata") or {}
        text_blob = combined_text([caption, *visible_text, *(e.get("surface", "") for e in entities)])

        person_surfaces: list[str] = []
        relation_terms = extract_relation_terms(text_blob)
        organizations: list[str] = []
        for ent in entities:
            surface = str(ent.get("surface") or "")
            etype = str(ent.get("entity_type") or "").lower()
            if etype == "person":
                name, rel_from_prefix = strip_relation_prefix(surface)
                relation_terms.extend(rel_from_prefix)
                if looks_like_person_name(name) and not _skip_visible_text_name(
                    name=name,
                    entity=ent,
                    caption=caption,
                    visible_text=visible_text,
                    relation_terms=relation_terms,
                    has_face=bool(photo.get("visible_face_ids") or []),
                ):
                    person_surfaces.append(normalize_name(name))
            elif etype == "organization":
                organizations.append(surface)

        person_surfaces = dedupe(person_surfaces)
        if not (person_surfaces or visible_text or caption):
            continue
        units.append(
            TextEvidenceUnit(
                unit_id=f"text::{photo_id}",
                photo_id=photo_id,
                year_month=str(photo.get("year_month") or ""),
                timestamp=str(photo.get("timestamp") or ""),
                gps_city=str(metadata.get("gps_city") or ""),
                gps_location=str(metadata.get("gps_location") or ""),
                caption=caption,
                visible_text=visible_text,
                person_surfaces=person_surfaces,
                relation_terms=dedupe(relation_terms),
                activity_terms=extract_activity_terms(text_blob),
                venue_terms=extract_venue_terms(
                    combined_text([text_blob, str(metadata.get("gps_location") or ""), str(metadata.get("gps_city") or "")])
                ),
                organization_terms=dedupe(organizations),
                owner_reference_score=owner_reference_score(text_blob),
                has_face=bool(photo.get("visible_face_ids") or []),
            )
        )
    return units


def _skip_visible_text_name(
    *,
    name: str,
    entity: dict[str, Any],
    caption: str,
    visible_text: list[str],
    relation_terms: list[str],
    has_face: bool,
) -> bool:
    """Filter likely OCR/person-entity hallucinations from casual face photos."""
    if str(entity.get("source") or "").lower() != "visible_text":
        return False
    lower_caption = caption.lower()
    artifact_context = any(term in lower_caption for term in TEXT_ARTIFACT_TERMS)
    if relation_terms:
        artifact_context = True
    if not has_face:
        return False
    if artifact_context:
        return False
    if len(visible_text) <= 4:
        return True
    return _ocr_noise_tokens(name, visible_text) >= 1


def _ocr_noise_tokens(name: str, visible_text: list[str]) -> int:
    noise_tokens = 0
    for text in visible_text:
        raw = str(text or "").strip()
        if raw == name:
            continue
        alpha = "".join(ch for ch in raw if ch.isalpha())
        if len(alpha) >= 5 and (raw.isupper() or sum(1 for ch in alpha if ch.lower() not in "aeiou") / len(alpha) > 0.72):
            noise_tokens += 1
    return noise_tokens


def _estimate_owner_name(
    text_units: list[TextEvidenceUnit],
    owner_face_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    surface_stats: dict[str, Counter] = defaultdict(Counter)
    surface_photos: dict[str, set[str]] = defaultdict(set)
    scores: Counter[str] = Counter()

    for unit in text_units:
        unit_text_weight = 1.0 + unit.owner_reference_score * 1.6
        if unit.has_face and owner_face_id:
            unit_text_weight += 0.4
        for surface in unit.person_surfaces:
            if not surface:
                continue
            surface_stats[surface]["mentions"] += 1
            surface_photos[surface].add(unit.photo_id)
            score = unit_text_weight
            if owner_face_id and unit.photo_id:
                # Same-photo evidence is strong but not exclusive; the owner's
                # full name is often carried by screenshots with no face.
                score += 0.2
            if is_full_name(surface):
                score += 0.8
            scores[surface] += score

    candidates = []
    for surface, score in scores.items():
        if not looks_like_person_name(surface):
            continue
        candidates.append(
            {
                "name": surface,
                "score": round(float(score), 3),
                "mentions": int(surface_stats[surface]["mentions"]),
                "n_photos": len(surface_photos[surface]),
                "is_full_name": is_full_name(surface),
            }
        )
    candidates.sort(key=lambda row: (-row["is_full_name"], -row["score"], -row["mentions"], row["name"]))
    full_candidates = [row for row in candidates if row["is_full_name"]]
    best = (full_candidates or candidates or [{"name": ""}])[0]["name"]
    return str(best or ""), candidates[:20]


def _build_face_units(photos: list[dict[str, Any]], owner_face_id: str) -> list[FaceEvidenceUnit]:
    units: list[FaceEvidenceUnit] = []
    for photo in photos:
        photo_id = str(photo.get("photo_id") or "")
        face_ids = [str(f) for f in (photo.get("visible_face_ids") or [])]
        if not face_ids:
            continue
        visible_text = [str(t) for t in (photo.get("visible_text") or []) if str(t).strip()]
        caption = str(photo.get("caption") or "")
        entities = photo.get("text_entities") or []
        metadata = photo.get("metadata") or {}
        text_blob = combined_text(
            [caption, *visible_text, *(str(e.get("surface") or "") for e in entities)]
        )
        relation_terms = extract_relation_terms(text_blob)
        activity_terms = extract_activity_terms(text_blob)
        venue_terms = extract_venue_terms(
            combined_text([text_blob, str(metadata.get("gps_location") or ""), str(metadata.get("gps_city") or "")])
        )
        for face_id in face_ids:
            units.append(
                FaceEvidenceUnit(
                    face_id=face_id,
                    photo_id=photo_id,
                    year_month=str(photo.get("year_month") or ""),
                    timestamp=str(photo.get("timestamp") or ""),
                    gps_city=str(metadata.get("gps_city") or ""),
                    gps_location=str(metadata.get("gps_location") or ""),
                    caption=caption,
                    visible_text=visible_text,
                    co_faces=[f for f in face_ids if f != face_id],
                    owner_present=bool(owner_face_id and owner_face_id in face_ids and face_id != owner_face_id),
                    relation_terms=relation_terms,
                    activity_terms=activity_terms,
                    venue_terms=venue_terms,
                )
            )
    return units

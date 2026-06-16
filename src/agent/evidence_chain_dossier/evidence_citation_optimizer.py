"""Post-readout evidence citation optimization for ECD profiles.

This pass is deliberately value-frozen: it never changes names, relations,
categories, owner fact text, or confidence. It only reorders cited public photos
and adds a short evidence note when the selected citations become stronger.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .config import ECDConfig
from .schemas import EvidenceInventory, ResolvedIdentity
from .text import dedupe, first_name, normalize_name


def optimize_evidence_citations(
    *,
    profile: dict[str, Any],
    inventory: EvidenceInventory,
    resolved: list[ResolvedIdentity],
    config: ECDConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Improve evidence photo ordering without changing predicted values."""
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "owner_updates": [],
        "person_updates": [],
    }
    owner_updates = _optimize_owner_fact_evidence(profile, inventory, config)
    person_updates = _optimize_person_evidence(profile, inventory, resolved, config)
    diagnostics["owner_updates"] = owner_updates
    diagnostics["person_updates"] = person_updates
    diagnostics["n_owner_updates"] = len(owner_updates)
    diagnostics["n_person_updates"] = len(person_updates)
    return profile, diagnostics


def _optimize_owner_fact_evidence(
    profile: dict[str, Any],
    inventory: EvidenceInventory,
    config: ECDConfig,
) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    facts = list((profile.get("owner") or {}).get("facts") or [])
    for idx, fact in enumerate(facts):
        text = str(fact.get("text") or "").strip()
        if not text:
            continue
        if _owner_fact_skip_evidence_optimization(text):
            continue
        reasoning = str(fact.get("reasoning_path") or "")
        reasoning_photo_ids = _photo_ids_in_text(reasoning, inventory)
        original_current = [
            str(pid)
            for pid in fact.get("evidence_photo_ids") or []
            if str(pid) in inventory.photo_lookup
        ]
        missing_reasoning_ids = [pid for pid in reasoning_photo_ids if pid not in original_current[:5]]
        if not missing_reasoning_ids and _owner_current_strong(text, original_current[:3], inventory):
            continue
        current = [
            str(pid)
            for pid in dedupe([*original_current, *reasoning_photo_ids])
            if str(pid) in inventory.photo_lookup
        ]
        ranked = _rank_owner_fact_photos(
            text,
            current,
            inventory,
            config,
            reasoning_photo_ids=reasoning_photo_ids,
        )
        if not ranked:
            continue
        optimized = ranked[: max(1, config.evidence_optimizer_owner_max_photos)]
        if optimized == current[: len(optimized)]:
            continue
        fact["evidence_photo_ids"] = optimized
        note = _owner_evidence_note(text, optimized, inventory)
        if note:
            fact["reasoning_path"] = _append_note(str(fact.get("reasoning_path") or ""), note)
        updates.append(
            {
                "index": idx,
                "text": text[:120],
                "before": current[:8],
                "after": optimized,
            }
        )
    return updates


def _optimize_person_evidence(
    profile: dict[str, Any],
    inventory: EvidenceInventory,
    resolved: list[ResolvedIdentity],
    config: ECDConfig,
) -> list[dict[str, Any]]:
    resolved_by_face = {row.face_id: row for row in resolved}
    updates: list[dict[str, Any]] = []
    for person in profile.get("persons") or []:
        face_id = str(person.get("face_id") or "")
        row = resolved_by_face.get(face_id)
        if not row:
            continue
        current = [str(pid) for pid in person.get("evidence_photo_ids") or [] if str(pid) in inventory.photo_lookup]
        ranked = _rank_person_photos(person, row, current, inventory)
        if not ranked:
            continue
        optimized = ranked[: max(1, config.evidence_optimizer_person_max_photos)]
        if optimized == current[: len(optimized)]:
            continue
        person["evidence_photo_ids"] = optimized
        identity_ids = [pid for pid in optimized if _photo_has_face(inventory, pid, face_id)][:6]
        relation_ids = [
            pid
            for pid in optimized
            if _photo_has_face(inventory, pid, face_id)
            and _photo_has_face(inventory, pid, inventory.owner_face_id)
        ][:6]
        if identity_ids:
            person["identity_evidence_photo_ids"] = identity_ids
        if relation_ids:
            person["relation_evidence_photo_ids"] = relation_ids
        note = _person_evidence_note(person, face_id, optimized, inventory)
        if note:
            person["reasoning_path"] = _append_note(str(person.get("reasoning_path") or ""), note)
        updates.append(
            {
                "face_id": face_id,
                "canonical_name": str(person.get("canonical_name") or ""),
                "before": current[:10],
                "after": optimized,
            }
        )
    return updates


def _rank_owner_fact_photos(
    fact_text: str,
    current: list[str],
    inventory: EvidenceInventory,
    config: ECDConfig,
    *,
    reasoning_photo_ids: list[str] | None = None,
) -> list[str]:
    query = _query_tokens(fact_text)
    if len(query) < 2:
        return current
    scored: dict[str, float] = defaultdict(float)
    current_set = set(current)
    reasoning_set = set(reasoning_photo_ids or [])
    for photo_id, photo in inventory.photo_lookup.items():
        blob = _photo_blob(photo)
        overlap = _token_overlap_score(query, blob)
        phrase_hit = _specific_phrase_hit(fact_text, blob)
        reasoning_hit = photo_id in reasoning_set
        if not phrase_hit and overlap < 2.0 and not reasoning_hit:
            continue
        score = overlap
        if photo_id in current_set:
            score += 0.8
        if reasoning_hit:
            score += 1.35
        if _photo_has_face(inventory, photo_id, inventory.owner_face_id):
            score += 0.4
        if photo.get("visible_text"):
            score += 0.25
        if phrase_hit:
            score += 2.0
        if score >= config.evidence_optimizer_min_score:
            scored[str(photo_id)] = max(scored[str(photo_id)], score)
    if not scored:
        return current
    for pos, photo_id in enumerate(current):
        if photo_id in inventory.photo_lookup:
            scored[photo_id] = max(scored[photo_id], 0.9 - pos * 0.03)
    return sorted(scored, key=lambda pid: (-scored[pid], 0 if pid in current_set else 1, pid))


def _owner_current_strong(
    fact_text: str,
    current: list[str],
    inventory: EvidenceInventory,
) -> bool:
    query = _query_tokens(fact_text)
    if len(query) < 2:
        return True
    best = 0.0
    for photo_id in current:
        photo = inventory.photo_lookup.get(photo_id) or {}
        blob = _photo_blob(photo)
        score = _token_overlap_score(query, blob)
        if _specific_phrase_hit(fact_text, blob):
            score += 2.0
        best = max(best, score)
    return best >= 5.0


def _owner_fact_skip_evidence_optimization(fact_text: str) -> bool:
    lower = fact_text.lower()
    broad_visual_terms = [
        "clothing",
        "wardrobe",
        "apparel",
        "wears",
        "wearing",
        "dresses",
        "dressed",
        "outfit",
        "style",
        "business-casual",
        "professional clothing",
        "casual clothing",
    ]
    return any(re.search(rf"\b{re.escape(term)}\b", lower) for term in broad_visual_terms)


def _rank_person_photos(
    person: dict[str, Any],
    row: ResolvedIdentity,
    current: list[str],
    inventory: EvidenceInventory,
) -> list[str]:
    packet = row.evidence_packet
    if _person_topk_balanced(current[:6], person, row, inventory):
        return current
    candidates = dedupe(
        current
        + packet.same_photo_ids
        + packet.text_photo_ids
        + packet.face_photo_ids
        + [
            str(pair.get("text_photo_id") or "")
            for pair in packet.bridge_photo_pairs
            if str(pair.get("text_photo_id") or "")
        ]
        + [
            str(pair.get("face_photo_id") or "")
            for pair in packet.bridge_photo_pairs
            if str(pair.get("face_photo_id") or "")
        ]
    )
    candidates = [pid for pid in candidates if pid in inventory.photo_lookup]
    if not candidates:
        return []
    current_set = set(current)
    name = str(person.get("canonical_name") or row.canonical_name or "")
    relation = str(person.get("relation_to_owner") or "")
    category = str(person.get("relation_category") or "")
    name_tokens = set(_query_tokens(name))
    first = first_name(name).lower()
    relation_tokens = set(_query_tokens(" ".join([relation, category, *packet.relation_clues, *packet.activity_clues, *packet.venue_clues])))
    scored: dict[str, float] = {}
    for pos, photo_id in enumerate(candidates):
        photo = inventory.photo_lookup.get(photo_id) or {}
        blob = _photo_blob(photo)
        score = 0.0
        if photo_id in current_set:
            score += max(0.2, 1.0 - pos * 0.02)
        if _photo_has_face(inventory, photo_id, row.face_id):
            score += 4.0
        if _photo_has_face(inventory, photo_id, inventory.owner_face_id):
            score += 0.8
        if photo_id in packet.same_photo_ids:
            score += 2.0
        if name_tokens and any(tok in blob for tok in name_tokens):
            score += 4.0
        elif first and first in blob:
            score += 1.5
        score += min(2.5, _token_overlap_score(relation_tokens, blob) * 0.5)
        if photo.get("visible_text"):
            score += 0.3
        scored[photo_id] = score
    ranked = sorted(scored, key=lambda pid: (-scored[pid], candidates.index(pid), pid))
    return _balanced_person_ranking(ranked, person, row, inventory)


def _person_topk_balanced(
    photo_ids: list[str],
    person: dict[str, Any],
    row: ResolvedIdentity,
    inventory: EvidenceInventory,
) -> bool:
    if not photo_ids:
        return False
    has_face = any(_photo_has_face(inventory, pid, row.face_id) for pid in photo_ids)
    has_text = any(_person_text_anchor(pid, person, row, inventory) for pid in photo_ids)
    return has_face and has_text


def _balanced_person_ranking(
    ranked: list[str],
    person: dict[str, Any],
    row: ResolvedIdentity,
    inventory: EvidenceInventory,
) -> list[str]:
    if not ranked:
        return []
    prefix: list[str] = []
    same_photo = [
        pid
        for pid in ranked
        if _photo_has_face(inventory, pid, row.face_id)
        and _person_text_anchor(pid, person, row, inventory)
    ]
    text_photos = [pid for pid in ranked if _person_text_anchor(pid, person, row, inventory)]
    face_photos = [pid for pid in ranked if _photo_has_face(inventory, pid, row.face_id)]
    owner_context = [
        pid
        for pid in ranked
        if _photo_has_face(inventory, pid, row.face_id)
        and _photo_has_face(inventory, pid, inventory.owner_face_id)
    ]
    bridge_photos: list[str] = []
    ranked_set = set(ranked)
    for pair in row.evidence_packet.bridge_photo_pairs:
        text_pid = str(pair.get("text_photo_id") or "")
        face_pid = str(pair.get("face_photo_id") or "")
        if text_pid in ranked_set:
            bridge_photos.append(text_pid)
        if face_pid in ranked_set:
            bridge_photos.append(face_pid)
    for group in [same_photo[:1], text_photos[:1], face_photos[:1], owner_context[:2], bridge_photos[:4], ranked]:
        for photo_id in group:
            if photo_id and photo_id not in prefix:
                prefix.append(photo_id)
    return prefix


def _person_text_anchor(
    photo_id: str,
    person: dict[str, Any],
    row: ResolvedIdentity,
    inventory: EvidenceInventory,
) -> bool:
    if photo_id in set(row.evidence_packet.text_photo_ids):
        return True
    name = str(person.get("canonical_name") or row.canonical_name or "").strip()
    if not name:
        return False
    return _name_in_photo(name, inventory.photo_lookup.get(photo_id) or {})


def _owner_evidence_note(fact_text: str, photo_ids: list[str], inventory: EvidenceInventory) -> str:
    if not photo_ids:
        return ""
    cues = []
    tokens = _query_tokens(fact_text)
    for pid in photo_ids[:3]:
        photo = inventory.photo_lookup.get(pid) or {}
        hit = _best_text_hit(photo, tokens)
        if hit:
            cues.append(f"{pid} contains `{hit}`")
    if not cues:
        return ""
    return "Evidence citation optimizer: " + "; ".join(cues) + "."


def _person_evidence_note(
    person: dict[str, Any],
    face_id: str,
    photo_ids: list[str],
    inventory: EvidenceInventory,
) -> str:
    if not photo_ids:
        return ""
    name = str(person.get("canonical_name") or "").strip()
    face_photos = [pid for pid in photo_ids if _photo_has_face(inventory, pid, face_id)]
    name_photos = [pid for pid in photo_ids if name and _name_in_photo(name, inventory.photo_lookup.get(pid) or {})]
    parts = []
    if face_photos:
        parts.append(f"{face_photos[0]} shows {face_id}")
    if name_photos and name:
        parts.append(f"{name_photos[0]} contains name/text evidence for {name}")
    if not parts:
        return ""
    return "Evidence citation optimizer: " + "; ".join(parts[:2]) + "."


def _append_note(existing: str, note: str, *, limit: int = 1650) -> str:
    if not note or note in existing:
        return existing
    combined = " ".join(part for part in [note, existing] if part).strip()
    if len(combined) <= limit:
        return combined
    shortened = combined[: max(0, limit - 1)].rsplit(" ", 1)[0].rstrip()
    return shortened if shortened.endswith((".", "!", "?")) else shortened + "."


def _photo_has_face(inventory: EvidenceInventory, photo_id: str, face_id: str) -> bool:
    photo = inventory.photo_lookup.get(photo_id) or {}
    return face_id in [str(fid) for fid in photo.get("visible_face_ids") or []]


def _photo_blob(photo: dict[str, Any]) -> str:
    parts = [
        str(photo.get("caption") or ""),
        *(str(item) for item in photo.get("visible_text") or []),
    ]
    metadata = photo.get("metadata") if isinstance(photo.get("metadata"), dict) else {}
    parts.extend([str(metadata.get("gps_city") or ""), str(metadata.get("gps_location") or "")])
    for entity in photo.get("text_entities") or []:
        if isinstance(entity, dict):
            parts.append(str(entity.get("surface") or entity.get("text") or ""))
        else:
            parts.append(str(entity))
    return " ".join(parts).lower()


def _query_tokens(text: str) -> list[str]:
    stop = {
        "album",
        "owner",
        "person",
        "photo",
        "face",
        "regularly",
        "appears",
        "uses",
        "use",
        "has",
        "have",
        "with",
        "from",
        "that",
        "this",
        "the",
        "and",
        "are",
        "was",
        "were",
        "she",
        "he",
        "they",
        "for",
        "her",
        "his",
        "their",
        "first",
        "last",
        "name",
        "age",
        "year",
        "years",
        "old",
        "based",
        "located",
        "connected",
        "appears",
        "woman",
        "women",
        "man",
        "men",
    }
    return dedupe(
        tok
        for tok in re.findall(r"[a-z0-9][a-z0-9'-]+", normalize_name(text).lower())
        if len(tok) >= 3 and tok not in stop
    )[:24]


def _token_overlap_score(query: list[str] | set[str], blob: str) -> float:
    if not query or not blob:
        return 0.0
    return sum(1.0 for tok in query if tok in blob)


def _specific_phrase_hit(text: str, blob: str) -> bool:
    for phrase in _important_phrases(text):
        if phrase and phrase in blob:
            return True
    return False


def _important_phrases(text: str) -> list[str]:
    raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9'&.-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'&.-]*){1,4}", text)
    out = []
    for phrase in raw:
        norm = normalize_name(phrase).lower()
        tokens = _query_tokens(norm)
        if len(tokens) >= 2:
            out.append(" ".join(tokens[:5]))
    return dedupe(out)[:8]


def _photo_ids_in_text(text: str, inventory: EvidenceInventory) -> list[str]:
    if not text:
        return []
    ids = []
    for match in re.findall(r"\bphoto[_ -]?(\d{1,5})\b", text, flags=re.IGNORECASE):
        photo_id = f"photo_{int(match):04d}"
        if photo_id in inventory.photo_lookup:
            ids.append(photo_id)
    return dedupe(ids)


def _best_text_hit(photo: dict[str, Any], tokens: list[str]) -> str:
    if not tokens:
        return ""
    items = [
        str(photo.get("caption") or ""),
        *(str(item) for item in photo.get("visible_text") or []),
    ]
    for entity in photo.get("text_entities") or []:
        if isinstance(entity, dict):
            items.append(str(entity.get("surface") or entity.get("text") or ""))
        else:
            items.append(str(entity))
    best = ""
    best_count = 0
    for item in items:
        lower = item.lower()
        count = sum(1 for tok in tokens if tok in lower)
        if count > best_count:
            best = " ".join(item.split())[:90]
            best_count = count
    return best if best_count else ""


def _name_in_photo(name: str, photo: dict[str, Any]) -> bool:
    blob = _photo_blob(photo)
    tokens = _query_tokens(name)
    if not tokens:
        return False
    if " ".join(tokens) and " ".join(tokens) in blob:
        return True
    return any(tok in blob for tok in tokens)

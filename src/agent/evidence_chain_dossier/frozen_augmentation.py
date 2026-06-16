"""Frozen-identity evidence augmentation for ECD profiles.

This pass runs after identity resolution and readout. It never changes face-name
bindings; it only appends conservative owner facts, repairs relation/category
fields, and reorders evidence photos.
"""

from __future__ import annotations

import re
from typing import Any

from .config import ECDConfig
from .owner_evidence_planner import build_owner_evidence_plan
from .schemas import EvidenceInventory, EvidencePacket, ResolvedIdentity
from .text import combined_text, dedupe, first_name, normalize_name, text_contains_name


RELATION_HINTS: tuple[tuple[tuple[str, ...], str, str, str], ...] = (
    (("mother", "mom", "mum", "mama", "mother's day"), "mother", "family", "direct_relation"),
    (("father", "dad", "papa", "father's day"), "father", "family", "direct_relation"),
    (("wife", "husband", "spouse", "partner", "boyfriend", "girlfriend", "date night"), "partner", "family", "direct_relation"),
    (("older brother", "younger brother", "brother"), "brother", "family", "direct_relation"),
    (("older sister", "younger sister", "sister"), "sister", "family", "direct_relation"),
    (("daughter", "son"), "child", "family", "direct_relation"),
    (("aunt", "auntie", "uncle", "cousin", "niece", "nephew", "grandmother", "grandfather", "granddaughter", "grandson", "in-law"), "family member", "family", "direct_relation"),
    (("coworker", "co-worker", "colleague", "supervisor", "manager", "boss", "advisor", "coordinator", "organizer", "director", "instructor", "employee", "business partner", "small business peer"), "colleague", "colleague", "direct_relation"),
    (("classmate", "schoolmate", "study partner", "study group member", "university friend", "college friend", "graduate classmate"), "classmate", "classmate", "direct_relation"),
    (("next-door neighbor", "next-door neighbour", "building neighbor", "neighbor", "neighbour"), "neighbor", "neighbor", "direct_relation"),
    (("close friend", "best friend", "longtime friend", "family friend", "friend", "friends"), "friend", "friend", "direct_relation"),
)

ACTIVITY_CATEGORY_HINTS: tuple[tuple[tuple[str, ...], str, str, str], ...] = (
    (("legal", "law", "attorney", "bar association", "office", "court", "conference", "workshop", "client", "presentation"), "colleague", "colleague", "activity_context"),
    (("hospital", "clinic", "doctor", "nurse", "medical"), "colleague", "colleague", "activity_context"),
    (("school", "class", "course", "campus", "college", "university", "study group"), "classmate", "classmate", "activity_context"),
    (("apartment lobby", "building", "neighbor", "neighbour"), "neighbor", "neighbor", "activity_context"),
    (("brunch", "birthday", "dinner", "cycling", "bike", "hiking", "yoga", "book club", "board game", "karaoke", "dance", "photography", "gallery", "museum", "film", "cinema", "choir", "drumming", "calligraphy"), "friend", "friend", "activity_context"),
    (("thanksgiving", "christmas", "holiday", "family dinner", "reunion"), "family member", "family", "activity_context"),
)

OWNER_DUPLICATE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("owner_name", ("first and last name", "full name", "name is")),
    ("owner_age", ("years old",)),
    ("owner_birth", ("birth year", "date of birth", "born")),
    ("owner_base", ("based in", "lives in", "living in")),
    ("owner_gender", ("appears to be a man", "appears to be a woman")),
)


def augment_frozen_evidence_profile(
    *,
    profile: dict[str, Any],
    inventory: EvidenceInventory,
    resolved: list[ResolvedIdentity],
    config: ECDConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply post-readout augmentation without changing identity bindings."""
    diagnostics: dict[str, Any] = {"enabled": True}
    if config.owner_addendum_max_new_facts > 0:
        diagnostics["owner_addendum"] = _augment_owner_facts(profile, inventory, config)
    else:
        diagnostics["owner_addendum"] = {"enabled": False, "reason": "max_new_facts_zero"}

    resolved_by_face = {row.face_id: row for row in resolved}
    relation_updates: list[dict[str, Any]] = []
    evidence_updates: list[dict[str, Any]] = []
    certificate_updates: list[dict[str, Any]] = []
    for person in profile.get("persons") or []:
        face_id = str(person.get("face_id") or "")
        row = resolved_by_face.get(face_id)
        if not row:
            continue
        if config.use_relation_evidence_repair:
            update = _repair_relation_fields(person, row)
            if update:
                relation_updates.append(update)
        if config.use_person_evidence_routing:
            routed = _reroute_person_evidence(person, row, inventory, config)
            if routed:
                evidence_updates.append(routed)
        if config.use_person_evidence_certificates:
            certificate = _attach_person_evidence_certificate(person, row, inventory, config)
            if certificate:
                certificate_updates.append(certificate)
    diagnostics["relation_repair"] = {
        "enabled": bool(config.use_relation_evidence_repair),
        "n_updates": len(relation_updates),
        "updates": relation_updates[:100],
    }
    diagnostics["evidence_routing"] = {
        "enabled": bool(config.use_person_evidence_routing),
        "n_updates": len(evidence_updates),
        "updates": evidence_updates[:100],
    }
    diagnostics["evidence_certificates"] = {
        "enabled": bool(config.use_person_evidence_certificates),
        "n_updates": len(certificate_updates),
        "updates": certificate_updates[:100],
    }
    return profile, diagnostics


def _augment_owner_facts(
    profile: dict[str, Any],
    inventory: EvidenceInventory,
    config: ECDConfig,
) -> dict[str, Any]:
    owner = profile.setdefault("owner", {})
    facts = list(owner.get("facts") or [])
    max_total = max(config.max_owner_facts, int(config.owner_addendum_max_total_facts))
    slots = max(0, min(int(config.owner_addendum_max_new_facts), max_total - len(facts)))
    if slots <= 0:
        return {"enabled": True, "n_added": 0, "reason": "no_slots"}

    seen_signatures = {_owner_fact_signature(str(f.get("text") or "")) for f in facts}
    cards = build_owner_evidence_plan(
        inventory,
        max_cards=max(48, config.owner_evidence_max_cards),
        balanced=True,
        min_cards_per_source=max(2, config.owner_census_min_cards_per_source),
    )
    cards.sort(
        key=lambda card: (
            0 if str(card.get("fact_type") or "") not in _BASE_OWNER_FACT_TYPES else 1,
            -float(card.get("score") or 0.0),
            str(card.get("fact_type") or ""),
        )
    )

    added: list[dict[str, Any]] = []
    for card in cards:
        if len(added) >= slots:
            break
        if not _owner_card_passes_addendum_gate(card, config):
            continue
        text = str(card.get("text") or "").strip()
        signature = _owner_fact_signature(text)
        if not text or not signature or signature in seen_signatures:
            continue
        evidence = [str(pid) for pid in (card.get("evidence_photo_ids") or []) if str(pid).startswith("photo_")]
        matched_terms = [str(term) for term in (card.get("matched_terms") or [])[:8]]
        fact = {
            "text": text,
            "evidence_photo_ids": dedupe(evidence)[:5],
            "confidence": round(min(0.82, max(0.55, float(card.get("confidence") or 0.6))), 3),
            "reasoning_path": (
                f"Frozen addendum fact `{card.get('fact_type')}` is appended after identity resolution; "
                f"it is supported by {len(evidence)} public photos with cues: {', '.join(matched_terms) or 'recurring evidence'}."
            ),
        }
        facts.append(fact)
        added.append(
            {
                "fact_type": str(card.get("fact_type") or ""),
                "source": str(card.get("source") or ""),
                "text": text,
                "score": round(float(card.get("score") or 0.0), 3),
                "evidence_photo_ids": fact["evidence_photo_ids"],
            }
        )
        seen_signatures.add(signature)
    owner["facts"] = facts[:max_total]
    return {"enabled": True, "n_added": len(added), "added": added}


_BASE_OWNER_FACT_TYPES = {
    "music_choir",
    "home_music_setup",
    "home_cooking",
    "coffee_routine",
    "home_environment",
    "work_consulting_office",
    "work_legal",
    "work_medical",
    "work_education",
    "education_degree",
    "hair_salon_work",
    "warehouse_logistics_work",
    "running",
    "cycling",
    "hiking_outdoors",
    "fishing",
    "sewing_crafts",
    "gardening",
    "photography",
    "volunteer_community",
    "religious_church",
    "public_transit",
    "driving_vehicle",
    "pet_owner",
    "smartphone_digital",
    "casual_clothing",
    "small_group_social",
    "sports_family_support",
}


def _owner_card_passes_addendum_gate(card: dict[str, Any], config: ECDConfig) -> bool:
    evidence = [str(pid) for pid in card.get("evidence_photo_ids") or []]
    if len(evidence) < 2:
        return False
    score = float(card.get("score") or 0.0)
    if score < float(config.owner_addendum_min_score):
        return False
    source = str(card.get("source") or "")
    fact_type = str(card.get("fact_type") or "")
    if source == "direct_ocr" and fact_type not in {
        "student_academic_terms",
        "social_media_business_account",
        "education_degree",
    }:
        return False
    if source in {"face_cooccurrence", "metadata_inference"} and len(evidence) < 3:
        return False
    return True


def _owner_fact_signature(text: str) -> str:
    lower = " ".join(str(text or "").lower().split())
    for signature, needles in OWNER_DUPLICATE_PATTERNS:
        if any(needle in lower for needle in needles):
            return signature
    normalized = re.sub(r"[^a-z0-9]+", " ", lower).strip()
    tokens = [tok for tok in normalized.split() if tok not in {"the", "album", "owner", "regularly", "recurring", "appears"}]
    return " ".join(tokens[:8])


def _repair_relation_fields(person: dict[str, Any], row: ResolvedIdentity) -> dict[str, Any] | None:
    relation, category, strength, source = _relation_hint(row.evidence_packet)
    if not category:
        return None
    current_relation = str(person.get("relation_to_owner") or "").strip()
    current_category = str(person.get("relation_category") or "").strip()
    should_update_relation = False
    should_update_category = False
    if source == "direct_relation":
        should_update_relation = bool(relation) and not current_relation
        should_update_category = bool(category) and (not current_category or current_category == "other")
    elif strength >= 0.62:
        should_update_relation = bool(relation) and not current_relation
        should_update_category = bool(category) and not current_category
    if not should_update_relation and not should_update_category:
        return None

    before = {
        "relation_to_owner": current_relation,
        "relation_category": current_category,
    }
    if should_update_relation:
        person["relation_to_owner"] = relation
    if should_update_category:
        person["relation_category"] = category
    person["reasoning_path"] = _append_relation_reasoning(
        str(person.get("reasoning_path") or ""),
        row.evidence_packet,
        relation or category,
    )
    person["confidence"] = round(max(float(person.get("confidence") or 0.0), min(0.86, 0.55 + strength * 0.25)), 3)
    return {
        "face_id": row.face_id,
        "canonical_name": row.canonical_name,
        "before": before,
        "after": {
            "relation_to_owner": person.get("relation_to_owner") or "",
            "relation_category": person.get("relation_category") or "",
        },
        "source": source,
        "strength": round(strength, 3),
    }


def _relation_hint(packet: EvidencePacket) -> tuple[str, str, float, str]:
    relation_blob = " ".join(packet.relation_clues).lower()
    context_blob = " ".join(
        [
            *packet.relation_clues,
            *packet.activity_clues,
            *packet.venue_clues,
            *packet.coface_clues,
            packet.narrative_summary,
        ]
    ).lower()
    for terms, relation, category, source in RELATION_HINTS:
        if _has_any_term(relation_blob, terms) or _has_any_term(context_blob, terms):
            return relation, category, 0.90, source
    for terms, relation, category, source in ACTIVITY_CATEGORY_HINTS:
        hits = sum(1 for term in terms if term in context_blob)
        if hits:
            return relation, category, min(0.78, 0.48 + hits * 0.10), source
    return "", "", 0.0, ""


def _has_any_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _relation_norm(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def _append_relation_reasoning(existing: str, packet: EvidencePacket, relation: str) -> str:
    cited = dedupe(packet.same_photo_ids + packet.text_photo_ids + packet.face_photo_ids)[:3]
    if not cited:
        return existing
    sentence = f"Frozen relation repair uses {', '.join(cited)} to support `{relation}` without changing the face-name binding."
    if sentence in existing:
        return existing
    return " ".join(part for part in [existing, sentence] if part).strip()


def _reroute_person_evidence(
    person: dict[str, Any],
    row: ResolvedIdentity,
    inventory: EvidenceInventory,
    config: ECDConfig,
) -> dict[str, Any] | None:
    current = [str(pid) for pid in person.get("evidence_photo_ids") or []]
    routed = _rank_person_evidence_ids(
        person,
        row,
        inventory,
        limit=config.max_evidence_photos_per_person,
        current=current,
    )
    if not routed:
        return None
    if routed == current[: config.max_evidence_photos_per_person]:
        return None
    person["evidence_photo_ids"] = routed
    return {
        "face_id": row.face_id,
        "canonical_name": row.canonical_name,
        "before": current[: config.max_evidence_photos_per_person],
        "after": routed,
    }


def _attach_person_evidence_certificate(
    person: dict[str, Any],
    row: ResolvedIdentity,
    inventory: EvidenceInventory,
    config: ECDConfig,
) -> dict[str, Any] | None:
    current = [str(pid) for pid in person.get("evidence_photo_ids") or []]
    routed = _rank_person_evidence_ids(
        person,
        row,
        inventory,
        limit=config.max_evidence_photos_per_person,
        current=current,
    )
    if routed and routed != current[: config.max_evidence_photos_per_person]:
        person["evidence_photo_ids"] = routed

    identity_ids = _identity_certificate_ids(person, row, inventory, routed)
    relation_ids = _relation_certificate_ids(person, row, inventory, routed)
    summary = _person_evidence_certificate(person, row, inventory, identity_ids, relation_ids)
    if not summary:
        return None

    before_reasoning = str(person.get("reasoning_path") or "")
    person["identity_evidence_photo_ids"] = identity_ids
    person["relation_evidence_photo_ids"] = relation_ids
    person["evidence_summary"] = summary
    person["reasoning_path"] = _prepend_certificate(
        before_reasoning,
        summary,
        max_chars=config.person_evidence_certificate_max_chars,
    )
    return {
        "face_id": row.face_id,
        "canonical_name": row.canonical_name,
        "identity_evidence_photo_ids": identity_ids,
        "relation_evidence_photo_ids": relation_ids,
        "evidence_changed": routed != current[: config.max_evidence_photos_per_person],
    }


def _rank_person_evidence_ids(
    person: dict[str, Any],
    row: ResolvedIdentity,
    inventory: EvidenceInventory,
    *,
    limit: int,
    current: list[str],
) -> list[str]:
    packet = row.evidence_packet
    bridge_ids = []
    for pair in _rank_bridge_pairs(packet.bridge_photo_pairs):
        bridge_ids.extend([str(pair.get("text_photo_id") or ""), str(pair.get("face_photo_id") or "")])
    candidate_ids = dedupe(
        packet.same_photo_ids
        + bridge_ids
        + packet.text_photo_ids
        + packet.face_photo_ids
        + current
    )
    candidate_ids = [pid for pid in candidate_ids if pid in inventory.photo_lookup]
    if not candidate_ids:
        return []

    full_name_ids = _photos_matching_name(candidate_ids, person, row, inventory, require_full=True)
    first_name_ids = _photos_matching_name(candidate_ids, person, row, inventory, require_full=False)
    same_ids = [pid for pid in candidate_ids if pid in packet.same_photo_ids]
    same_direct_ids = [
        pid
        for pid in same_ids
        if _photo_has_face(inventory, pid, row.face_id)
        and _photo_matches_person_name(pid, person, row, inventory, require_full=False)
    ]
    face_owner_ids = [
        pid
        for pid in candidate_ids
        if _photo_has_face(inventory, pid, row.face_id)
        and _photo_has_face(inventory, pid, inventory.owner_face_id)
    ]
    face_ids = [pid for pid in candidate_ids if _photo_has_face(inventory, pid, row.face_id)]
    relation_ids = _relation_photo_ids(candidate_ids, person, row, inventory)

    out: list[str] = []
    _extend_ranked(out, same_direct_ids, person, row, inventory, candidate_ids)
    _extend_ranked(out, full_name_ids, person, row, inventory, candidate_ids, cap=2)
    _extend_bridge_ids(out, packet.bridge_photo_pairs, candidate_ids, person, row, inventory)
    _extend_ranked(out, face_owner_ids, person, row, inventory, candidate_ids, cap=3)
    _extend_ranked(out, same_ids, person, row, inventory, candidate_ids)
    _extend_ranked(out, relation_ids, person, row, inventory, candidate_ids, cap=3)
    _extend_ranked(out, first_name_ids, person, row, inventory, candidate_ids, cap=2)
    _extend_ranked(out, face_ids, person, row, inventory, candidate_ids)
    _extend_ranked(out, candidate_ids, person, row, inventory, candidate_ids)
    return out[:limit]


def _identity_certificate_ids(
    person: dict[str, Any],
    row: ResolvedIdentity,
    inventory: EvidenceInventory,
    routed: list[str],
) -> list[str]:
    candidates = routed or [str(pid) for pid in person.get("evidence_photo_ids") or []]
    text_ids = _photos_matching_name(candidates, person, row, inventory, require_full=True)
    if not text_ids:
        text_ids = _photos_matching_name(candidates, person, row, inventory, require_full=False)
    face_ids = [
        pid
        for pid in candidates
        if _photo_has_face(inventory, pid, row.face_id)
    ]
    owner_face_ids = [
        pid
        for pid in face_ids
        if _photo_has_face(inventory, pid, inventory.owner_face_id)
    ]
    bridge_ids: list[str] = []
    for pair in _rank_bridge_pairs(row.evidence_packet.bridge_photo_pairs):
        for pid in [str(pair.get("text_photo_id") or ""), str(pair.get("face_photo_id") or "")]:
            if pid in candidates:
                bridge_ids.append(pid)
    return dedupe(row.evidence_packet.same_photo_ids + text_ids[:2] + owner_face_ids[:2] + bridge_ids[:4] + face_ids[:3])[:6]


def _relation_certificate_ids(
    person: dict[str, Any],
    row: ResolvedIdentity,
    inventory: EvidenceInventory,
    routed: list[str],
) -> list[str]:
    candidates = routed or [str(pid) for pid in person.get("evidence_photo_ids") or []]
    relation_ids = _relation_photo_ids(candidates, person, row, inventory)
    owner_face_ids = [
        pid
        for pid in candidates
        if _photo_has_face(inventory, pid, row.face_id)
        and _photo_has_face(inventory, pid, inventory.owner_face_id)
    ]
    return dedupe(relation_ids[:3] + owner_face_ids[:3] + row.evidence_packet.same_photo_ids[:2] + candidates[:3])[:6]


def _person_evidence_certificate(
    person: dict[str, Any],
    row: ResolvedIdentity,
    inventory: EvidenceInventory,
    identity_ids: list[str],
    relation_ids: list[str],
) -> str:
    name = str(person.get("canonical_name") or row.canonical_name or "").strip()
    relation = str(person.get("relation_to_owner") or row.relation_to_owner or "").strip()
    category = str(person.get("relation_category") or row.relation_category or "").strip()
    if not name and not relation and not category:
        return ""

    full_name_ids = _photos_matching_name(identity_ids, person, row, inventory, require_full=True)
    first_name_ids = _photos_matching_name(identity_ids, person, row, inventory, require_full=False)
    face_ids = [pid for pid in identity_ids if _photo_has_face(inventory, pid, row.face_id)]
    same_direct_ids = [
        pid
        for pid in identity_ids
        if _photo_has_face(inventory, pid, row.face_id)
        and _photo_matches_person_name(pid, person, row, inventory, require_full=False)
    ]
    parts: list[str] = []
    if same_direct_ids and name:
        parts.append(f"{same_direct_ids[0]} directly shows {row.face_id} with text/name evidence for {name}.")
    elif full_name_ids and face_ids and name:
        parts.append(f"{full_name_ids[0]} names {name}; {face_ids[0]} shows {row.face_id} in the matched album context.")
    elif first_name_ids and face_ids and name:
        parts.append(f"{first_name_ids[0]} names {first_name(name) or name}; {face_ids[0]} shows {row.face_id} in the matched album context.")
    elif face_ids and name:
        parts.append(f"{face_ids[0]} shows {row.face_id}; the surrounding evidence chain supports the name {name}.")

    bridge = _best_bridge_pair(row.evidence_packet.bridge_photo_pairs, identity_ids)
    if bridge:
        text_pid = str(bridge.get("text_photo_id") or "")
        face_pid = str(bridge.get("face_photo_id") or "")
        bridge_label = str(bridge.get("bridge") or "event")
        cues = [str(c) for c in (bridge.get("event_channels") or bridge.get("shared_keywords") or [])[:3]]
        suffix = f" using {', '.join(cues)}" if cues else ""
        parts.append(f"The bridge {text_pid}->{face_pid} links the name/text anchor to the face photo by {bridge_label}{suffix}.")

    relation_photo = relation_ids[0] if relation_ids else ""
    if relation_photo and (relation or category):
        rel_text = relation or category
        parts.append(f"{relation_photo} and repeated co-presence/context support the `{rel_text}` relation/category.")

    if not parts:
        return ""
    return "Evidence certificate: " + " ".join(parts[:3])


def _prepend_certificate(existing: str, certificate: str, *, max_chars: int) -> str:
    if not certificate:
        return existing
    if certificate in existing:
        return existing
    combined = " ".join(part for part in [certificate, existing] if part).strip()
    if len(combined) <= max_chars:
        return combined
    shortened = combined[: max(0, max_chars - 1)].rsplit(" ", 1)[0].rstrip()
    if not shortened.endswith((".", "!", "?")):
        shortened += "."
    return shortened


def _extend_ranked(
    out: list[str],
    ids: list[str],
    person: dict[str, Any],
    row: ResolvedIdentity,
    inventory: EvidenceInventory,
    candidate_ids: list[str],
    *,
    cap: int | None = None,
) -> None:
    ranked = sorted(
        dedupe(ids),
        key=lambda pid: (
            -_person_evidence_score(pid, person, row, inventory),
            candidate_ids.index(pid) if pid in candidate_ids else len(candidate_ids),
            pid,
        ),
    )
    added = 0
    for pid in ranked:
        if pid and pid not in out:
            out.append(pid)
            added += 1
            if cap is not None and added >= cap:
                return


def _extend_bridge_ids(
    out: list[str],
    bridge_pairs: list[dict[str, Any]],
    candidate_ids: list[str],
    person: dict[str, Any],
    row: ResolvedIdentity,
    inventory: EvidenceInventory,
) -> None:
    for pair in _rank_bridge_pairs(bridge_pairs):
        pair_ids = [str(pair.get("text_photo_id") or ""), str(pair.get("face_photo_id") or "")]
        for pid in pair_ids:
            if pid in candidate_ids and pid not in out:
                out.append(pid)
        if len([pid for pid in pair_ids if pid in out]) == 2:
            return


def _rank_bridge_pairs(bridge_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {"location_month": 0, "near_date_city": 1, "city_month": 2, "event_semantic": 3}
    return sorted(
        bridge_pairs,
        key=lambda pair: (
            priority.get(str(pair.get("bridge") or ""), 9),
            -float(pair.get("event_score") or 0.0),
            0 if pair.get("strong_signal") else 1,
            str(pair.get("text_photo_id") or ""),
            str(pair.get("face_photo_id") or ""),
        ),
    )


def _best_bridge_pair(bridge_pairs: list[dict[str, Any]], allowed_ids: list[str]) -> dict[str, Any] | None:
    allowed = set(allowed_ids)
    for pair in _rank_bridge_pairs(bridge_pairs):
        text_pid = str(pair.get("text_photo_id") or "")
        face_pid = str(pair.get("face_photo_id") or "")
        if text_pid in allowed and face_pid in allowed:
            return pair
    return None


def _photos_matching_name(
    photo_ids: list[str],
    person: dict[str, Any],
    row: ResolvedIdentity,
    inventory: EvidenceInventory,
    *,
    require_full: bool,
) -> list[str]:
    return [
        pid
        for pid in photo_ids
        if _photo_matches_person_name(pid, person, row, inventory, require_full=require_full)
    ]


def _photo_matches_person_name(
    photo_id: str,
    person: dict[str, Any],
    row: ResolvedIdentity,
    inventory: EvidenceInventory,
    *,
    require_full: bool,
) -> bool:
    name = normalize_name(str(person.get("canonical_name") or row.canonical_name or ""))
    if not name:
        return False
    text = _photo_text(inventory.photo_lookup.get(photo_id) or {})
    if require_full:
        return text_contains_name(text, name)
    first = first_name(name)
    return bool(first and re.search(rf"\b{re.escape(first.lower())}\b", text.lower()))


def _relation_photo_ids(
    photo_ids: list[str],
    person: dict[str, Any],
    row: ResolvedIdentity,
    inventory: EvidenceInventory,
) -> list[str]:
    relation = _relation_norm(str(person.get("relation_to_owner") or row.relation_to_owner or ""))
    category = _relation_norm(str(person.get("relation_category") or row.relation_category or ""))
    clues = [
        term
        for term in dedupe(row.evidence_packet.relation_clues + row.evidence_packet.activity_clues + row.evidence_packet.venue_clues)
        if term
    ][:8]
    needles = [
        token
        for value in [relation, category, *(_relation_norm(c) for c in clues)]
        for token in value.split()
        if len(token) >= 4
    ]
    matches = []
    for pid in photo_ids:
        text = _photo_text(inventory.photo_lookup.get(pid) or {}).lower()
        if any(re.search(rf"\b{re.escape(token)}\b", text) for token in needles):
            matches.append(pid)
    return matches


def _photo_has_face(inventory: EvidenceInventory, photo_id: str, face_id: str) -> bool:
    if not face_id:
        return False
    photo = inventory.photo_lookup.get(photo_id) or {}
    return face_id in {str(fid) for fid in photo.get("visible_face_ids") or []}


def _person_evidence_score(
    photo_id: str,
    person: dict[str, Any],
    row: ResolvedIdentity,
    inventory: EvidenceInventory,
) -> float:
    photo = inventory.photo_lookup.get(photo_id) or {}
    text = _photo_text(photo)
    score = 0.0
    faces = {str(fid) for fid in photo.get("visible_face_ids") or []}
    full_name = normalize_name(str(person.get("canonical_name") or row.canonical_name or ""))
    if full_name and text_contains_name(text, full_name):
        score += 5.5
    elif first_name(full_name) and re.search(rf"\b{re.escape(first_name(full_name).lower())}\b", text.lower()):
        score += 1.2
    if row.face_id in faces:
        score += 4.0
    if inventory.owner_face_id in faces:
        score += 0.8
    relation = str(person.get("relation_to_owner") or "")
    category = str(person.get("relation_category") or "")
    if relation and _has_any_term(text.lower(), tuple(_relation_norm(relation).split())):
        score += 1.0
    if category and category.lower() in text.lower():
        score += 0.5
    if photo_id in row.evidence_packet.same_photo_ids:
        score += 1.4
    if photo_id in row.evidence_packet.text_photo_ids:
        score += 0.9
    if photo_id in row.evidence_packet.face_photo_ids:
        score += 0.7
    return score


def _photo_text(photo: dict[str, Any]) -> str:
    metadata = photo.get("metadata") or {}
    entities = [str(entity.get("surface") or "") for entity in photo.get("text_entities") or []]
    return combined_text(
        [
            photo.get("caption") or "",
            *(photo.get("visible_text") or []),
            *entities,
            metadata.get("gps_city") or "",
            metadata.get("gps_location") or "",
        ]
    )

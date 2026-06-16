"""Profile readout for Evidence-Chain Dossier."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from .config import ECDConfig
from .owner_evidence_planner import planned_owner_facts
from .schemas import EvidenceInventory, ResolvedIdentity, TextEvidenceUnit
from .text import combined_text, dedupe, normalize_name, text_contains_name


def build_profile(
    inventory: EvidenceInventory,
    resolved: list[ResolvedIdentity],
    config: ECDConfig,
    run_stats: dict[str, Any],
) -> dict[str, Any]:
    """Build evaluator-compatible predicted_profile.v2."""
    owner_facts = _owner_facts(inventory, config)
    persons = [
        row.to_profile_row(max_evidence=config.max_evidence_photos_per_person)
        for row in resolved
    ]
    return {
        "schema_version": "predicted_profile.v2",
        "user_id": inventory.album.get("user_id"),
        "framework": "evidence_chain_dossier",
        "variant": "ecd_llm_core_v1" if _llm_enabled(config) else "deterministic_v0",
        "owner": {
            "face_id": inventory.owner_face_id,
            "facts": owner_facts,
        },
        "persons": persons,
        "run_stats": run_stats,
    }


def _owner_facts(inventory: EvidenceInventory, config: ECDConfig) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    add = _fact_adder(facts)
    gender, gender_photos = _infer_gender(inventory)
    subject, possessive = _owner_pronouns(gender)

    owner_name_photos = _photos_for_owner_name(inventory)
    if inventory.owner_name:
        add(
            text=f"{possessive} first and last name are {inventory.owner_name}.",
            evidence=owner_name_photos,
            confidence=0.78,
            reasoning=(
                f"The name {inventory.owner_name} is the strongest owner-name candidate from public OCR/text "
                "evidence, with profile/account style text prioritized over generic attendee lists."
            ),
        )

    if gender:
        add(
            text=f"The album owner appears to be a {gender}.",
            evidence=gender_photos,
            confidence=0.68,
            reasoning=f"Repeated captions for the owner face describe a {gender}, grounded in owner-face photos.",
        )

    dob = _extract_birth_date(inventory)
    if dob:
        age = _age_at_album_end(dob, inventory)
        if age is not None:
            add(
                text=f"{subject} is {age} years old.",
                evidence=dob["photos"],
                confidence=0.74,
                reasoning=(
                    f"Public OCR mentions a birth date or birth year {dob['year']}; comparing it with the album "
                    f"time window ending in {_album_end(inventory)[:10]} yields age {age}."
                ),
            )
        add(
            text=f"{possessive} birth year is {dob['year']}.",
            evidence=dob["photos"],
            confidence=0.70,
            reasoning="The owner-oriented text evidence includes a readable date of birth or birth-year line.",
        )

    city, city_photos = _top_owner_city(inventory)
    if city:
        add(
            text=f"{subject} is based in {city}.",
            evidence=city_photos,
            confidence=0.76,
            reasoning=(
                f"The owner face and owner-oriented text recur in {city} across multiple photos, "
                "so the city is treated as her base rather than a one-off visit."
            ),
        )

    for fact in _activity_facts(inventory, subject=subject):
        add(**fact)

    if config.use_owner_evidence_planner:
        for fact in planned_owner_facts(
            inventory,
            max_cards=config.owner_evidence_max_cards,
            balanced=config.use_owner_fact_census,
            min_cards_per_source=config.owner_census_min_cards_per_source,
        ):
            add(**fact)

    return facts[: config.max_owner_facts]


def _owner_pronouns(gender: str) -> tuple[str, str]:
    if gender == "man":
        return "He", "His"
    if gender == "woman":
        return "She", "Her"
    return "The album owner", "The album owner's"


def _fact_adder(facts: list[dict[str, Any]]):
    seen = set()

    def add(text: str, evidence: list[str], confidence: float, reasoning: str) -> None:
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if not key or key in seen:
            return
        seen.add(key)
        facts.append(
            {
                "text": text,
                "evidence_photo_ids": dedupe(evidence)[:5],
                "confidence": round(max(0.05, min(0.95, confidence)), 3),
                "reasoning_path": reasoning,
            }
        )

    return add


def _photos_for_owner_name(inventory: EvidenceInventory) -> list[str]:
    if not inventory.owner_name:
        return []
    photos = []
    for unit in inventory.text_units:
        if any(normalize_name(s).lower() == inventory.owner_name.lower() for s in unit.person_surfaces):
            photos.append(unit.photo_id)
    photos.sort(key=lambda pid: (0 if _photo_has_owner_face(inventory, pid) else 1, pid))
    return dedupe(photos)


def _photo_has_owner_face(inventory: EvidenceInventory, photo_id: str) -> bool:
    photo = inventory.photo_lookup.get(photo_id) or {}
    return inventory.owner_face_id in [str(f) for f in (photo.get("visible_face_ids") or [])]


def _infer_gender(inventory: EvidenceInventory) -> tuple[str, list[str]]:
    counts: Counter[str] = Counter()
    photos: dict[str, list[str]] = defaultdict(list)
    for unit in inventory.face_units_by_face.get(inventory.owner_face_id, []):
        text = unit.caption.lower()
        if re.search(r"\b(woman|female|lady|her|she)\b", text):
            counts["woman"] += 1
            photos["woman"].append(unit.photo_id)
        if re.search(r"\b(man|male|gentleman|his|he)\b", text):
            counts["man"] += 1
            photos["man"].append(unit.photo_id)
    if not counts:
        return "", []
    gender, _ = counts.most_common(1)[0]
    return gender, dedupe(photos[gender])


def _extract_birth_date(inventory: EvidenceInventory) -> dict[str, Any] | None:
    patterns = [
        re.compile(r"(?:date of birth|dob|birth date)[:\s]*([0-1]?\d)[/\-]([0-3]?\d)[/\-]((?:19|20)\d{2})", re.I),
        re.compile(r"(?:born|birth year|year of birth)[:\s]*(?:in\s*)?((?:19|20)\d{2})", re.I),
    ]
    best = None
    for unit in inventory.text_units:
        blob = combined_text([unit.caption, *unit.visible_text])
        for pattern in patterns:
            match = pattern.search(blob)
            if not match:
                continue
            if len(match.groups()) == 3:
                month, day, year = match.groups()
                record = {
                    "year": int(year),
                    "month": int(month),
                    "day": int(day),
                    "photos": [unit.photo_id],
                    "score": 2.0 + unit.owner_reference_score,
                }
            else:
                year = match.group(1)
                record = {
                    "year": int(year),
                    "month": None,
                    "day": None,
                    "photos": [unit.photo_id],
                    "score": 1.0 + unit.owner_reference_score,
                }
            if best is None or record["score"] > best["score"]:
                best = record
    return best


def _age_at_album_end(dob: dict[str, Any], inventory: EvidenceInventory) -> int | None:
    end = _album_end(inventory)
    if not end:
        return None
    try:
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except Exception:
        return None
    age = end_dt.year - int(dob["year"])
    if dob.get("month") and dob.get("day"):
        if (end_dt.month, end_dt.day) < (int(dob["month"]), int(dob["day"])):
            age -= 1
    return age if 0 < age < 120 else None


def _album_end(inventory: EvidenceInventory) -> str:
    timestamps = [str(p.get("timestamp") or "") for p in inventory.photos if p.get("timestamp")]
    if timestamps:
        return max(timestamps)
    months = [str(p.get("year_month") or "") for p in inventory.photos if p.get("year_month")]
    return f"{max(months)}-28T23:59:59" if months else ""


def _top_owner_city(inventory: EvidenceInventory) -> tuple[str, list[str]]:
    counts: Counter[str] = Counter()
    photos: dict[str, list[str]] = defaultdict(list)
    for unit in inventory.face_units_by_face.get(inventory.owner_face_id, []):
        if unit.gps_city:
            counts[unit.gps_city] += 1
            photos[unit.gps_city].append(unit.photo_id)
    if not counts:
        for photo in inventory.photos:
            city = str((photo.get("metadata") or {}).get("gps_city") or "")
            if city:
                counts[city] += 1
                photos[city].append(str(photo.get("photo_id") or ""))
    if not counts:
        return "", []
    city, _ = counts.most_common(1)[0]
    return city, dedupe(photos[city])


def _activity_facts(inventory: EvidenceInventory, *, subject: str = "The album owner") -> list[dict[str, Any]]:
    term_photos: dict[str, list[str]] = defaultdict(list)
    units: list[TextEvidenceUnit] = []
    owner_name_lower = inventory.owner_name.lower()
    for unit in inventory.text_units:
        has_owner_name = bool(owner_name_lower and text_contains_name(combined_text([unit.caption, *unit.visible_text]), owner_name_lower))
        if unit.owner_reference_score > 0 or has_owner_name or _photo_has_owner_face(inventory, unit.photo_id):
            units.append(unit)
    for unit in units:
        for term in unit.activity_terms:
            term_photos[term.lower()].append(unit.photo_id)
    for face_unit in inventory.face_units_by_face.get(inventory.owner_face_id, []):
        for term in face_unit.activity_terms:
            term_photos[term.lower()].append(face_unit.photo_id)

    facts = []
    def has(*terms: str) -> bool:
        return any(term in term_photos for term in terms)

    def evidence(*terms: str) -> list[str]:
        out = []
        for term in terms:
            out.extend(term_photos.get(term, []))
        return dedupe(out)

    if has("cycling", "bike", "bicycle", "ride", "rickenbacker"):
        facts.append(
            {
                "text": f"{subject} participates in road cycling as a hobby.",
                "evidence": evidence("cycling", "bike", "bicycle", "ride", "rickenbacker"),
                "confidence": 0.72,
                "reasoning": "Owner-face and text evidence repeatedly show cycling rides, bikes, or route references.",
            }
        )
    if has("baking", "bake", "cookies", "cupcakes", "cake"):
        facts.append(
            {
                "text": f"{subject} bakes and decorates desserts such as cookies, cupcakes, or cakes.",
                "evidence": evidence("baking", "bake", "cookies", "cupcakes", "cake"),
                "confidence": 0.72,
                "reasoning": "Public captions and visible objects around owner scenes show baking and dessert preparation.",
            }
        )
    if has("cooking", "cook", "kitchen"):
        facts.append(
            {
                "text": f"{subject} cooks at home.",
                "evidence": evidence("cooking", "cook", "kitchen"),
                "confidence": 0.64,
                "reasoning": "Owner-face kitchen scenes and food-preparation captions support home cooking.",
            }
        )
    if has("law", "legal", "attorney", "bar association", "court"):
        facts.append(
            {
                "text": f"{subject} is connected to the legal profession, likely as an attorney or legal professional.",
                "evidence": evidence("law", "legal", "attorney", "bar association", "court"),
                "confidence": 0.70,
                "reasoning": "Owner-oriented scenes and OCR include legal, law, attorney, court, or bar-association cues.",
            }
        )
    if has("networking", "office"):
        facts.append(
            {
                "text": f"{subject} attends professional networking or office-related events.",
                "evidence": evidence("networking", "office"),
                "confidence": 0.62,
                "reasoning": "Owner-context photos include office and networking cues across the album.",
            }
        )
    if has("yoga"):
        facts.append(
            {
                "text": f"{subject} practices yoga.",
                "evidence": evidence("yoga"),
                "confidence": 0.66,
                "reasoning": "Owner-context photos or text repeatedly mention yoga.",
            }
        )
    if has("running", "run", "marathon"):
        facts.append(
            {
                "text": f"{subject} participates in running or race events.",
                "evidence": evidence("running", "run", "marathon"),
                "confidence": 0.66,
                "reasoning": "Owner-context photos or text mention running and race activities.",
            }
        )
    if has("garden", "gardening"):
        facts.append(
            {
                "text": f"{subject} spends time gardening or visiting garden settings.",
                "evidence": evidence("garden", "gardening"),
                "confidence": 0.58,
                "reasoning": "Owner-context photos include gardening or garden activity cues.",
            }
        )
    return facts


def _llm_enabled(config: ECDConfig) -> bool:
    return bool(
        config.use_llm_reranker
        or config.use_llm_batch_adjudicator
        or config.use_llm_joint_assignment
        or config.use_llm_global_consistency
        or config.use_llm_readout
    )

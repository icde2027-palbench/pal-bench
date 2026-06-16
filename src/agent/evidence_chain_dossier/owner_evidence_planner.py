"""Owner evidence planning for Evidence-Chain Dossier readout.

The planner converts public album evidence into typed owner-profile evidence
cards.  It is intentionally deterministic: LLM readout can compose better text
from these cards, while no-LLM readout can still emit useful grounded facts.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .schemas import EvidenceInventory, FaceEvidenceUnit, TextEvidenceUnit
from .text import combined_text, dedupe, text_contains_name


@dataclass(frozen=True)
class OwnerFactSpec:
    fact_type: str
    text: str
    terms: tuple[str, ...]
    min_photos: int = 2
    min_score: float = 3.0
    confidence: float = 0.62
    source: str = "scene_accumulation"


OWNER_FACT_SPECS: tuple[OwnerFactSpec, ...] = (
    OwnerFactSpec(
        "music_choir",
        "The album owner participates in choir, singing, or vocal music activities.",
        ("choir", "rehearsal", "singing", "vocal", "lyrics", "songs", "music stand", "microphone", "piano"),
        min_photos=2,
        min_score=3.2,
        confidence=0.70,
        source="cross_modal_synthesis",
    ),
    OwnerFactSpec(
        "home_music_setup",
        "The album owner's home includes music equipment such as a microphone, music stand, lyrics, or song binders.",
        ("microphone", "music stand", "lyrics", "song binder", "songs", "home recording", "piano", "guitar"),
        min_photos=2,
        min_score=3.1,
        confidence=0.68,
        source="repeated_visual_pattern",
    ),
    OwnerFactSpec(
        "home_cooking",
        "The album owner regularly prepares meals at home.",
        ("kitchen", "cooking", "cook", "meal", "breakfast", "lunch", "dinner", "stove", "pan", "pot", "eggs", "rice", "beans", "toast"),
        min_photos=3,
        min_score=3.4,
        confidence=0.66,
        source="temporal_pattern",
    ),
    OwnerFactSpec(
        "coffee_routine",
        "Coffee is part of the album owner's regular home routine.",
        ("coffee", "mug", "drip coffee", "espresso", "kettle", "grounds", "brew", "chai"),
        min_photos=3,
        min_score=3.4,
        confidence=0.66,
        source="temporal_pattern",
    ),
    OwnerFactSpec(
        "home_environment",
        "The album owner's home includes a lived-in living space with recurring personal decor.",
        ("apartment", "living room", "sofa", "couch", "family photos", "framed photos", "gallery wall", "home office", "desk"),
        min_photos=3,
        min_score=3.4,
        confidence=0.63,
        source="scene_accumulation",
    ),
    OwnerFactSpec(
        "work_consulting_office",
        "The album owner's work involves consulting, office tasks, or professional check-ins.",
        ("consulting", "consultant", "office", "work stuff", "zoom", "client", "check-in", "meeting", "presentation", "task"),
        min_photos=1,
        min_score=2.4,
        confidence=0.66,
        source="cross_modal_synthesis",
    ),
    OwnerFactSpec(
        "work_legal",
        "The album owner is connected to legal, court, or attorney-related work.",
        ("legal", "law", "attorney", "court", "bar association", "case file", "paralegal"),
        min_photos=1,
        min_score=2.2,
        confidence=0.69,
        source="direct_ocr",
    ),
    OwnerFactSpec(
        "work_medical",
        "The album owner is connected to medical, clinic, or healthcare work.",
        ("nurse", "doctor", "healthcare worker", "medical badge", "hospital staff", "scrubs", "medicine rounds"),
        min_photos=1,
        min_score=2.4,
        confidence=0.64,
        source="cross_modal_synthesis",
    ),
    OwnerFactSpec(
        "work_education",
        "The album owner is connected to education, school, college, or teaching contexts.",
        ("college", "university", "teacher", "classroom", "undergraduate", "campus", "student id", "course schedule"),
        min_photos=1,
        min_score=2.4,
        confidence=0.64,
        source="cross_modal_synthesis",
    ),
    OwnerFactSpec(
        "education_degree",
        "The album owner has education or certification evidence in the album.",
        ("degree", "bachelor", "bachelor's", "master", "master's", "certificate", "certification", "diploma", "transcript", "alumni"),
        min_photos=1,
        min_score=2.2,
        confidence=0.67,
        source="direct_ocr",
    ),
    OwnerFactSpec(
        "hair_salon_work",
        "The album owner is connected to hair, salon, barbering, or styling work.",
        ("salon", "haircut", "hairstylist", "stylist station", "barber", "wahl", "clippers", "shears", "cosmetology"),
        min_photos=1,
        min_score=2.4,
        confidence=0.67,
        source="direct_ocr",
    ),
    OwnerFactSpec(
        "warehouse_logistics_work",
        "The album owner's work involves warehouse, logistics, inventory, or safety operations.",
        ("warehouse", "inventory", "logistics", "forklift", "barcode scanner", "clipboard", "safety audit", "supply chain"),
        min_photos=1,
        min_score=2.4,
        confidence=0.66,
        source="cross_modal_synthesis",
    ),
    OwnerFactSpec(
        "running",
        "The album owner has a recurring running or race routine.",
        ("running", "run", "race", "marathon", "jog", "jogging", "trail run"),
        min_photos=2,
        min_score=3.0,
        confidence=0.66,
        source="temporal_pattern",
    ),
    OwnerFactSpec(
        "cycling",
        "The album owner participates in cycling or bike rides.",
        ("cycling", "bike", "bicycle", "ride", "cycling route", "helmet"),
        min_photos=2,
        min_score=3.0,
        confidence=0.68,
        source="cross_modal_synthesis",
    ),
    OwnerFactSpec(
        "hiking_outdoors",
        "The album owner regularly spends leisure time hiking or outdoors.",
        ("hiking", "hike", "trail", "mountains", "park", "outdoor", "reservoir", "beach", "waterfront"),
        min_photos=2,
        min_score=3.2,
        confidence=0.62,
        source="scene_accumulation",
    ),
    OwnerFactSpec(
        "fishing",
        "Fishing appears as a recurring hobby for the album owner.",
        ("fishing", "fish", "rod", "tackle", "reservoir", "canal"),
        min_photos=2,
        min_score=3.0,
        confidence=0.67,
        source="temporal_pattern",
    ),
    OwnerFactSpec(
        "sewing_crafts",
        "The album owner does sewing, fabric craft, or handmade craft projects.",
        ("sewing", "fabric", "craft", "craft room", "singer", "macrame", "knitting", "crochet", "pegboard"),
        min_photos=2,
        min_score=3.0,
        confidence=0.67,
        source="repeated_visual_pattern",
    ),
    OwnerFactSpec(
        "gardening",
        "The album owner spends time gardening or maintaining plants.",
        ("garden", "gardening", "plants", "potted plants", "herbs", "balcony garden"),
        min_photos=2,
        min_score=3.0,
        confidence=0.61,
        source="scene_accumulation",
    ),
    OwnerFactSpec(
        "photography",
        "Photography or repeated photo outings appear as an owner hobby.",
        ("photography", "camera", "street scenes", "photo walk", "architecture", "gallery"),
        min_photos=2,
        min_score=3.0,
        confidence=0.60,
        source="scene_accumulation",
    ),
    OwnerFactSpec(
        "volunteer_community",
        "The album owner participates in community, volunteer, or outreach activities.",
        ("volunteer", "community", "outreach", "resource table", "neighborhood", "cleanup", "community centre", "community center"),
        min_photos=2,
        min_score=3.0,
        confidence=0.63,
        source="cross_modal_synthesis",
    ),
    OwnerFactSpec(
        "religious_church",
        "Church or faith-community activities recur in the album owner's routines.",
        ("church", "service", "faith", "parish", "choir rehearsal @ church"),
        min_photos=2,
        min_score=3.0,
        confidence=0.60,
        source="scene_accumulation",
    ),
    OwnerFactSpec(
        "public_transit",
        "Public transit or bus stops appear in the album owner's routine environment.",
        ("bus stop", "metro", "transit", "train", "station", "subway"),
        min_photos=2,
        min_score=3.1,
        confidence=0.58,
        source="metadata_inference",
    ),
    OwnerFactSpec(
        "driving_vehicle",
        "The album owner appears to rely on a recurring personal vehicle or driving routine.",
        ("car", "honda", "civic", "driver", "parking", "commuter", "drive", "driving"),
        min_photos=2,
        min_score=3.2,
        confidence=0.58,
        source="repeated_visual_pattern",
    ),
    OwnerFactSpec(
        "pet_owner",
        "A pet appears in the album owner's home or routine environment.",
        ("cat", "dog", "pet", "cat tree", "leash", "pet bowl"),
        min_photos=2,
        min_score=3.0,
        confidence=0.58,
        source="scene_accumulation",
    ),
    OwnerFactSpec(
        "smartphone_digital",
        "The album owner regularly uses a smartphone or digital scheduling tools.",
        ("smartphone", "phone screen", "reminders", "calendar", "notes", "text message", "status bar", "iphone", "android"),
        min_photos=2,
        min_score=3.1,
        confidence=0.61,
        source="cross_modal_synthesis",
    ),
    OwnerFactSpec(
        "casual_clothing",
        "The album owner often wears comfortable casual clothing in everyday scenes.",
        ("casual", "polo", "jeans", "zip-up", "jacket", "sneakers", "t-shirt", "flannel", "comfortable"),
        min_photos=3,
        min_score=3.4,
        confidence=0.57,
        source="repeated_visual_pattern",
    ),
    OwnerFactSpec(
        "business_casual_clothing",
        "The album owner's everyday wardrobe often fits a business-casual or polished professional style.",
        ("business casual", "professional outfit", "professional attire", "button-down", "collared shirt", "blouse", "slacks", "work outfit", "office attire"),
        min_photos=2,
        min_score=3.0,
        confidence=0.60,
        source="repeated_visual_pattern",
    ),
    OwnerFactSpec(
        "formal_professional_clothing",
        "The album owner regularly wears formal or professional clothing.",
        ("blazer", "suit", "tie", "dress shirt", "formal", "tailored", "heels", "court outfit", "professional clothing"),
        min_photos=2,
        min_score=3.0,
        confidence=0.61,
        source="repeated_visual_pattern",
    ),
    OwnerFactSpec(
        "small_group_social",
        "The album owner regularly spends time in small group social settings.",
        ("small group", "friends", "group", "gathered", "conversation", "social", "meetup", "shared meal"),
        min_photos=3,
        min_score=3.5,
        confidence=0.58,
        source="face_cooccurrence",
    ),
    OwnerFactSpec(
        "recurring_friend_group",
        "A recurring local friend group appears across the album owner's activities.",
        ("friend group", "same friends", "friends", "brunch", "birthday", "party", "celebration", "game night", "board game", "shared outing"),
        min_photos=3,
        min_score=3.4,
        confidence=0.60,
        source="face_cooccurrence",
    ),
    OwnerFactSpec(
        "family_meals",
        "Family or holiday meals recur in the album owner's home and social routine.",
        ("family dinner", "family meal", "holiday meal", "thanksgiving", "mid-autumn", "birthday dinner", "dining table", "reunion", "shared meal"),
        min_photos=2,
        min_score=3.0,
        confidence=0.60,
        source="face_cooccurrence",
    ),
    OwnerFactSpec(
        "sports_family_support",
        "The album owner repeatedly attends sports or youth activity events.",
        ("basketball", "soccer", "baseball", "team", "game", "league", "tournament"),
        min_photos=2,
        min_score=3.0,
        confidence=0.60,
        source="repeated_visual_pattern",
    ),
    OwnerFactSpec(
        "sports_apparel",
        "Team-branded apparel or sports-event clothing appears repeatedly in the album owner's outings.",
        ("team-branded", "jersey", "team shirt", "cap", "baseball cap", "game day", "sports apparel"),
        min_photos=2,
        min_score=3.0,
        confidence=0.58,
        source="repeated_visual_pattern",
    ),
    OwnerFactSpec(
        "apartment_living",
        "The album owner's living environment includes apartment, flat, or distinctive home-structure details.",
        ("apartment", "flat", "condo", "terraced house", "victorian-era", "cornicing", "high ceilings", "sectional sofa"),
        min_photos=2,
        min_score=3.0,
        confidence=0.60,
        source="scene_accumulation",
    ),
    OwnerFactSpec(
        "home_decor_furnishings",
        "The album owner's home shows recurring decor, furnishings, or displayed objects.",
        ("gallery wall", "framed wall art", "framed photos", "wall art", "prints", "macrame", "calligraphy", "ceramic bowls", "side tables", "sofa"),
        min_photos=2,
        min_score=3.0,
        confidence=0.60,
        source="scene_accumulation",
    ),
    OwnerFactSpec(
        "home_workstation",
        "The album owner's home includes a recurring desk, monitor, or workstation setup.",
        ("home office", "desk", "workstation", "standing desk", "external monitor", "large monitor", "laptop setup", "task board"),
        min_photos=2,
        min_score=3.0,
        confidence=0.61,
        source="repeated_visual_pattern",
    ),
    OwnerFactSpec(
        "home_baking_desserts",
        "Baking, desserts, or homemade baked goods recur in the album owner's activities.",
        ("baking", "baked goods", "cake", "cakes", "cookies", "cupcakes", "brownies", "muffins", "pastry", "dessert table", "homemade dessert"),
        min_photos=2,
        min_score=3.0,
        confidence=0.61,
        source="temporal_pattern",
    ),
    OwnerFactSpec(
        "restaurant_cafe_routine",
        "Restaurants, cafes, or dining-out settings recur in the album owner's routine.",
        ("restaurant", "korean bbq", "bbq restaurant", "cafe", "coffee shop", "brunch", "takeout", "menu", "dining out"),
        min_photos=2,
        min_score=3.0,
        confidence=0.58,
        source="metadata_inference",
    ),
    OwnerFactSpec(
        "coffee_shop_work",
        "Coffee shops or laptop-based out-of-home work settings appear in the album owner's routine.",
        ("coffee shop", "cafe work", "laptop work", "work session", "coworking", "remote work", "task-tracking"),
        min_photos=2,
        min_score=3.0,
        confidence=0.59,
        source="cross_modal_synthesis",
    ),
    OwnerFactSpec(
        "professional_networking_events",
        "The album owner attends professional networking, conference, workshop, or association events.",
        ("networking", "mixer", "conference", "workshop", "event badge", "association", "bar association", "client reception", "registration table"),
        min_photos=1,
        min_score=2.4,
        confidence=0.62,
        source="cross_modal_synthesis",
    ),
    OwnerFactSpec(
        "organized_community_events",
        "Organized community, charity, outreach, or registration-based events appear in the album owner's routine.",
        ("volunteer", "outreach", "charity", "fundraiser", "registration", "rsvp", "sign-up", "community event", "resource table", "cleanup"),
        min_photos=2,
        min_score=3.0,
        confidence=0.61,
        source="cross_modal_synthesis",
    ),
    OwnerFactSpec(
        "student_academic_terms",
        "The album owner appears connected to ongoing coursework, graduate study, or academic terms.",
        ("phd", "doctoral", "graduate student", "course schedule", "semester", "class notes", "campus", "student id", "thesis"),
        min_photos=1,
        min_score=2.4,
        confidence=0.61,
        source="direct_ocr",
    ),
    OwnerFactSpec(
        "social_media_business_account",
        "A social-media, portfolio, or business account is part of the album owner's public work evidence.",
        ("instagram", "business account", "portfolio", "followers", "profile page", "before and after", "transformation photos", "social media"),
        min_photos=1,
        min_score=2.4,
        confidence=0.61,
        source="direct_ocr",
    ),
    OwnerFactSpec(
        "specialized_work_tools",
        "Specialized tools, software, or equipment recur in the album owner's work routine.",
        ("barcode scanner", "clipboard", "inventory walk", "safety audit", "lightroom", "styling tools", "shears", "clippers", "projector", "presentation tools"),
        min_photos=1,
        min_score=2.4,
        confidence=0.61,
        source="repeated_visual_pattern",
    ),
    OwnerFactSpec(
        "outdoor_walks_recreation",
        "Local walks, parks, trails, or light outdoor recreation recur in the album owner's leisure routine.",
        ("walk", "walking", "neighborhood walk", "park", "trail", "light hike", "day trip", "outdoor recreation", "morning walk"),
        min_photos=3,
        min_score=3.4,
        confidence=0.58,
        source="scene_accumulation",
    ),
    OwnerFactSpec(
        "waterside_leisure",
        "Waterfront, beach, seaside, surfing, or waterside places recur in the album owner's leisure photos.",
        ("waterfront", "beach", "seaside", "surf", "surfing", "marina", "lake", "reservoir", "canal", "river"),
        min_photos=2,
        min_score=3.0,
        confidence=0.60,
        source="scene_accumulation",
    ),
    OwnerFactSpec(
        "daypack_water_bottle",
        "The album owner often carries a small bag, daypack, or reusable water bottle during outings.",
        ("daypack", "backpack", "water bottle", "reusable bottle", "hydration", "tote bag", "small bag"),
        min_photos=2,
        min_score=3.0,
        confidence=0.57,
        source="repeated_visual_pattern",
    ),
    OwnerFactSpec(
        "board_games_home_social",
        "Board games or in-person game sessions appear in the album owner's social routine.",
        ("board game", "board games", "game night", "tabletop", "card game", "dice", "strategy game"),
        min_photos=2,
        min_score=3.0,
        confidence=0.60,
        source="face_cooccurrence",
    ),
    OwnerFactSpec(
        "collections_memorabilia",
        "Collections, vintage objects, memorabilia, or rotating displays appear in the album owner's environment.",
        ("collection", "collects", "vintage", "memorabilia", "movie posters", "old tools", "vintage signs", "rotating display", "display shelf"),
        min_photos=2,
        min_score=3.0,
        confidence=0.59,
        source="scene_accumulation",
    ),
    OwnerFactSpec(
        "cinema_gallery_culture",
        "Cinema, galleries, museums, or cultural venues recur in the album owner's leisure activities.",
        ("cinema", "film", "movie", "movie poster", "gallery", "art gallery", "museum", "theater", "theatre", "lightbox"),
        min_photos=2,
        min_score=3.0,
        confidence=0.59,
        source="scene_accumulation",
    ),
    OwnerFactSpec(
        "grocery_household_shopping",
        "Grocery, household, or home-goods shopping recurs in the album owner's routine.",
        ("grocery", "groceries", "asian grocery", "home goods", "home decor", "shopping bag", "packaged coffee", "household items"),
        min_photos=2,
        min_score=3.0,
        confidence=0.58,
        source="metadata_inference",
    ),
    OwnerFactSpec(
        "cultural_holiday_home",
        "Cultural or festive foods, decor, or holiday traditions appear in the album owner's home life.",
        ("mid-autumn", "moon cake", "mooncake", "lantern", "calligraphy", "holiday decor", "cultural festival", "festival foods"),
        min_photos=1,
        min_score=2.4,
        confidence=0.59,
        source="cross_modal_synthesis",
    ),
)

OWNER_CENSUS_SOURCE_ORDER: tuple[str, ...] = (
    "face_cooccurrence",
    "repeated_visual_pattern",
    "temporal_pattern",
    "scene_accumulation",
    "cross_modal_synthesis",
    "metadata_inference",
    "direct_ocr",
)

BASE_OWNER_FACT_TYPES = frozenset(
    {
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
)


def build_owner_evidence_plan(
    inventory: EvidenceInventory,
    *,
    max_cards: int = 28,
    balanced: bool = False,
    min_cards_per_source: int = 2,
) -> list[dict[str, Any]]:
    """Return ranked owner evidence cards from public album evidence."""
    text_rows = _scored_text_rows(inventory)
    face_rows = _scored_face_rows(inventory)
    cards = []
    for spec in OWNER_FACT_SPECS:
        if not balanced and spec.fact_type not in BASE_OWNER_FACT_TYPES:
            continue
        card = _build_card(spec, text_rows, face_rows)
        if not card:
            continue
        cards.append(card)
    cards.sort(
        key=lambda row: (
            -float(row["score"]),
            -len(row["evidence_photo_ids"]),
            row["fact_type"],
        )
    )
    if balanced:
        return _balanced_cards(
            cards,
            max_cards=max_cards,
            min_cards_per_source=min_cards_per_source,
        )
    return cards[:max_cards]


def planned_owner_facts(
    inventory: EvidenceInventory,
    *,
    max_cards: int = 28,
    balanced: bool = False,
    min_cards_per_source: int = 2,
) -> list[dict[str, Any]]:
    """Convert evidence cards into evaluator-facing owner fact rows."""
    facts = []
    for card in build_owner_evidence_plan(
        inventory,
        max_cards=max_cards,
        balanced=balanced,
        min_cards_per_source=min_cards_per_source,
    ):
        evidence = [str(pid) for pid in card.get("evidence_photo_ids") or []]
        fact_type = str(card.get("fact_type") or "")
        matched_terms = [str(t) for t in card.get("matched_terms") or []][:8]
        facts.append(
            {
                "text": str(card.get("text") or ""),
                "evidence": evidence,
                "confidence": float(card.get("confidence") or 0.6),
                "reasoning": (
                    f"Evidence-planned {card.get('source', 'owner')} fact `{fact_type}` is supported by "
                    f"{len(evidence)} public photos. Matching cues include {', '.join(matched_terms) or 'recurring visual/text cues'}."
                ),
            }
        )
    return facts


def compact_owner_evidence_plan(
    inventory: EvidenceInventory,
    *,
    max_cards: int = 18,
    max_examples_per_card: int = 4,
    chars: int = 220,
    balanced: bool = False,
    min_cards_per_source: int = 2,
) -> list[dict[str, Any]]:
    """Small JSON-serializable plan for LLM prompts."""
    cards = build_owner_evidence_plan(
        inventory,
        max_cards=max_cards,
        balanced=balanced,
        min_cards_per_source=min_cards_per_source,
    )
    out = []
    for card in cards:
        examples = []
        for photo_id in card.get("evidence_photo_ids", [])[:max_examples_per_card]:
            photo = inventory.photo_lookup.get(str(photo_id)) or {}
            examples.append(_photo_excerpt(photo, chars=chars))
        out.append(
            {
                "fact_type": card["fact_type"],
                "source": card["source"],
                "candidate_fact": card["text"],
                "score": round(float(card["score"]), 3),
                "confidence": card["confidence"],
                "matched_terms": card["matched_terms"][:10],
                "evidence_photo_ids": card["evidence_photo_ids"][:6],
                "evidence_examples": examples,
            }
        )
    return out


def _balanced_cards(
    cards: list[dict[str, Any]],
    *,
    max_cards: int,
    min_cards_per_source: int,
) -> list[dict[str, Any]]:
    """Keep weaker but useful owner evidence sources visible to the readout LLM."""
    if max_cards <= 0:
        return []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        by_source[str(card.get("source") or "")].append(card)

    selected: list[dict[str, Any]] = []
    seen = set()

    def add(card: dict[str, Any]) -> None:
        fact_type = str(card.get("fact_type") or "")
        if not fact_type or fact_type in seen or len(selected) >= max_cards:
            return
        selected.append(card)
        seen.add(fact_type)

    source_order = [
        source
        for source in OWNER_CENSUS_SOURCE_ORDER
        if source in by_source
    ] + sorted(source for source in by_source if source not in OWNER_CENSUS_SOURCE_ORDER)
    quota = max(1, min_cards_per_source)
    for source in source_order:
        for card in by_source[source][:quota]:
            add(card)

    for card in cards:
        add(card)
    return selected[:max_cards]


def _build_card(
    spec: OwnerFactSpec,
    text_rows: list[dict[str, Any]],
    face_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    scored: dict[str, dict[str, Any]] = {}
    matched_terms = []
    for row in [*text_rows, *face_rows]:
        hits = _term_hits(row["text"], spec.terms)
        if not hits:
            continue
        photo_id = str(row["photo_id"])
        hit_weight = 0.9 + min(1.8, len(hits) * 0.35)
        score = float(row["owner_score"]) * hit_weight
        if row.get("has_owner_face"):
            score += 0.8
        if row.get("has_visible_text"):
            score += 0.35
        record = scored.setdefault(
            photo_id,
            {
                "photo_id": photo_id,
                "score": 0.0,
                "terms": set(),
                "owner_score": row["owner_score"],
            },
        )
        record["score"] = max(float(record["score"]), score)
        record["terms"].update(hits)
        matched_terms.extend(hits)

    rows = sorted(scored.values(), key=lambda row: (-float(row["score"]), row["photo_id"]))
    if len(rows) < spec.min_photos:
        return None
    total_score = sum(float(row["score"]) for row in rows[:8])
    if total_score < spec.min_score:
        return None
    evidence_ids = [str(row["photo_id"]) for row in rows[:8]]
    unique_terms = dedupe(str(term) for term in matched_terms)
    confidence = min(0.88, spec.confidence + min(0.12, len(evidence_ids) * 0.015))
    return {
        "fact_type": spec.fact_type,
        "source": spec.source,
        "text": spec.text,
        "score": round(total_score, 3),
        "confidence": round(confidence, 3),
        "matched_terms": unique_terms,
        "evidence_photo_ids": evidence_ids,
        "support": [
            {
                "photo_id": row["photo_id"],
                "score": round(float(row["score"]), 3),
                "terms": sorted(str(term) for term in row["terms"]),
            }
            for row in rows[:8]
        ],
    }


def _scored_text_rows(inventory: EvidenceInventory) -> list[dict[str, Any]]:
    owner_name = inventory.owner_name.lower()
    rows = []
    for unit in inventory.text_units:
        blob = _unit_text(unit)
        photo = inventory.photo_lookup.get(unit.photo_id) or {}
        metadata = photo.get("metadata") or {}
        owner_score = 0.65 + float(unit.owner_reference_score) * 0.9
        if owner_name and text_contains_name(blob, owner_name):
            owner_score += 1.2
        if _photo_has_owner(inventory, unit.photo_id):
            owner_score += 1.0
        if _home_like(str(metadata.get("gps_location") or "")):
            owner_score += 0.55
        if unit.visible_text:
            owner_score += 0.25
        if not unit.has_face and unit.visible_text:
            owner_score += 0.20
        rows.append(
            {
                "photo_id": unit.photo_id,
                "text": blob,
                "owner_score": owner_score,
                "has_owner_face": _photo_has_owner(inventory, unit.photo_id),
                "has_visible_text": bool(unit.visible_text),
            }
        )
    return rows


def _scored_face_rows(inventory: EvidenceInventory) -> list[dict[str, Any]]:
    rows = []
    owner_units = list(inventory.face_units_by_face.get(inventory.owner_face_id, []))
    for unit in owner_units:
        blob = _face_unit_text(unit)
        owner_score = 1.25
        if unit.visible_text:
            owner_score += 0.35
        if _home_like(unit.gps_location):
            owner_score += 0.35
        rows.append(
            {
                "photo_id": unit.photo_id,
                "text": blob,
                "owner_score": owner_score,
                "has_owner_face": True,
                "has_visible_text": bool(unit.visible_text),
            }
        )
    return rows


def _unit_text(unit: TextEvidenceUnit) -> str:
    return combined_text(
        [
            unit.caption,
            *unit.visible_text,
            *unit.person_surfaces,
            *unit.relation_terms,
            *unit.activity_terms,
            *unit.venue_terms,
            *unit.organization_terms,
            unit.gps_city,
            unit.gps_location,
        ]
    ).lower()


def _face_unit_text(unit: FaceEvidenceUnit) -> str:
    return combined_text(
        [
            unit.caption,
            *unit.visible_text,
            *unit.relation_terms,
            *unit.activity_terms,
            *unit.venue_terms,
            unit.gps_city,
            unit.gps_location,
        ]
    ).lower()


def _term_hits(text: str, terms: tuple[str, ...]) -> list[str]:
    hits = []
    lower = str(text or "").lower()
    for term in terms:
        term_lower = term.lower()
        if " " in term_lower:
            if term_lower in lower:
                hits.append(term)
        elif re.search(rf"\b{re.escape(term_lower)}\b", lower):
            hits.append(term)
    return hits


def _photo_has_owner(inventory: EvidenceInventory, photo_id: str) -> bool:
    photo = inventory.photo_lookup.get(photo_id) or {}
    return inventory.owner_face_id in [str(fid) for fid in photo.get("visible_face_ids") or []]


def _home_like(location: str) -> bool:
    lower = str(location or "").lower()
    return any(term in lower for term in ("home", "apartment", "house", "living room", "kitchen", "bedroom"))


def _photo_excerpt(photo: dict[str, Any], *, chars: int) -> dict[str, Any]:
    metadata = photo.get("metadata") or {}
    visible_text = [str(t) for t in photo.get("visible_text") or [] if str(t).strip()]
    entities = [
        {
            "surface": str(entity.get("surface") or ""),
            "type": str(entity.get("entity_type") or ""),
            "source": str(entity.get("source") or ""),
        }
        for entity in (photo.get("text_entities") or [])[:8]
    ]
    return {
        "photo_id": str(photo.get("photo_id") or ""),
        "year_month": str(photo.get("year_month") or ""),
        "gps_city": str(metadata.get("gps_city") or ""),
        "gps_location": _truncate(str(metadata.get("gps_location") or ""), 100),
        "faces": [str(fid) for fid in photo.get("visible_face_ids") or []],
        "caption": _truncate(str(photo.get("caption") or ""), chars),
        "visible_text": [_truncate(text, 120) for text in visible_text[:8]],
        "entities": entities,
    }


def _truncate(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."

"""Event and role bridge features for ECD identity recall."""

from __future__ import annotations

from typing import Any

from .text import tokens


EVENT_KEYWORDS = {
    "anniversary",
    "association",
    "badge",
    "bake",
    "baked",
    "bakery",
    "bar",
    "birthday",
    "breakfast",
    "brunch",
    "cafe",
    "cake",
    "campus",
    "candles",
    "card",
    "choir",
    "christmas",
    "church",
    "class",
    "classmate",
    "clinic",
    "coffee",
    "community",
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
    "music",
    "neighbor",
    "neighborhood",
    "networking",
    "office",
    "parent",
    "park",
    "party",
    "reception",
    "rehearsal",
    "reservation",
    "restaurant",
    "school",
    "study",
    "thanksgiving",
    "volunteer",
}


GENERIC_EVENT_KEYWORDS = {
    "bar",
    "cafe",
    "downtown",
    "friend",
    "home",
    "office",
    "park",
    "restaurant",
}


SPECIFIC_CHANNELS = {
    "birthday_social",
    "clinic_health",
    "neighbor_residential",
    "professional_legal",
    "school_class",
    "volunteer_community",
}


STOPWORDS = {
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


CHANNEL_TERMS: dict[str, set[str]] = {
    "birthday_social": {"birthday", "cake", "candles", "party"},
    "family_home": {
        "anniversary",
        "card",
        "christmas",
        "father",
        "holiday",
        "home",
        "kitchen",
        "mother",
        "parent",
        "thanksgiving",
    },
    "professional_legal": {
        "association",
        "badge",
        "conference",
        "court",
        "courthouse",
        "law",
        "legal",
        "meeting",
        "networking",
        "office",
        "reception",
    },
    "school_class": {"campus", "class", "classmate", "school", "study"},
    "neighbor_residential": {
        "community",
        "condo",
        "lobby",
        "neighbor",
        "neighborhood",
    },
    "clinic_health": {"clinic", "doctor", "hospital"},
    "volunteer_community": {"church", "community", "neighborhood", "volunteer"},
    "meal_social": {
        "bar",
        "breakfast",
        "brunch",
        "cafe",
        "coffee",
        "dinner",
        "reservation",
        "restaurant",
    },
    "hobby_activity": {
        "bake",
        "baked",
        "bakery",
        "choir",
        "cookies",
        "cycling",
        "garden",
        "marina",
        "music",
        "rehearsal",
    },
}


PHRASE_CHANNELS: dict[str, tuple[str, tuple[str, ...]]] = {
    "birthday cake": ("birthday_social", ("birthday", "cake")),
    "coffee cups": ("meal_social", ("coffee",)),
    "legal association": ("professional_legal", ("legal", "association")),
    "mother's day": ("family_home", ("mother", "card")),
    "mothers day": ("family_home", ("mother", "card")),
    "neighborhood cleanup": ("neighbor_residential", ("neighborhood", "volunteer")),
    "open table": ("meal_social", ("reservation", "restaurant")),
    "opentable": ("meal_social", ("reservation", "restaurant")),
    "text message": ("meal_social", tuple()),
}


RELATION_CHANNEL_TERMS: dict[str, set[str]] = {
    "family_home": {
        "aunt",
        "auntie",
        "brother",
        "cousin",
        "dad",
        "daughter",
        "father",
        "family",
        "husband",
        "mom",
        "mother",
        "parent",
        "parents",
        "partner",
        "sister",
        "son",
        "spouse",
        "uncle",
        "wife",
    },
    "professional_legal": {
        "associate",
        "attorney",
        "bar association",
        "colleague",
        "co-worker",
        "coworker",
        "law",
        "legal",
        "office",
    },
    "school_class": {"class", "classmate", "school", "teacher"},
    "neighbor_residential": {"neighbor", "neighbour", "apartment", "condo", "lobby"},
    "clinic_health": {"clinic", "doctor", "hospital", "nurse"},
    "volunteer_community": {"church", "community", "volunteer"},
    "meal_social": {"best friend", "brunch", "close friend", "dinner", "friend", "friends"},
    "hobby_activity": {"book club", "cycling", "garden", "gardening", "music", "photography"},
}


def photo_event_features(photo: dict[str, Any]) -> dict[str, Any]:
    """Return normalized event terms and broad social channels for a photo."""
    parts = [str(photo.get("caption") or "")]
    parts.extend(str(t) for t in (photo.get("visible_text") or []))
    parts.extend(str(e.get("surface") or "") for e in (photo.get("text_entities") or []))
    metadata = photo.get("metadata") or {}
    parts.append(str(metadata.get("gps_location") or ""))
    parts.append(str(metadata.get("gps_city") or ""))
    joined = " ".join(parts).lower()

    terms = {
        token
        for token in tokens(joined)
        if len(token) >= 4 and token not in STOPWORDS and token in EVENT_KEYWORDS
    }
    channels: set[str] = set()
    for phrase, (channel, phrase_terms) in PHRASE_CHANNELS.items():
        if phrase in joined:
            channels.add(channel)
            terms.update(phrase_terms)
    channels.update(social_channels_from_terms(terms))
    return {
        "terms": sorted(terms),
        "channels": sorted(channels),
    }


def social_channels_from_terms(terms: Any) -> set[str]:
    """Map relation/activity/venue terms to coarse social-event channels."""
    normalized = {str(term).lower() for term in (terms or []) if str(term).strip()}
    channels = {
        channel
        for channel, channel_terms in CHANNEL_TERMS.items()
        if normalized & channel_terms
    }
    for channel, channel_terms in RELATION_CHANNEL_TERMS.items():
        if normalized & channel_terms:
            channels.add(channel)
    return channels


def score_event_bridge(text_photo: dict[str, Any], face_photo: dict[str, Any]) -> dict[str, Any]:
    """Score whether two photos plausibly describe the same social event context."""
    return score_event_bridge_from_features(
        photo_event_features(text_photo),
        photo_event_features(face_photo),
        text_photo_has_faces=bool(text_photo.get("visible_face_ids")),
        face_photo_face_count=len(face_photo.get("visible_face_ids") or []),
        text_photo_person_entities=_person_entity_count(text_photo),
    )


def score_event_bridge_from_features(
    text_features: dict[str, Any],
    face_features: dict[str, Any],
    *,
    text_photo_has_faces: bool,
    face_photo_face_count: int,
    text_photo_person_entities: int,
) -> dict[str, Any]:
    """Score a photo bridge from precomputed event features."""
    text_terms = set(text_features["terms"])
    face_terms = set(face_features["terms"])
    text_channels = set(text_features["channels"])
    face_channels = set(face_features["channels"])

    shared_terms = sorted(text_terms & face_terms)
    shared_channels = sorted(text_channels & face_channels)
    strong_terms = [term for term in shared_terms if term not in GENERIC_EVENT_KEYWORDS]
    generic_terms = [term for term in shared_terms if term in GENERIC_EVENT_KEYWORDS]
    strong_signal = bool(strong_terms or (set(shared_channels) & SPECIFIC_CHANNELS))

    score = 0.0
    score += min(0.38, 0.12 * len(strong_terms) + 0.04 * len(generic_terms))
    score += min(0.42, 0.18 * len(shared_channels))
    if shared_channels and strong_terms:
        score += 0.08
    if shared_channels and not strong_signal:
        score *= 0.35
    if not text_photo_has_faces:
        score += 0.08
    else:
        score -= 0.04
    if text_photo_person_entities >= 2 and text_photo_has_faces:
        score -= 0.06
    if face_photo_face_count <= 2:
        score += 0.03

    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "shared_keywords": shared_terms[:12],
        "shared_channels": shared_channels[:8],
        "text_keywords": text_features["terms"][:12],
        "face_keywords": face_features["terms"][:12],
        "text_channels": text_features["channels"][:8],
        "face_channels": face_features["channels"][:8],
        "text_photo_has_faces": text_photo_has_faces,
        "strong_signal": strong_signal,
    }


def _person_entity_count(photo: dict[str, Any]) -> int:
    return sum(
        1
        for ent in (photo.get("text_entities") or [])
        if str(ent.get("entity_type") or "").lower() == "person"
    )

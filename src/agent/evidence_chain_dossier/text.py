"""Text normalization and lightweight semantic lexicons for ECD."""

from __future__ import annotations

import re
from collections.abc import Iterable

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*")

RELATION_MAP: dict[str, tuple[str, str]] = {
    "mom": ("mother", "family"),
    "mother": ("mother", "family"),
    "mother's day": ("mother", "family"),
    "mothers day": ("mother", "family"),
    "dad": ("father", "family"),
    "father": ("father", "family"),
    "father's day": ("father", "family"),
    "fathers day": ("father", "family"),
    "parents": ("parent", "family"),
    "parent": ("parent", "family"),
    "sister": ("sister", "family"),
    "brother": ("brother", "family"),
    "aunt": ("aunt", "family"),
    "auntie": ("aunt", "family"),
    "uncle": ("uncle", "family"),
    "cousin": ("cousin", "family"),
    "daughter": ("daughter", "family"),
    "son": ("son", "family"),
    "wife": ("partner", "family"),
    "husband": ("partner", "family"),
    "spouse": ("partner", "family"),
    "partner": ("partner", "family"),
    "boyfriend": ("partner", "family"),
    "girlfriend": ("partner", "family"),
    "date night": ("partner", "family"),
    "friend": ("friend", "friend"),
    "friends": ("friend", "friend"),
    "close friend": ("close friend", "friend"),
    "best friend": ("best friend", "friend"),
    "coworker": ("coworker", "colleague"),
    "co-worker": ("coworker", "colleague"),
    "colleague": ("colleague", "colleague"),
    "associate": ("colleague", "colleague"),
    "classmate": ("classmate", "classmate"),
    "neighbor": ("neighbor", "neighbor"),
    "neighbour": ("neighbor", "neighbor"),
}

RELATION_PREFIXES = {
    "aunt",
    "auntie",
    "uncle",
    "cousin",
    "mom",
    "mother",
    "dad",
    "father",
    "grandma",
    "grandpa",
    "grandmother",
    "grandfather",
    "nana",
    "papa",
    "mama",
    "tia",
    "tio",
    "zia",
    "zio",
}

ORG_NOISE = {
    "facebook",
    "instagram",
    "twitter",
    "youtube",
    "google",
    "apple",
    "kitchenaid",
    "starbucks",
    "costco",
    "target",
    "walmart",
}

NON_NAME_WORDS = {
    "ride",
    "run",
    "walk",
    "park",
    "street",
    "road",
    "cafe",
    "restaurant",
    "bar",
    "club",
    "center",
    "centre",
    "market",
    "hotel",
    "salad",
    "recipe",
    "kitchen",
    "office",
    "school",
    "church",
    "hospital",
    "clinic",
    "beach",
    "marina",
    "garden",
    "gardens",
    "festival",
    "meeting",
    "meetup",
    "conference",
    "association",
}

ACTIVITY_KEYWORDS = {
    "cycling",
    "bike",
    "bicycle",
    "ride",
    "rickenbacker",
    "run",
    "running",
    "marathon",
    "yoga",
    "hiking",
    "hike",
    "baking",
    "bake",
    "cookies",
    "cupcakes",
    "cake",
    "cooking",
    "cook",
    "kitchen",
    "law",
    "legal",
    "attorney",
    "bar association",
    "court",
    "office",
    "networking",
    "hospital",
    "clinic",
    "nurse",
    "doctor",
    "school",
    "teacher",
    "class",
    "church",
    "volunteer",
    "book club",
    "garden",
    "gardening",
    "photography",
    "piano",
    "music",
    "brunch",
    "dinner",
    "birthday",
    "thanksgiving",
    "christmas",
    "holiday",
}

VENUE_KEYWORDS = {
    "home",
    "apartment",
    "house",
    "office",
    "downtown",
    "marina",
    "beach",
    "cafe",
    "restaurant",
    "church",
    "school",
    "hospital",
    "clinic",
    "park",
    "garden",
    "gardens",
    "patio",
    "lobby",
    "kitchen",
    "living room",
}

OWNER_REFERENCE_TERMS = {
    "profile": 2.0,
    "account": 2.0,
    "date of birth": 2.5,
    "dob": 2.5,
    "birth": 1.5,
    "address": 1.5,
    "mailing": 1.2,
    "license": 1.5,
    "resume": 1.8,
    "appointment": 1.0,
    "confirmation": 0.6,
    "my": 0.5,
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_name(name: str) -> str:
    """Normalize a name surface while preserving useful surname particles."""
    value = normalize_space(name)
    value = re.sub(r"^[#@\-\s:]+", "", value)
    value = re.sub(r"\s+\([^)]+\)$", "", value)
    value = value.strip(" ,.;:!?")
    if not value:
        return ""
    tokens = []
    for raw in value.split():
        if raw.lower() in {"de", "del", "dela", "de la", "van", "von", "da", "di"}:
            tokens.append(raw.lower())
        elif re.fullmatch(r"[A-Za-z]\.", raw):
            tokens.append(raw.upper())
        else:
            pieces = re.split(r"([\-'])", raw)
            tokens.append("".join(p.capitalize() if p.isalpha() else p for p in pieces))
    return normalize_space(" ".join(tokens))


def strip_relation_prefix(name: str) -> tuple[str, list[str]]:
    normalized = normalize_name(name)
    parts = normalized.split()
    if len(parts) >= 2 and parts[0].lower().strip(".") in RELATION_PREFIXES:
        prefix = parts[0].lower().strip(".")
        relation = RELATION_MAP.get(prefix, (prefix, "family"))[0]
        return normalize_name(" ".join(parts[1:])), [relation]
    return normalized, []


def looks_like_person_name(name: str) -> bool:
    normalized = normalize_name(name)
    if not normalized:
        return False
    if "_" in normalized or "@" in normalized or "#" in normalized:
        return False
    lower = normalized.lower()
    if lower in ORG_NOISE or lower in RELATION_MAP:
        return False
    if any(ch.isdigit() for ch in normalized):
        return False
    if "&" in normalized or "/" in normalized:
        return False
    if normalized.isupper() and len(normalized) > 6:
        return False
    parts = normalized.split()
    if not parts:
        return False
    if parts[0].lower() in {"the", "new", "old"}:
        return False
    for word in parts[1:]:
        if word.lower().strip(".,") in NON_NAME_WORDS:
            return False
    if len(parts) == 1:
        return len(parts[0]) >= 2 and lower not in NON_NAME_WORDS
    return True


def first_name(name: str) -> str:
    parts = normalize_name(name).split()
    return parts[0] if parts else ""


def last_name(name: str) -> str:
    parts = normalize_name(name).split()
    if len(parts) < 2:
        return ""
    return parts[-1].strip(".")


def is_full_name(name: str) -> bool:
    return len(normalize_name(name).split()) >= 2


def tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text or "")}


def combined_text(parts: Iterable[str]) -> str:
    return "\n".join(str(p or "") for p in parts if str(p or "").strip())


def extract_relation_terms(text: str) -> list[str]:
    lower = normalize_space(text).lower()
    terms = []
    for key in sorted(RELATION_MAP, key=len, reverse=True):
        if " " in key:
            if key in lower:
                terms.append(key)
        elif re.search(rf"\b{re.escape(key)}\b", lower):
            terms.append(key)
    return dedupe(terms)


def relation_value(term: str) -> tuple[str, str]:
    return RELATION_MAP.get(term.lower(), ("", ""))


def extract_activity_terms(text: str) -> list[str]:
    lower = normalize_space(text).lower()
    found = []
    for key in sorted(ACTIVITY_KEYWORDS, key=len, reverse=True):
        if " " in key:
            if key in lower:
                found.append(key)
        elif re.search(rf"\b{re.escape(key)}\b", lower):
            found.append(key)
    return dedupe(found)


def extract_venue_terms(text: str) -> list[str]:
    lower = normalize_space(text).lower()
    found = []
    for key in sorted(VENUE_KEYWORDS, key=len, reverse=True):
        if " " in key:
            if key in lower:
                found.append(key)
        elif re.search(rf"\b{re.escape(key)}\b", lower):
            found.append(key)
    return dedupe(found)


def owner_reference_score(text: str) -> float:
    lower = normalize_space(text).lower()
    score = 0.0
    for key, weight in OWNER_REFERENCE_TERMS.items():
        if key in lower:
            score += weight
    return min(score, 6.0)


def norm_key(value: str) -> str:
    value = normalize_space(value).lower()
    value = value.replace("—", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return normalize_space(value)


def dedupe(items: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        value = normalize_space(item)
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def text_contains_name(text: str, name: str) -> bool:
    surface = normalize_name(name)
    if not surface:
        return False
    lower = normalize_space(text).lower()
    full = re.escape(surface.lower())
    if re.search(rf"\b{full}\b", lower):
        return True
    first = first_name(surface)
    return bool(first and re.search(rf"\b{re.escape(first.lower())}\b", lower))

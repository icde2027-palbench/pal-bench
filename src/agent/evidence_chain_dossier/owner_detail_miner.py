"""LLM owner detail mining for ECD profiles.

The miner is a bounded readout-side pass. It never changes identities; it only
adds evidence-cited owner facts from high-information owner-related photos.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from .config import ECDConfig
from .llm_context import photo_brief
from .schemas import EvidenceInventory
from .text import dedupe

_SYSTEM = """You are the Owner Detail Miner for Evidence-Chain Dossier.
Use only the supplied public album evidence to add missing owner facts.

Rules:
- Extract atomic, evaluator-friendly owner facts.
- Focus on specific details: work/education, age or location, recurring routines, hobbies, venues, tools, brands, vehicles, home/living-space details, languages, schedules, and social/family patterns.
- A fact must be about the album owner, not merely another person or the album in general.
- Every fact must cite 1-4 evidence_photo_ids from the supplied photos.
- Avoid duplicates or near-duplicates of existing facts.
- Do not extract exact contact details such as email addresses, phone numbers, or street addresses; use city/base-level wording instead.
- Do not enumerate named friends, relatives, or partners as owner facts. Those belong in person profiles unless the fact is a broad social pattern.
- Do not write disjunctive facts with "or"; split only when each atomic fact is separately supported.
- Do not invent facts, names, or photo ids. Abstain when evidence is generic.
- Prefer concrete wording: "She uses a Singer sewing machine" rather than "She likes crafts".
- Output ONLY JSON:
{"facts":[{"text":"...","evidence_photo_ids":["photo_..."],"confidence":0.0,"reasoning_path":"..."}],"abstentions":[{"photo_id":"photo_...","reason":"..."}]}"""


def mine_owner_details(
    *,
    profile: dict[str, Any],
    inventory: EvidenceInventory,
    config: ECDConfig,
    llm: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append high-signal owner detail facts using a small LLM budget."""
    if llm is None:
        return profile, {"enabled": False, "reason": "no_llm"}
    if config.owner_detail_miner_max_calls <= 0 or config.owner_detail_miner_max_new_facts <= 0:
        return profile, {"enabled": False, "reason": "owner_detail_budget_zero"}

    owner = profile.setdefault("owner", {})
    existing = list(owner.get("facts") or [])
    max_total = max(config.max_owner_facts, int(config.owner_addendum_max_total_facts))
    slots = max(0, min(int(config.owner_detail_miner_max_new_facts), max_total - len(existing)))
    if slots <= 0:
        return profile, {"enabled": True, "n_added": 0, "reason": "no_slots"}

    photo_ids = _select_owner_detail_photos(
        inventory,
        limit=max(1, config.owner_detail_miner_max_calls) * max(1, config.owner_detail_miner_photos_per_call),
    )
    batches = [
        photo_ids[i : i + max(1, config.owner_detail_miner_photos_per_call)]
        for i in range(0, len(photo_ids), max(1, config.owner_detail_miner_photos_per_call))
    ][: config.owner_detail_miner_max_calls]
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "n_candidate_photos": len(photo_ids),
        "batches": {},
        "errors": [],
    }

    facts = list(existing)
    seen = {_fact_signature(str(fact.get("text") or "")) for fact in facts}
    added: list[dict[str, Any]] = []
    for idx, batch in enumerate(batches, start=1):
        if len(added) >= slots:
            break
        batch_id = f"owner_detail_batch_{idx:02d}"
        payload = _build_payload(profile, inventory, batch, config, remaining=slots - len(added))
        result = llm.call_json(
            prompt=json.dumps(payload, ensure_ascii=False),
            system=_SYSTEM,
            temperature=0.1,
            max_tokens=6144,
            retries=1,
            stage="owner_detail_miner",
        )
        if not isinstance(result, dict):
            diagnostics["errors"].append({"batch_id": batch_id, "error": "no_json"})
            diagnostics["batches"][batch_id] = {"photos": batch, "error": "no_json"}
            continue
        cleaned, rejected = _clean_new_facts(
            result.get("facts") or [],
            allowed_photo_ids=set(batch),
            seen=seen,
            config=config,
            limit=slots - len(added),
        )
        for fact in cleaned:
            facts.append(fact)
            added.append(fact)
            seen.add(_fact_signature(str(fact.get("text") or "")))
        diagnostics["batches"][batch_id] = {
            "photos": batch,
            "raw_result": result,
            "accepted": cleaned,
            "rejected": rejected[:50],
        }

    owner["facts"] = facts[:max_total]
    diagnostics["n_added"] = len(added)
    diagnostics["added"] = added
    return profile, diagnostics


def _build_payload(
    profile: dict[str, Any],
    inventory: EvidenceInventory,
    photo_ids: list[str],
    config: ECDConfig,
    *,
    remaining: int,
) -> dict[str, Any]:
    existing_facts = [
        str(fact.get("text") or "").strip()
        for fact in (profile.get("owner") or {}).get("facts") or []
        if str(fact.get("text") or "").strip()
    ]
    return {
        "user_id": inventory.album.get("user_id"),
        "owner": {
            "face_id": inventory.owner_face_id,
            "name_candidate": inventory.owner_name,
            "last_name": inventory.owner_last_name,
        },
        "existing_owner_facts": existing_facts[:80],
        "max_new_facts": max(1, remaining),
        "evidence_photos": [
            photo_brief(inventory, photo_id, chars=min(max(config.llm_photo_snippet_chars, 240), 360))
            for photo_id in photo_ids
        ],
    }


def _select_owner_detail_photos(inventory: EvidenceInventory, *, limit: int) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    reasons: dict[str, set[str]] = defaultdict(set)
    owner_face = inventory.owner_face_id

    for unit in inventory.text_units:
        photo_id = str(unit.photo_id)
        score = 0.55 + float(unit.owner_reference_score) * 1.6
        if _photo_has_owner(inventory, photo_id):
            score += 1.25
            reasons[photo_id].add("owner_face")
        if unit.visible_text:
            score += 0.45 + min(1.2, len(unit.visible_text) * 0.12)
            reasons[photo_id].add("visible_text")
        if unit.organization_terms:
            score += 0.35
            reasons[photo_id].add("organization")
        if unit.activity_terms:
            score += 0.30
            reasons[photo_id].add("activity")
        if unit.venue_terms:
            score += 0.25
            reasons[photo_id].add("venue")
        if unit.person_surfaces:
            score += 0.15
        if not unit.has_face and unit.visible_text:
            score += 0.35
            reasons[photo_id].add("text_only_document")
        scores[photo_id] = max(scores[photo_id], score)

    for unit in inventory.face_units_by_face.get(owner_face, []):
        photo_id = str(unit.photo_id)
        score = 1.35
        if unit.visible_text:
            score += 0.45 + min(1.0, len(unit.visible_text) * 0.10)
            reasons[photo_id].add("owner_visible_text")
        if unit.activity_terms:
            score += 0.35
            reasons[photo_id].add("owner_activity")
        if unit.venue_terms:
            score += 0.25
            reasons[photo_id].add("owner_venue")
        if unit.gps_location:
            score += 0.12
        scores[photo_id] = max(scores[photo_id], score)

    for photo_id, photo in inventory.photo_lookup.items():
        caption = str(photo.get("caption") or "")
        visible_text = [str(text) for text in photo.get("visible_text") or [] if str(text).strip()]
        entities = photo.get("text_entities") or []
        if not caption and not visible_text and not entities:
            continue
        face_ids = [str(fid) for fid in photo.get("visible_face_ids") or []]
        score = 0.10
        if owner_face in face_ids:
            score += 0.85
        if visible_text:
            score += min(1.0, 0.22 * len(visible_text))
        if entities:
            score += min(0.8, 0.08 * len(entities))
        if len(caption) >= 80:
            score += 0.20
        if _specific_detail_text(caption, visible_text):
            score += 0.45
        scores[str(photo_id)] = max(scores[str(photo_id)], score)

    ranked = sorted(
        scores,
        key=lambda pid: (
            -scores[pid],
            -len(reasons.get(pid, set())),
            pid,
        ),
    )
    return dedupe(ranked)[:limit]


def _clean_new_facts(
    facts: list[Any],
    *,
    allowed_photo_ids: set[str],
    seen: set[str],
    config: ECDConfig,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in facts:
        if len(accepted) >= limit:
            break
        if not isinstance(item, dict):
            continue
        text = _clean_text(str(item.get("text") or ""))
        signature = _fact_signature(text)
        evidence = [
            str(pid)
            for pid in item.get("evidence_photo_ids") or []
            if str(pid) in allowed_photo_ids
        ]
        confidence = _bounded_float(item.get("confidence"), 0.0)
        reason = ""
        if not text or len(text) < 12:
            reason = "empty_or_too_short"
        elif signature in seen or any(signature == _fact_signature(str(row.get("text") or "")) for row in accepted):
            reason = "duplicate"
        elif not evidence:
            reason = "no_allowed_evidence"
        elif confidence < config.owner_detail_miner_min_confidence:
            reason = "low_confidence"
        elif _hedged_or_non_owner(text):
            reason = "hedged_non_owner_or_private"
        if reason:
            rejected.append({"text": text, "reason": reason, "confidence": round(confidence, 3)})
            continue
        accepted.append(
            {
                "text": text,
                "evidence_photo_ids": dedupe(evidence)[:4],
                "confidence": round(min(0.86, max(config.owner_detail_miner_min_confidence, confidence)), 3),
                "reasoning_path": str(item.get("reasoning_path") or "Owner detail miner extracted this fact from cited public album evidence.")[:420],
            }
        )
    return accepted, rejected


def _photo_has_owner(inventory: EvidenceInventory, photo_id: str) -> bool:
    photo = inventory.photo_lookup.get(photo_id) or {}
    return inventory.owner_face_id in [str(fid) for fid in photo.get("visible_face_ids") or []]


def _specific_detail_text(caption: str, visible_text: list[str]) -> bool:
    blob = " ".join([caption, *visible_text]).lower()
    return bool(
        re.search(r"\b[A-Z][A-Za-z0-9&.-]{2,}\b", caption)
        or re.search(r"\b\d{1,2}\s*(am|pm)\b", blob)
        or re.search(r"\b(brand|model|degree|certificate|license|badge|route|district|restaurant|cafe|clinic|school|studio|machine|camera|tablet|car|truck|bike|guitar|sewing|scanner|monitor|boots|glasses)\b", blob)
    )


def _clean_text(text: str) -> str:
    value = " ".join(str(text or "").split()).strip(" -")
    if value and value[-1] not in ".!?":
        value += "."
    return value[:260]


def _hedged_or_non_owner(text: str) -> bool:
    lower = text.lower()
    if re.search(r"\b(may|might|could|possibly|probably)\b", lower):
        return True
    if re.search(r"\bor\b", lower):
        return True
    if lower.startswith(("someone ", "a person ", "another person ", "the photo ")):
        return True
    if re.search(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", lower):
        return True
    if re.search(r"\b\d{1,5}\s+[a-z0-9 .'-]+\s+(street|st|road|rd|avenue|ave|drive|dr|lane|ln|court|ct|way|boulevard|blvd|place|pl|apartment|apt)\b", lower):
        return True
    if re.search(r"\b(friend|relative|husband|wife|partner|boyfriend|girlfriend|mother|father|sister|brother|cousin|aunt|uncle|colleague|neighbor|neighbour)\s+named\b", lower):
        return True
    if re.search(r"\b(has|knows|met|texts|messages|communicates with)\s+(a\s+)?(friend|relative|person|man|woman|colleague|neighbor|neighbour)\s+named\b", lower):
        return True
    if re.search(r"\b(married to|dating|partnered with)\s+(a\s+)?(man|woman|person)?\s*named\b", lower):
        return True
    if re.search(r"\b(friendship|relationship|connection)\s+with\s+[a-z]+", lower):
        return True
    if "owner's name is" in lower or "album owner's name is" in lower:
        return True
    if "album owner" in lower and re.search(r"\b(full\s+name|name)\s+is\b", lower):
        return True
    return False


def _fact_signature(text: str) -> str:
    lower = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
    if not lower:
        return ""
    tokens = [
        tok
        for tok in lower.split()
        if tok not in {"the", "album", "owner", "regularly", "often", "frequently", "appears", "uses", "has", "have", "is", "are"}
    ]
    return " ".join(tokens[:10])


def _bounded_float(value: Any, fallback: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return fallback

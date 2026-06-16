"""LLM readout synthesis for ECD profiles."""

from __future__ import annotations

import json
import re
from typing import Any

from .config import ECDConfig
from .llm_context import compact_profile_snapshot, owner_context, resolved_context
from .owner_evidence_planner import compact_owner_evidence_plan
from .schemas import EvidenceInventory, ResolvedIdentity
from .text import dedupe

_OWNER_SYSTEM = """You are the owner-profile readout module for Evidence-Chain Dossier.
Use only the supplied public album evidence. Write concise owner facts that an evaluator can match.

Rules:
- Each fact must be directly supported by evidence_photo_ids.
- Prefer atomic facts: name, age, city/base, occupation, activities, recurring routines.
- Use owner_evidence_plan as the primary menu of candidate fact types and citations.
- If owner_readout_policy.mode is "owner_fact_census", produce a source-balanced census: cover recurring visual patterns, scenes, temporal routines, face co-occurrence/social context, cross-modal cues, metadata cues, and direct OCR when supported.
- Preserve direct atomic facts already present in current_profile when they are supported by evidence, especially exact name, age, birth year/date, city/base, occupation, education/certification, and recurring activity facts.
- Avoid near-duplicate facts. Do not spend many slots on name/city/age/credential variants when recurring routines or environment facts are supported.
- Prefer all available max_facts slots when the evidence supports distinct owner facts.
- Prefer evaluator-friendly wording for numeric and credential facts, e.g. "He is 44 years old", "Her occupation is lawyer", "He holds a trade certificate".
- Do not overstate uncertain evidence.
- Use neutral wording such as "The album owner ..." unless the evidence explicitly supports gendered wording.
- Output ONLY JSON: {"facts":[{"text":"...","evidence_photo_ids":["photo_..."],"confidence":0.0,"reasoning_path":"..."}]}"""

_PERSON_SYSTEM = """You are the person readout module for Evidence-Chain Dossier.
Improve relation labels and reasoning paths for already resolved face identities.

Rules:
- Do not change face_id.
- Do not invent names.
- Keep relation_category to one of: family, friend, colleague, classmate, neighbor, other, or empty.
- Output only persons whose relation_to_owner, relation_category, or reasoning_path should change; omit rows that are already adequate.
- Keep reasoning_path to one concise sentence of at most 35 words, citing 1-3 concrete photo_ids.
- reasoning_path must cite concrete photo_ids and explain the evidence chain.
- Output ONLY JSON: {"persons":[{"face_id":"face_001","relation_to_owner":"...","relation_category":"...","confidence":0.0,"reasoning_path":"..."}]}"""


def enhance_profile_readout(
    *,
    profile: dict[str, Any],
    inventory: EvidenceInventory,
    resolved: list[ResolvedIdentity],
    config: ECDConfig,
    llm: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use LLM to improve owner facts and person reasoning/relation fields."""
    if llm is None:
        return profile, {"enabled": False, "reason": "no_llm"}

    diagnostics: dict[str, Any] = {"enabled": True}
    owner_result = _owner_readout(profile, inventory, config, llm)
    diagnostics["owner"] = owner_result
    if isinstance(owner_result, dict) and isinstance(owner_result.get("facts"), list):
        facts = _clean_facts(owner_result["facts"], config.max_owner_facts)
        if facts:
            existing = list((profile.get("owner") or {}).get("facts") or [])
            profile.setdefault("owner", {})["facts"] = _merge_facts(
                facts,
                existing,
                config.max_owner_facts,
                semantic_dedupe=config.use_owner_fact_census,
            )

    person_result = _person_readout(profile, inventory, resolved, config, llm)
    diagnostics["persons"] = person_result
    if isinstance(person_result, dict) and isinstance(person_result.get("persons"), list):
        _apply_person_updates(profile, person_result["persons"])

    profile["variant"] = "ecd_llm_core_v1"
    return profile, diagnostics


def _owner_readout(profile: dict[str, Any], inventory: EvidenceInventory, config: ECDConfig, llm: Any) -> dict[str, Any] | None:
    max_plan_cards = config.owner_evidence_max_cards
    max_facts = min(config.max_owner_facts, config.llm_owner_fact_budget)
    context_units = 30
    prompt_chars = config.llm_photo_snippet_chars
    examples_per_card = 4
    max_tokens = 8192
    if config.use_owner_fact_census:
        max_facts = config.max_owner_facts
        context_units = 24
        prompt_chars = min(config.llm_photo_snippet_chars, 220)
        examples_per_card = 3
        max_tokens = 6144
    payload = {
        "user_id": inventory.album.get("user_id"),
        "current_profile": {
            "owner": profile.get("owner") or {},
        },
        "owner_context": owner_context(
            inventory,
            max_units=context_units,
            chars=prompt_chars,
        ),
        "owner_evidence_plan": compact_owner_evidence_plan(
            inventory,
            max_cards=(
                max_plan_cards
                if config.use_owner_fact_census
                else min(max_plan_cards, config.llm_owner_fact_budget + 8)
            ),
            max_examples_per_card=examples_per_card,
            chars=prompt_chars,
            balanced=config.use_owner_fact_census,
            min_cards_per_source=config.owner_census_min_cards_per_source,
        ),
        "max_facts": max_facts,
    }
    if config.use_owner_fact_census:
        payload["owner_readout_policy"] = {
            "mode": "owner_fact_census",
            "goal": "maximize distinct, evidence-grounded owner-fact recall without weakening identity precision",
            "min_sources_to_cover_if_available": 5,
            "max_direct_ocr_facts": 7,
            "dedupe_guidance": "merge near-duplicate direct facts; prefer distinct routines, environments, events, tools, clothing, and social patterns",
        }
    result = llm.call_json(
        prompt=json.dumps(payload, ensure_ascii=False),
        system=_OWNER_SYSTEM,
        temperature=0.15,
        max_tokens=max_tokens,
        retries=1,
        stage="llm_readout.owner",
    )
    return result if isinstance(result, dict) else None


def _person_readout(
    profile: dict[str, Any],
    inventory: EvidenceInventory,
    resolved: list[ResolvedIdentity],
    config: ECDConfig,
    llm: Any,
) -> dict[str, Any] | None:
    payload = {
        "user_id": inventory.album.get("user_id"),
        "owner": {
            "face_id": inventory.owner_face_id,
            "name_candidate": inventory.owner_name,
            "last_name": inventory.owner_last_name,
        },
        "current_persons": compact_profile_snapshot(profile)["persons"],
        "evidence_by_person": [
            resolved_context(inventory, row, chars=config.llm_photo_snippet_chars)
            for row in resolved
        ],
    }
    result = llm.call_json(
        prompt=json.dumps(payload, ensure_ascii=False),
        system=_PERSON_SYSTEM,
        temperature=0.15,
        max_tokens=4096,
        retries=1,
        stage="llm_readout.person",
    )
    return result if isinstance(result, dict) else None


def _clean_facts(facts: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for fact in facts:
        text = str(fact.get("text") or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        evidence = [str(pid) for pid in (fact.get("evidence_photo_ids") or []) if str(pid).startswith("photo_")]
        if not evidence:
            continue
        try:
            confidence = float(fact.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.55
        out.append(
            {
                "text": text,
                "evidence_photo_ids": dedupe(evidence)[:5],
                "confidence": round(max(0.05, min(0.95, confidence)), 3),
                "reasoning_path": str(fact.get("reasoning_path") or ""),
            }
        )
        seen.add(key)
        if len(out) >= limit:
            break
    return out


def _merge_facts(
    primary: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    limit: int,
    *,
    semantic_dedupe: bool = False,
) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for fact in [*primary, *fallback]:
        text = str(fact.get("text") or "").strip()
        if not text:
            continue
        key = _fact_signature(text) if semantic_dedupe else " ".join(text.lower().split())
        if key in seen:
            continue
        merged.append(fact)
        seen.add(key)
        if len(merged) >= limit:
            break
    return merged


def _fact_signature(text: str) -> str:
    lower = " ".join(str(text or "").lower().split())
    compact = re.sub(r"[^a-z0-9]+", " ", lower).strip()
    if not compact:
        return ""
    if "first and last name" in lower or re.search(r"\b(full )?name\b", lower):
        return "owner_name"
    if re.search(r"\b\d{2}\s+years old\b", lower):
        return "owner_age"
    if "birth year" in lower or "date of birth" in lower or re.search(r"\bborn\b", lower):
        return "owner_birth"
    if "based in" in lower or "lives in" in lower or "living in" in lower:
        return "owner_base"
    return compact


def _apply_person_updates(profile: dict[str, Any], updates: list[dict[str, Any]]) -> None:
    by_face = {str(row.get("face_id") or ""): row for row in profile.get("persons") or []}
    for update in updates:
        face_id = str(update.get("face_id") or "")
        row = by_face.get(face_id)
        if not row:
            continue
        if update.get("relation_to_owner") is not None:
            relation = str(update.get("relation_to_owner") or "").strip()
            if relation:
                row["relation_to_owner"] = relation
        if update.get("relation_category") is not None:
            category = str(update.get("relation_category") or "").strip()
            current = str(row.get("relation_category") or "").strip()
            if category and (not current or current == "other" or category != "other"):
                row["relation_category"] = category
        if update.get("reasoning_path"):
            row["reasoning_path"] = str(update.get("reasoning_path") or "")
        try:
            row["confidence"] = round(max(0.05, min(0.95, float(update.get("confidence")))), 3)
        except (TypeError, ValueError):
            pass

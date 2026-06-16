from __future__ import annotations

import hashlib
import json
import logging
import random
import time
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm.base import LLMClient
from src.step6.photo_generator import (
    ImageBackend,
    PhotoSignalGenerator,
    _build_recurring_people_summary,
    _build_user_device_plan,
    _default_ambient_people_plan,
    _default_dimensions_for_aspect_ratio,
    _device_for_year_month,
    _normalize_people_plan,
    _resolve_people_to_face_ids,
)
from src.utils.io import load_json, save_json

logger = logging.getLogger(__name__)

_SCENE_TYPES = {"scene", "scene_with_text"}
_TEXT_ONLY_TYPES = {"screenshot", "document", "object_detail"}


class _NullBackend(ImageBackend):
    @property
    def name(self) -> str:
        return "null"

    def generate(
        self,
        prompt: str,
        ref_image_paths: list[Path] | None = None,
        width: int = 1024,
        height: int = 1024,
        aspect_ratio: str | None = None,
        seed: int = -1,
    ) -> bytes:
        raise RuntimeError("BenchmarkDatasetBuilder does not generate images")


def _load_prompt(name: str) -> str:
    path = Path(__file__).parents[2] / "prompts" / "benchmark" / name
    return path.read_text(encoding="utf-8")


def _coerce_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _dedupe_str_list(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _extract_json(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty llm response")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fenced:
        return json.loads(fenced.group(1).strip())

    obj = re.search(r"\{[\s\S]*\}", raw)
    if obj:
        return json.loads(obj.group(0))
    raise ValueError(f"cannot extract json from llm response: {raw[:300]}")


def _chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    chunk_size = max(int(size or 1), 1)
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _format_year_month(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "an unspecified time"
    try:
        return datetime.strptime(text, "%Y-%m").strftime("%B %Y")
    except ValueError:
        return text


def _sanitize_free_text(text: object, forbidden_names: list[str]) -> str:
    result = str(text or "").strip()
    if not result:
        return ""
    for name in sorted({n.strip() for n in forbidden_names if str(n).strip()}, key=len, reverse=True):
        result = re.sub(re.escape(name), "someone", result, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", result).strip()


def _build_name_replacements(profile: dict[str, Any], graph: dict[str, Any]) -> list[tuple[str, str]]:
    replacements: dict[str, str] = {}

    owner_name = str(profile.get("name", "")).strip()
    if owner_name:
        replacements[owner_name] = "the owner"
        owner_first = owner_name.split()[0].strip()
        if owner_first:
            replacements.setdefault(owner_first, "the owner")

    for node in graph.get("nodes", []):
        name = str(node.get("name", "")).strip()
        if not name:
            continue
        replacements.setdefault(name, "someone")
        first = name.split()[0].strip()
        if first:
            replacements.setdefault(first, "someone")

    return sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)


def _replace_name_mentions(
    text: object,
    replacements: list[tuple[str, str]],
    *,
    allowed_text: str = "",
) -> str:
    result = str(text or "").strip()
    if not result:
        return ""

    allowed = allowed_text.lower()
    for name, replacement in replacements:
        if not name or name.lower() in allowed:
            continue
        pattern = re.compile(rf"\b{re.escape(name)}(?P<possessive>'s|’s)?\b", re.IGNORECASE)

        def _sub(match: re.Match[str]) -> str:
            possessive = match.group("possessive") or ""
            if possessive:
                if replacement == "the owner":
                    return "the owner's"
                if replacement == "someone":
                    return "someone's"
            return replacement

        result = pattern.sub(_sub, result)

    result = re.sub(r"\s+", " ", result).strip()
    result = result.replace("someone's home in someone's", "someone's home in")
    result = result.replace("the owner's home in the owner's", "the owner's home in")
    return result


def _sanitize_metadata(metadata: dict[str, Any], replacements: list[tuple[str, str]]) -> dict[str, Any]:
    sanitized = dict(metadata)
    for key in ["gps_city", "gps_location"]:
        value = sanitized.get(key)
        if isinstance(value, str):
            sanitized[key] = _replace_name_mentions(value, replacements)
    return sanitized


def _default_render_layout(aspect_ratio: str, layout_policy: str) -> dict[str, Any]:
    width, height = _default_dimensions_for_aspect_ratio(aspect_ratio)
    return {
        "aspect_ratio": aspect_ratio,
        "layout_policy": layout_policy,
        "requested_width": width,
        "requested_height": height,
        "actual_width": None,
        "actual_height": None,
    }


# ---------------------------------------------------------------------------
# Entity alignment GT builder
# ---------------------------------------------------------------------------


def _event_for_photo(
    photo_id: str,
    photo_annotations: list[dict[str, Any]],
) -> str | None:
    for ann in photo_annotations:
        if str(ann.get("photo_id") or "") == photo_id:
            ev = ann.get("event_id")
            return str(ev) if ev else None
    return None


def _face_visible_in_event(
    event_id: str | None,
    face_id: str,
    photo_annotations: list[dict[str, Any]],
) -> bool:
    if not event_id or not face_id:
        return False
    for ann in photo_annotations:
        if str(ann.get("event_id") or "") != event_id:
            continue
        if face_id in (ann.get("visible_face_ids") or []):
            return True
    return False


def _photos_with_face(
    face_id: str,
    photo_annotations: list[dict[str, Any]],
) -> list[str]:
    out: list[str] = []
    for ann in photo_annotations:
        if face_id in (ann.get("visible_face_ids") or []):
            out.append(str(ann.get("photo_id") or ""))
    return [pid for pid in out if pid]


def _classify_alignment_difficulty(
    *,
    source_photo_id: str,
    target_face_id: str,
    photo_annotations: list[dict[str, Any]],
) -> str:
    """Deterministic alignment difficulty classifier.

    easy   = text mention and target face co-occur in the same photo
    medium = target face does not appear in the mention photo but appears
             somewhere else in the same event
    hard   = target face does not appear in the same event at all; alignment
             requires cross-event / cross-time reasoning
    """
    if not source_photo_id or not target_face_id:
        return "hard"
    # Locate the mention's photo annotation
    host = None
    for ann in photo_annotations:
        if str(ann.get("photo_id") or "") == source_photo_id:
            host = ann
            break
    if host is None:
        return "hard"
    if target_face_id in (host.get("visible_face_ids") or []):
        return "easy"
    event_id = host.get("event_id")
    if event_id and _face_visible_in_event(str(event_id), target_face_id, photo_annotations):
        return "medium"
    return "hard"


def _build_entity_alignment(
    *,
    profile: dict[str, Any],
    graph: dict[str, Any],
    face_catalog: list[dict[str, Any]],
    photo_annotations: list[dict[str, Any]],
    event_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate per-mention alignment records from photo_annotations.

    Each annotation's ``text_mentions`` already carries
    ``resolves_to_person_ids``; here we:

    * group mentions by (canonical_target_person_id, mention_kind, text_lower)
    * attach face_id via the face_catalog / graph mapping
    * compute alignment_difficulty per record
    * assign a stable ``mention_id``
    """
    person_to_face: dict[str, str] = {}
    for item in face_catalog or []:
        pid = str(item.get("person_id") or "").strip()
        fid = str(item.get("face_id") or "").strip()
        if pid and fid:
            person_to_face[pid] = fid

    person_info: dict[str, dict[str, Any]] = {"owner": {
        "canonical_name": profile.get("name") or "",
        "relation_category": "self",
        "relation": "self",
    }}
    for node in graph.get("nodes", []) or []:
        pid = str(node.get("person_id") or "").strip()
        if pid:
            person_info[pid] = {
                "canonical_name": node.get("name") or "",
                "relation_category": node.get("relation_category") or "",
                "relation": node.get("relation") or "",
            }

    # group key: (person_id, mention_kind, normalized_text)
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ann in photo_annotations:
        photo_id = str(ann.get("photo_id") or "")
        mentions = ann.get("text_mentions") or []
        for m in mentions:
            if not isinstance(m, dict):
                continue
            kind = str(m.get("mention_kind") or "").strip().lower()
            text = str(m.get("text") or "").strip()
            if kind not in {"name", "relation"} or not text:
                continue
            resolves = [str(p).strip() for p in (m.get("resolves_to_person_ids") or []) if str(p).strip()]
            if not resolves:
                continue
            alias_of = str(m.get("alias_of") or "").strip()
            for pid in resolves:
                key = (pid, kind, text.lower())
                if key not in groups:
                    groups[key] = {
                        "person_id": pid,
                        "face_id": person_to_face.get(pid) or ("owner" if pid == "owner" else ""),
                        "mention_kind": kind,
                        "text": text,
                        "aliases": [alias_of] if alias_of else [],
                        "canonical_name": (
                            alias_of
                            or (person_info.get(pid) or {}).get("canonical_name")
                            or ""
                        ),
                        "relation_hint": (person_info.get(pid) or {}).get("relation", ""),
                        "relation_category": (person_info.get(pid) or {}).get("relation_category", ""),
                        "source_photo_ids": [],
                        "source_texts": [],
                    }
                g = groups[key]
                if photo_id and photo_id not in g["source_photo_ids"]:
                    g["source_photo_ids"].append(photo_id)
                if text and text not in g["source_texts"]:
                    g["source_texts"].append(text)
                if alias_of and alias_of not in g["aliases"]:
                    g["aliases"].append(alias_of)

    # Phase 5: drop owner self-reference name mentions BEFORE assigning
    # mention_id (so IDs stay sequential without gaps). Keep
    # ``person_id="owner"`` + ``mention_kind="relation"`` entries
    # (e.g. visible "my dad" pointing AWAY from the owner) — those are
    # legitimate cross-modal anchors. Only the literal-name shortcut
    # ("the owner's name appears on this artifact") is filtered.
    filtered_groups = {
        key: g for key, g in groups.items()
        if not (key[0] == "owner" and key[1] == "name")
    }

    records: list[dict[str, Any]] = []
    for index, ((pid, kind, _text_lower), g) in enumerate(sorted(filtered_groups.items()), 1):
        mention_id = f"ment_{index:03d}"
        target_face = g.get("face_id") or ""
        # difficulty = min across all source photos (easier wins)
        difficulty = "hard"
        rank = {"easy": 0, "medium": 1, "hard": 2}
        for sp in g["source_photo_ids"]:
            d = _classify_alignment_difficulty(
                source_photo_id=sp,
                target_face_id=target_face,
                photo_annotations=photo_annotations,
            )
            if rank[d] < rank[difficulty]:
                difficulty = d
        records.append({
            "mention_id": mention_id,
            "canonical_name": g["canonical_name"],
            "aliases": g["aliases"],
            "mention_kind": kind,
            "person_id": pid,
            "face_id": target_face,
            "relation_hint": g["relation_hint"],
            "relation_category": g["relation_category"],
            "source_photo_ids": list(g["source_photo_ids"]),
            "source_texts": list(g["source_texts"]),
            "alignment_difficulty": difficulty,
        })
    return records


_VALID_AMBIENT_SUPPORT_TYPES = {
    "recurring_face", "repeated_location", "temporal_anchor",
    "routine_object", "text_trace", "social_setting", "other",
}


def _build_ambient_supports_index(
    *,
    fact_paths: list[dict[str, Any]],
    node_paths: list[dict[str, Any]],
    fact_id_by_target: dict[str, str],
) -> list[dict[str, Any]]:
    """Lift ``ambient_support_needs`` from Step-4 paths into a flat list.

    Phase 5: each support entry from a fact_path or node_path is wrapped
    with ``target_kind`` (fact|node) and ``target_ref`` (fact_id or
    person_id) so the evaluator can filter / aggregate without having
    to re-traverse the original reasoning paths.

    Accepts both shapes for back-compat with v2/v3 data:
    - new dict shape: ``{"support_type": <enum>, "description": str}``
    - legacy str shape: free-text string (becomes support_type="other").
    """
    out: list[dict[str, Any]] = []
    seq = 0

    def _emit(target_kind: str, target_ref: str, raw: object) -> None:
        nonlocal seq
        if not isinstance(raw, list):
            return
        for item in raw:
            stype: str
            desc: str
            if isinstance(item, dict):
                desc = str(item.get("description") or "").strip()
                stype = str(item.get("support_type") or "other").strip().lower()
                if stype not in _VALID_AMBIENT_SUPPORT_TYPES:
                    stype = "other"
            elif isinstance(item, str) and item.strip():
                desc = item.strip()
                stype = "other"
            else:
                continue
            if not desc:
                continue
            seq += 1
            out.append({
                "support_id": f"supp_{seq:03d}",
                "target_kind": target_kind,
                "target_ref": target_ref,
                "support_type": stype,
                "description": desc[:500],
            })

    for fp in fact_paths or []:
        target_text = str(fp.get("target") or "").strip()
        fid = fact_id_by_target.get(target_text) or ""
        target_ref = fid or target_text[:64]
        _emit("fact", target_ref, fp.get("ambient_support_needs"))
        # Legacy alias some old paths used; harmless to harvest.
        _emit("fact", target_ref, fp.get("auxiliary_album_signals"))

    for npath in node_paths or []:
        pid = str(npath.get("person_id") or "").strip()
        if not pid:
            continue
        _emit("node", pid, npath.get("ambient_support_needs"))
        _emit("node", pid, npath.get("auxiliary_album_signals"))

    return out


def _build_persons_roster(
    *,
    profile: dict[str, Any],
    graph: dict[str, Any],
    face_catalog: list[dict[str, Any]],
    node_path_map: dict[str, dict[str, Any]],
    entity_alignment: list[dict[str, Any]],
    photo_annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Unified persons roster. Owner first, then non-owner persons.

    Each record mirrors the paper-facing PersonRecord schema and combines
    info from profile / graph / node_paths / face_catalog / entity_alignment.
    """
    face_by_person = {
        str(item.get("person_id") or ""): str(item.get("face_id") or "")
        for item in face_catalog or []
    }
    mentions_by_person: dict[str, list[str]] = {}
    mention_photos_by_person: dict[str, list[str]] = {}
    for rec in entity_alignment:
        pid = rec.get("person_id") or ""
        mentions_by_person.setdefault(pid, []).append(rec.get("mention_id") or "")
        for sp in rec.get("source_photo_ids") or []:
            mention_photos_by_person.setdefault(pid, [])
            if sp not in mention_photos_by_person[pid]:
                mention_photos_by_person[pid].append(sp)

    persons: list[dict[str, Any]] = []
    owner_face = "owner"
    owner_face_photos = _photos_with_face(owner_face, photo_annotations)
    owner_name_photos = mention_photos_by_person.get("owner", [])
    persons.append({
        "person_id": "owner",
        "face_id": "owner",
        "is_owner": True,
        "canonical_name": profile.get("name") or "",
        "aliases": [profile.get("name") or ""] if profile.get("name") else [],
        "relation_category": "self",
        "relation_to_owner": "self",
        "attributes": dict(profile.get("raw_attributes") or {}),
        "evidence_photo_ids": {
            "face_visible": owner_face_photos,
            "name_mentions": owner_name_photos,
            "joint": sorted(set(owner_face_photos) & set(owner_name_photos)),
        },
        "reasoning_trace": {
            "identification": "",
            "relation_reasoning": "",
        },
        "mention_ids": mentions_by_person.get("owner", []),
    })
    for node in graph.get("nodes", []) or []:
        pid = str(node.get("person_id") or "").strip()
        if not pid:
            continue
        fid = face_by_person.get(pid) or ""
        node_path = node_path_map.get(pid, {})
        face_photos = _photos_with_face(fid, photo_annotations) if fid else []
        name_photos = mention_photos_by_person.get(pid, [])
        persons.append({
            "person_id": pid,
            "face_id": fid,
            "is_owner": False,
            "canonical_name": node.get("name") or "",
            "aliases": list(dict.fromkeys([node.get("name") or "", *(node_path.get("canonical_mentions") or [])])),
            "relation_category": node.get("relation_category") or "",
            "relation_to_owner": node.get("relation") or "",
            "attributes": {
                "age": node.get("age"),
                "gender": node.get("gender"),
                "occupation": node.get("occupation"),
                "city": node.get("city"),
            },
            "evidence_photo_ids": {
                "face_visible": face_photos,
                "name_mentions": name_photos,
                "joint": sorted(set(face_photos) & set(name_photos)),
            },
            "reasoning_trace": {
                "identification": node_path.get("identification") or "",
                "relation_reasoning": node_path.get("relation_reasoning") or "",
            },
            "mention_ids": mentions_by_person.get(pid, []),
        })
    return persons


def _invert_mapping(mapping: dict[str, str]) -> dict[str, str]:
    return {str(v): str(k) for k, v in (mapping or {}).items()}


def _map_ids(ids: list[Any], mapping: dict[str, str]) -> list[str]:
    out: list[str] = []
    for value in ids or []:
        mapped = mapping.get(str(value), str(value))
        if mapped and mapped not in out:
            out.append(mapped)
    return out


def _photo_public_to_private_map(ground_truth: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    trace = ground_truth.get("traceability") or {}
    private_to_public = {str(k): str(v) for k, v in (trace.get("photo_id_map") or {}).items()}
    public_to_private = _invert_mapping(private_to_public)
    return public_to_private, private_to_public


def _face_public_to_private_map(ground_truth: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    trace = ground_truth.get("traceability") or {}
    private_to_public = {str(k): str(v) for k, v in (trace.get("face_id_map") or {}).items()}
    public_to_private = _invert_mapping(private_to_public)
    return public_to_private, private_to_public


def _safe_year_month(photo: dict[str, Any]) -> str:
    ym = str(photo.get("year_month") or "").strip()
    if ym:
        return ym
    ts = str((photo.get("metadata") or {}).get("timestamp") or "").strip()
    return ts[:7] if len(ts) >= 7 else ""


def build_dual_benchmark_views(internal: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build public Agent and private evaluator views from the internal export state."""
    user_id = str(internal.get("user_id") or "")
    generated_at = str(internal.get("generated_at") or datetime.now(timezone.utc).isoformat())
    photos = list((internal.get("album_data") or {}).get("photos") or [])
    ground_truth = dict(internal.get("ground_truth") or {})
    public_to_private_photo, private_to_public_photo = _photo_public_to_private_map(ground_truth)
    public_to_private_face, private_to_public_face = _face_public_to_private_map(ground_truth)

    public_photos: list[dict[str, Any]] = []
    face_months: dict[str, list[str]] = {}
    for photo in photos:
        meta = dict(photo.get("metadata") or {})
        year_month = _safe_year_month(photo)
        timestamp = str(meta.get("timestamp") or "").strip()
        face_ids = [str(fid) for fid in (photo.get("visible_face_ids") or []) if str(fid).strip()]
        for fid in face_ids:
            face_months.setdefault(fid, [])
            if year_month:
                face_months[fid].append(year_month)
        public_photos.append({
            "photo_id": str(photo.get("photo_id") or ""),
            "year_month": year_month,
            "timestamp": timestamp,
            "caption": str(photo.get("description") or photo.get("caption") or ""),
            "visible_text": list(photo.get("visible_text") or []),
            "text_entities": [
                {
                    "surface": str(ent.get("surface") or ""),
                    "entity_type": str(ent.get("entity_type") or "person"),
                    "source": str(ent.get("source") or ""),
                    **({"confidence": ent.get("confidence")} if "confidence" in ent else {}),
                }
                for ent in (photo.get("text_entities") or [])
                if isinstance(ent, dict) and str(ent.get("surface") or "").strip()
            ],
            "visible_face_ids": face_ids,
            "metadata": {
                "gps_city": meta.get("gps_city"),
                "gps_location": meta.get("gps_location"),
                "device": meta.get("device"),
            },
        })

    all_months = [_safe_year_month(p) for p in public_photos if _safe_year_month(p)]
    faces = []
    for fid in sorted(face_months):
        months = sorted(face_months[fid])
        faces.append({
            "face_id": fid,
            "n_appearances": len(face_months[fid]),
            "first_seen": months[0] if months else "",
            "last_seen": months[-1] if months else "",
        })

    agent_album = {
        "schema_version": "agent_album.v1",
        "user_id": user_id,
        "album_id": str(user_id or ""),
        "generated_at": generated_at,
        "source_benchmark_version": str(internal.get("benchmark_version") or ""),
        "visibility_contract": {
            "contains_ground_truth": False,
            "contains_questions": False,
            "contains_reasoning_paths": False,
            "contains_private_traceability": False,
            "agent_may_read_entire_file": True,
        },
        "album_summary": {
            "n_photos": len(public_photos),
            "time_start": min(all_months) if all_months else "",
            "time_end": max(all_months) if all_months else "",
            "available_modalities": [
                "caption",
                "visible_text",
                "text_entities",
                "visible_face_ids",
                "timestamp",
                "location",
            ],
        },
        "faces": faces,
        "photos": public_photos,
    }

    photo_annotations = {
        str(ann.get("photo_id") or ""): ann
        for ann in (ground_truth.get("photo_annotations") or [])
        if str(ann.get("photo_id") or "")
    }

    ambient_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for support in ground_truth.get("ambient_supports") or []:
        if not isinstance(support, dict):
            continue
        key = (str(support.get("target_kind") or ""), str(support.get("target_ref") or ""))
        if key[0] and key[1]:
            ambient_by_key.setdefault(key, []).append(dict(support))

    def _key_photo_roles(private_ids: list[str]) -> list[dict[str, Any]]:
        roles: list[dict[str, Any]] = []
        for private_id in private_ids:
            ann = photo_annotations.get(str(private_id), {})
            roles.append({
                "photo_id_public": private_to_public_photo.get(str(private_id), str(private_id)),
                "photo_id_private": str(private_id),
                "evidence_role": str(ann.get("evidence_role") or ""),
                "planned_description": str(ann.get("planned_description") or ""),
                "must_show": list(ann.get("must_show") or []),
                "must_read_text": list(ann.get("must_read_text") or []),
            })
        return roles

    facts = []
    evaluation_targets: list[dict[str, Any]] = []
    for fact in ground_truth.get("facts", []) or []:
        fact_id = str(fact.get("fact_id") or "")
        private_kps = [str(pid) for pid in (fact.get("key_photo_ids") or []) if str(pid).strip()]
        public_kps = _map_ids(private_kps, private_to_public_photo)
        key_photo_roles = _key_photo_roles(private_kps)
        ambient_support_needs = ambient_by_key.get(("fact", fact_id), [])
        reasoning_path_text = str(fact.get("reasoning") or "")
        atomic_targets = []
        for atomic in fact.get("atomic_targets") or []:
            if not isinstance(atomic, dict):
                continue
            role = str(atomic.get("role") or "evaluation_target")
            atom_id = str(atomic.get("atomic_id") or "atom_001")
            target_id = f"{fact_id}.{atom_id}" if fact_id else atom_id
            atomic_record = {
                "atom_id": atom_id,
                "target_id": target_id,
                "text": str(atomic.get("text") or ""),
                "role": role,
                "answer_type": str(atomic.get("answer_type") or "boolean"),
            }
            atomic_targets.append(atomic_record)
            if role == "evaluation_target":
                evidence_private = [str(pid) for pid in (atomic.get("supporting_photo_ids") or private_kps) if str(pid).strip()]
                evidence_public = _map_ids(evidence_private, private_to_public_photo)
                evaluation_targets.append({
                    "target_id": target_id,
                    "target_type": "owner_fact_atom",
                    "target_group": "owner",
                    "judge_prompt_type": "owner_atom",
                    "evidence_cap": 5,
                    "rpf_enabled": True,
                    "gt_text": str(atomic.get("text") or ""),
                    "answer_type": str(atomic.get("answer_type") or "boolean"),
                    "parent_fact_id": fact_id,
                    "parent_fact_text": str(fact.get("target") or ""),
                    "evaluation_role": str(fact.get("evaluation_role") or ""),
                    "inference_type": str(fact.get("inference_type") or ""),
                    "difficulty": str(fact.get("difficulty") or ""),
                    "key_photo_ids_public": evidence_public,
                    "key_photo_ids_private": evidence_private,
                    "reasoning_path_text": reasoning_path_text,
                    "key_photo_roles": [
                        role for role in key_photo_roles
                        if role.get("photo_id_private") in evidence_private
                    ],
                    "ambient_support_needs": ambient_support_needs,
                })
        facts.append({
            "fact_id": fact_id,
            "text": str(fact.get("target") or ""),
            "evaluation_role": str(fact.get("evaluation_role") or ""),
            "category": str(fact.get("fact_category") or fact.get("category") or ""),
            "inference_type": str(fact.get("inference_type") or ""),
            "difficulty": str(fact.get("difficulty") or ""),
            "observable": bool(fact.get("observable", False)),
            "self_check_pass": bool((fact.get("self_check") or {}).get("pass", False)),
            "key_photo_ids_public": public_kps,
            "key_photo_ids_private": private_kps,
            "reasoning_path_text": reasoning_path_text,
            "key_photo_roles": key_photo_roles,
            "ambient_support_needs": ambient_support_needs,
            "atomic_targets": atomic_targets,
        })

    difficulty_by_person: dict[str, str] = {}
    for q in internal.get("questions", []) or []:
        ref = str(q.get("source_ref") or "")
        if ref.startswith("node:") and q.get("alignment_difficulty"):
            difficulty_by_person[ref.split(":", 1)[1]] = str(q.get("alignment_difficulty"))

    persons = []
    for person in ground_truth.get("people", []) or []:
        pid = str(person.get("person_id") or "")
        if not pid:
            continue
        private_face = str(person.get("face_id") or "")
        public_face = private_to_public_face.get(private_face, private_face)
        private_kps = [str(pid_) for pid_ in (person.get("key_photo_ids") or []) if str(pid_).strip()]
        public_kps = _map_ids(private_kps, private_to_public_photo)
        key_photo_roles = _key_photo_roles(private_kps)
        ambient_support_needs = ambient_by_key.get(("node", pid), [])
        identification_text = str(person.get("identification") or "")
        relation_reasoning_text = str(person.get("relation_reasoning") or "")
        aliases = [str(alias).strip() for alias in (person.get("aliases") or []) if str(alias).strip()]
        canonical_name = str(person.get("name") or "").strip()
        if canonical_name:
            aliases = list(dict.fromkeys([canonical_name, *aliases]))
        rec = {
            "person_id": pid,
            "public_face_id": public_face,
            "private_face_id": private_face,
            "canonical_name": canonical_name,
            "aliases": aliases,
            "relation_to_owner": str(person.get("relation") or ""),
            "relation_category": str(person.get("relation_category") or ""),
            "alignment_difficulty": difficulty_by_person.get(pid, ""),
            "observable": bool(private_kps),
            "self_check_pass": True,
            "key_photo_ids_public": public_kps,
            "key_photo_ids_private": private_kps,
            "identification_text": identification_text,
            "relation_reasoning_text": relation_reasoning_text,
            "key_photo_roles": key_photo_roles,
            "ambient_support_needs": ambient_support_needs,
        }
        persons.append(rec)
        if pid != "owner":
            for field, prompt_type, target_type in [
                ("canonical_name", "person_name", "person_name"),
                ("relation_to_owner", "person_relation", "person_relation"),
                ("relation_category", "person_category", "person_category"),
            ]:
                value = str(rec.get(field) or "")
                if not value:
                    continue
                evaluation_targets.append({
                    "target_id": f"{pid}.{field}",
                    "target_type": target_type,
                    "target_group": "person",
                    "judge_prompt_type": prompt_type,
                    "evidence_cap": 8,
                    "rpf_enabled": target_type in {"person_name", "person_relation"},
                    "public_face_id": public_face,
                    "private_face_id": private_face,
                    "person_id": pid,
                    "gt_value": value,
                    "canonical_name": rec["canonical_name"],
                    "aliases": rec["aliases"] if target_type == "person_name" else [],
                    "relation_category": rec["relation_category"],
                    "alignment_difficulty": rec["alignment_difficulty"],
                    "key_photo_ids_public": public_kps,
                    "key_photo_ids_private": private_kps,
                    "reasoning_path_text": (
                        identification_text if field == "canonical_name" else relation_reasoning_text
                    ),
                    "identification_text": identification_text,
                    "relation_reasoning_text": relation_reasoning_text,
                    "key_photo_roles": key_photo_roles,
                    "ambient_support_needs": ambient_support_needs,
                })

    owner_public_face = private_to_public_face.get("owner", "")
    eval_gt = {
        "schema_version": "eval_gt.v1",
        "user_id": user_id,
        "generated_at": generated_at,
        "public_private_map": {
            "photo_id": public_to_private_photo,
            "private_photo_id": private_to_public_photo,
            "face_id": public_to_private_face,
            "private_face_id": private_to_public_face,
        },
        "owner": {
            "canonical_name": str((ground_truth.get("owner_profile") or {}).get("name") or ""),
            "canonical_owner_face_id": owner_public_face,
            "private_owner_face_id": "owner",
            "facts": facts,
        },
        "persons": [p for p in persons if p.get("person_id") != "owner"],
        "evaluation_targets": evaluation_targets,
        "private_sources": {
            "internal_export_schema": str(internal.get("benchmark_version") or ""),
        },
    }
    audit = audit_dual_view(agent_album, eval_gt)
    eval_gt["audit_summary"] = audit.get("summary", {})
    return agent_album, eval_gt, audit


def audit_dual_view(agent_album: dict[str, Any], eval_gt: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    photo_re = re.compile(r"^photo_\d{4,}$")
    face_re = re.compile(r"^face_\d{3,}$")
    forbidden_top = ["ground_truth", "questions", "reasoning_paths", "traceability"]
    for key in forbidden_top:
        if key in agent_album:
            errors.append(f"agent top-level contains forbidden key: {key}")
    agent_photos = agent_album.get("photos") or []
    agent_photo_ids = {str(p.get("photo_id") or "") for p in agent_photos}
    agent_face_ids = {str(f.get("face_id") or "") for f in (agent_album.get("faces") or [])}
    visible_face_ids = {
        str(fid)
        for p in agent_photos
        for fid in (p.get("visible_face_ids") or [])
        if str(fid).strip()
    }
    forbidden_photo_keys = {"photo_role", "event_id", "has_image", "text_verified", "image_path", "traceability", "category", "visual_entities", "event_context_tags"}
    private_id_pattern = re.compile(r"\b(kp_fact_|kp_node_|evt_\d|amb_\d)")
    for p in agent_photos:
        pid = str(p.get("photo_id") or "")
        if not photo_re.match(pid):
            errors.append(f"bad public photo_id: {pid}")
        leaked = forbidden_photo_keys.intersection(p.keys())
        if leaked:
            errors.append(f"photo {pid} contains forbidden keys: {sorted(leaked)}")
        if private_id_pattern.search(pid):
            errors.append(f"photo_id leaks private structure: {pid}")
        for ent in p.get("text_entities") or []:
            if not isinstance(ent, dict):
                errors.append(f"photo {pid} has non-dict text_entity")
                continue
            if not str(ent.get("surface") or "").strip():
                errors.append(f"photo {pid} has text_entity without surface")
            if str(ent.get("source") or "") not in {"caption", "visible_text"}:
                errors.append(f"photo {pid} has text_entity with bad source: {ent.get('source')}")
            if str(ent.get("entity_type") or "") not in {"person", "organization", "location", "event"}:
                errors.append(f"photo {pid} has text_entity with bad entity_type: {ent.get('entity_type')}")
        for fid in p.get("visible_face_ids") or []:
            sfid = str(fid)
            if sfid == "owner" or not face_re.match(sfid):
                errors.append(f"photo {pid} has bad face_id: {sfid}")
    if agent_face_ids != visible_face_ids:
        errors.append(f"face summary mismatch: faces={len(agent_face_ids)} visible={len(visible_face_ids)}")
    maps = eval_gt.get("public_private_map") or {}
    public_to_private_photo = maps.get("photo_id") or {}
    private_to_public_photo = maps.get("private_photo_id") or {}
    public_to_private_face = maps.get("face_id") or {}
    private_to_public_face = maps.get("private_face_id") or {}
    if _invert_mapping(public_to_private_photo) != private_to_public_photo:
        errors.append("photo_id maps are not bijective")
    if _invert_mapping(public_to_private_face) != private_to_public_face:
        errors.append("face_id maps are not bijective")
    mapped_public_photos = set(public_to_private_photo.keys())
    if agent_photo_ids != mapped_public_photos:
        errors.append(f"photo map mismatch: agent={len(agent_photo_ids)} map={len(mapped_public_photos)}")
    for target in eval_gt.get("evaluation_targets") or []:
        tid = str(target.get("target_id") or "")
        if not target.get("judge_prompt_type"):
            errors.append(f"target {tid} missing judge_prompt_type")
        if not target.get("target_group"):
            errors.append(f"target {tid} missing target_group")
        if not target.get("evidence_cap"):
            errors.append(f"target {tid} missing evidence_cap")
        if "rpf_enabled" not in target:
            errors.append(f"target {tid} missing rpf_enabled")
        if target.get("target_type") == "person_name" and not target.get("aliases"):
            errors.append(f"person_name target {tid} missing aliases")
        if target.get("rpf_enabled"):
            if not str(target.get("reasoning_path_text") or "").strip():
                errors.append(f"RPF target {tid} missing reasoning_path_text")
            if not target.get("key_photo_roles"):
                errors.append(f"RPF target {tid} missing key_photo_roles")
        for pid in target.get("key_photo_ids_public") or []:
            if str(pid) not in agent_photo_ids:
                errors.append(f"target {tid} references missing public photo {pid}")
        face_id = target.get("public_face_id")
        if face_id and str(face_id) not in agent_face_ids:
            warnings.append(f"target {tid} references face not visible in agent face summary: {face_id}")
    passed = not errors
    return {
        "schema_version": "export_audit.v1",
        "user_id": agent_album.get("user_id") or eval_gt.get("user_id") or "",
        "passed": passed,
        "summary": {
            "passed": passed,
            "n_errors": len(errors),
            "n_warnings": len(warnings),
            "n_agent_photos": len(agent_photos),
            "n_eval_targets": len(eval_gt.get("evaluation_targets") or []),
        },
        "errors": errors,
        "warnings": warnings,
    }


class BenchmarkDatasetBuilder:
    TEXT_ENTITY_SYSTEM = "You are a careful entity extractor. Output strict JSON only."

    def __init__(
        self,
        llm: LLMClient,
        description_batch_size: int = 12,
        rng: random.Random | None = None,
        target_album_photo_min: int = 500,
        target_album_photo_max: int = 1000,
        text_entity_max_workers: int = 8,
        text_entity_max_retries: int = 2,
    ) -> None:
        self._llm = llm
        self._rng = rng or random.Random(0)
        self._description_batch_size = max(int(description_batch_size or 1), 1)
        self._text_entity_max_workers = max(int(text_entity_max_workers or 1), 1)
        self._text_entity_max_retries = max(int(text_entity_max_retries or 0), 0)
        self._text_entities_prompt_tpl = _load_prompt("text_entities.txt")
        self._photo_helper = PhotoSignalGenerator(
            llm=llm,
            backend=_NullBackend(),
            max_workers=1,
            rng=self._rng,
            target_album_photo_min=target_album_photo_min,
            target_album_photo_max=target_album_photo_max,
        )

    def build_from_user_dir(
        self,
        user_dir: Path,
        *,
        preferred_backend: str = "gemini",
        save_ambient_plan: bool = True,
    ) -> dict[str, Any]:
        uid = user_dir.name
        profile = load_json(user_dir / f"{uid}.json")
        graph = load_json(user_dir / f"{uid}_social_graph.json")
        reasoning = load_json(user_dir / f"{uid}_reasoning_paths.json")
        timeline_path = user_dir / f"{uid}_adjusted_timeline.json"
        if not timeline_path.exists():
            timeline_path = user_dir / f"{uid}_timeline.json"
        timeline = load_json(timeline_path)

        manifest_records, manifest_backend = self._load_manifest(user_dir, preferred_backend)

        # Load Step 6.5 outputs (vlm_extraction / face_detection / verification).
        # These provide the ONLY agent-visible signals: caption → description,
        # measured OCR → visible_text, measured face clusters → visible_face_ids.
        step65 = self._load_step65_outputs(user_dir)

        ambient_plan_path = user_dir / "ambient_photo_plan.json"
        return self.build(
            profile=profile,
            graph=graph,
            reasoning=reasoning,
            timeline=timeline,
            manifest_records=manifest_records,
            manifest_backend=manifest_backend,
            ambient_plan_path=ambient_plan_path,
            save_ambient_plan=save_ambient_plan,
            step65=step65,
        )

    @staticmethod
    def _load_step65_outputs(user_dir: Path) -> dict[str, dict[str, Any]]:
        """Load Step 6.5 measurement outputs into a per-photo lookup.

        Returns
        -------
        dict with keys ``vlm`` / ``faces`` / ``verify``, each a dict keyed by
        ``photo_id``.  Missing files yield empty dicts (treated as "no
        measurement available").
        """
        def _safe(p: Path) -> dict[str, Any]:
            if not p.exists():
                return {}
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Cannot read %s: %s", p.name, exc)
                return {}

        return {
            "vlm": _safe(user_dir / "vlm_extraction_results.json"),
            "faces": _safe(user_dir / "face_detection_results.json"),
            "verify": _safe(user_dir / "verification_results.json"),
        }

    @staticmethod
    def _apply_step65_overrides(
        photo_sources: list[dict[str, Any]],
        step65: dict[str, dict[str, Any]],
    ) -> dict[str, str]:
        """Install VLM/InsightFace measurements as the agent-visible signals.

        VLM-first design: agent-visible fields (description, visible_text,
        visible_face_ids) MUST come from the rendered PNG, never from GT
        design intent. This mirrors a real personal-album setting where the
        only ground truth available to a downstream agent is what a VLM can
        actually see in the photo.

        Returns a ``{photo_id: verification_status}`` map for downstream
        ``evidence_integrity`` computation.

        Behaviour:
        - VLM extraction exists & no error → ``description`` = caption,
          ``visible_text`` = measured OCR, ``text_verified=True``
        - VLM missing / errored → ``description`` defaults to empty string,
          ``visible_text`` defaults to empty list, ``text_verified=None``.
          The agent will see "no description" rather than a GT-derived stub —
          this is the honest signal that Step 6.5 did not run for this photo.
        - Face detection exists → ``visible_face_ids`` = measured ids
          (excluding ``unknown_*`` clusters)
        - Verification exists → status stashed into
          ``ground_truth.verification_status``
        """
        vlm_map = step65.get("vlm", {})
        faces_map = step65.get("faces", {})
        verify_map = step65.get("verify", {})
        verify_status: dict[str, str] = {}

        for src in photo_sources:
            pid = str(src.get("photo_id") or "")
            if not pid:
                continue

            vlm_entry = vlm_map.get(pid)
            if isinstance(vlm_entry, dict) and not vlm_entry.get("error"):
                caption = vlm_entry.get("caption")
                if isinstance(caption, str) and caption.strip():
                    src["description"] = caption.strip()
                else:
                    src.setdefault("description", "")
                measured_text = vlm_entry.get("visible_text") or []
                if isinstance(measured_text, list):
                    src["visible_text"] = [str(t).strip() for t in measured_text if str(t).strip()]
                else:
                    src.setdefault("visible_text", [])
                src["text_verified"] = True
            else:
                # VLM did not run (or errored) — agent gets no description.
                # This is intentional: do not synthesise a GT-derived stub.
                src.setdefault("description", "")
                src.setdefault("visible_text", [])
                src.setdefault("text_verified", None)

            faces_entry = faces_map.get(pid)
            if isinstance(faces_entry, list):
                detected = [
                    str(f.get("face_id") or "").strip()
                    for f in faces_entry
                    if isinstance(f, dict)
                    and str(f.get("face_id") or "").strip()
                    and not str(f.get("face_id") or "").startswith("unknown_")
                ]
                src["visible_face_ids"] = list(dict.fromkeys(detected))

            verify_entry = verify_map.get(pid)
            if isinstance(verify_entry, dict):
                status = str(verify_entry.get("status") or "").strip()
                if status:
                    verify_status[pid] = status
                    gt = src.setdefault("ground_truth", {})
                    if isinstance(gt, dict):
                        gt["verification_status"] = status

        return verify_status

    def build(
        self,
        *,
        profile: dict[str, Any],
        graph: dict[str, Any],
        reasoning: dict[str, Any],
        timeline: dict[str, Any],
        manifest_records: list[dict[str, Any]] | None = None,
        manifest_backend: str | None = None,
        ambient_plan_path: Path | None = None,
        save_ambient_plan: bool = True,
        step65: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        manifest_records = manifest_records or []
        manifest_map = {
            str(item.get("photo_id", "")).strip(): dict(item)
            for item in manifest_records
            if str(item.get("photo_id", "")).strip()
        }
        device_plan = _build_user_device_plan(profile, timeline)
        event_map, kp_event_map = self._build_event_maps(timeline)
        face_catalog, person_to_face, face_to_person = self._build_face_catalog(profile, graph, reasoning)
        ambient_plan = self._load_or_generate_ambient_plan(
            profile=profile,
            graph=graph,
            timeline=timeline,
            ambient_plan_path=ambient_plan_path,
            save_ambient_plan=save_ambient_plan,
        )

        photo_sources: list[dict[str, Any]] = []
        photo_sources.extend(
            self._build_key_evidence_sources(
                profile=profile,
                reasoning=reasoning,
                manifest_map=manifest_map,
                kp_event_map=kp_event_map,
                face_to_person=face_to_person,
                device_plan=device_plan,
            )
        )
        photo_sources.extend(
            self._build_event_sources(
                profile=profile,
                graph=graph,
                timeline=timeline,
                manifest_map=manifest_map,
                person_to_face=person_to_face,
                face_to_person=face_to_person,
                device_plan=device_plan,
            )
        )
        photo_sources.extend(
            self._build_ambient_sources(
                profile=profile,
                graph=graph,
                reasoning=reasoning,
                timeline=timeline,
                ambient_plan=ambient_plan,
                manifest_map=manifest_map,
                person_to_face=person_to_face,
                face_to_person=face_to_person,
                device_plan=device_plan,
            )
        )

        # VLM-first: install Step 6.5 measurements as the agent-visible signals.
        # description ← caption, visible_text ← measured OCR, visible_face_ids
        # ← detected face clusters. No GT short-circuit.
        verify_status_map = self._apply_step65_overrides(photo_sources, step65 or {})

        # Text-side NER over (caption, visible_text). Each photo is extracted
        # independently to keep JSON outputs small and stable; calls are made
        # concurrently across the local Qwen3.6 endpoint pool.
        text_entities_map = self._extract_text_entities(photo_sources)

        # gps_* metadata still needs scrubbing — anchor labels like
        # "Kevin and Emily apartment" would otherwise reveal owner identity.
        # This is independent of agent-visible description (which now comes
        # from VLM caption and is allowed to mention OCR-visible names).
        name_replacements = _build_name_replacements(profile, graph)

        album_photos: list[dict[str, Any]] = []
        photo_annotations: list[dict[str, Any]] = []
        for source in photo_sources:
            visible_text = source.get("visible_text", [])
            description = source.get("description", "")
            raw_meta = source.get("metadata", {}) or {}
            allowed = " | ".join(str(t) for t in (visible_text or []))
            scrubbed_meta = dict(raw_meta)
            for k in ("gps_city", "gps_location"):
                v = scrubbed_meta.get(k)
                if isinstance(v, str) and name_replacements:
                    scrubbed_meta[k] = _replace_name_mentions(
                        v, name_replacements, allowed_text=allowed,
                    )
            album_photos.append(
                {
                    "photo_id": source["photo_id"],
                    "photo_role": source["photo_role"],
                    "event_id": source.get("event_id"),
                    "year_month": source.get("year_month", ""),
                    "category": source.get("category"),
                    "description": description,
                    "visible_face_ids": source.get("visible_face_ids", []),
                    "visible_text": visible_text,
                    "text_entities": text_entities_map.get(source["photo_id"], []),
                    "text_verified": source.get("text_verified"),
                    "metadata": scrubbed_meta,
                    "render_layout": source.get("render_layout", {}),
                    "has_image": bool(source.get("image_path")),
                    "image_path": source.get("image_path"),
                }
            )
            photo_annotations.append(dict(source["ground_truth"]))

        album_photos.sort(key=lambda item: (item.get("metadata", {}).get("timestamp", ""), item.get("photo_id", "")))
        photo_annotations.sort(key=lambda item: item.get("photo_id", ""))

        # Obfuscate the agent-visible album view so construction-time IDs,
        # owner labels, roles, and event IDs remain evaluator-only.
        # photo_annotations remains keyed by original photo_id because it lives
        # only in ground_truth.
        user_id_for_seed = str(profile.get("user_id", ""))
        seed_material = f"{user_id_for_seed}:album_obfuscation_v1".encode("utf-8")
        obfuscation_seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
        album_photos, traceability = self._obfuscate_album_for_agent(
            album_photos=album_photos,
            face_catalog=face_catalog,
            rng_seed=obfuscation_seed,
        )

        # role_counts uses photo_annotations (GT-side, retains photo_role)
        # since album_photos no longer carries photo_role after obfuscation.
        role_counts: dict[str, int] = {}
        for ann in photo_annotations:
            role = str(ann.get("photo_role", "unknown"))
            role_counts[role] = role_counts.get(role, 0) + 1

        result = {
            "user_id": profile.get("user_id", ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "benchmark_version": "v1",
            "album_backend": manifest_backend,
            "stats": {
                "total_photos": len(album_photos),
                "photos_with_images": sum(1 for item in album_photos if item.get("has_image")),
                "unique_face_ids": len({fid for item in album_photos for fid in item.get("visible_face_ids", [])}),
                "role_counts": role_counts,
            },
            "album_data": {
                "photos": album_photos,
            },
            "ground_truth": self._build_ground_truth(
                profile=profile,
                graph=graph,
                reasoning=reasoning,
                timeline=timeline,
                face_catalog=face_catalog,
                photo_annotations=photo_annotations,
                ambient_plan=ambient_plan,
                event_map=event_map,
                verify_status_map=verify_status_map,
            ),
        }
        # Stash obfuscation traceability inside ground_truth so evaluators can
        # map agent-produced public ids
        # back to original GT ids during evaluation. Agent never reads
        # ground_truth, so this does not leak.
        result["ground_truth"]["traceability"] = traceability
        # Phase 5: synthesize a top-level ``questions`` array directly from
        # the just-built ground_truth so the HEIR evaluator can score
        # without cross-referencing facts[].atomic_targets[] against parent
        # fact.target / person.canonical_name. Each question carries
        # evidence_kp_ids for traceability.
        node_difficulty = self._derive_node_alignment_difficulty(reasoning)
        result["questions"] = self._build_questions(
            ground_truth=result["ground_truth"],
            owner_name=str(profile.get("name") or ""),
            node_alignment_difficulty=node_difficulty,
        )
        # Augment top-level stats with Q&A coverage so consumers can sanity-check
        # the export at a glance.
        result["stats"]["total_questions"] = len(result["questions"])
        result["stats"]["questions_by_source"] = dict(
            Counter(q.get("source", "") for q in result["questions"])
        )
        return result

    def _load_manifest(self, user_dir: Path, preferred_backend: str) -> tuple[list[dict[str, Any]], str | None]:
        order = [preferred_backend, "gemini", "flux", "qwen"]
        seen: set[str] = set()
        for backend in order:
            if backend in seen:
                continue
            seen.add(backend)
            manifest_path = user_dir / f"photo_manifest_{backend}.json"
            if manifest_path.exists():
                loaded = load_json(manifest_path)
                # Phase 5 envelope compatibility: manifest is now wrapped
                # ``{"schema_version", "stats", "items": [...]}``; legacy
                # bare-array manifests are still accepted.
                if isinstance(loaded, dict) and isinstance(loaded.get("items"), list):
                    return loaded["items"], backend
                if isinstance(loaded, list):
                    return loaded, backend
        return [], None

    def _build_event_maps(self, timeline: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        event_map: dict[str, dict[str, Any]] = {}
        kp_event_map: dict[str, str] = {}
        for event in timeline.get("events", []):
            event_id = str(event.get("event_id", "")).strip()
            if not event_id:
                continue
            event_map[event_id] = dict(event)
            for kp_id in _coerce_str_list(event.get("kp_ids")):
                kp_event_map[kp_id] = event_id
        return event_map, kp_event_map

    def _build_face_catalog(
        self,
        profile: dict[str, Any],
        graph: dict[str, Any],
        reasoning: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
        person_to_face: dict[str, str] = {}
        for node_path in reasoning.get("node_paths", []):
            person_id = str(node_path.get("person_id", "")).strip()
            face_id = str(node_path.get("face_id", "")).strip()
            if person_id and face_id:
                person_to_face[person_id] = face_id

        catalog = [
            {
                "face_id": "owner",
                "person_id": "owner",
                "name": profile.get("name", ""),
                "relation": "self",
                "relation_category": "self",
            }
        ]
        face_to_person = {"owner": "owner"}
        for node in graph.get("nodes", []):
            person_id = str(node.get("person_id", "")).strip()
            face_id = person_to_face.get(person_id, "")
            if not person_id or not face_id:
                continue
            catalog.append(
                {
                    "face_id": face_id,
                    "person_id": person_id,
                    "name": node.get("name", ""),
                    "relation": node.get("relation", ""),
                    "relation_category": node.get("relation_category", ""),
                }
            )
            face_to_person[face_id] = person_id
        return catalog, person_to_face, face_to_person

    def _load_or_generate_ambient_plan(
        self,
        *,
        profile: dict[str, Any],
        graph: dict[str, Any],
        timeline: dict[str, Any],
        ambient_plan_path: Path | None,
        save_ambient_plan: bool,
    ) -> dict[str, Any]:
        if ambient_plan_path is not None and ambient_plan_path.exists():
            plan = load_json(ambient_plan_path)
            if isinstance(plan, dict):
                return plan
        plan = self._photo_helper._generate_ambient_plan(profile, graph, timeline)
        if ambient_plan_path is not None and save_ambient_plan:
            save_json(plan, ambient_plan_path)
        return plan

    def _build_key_evidence_sources(
        self,
        *,
        profile: dict[str, Any],
        reasoning: dict[str, Any],
        manifest_map: dict[str, dict[str, Any]],
        kp_event_map: dict[str, str],
        face_to_person: dict[str, str],
        device_plan: dict[str, Any],
    ) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        owner_name = profile.get("name", "")
        fact_paths = reasoning.get("fact_paths", [])
        for fact_index, fact_path in enumerate(fact_paths, 1):
            for kp in fact_path.get("key_photos", []):
                sources.append(
                    self._build_key_photo_source(
                        kp=kp,
                        source_kind="fact_path",
                        target=fact_path.get("target", ""),
                        evidence_id=f"fact_{fact_index:03d}",
                        person_id=None,
                        fact_metadata={
                            "fact_category": fact_path.get("fact_category", ""),
                            "evaluation_role": fact_path.get("evaluation_role", ""),
                            "evidence_policy": fact_path.get("evidence_policy", {}),
                            "evidence_distribution": fact_path.get("evidence_distribution", {}),
                            "atomic_targets": fact_path.get("atomic_targets", []),
                        },
                        manifest_entry=manifest_map.get(str(kp.get("kp_id", "")).strip(), {}),
                        event_id=kp_event_map.get(str(kp.get("kp_id", "")).strip()),
                        face_to_person=face_to_person,
                        owner_name=owner_name,
                        device_plan=device_plan,
                        profile=profile,
                    )
                )

        node_paths = reasoning.get("node_paths", [])
        for node_path in node_paths:
            related_face_id = str(node_path.get("face_id", "")).strip()
            person_id = str(node_path.get("person_id", "")).strip() or None
            node_target = node_path.get("name", "") or person_id or ""
            for kp in node_path.get("key_photos", []):
                sources.append(
                    self._build_key_photo_source(
                        kp=kp,
                        source_kind="node_path",
                        target=node_target,
                        evidence_id=person_id or "",
                        person_id=person_id,
                        manifest_entry=manifest_map.get(str(kp.get("kp_id", "")).strip(), {}),
                        event_id=kp_event_map.get(str(kp.get("kp_id", "")).strip()),
                        face_to_person=face_to_person,
                        owner_name=owner_name,
                        device_plan=device_plan,
                        related_face_id=related_face_id,
                        profile=profile,
                    )
                )
        return sources

    def _build_key_photo_source(
        self,
        *,
        kp: dict[str, Any],
        source_kind: str,
        target: str,
        evidence_id: str,
        person_id: str | None,
        manifest_entry: dict[str, Any],
        event_id: str | None,
        face_to_person: dict[str, str],
        owner_name: str,
        device_plan: dict[str, Any],
        related_face_id: str = "",
        fact_metadata: dict[str, Any] | None = None,
        profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        photo_id = str(kp.get("kp_id", "")).strip()
        photo_type = str(kp.get("photo_type", "scene")).strip() or "scene"
        text_subtype = str(kp.get("text_subtype", "")).strip()
        visible_text = _coerce_str_list(kp.get("must_read_text"))
        visible_face_ids = self._infer_key_photo_visible_faces(kp, related_face_id=related_face_id)
        text_mentions_raw = kp.get("text_mentions") or []
        text_mentions: list[dict[str, Any]] = []
        if isinstance(text_mentions_raw, list):
            for m in text_mentions_raw:
                if not isinstance(m, dict):
                    continue
                kind = str(m.get("mention_kind") or "").strip().lower()
                if kind not in {"name", "relation"}:
                    continue
                text = str(m.get("text") or "").strip()
                if not text:
                    continue
                text_mentions.append({
                    "mention_kind": kind,
                    "text": text,
                    "resolves_to_person_ids": [
                        str(pid).strip()
                        for pid in (m.get("resolves_to_person_ids") or [])
                        if str(pid).strip()
                    ],
                    "alias_of": str(m.get("alias_of") or "").strip(),
                })
        year_month = str(kp.get("time_hint", "")).strip() or "2024-01"
        device = _device_for_year_month(device_plan or {"primary_device": "iPhone 12"}, year_month)
        # Use full profile when available so _parse_location can fall back to
        # profile.raw_attributes.city for location_hints that don't contain a
        # comma-separated city tail. Without this, gps_city ends up as "" for
        # most natural-language location_hints (e.g. "office near campus").
        profile_for_meta = profile or {"name": owner_name, "raw_attributes": {}}
        metadata = dict(manifest_entry.get("metadata") or self._photo_helper._build_metadata_from_fields(
            year_month,
            kp.get("location_hint", ""),
            profile_for_meta,
            0,
            device=device,
        ))
        metadata["device"] = device
        render_layout = manifest_entry.get("render_layout") or self._default_key_photo_layout(photo_type, text_subtype)
        notes = _coerce_str_list(kp.get("must_show"))[:4]
        atomic_target_ids = [
            f"{evidence_id}_atom_{idx:03d}"
            for idx, item in enumerate((fact_metadata or {}).get("atomic_targets", []) or [], 1)
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        safe_location = str(kp.get("location_hint", "")).strip()
        description_input = {
            "photo_id": photo_id,
            "photo_role": "key_evidence",
            "year_month": str(kp.get("time_hint", "")).strip(),
            "location": safe_location,
            "summary": str(kp.get("description", "")).strip(),
            "visible_people": self._visible_people_summary(visible_face_ids),
            "visual_notes": notes,
            "visible_text": visible_text,
        }
        return {
            "photo_id": photo_id,
            "photo_role": "key_evidence",
            "event_id": event_id,
            "year_month": str(kp.get("time_hint", "")).strip(),
            "category": None,
            "visible_face_ids": visible_face_ids,
            "visible_text": visible_text,
            "metadata": metadata,
            "render_layout": render_layout,
            "image_path": manifest_entry.get("photo_path"),
            "description_input": description_input,
            "ground_truth": {
                "photo_id": photo_id,
                "photo_role": "key_evidence",
                "source_ref_type": source_kind,
                "source_ref_id": photo_id,
                "evidence_id": evidence_id,
                "event_id": event_id,
                "year_month": str(kp.get("time_hint", "")).strip(),
                "photo_type": photo_type,
                "text_subtype": text_subtype,
                "target": target,
                "fact_category": (fact_metadata or {}).get("fact_category", ""),
                "evaluation_role": (fact_metadata or {}).get("evaluation_role", ""),
                "evidence_policy": (fact_metadata or {}).get("evidence_policy", {}),
                "evidence_distribution": (fact_metadata or {}).get("evidence_distribution", {}),
                "atomic_target_ids": atomic_target_ids,
                "planned_description": str(kp.get("description", "")).strip(),
                "evidence_role": str(kp.get("evidence_role", "")).strip(),
                "must_show": _coerce_str_list(kp.get("must_show")),
                "must_read_text": visible_text,
                "text_mentions": text_mentions,
                "visible_face_ids": visible_face_ids,
                "person_ids_in_frame": [face_to_person.get(face_id, face_id) for face_id in visible_face_ids],
                # Phase 2: KP linkage to its planned location anchor
                # (eval-only; agent never sees loc_id in album_data).
                "loc_id": str(
                    (kp.get("location") or {}).get("loc_id")
                    if isinstance(kp.get("location"), dict)
                    else manifest_entry.get("traceability", {}).get("loc_id", "")
                ).strip(),
                "expected_metadata": (
                    kp.get("expected_metadata")
                    if isinstance(kp.get("expected_metadata"), dict)
                    else manifest_entry.get("traceability", {}).get("expected_metadata", {})
                ) or {},
            },
        }

    def _build_event_sources(
        self,
        *,
        profile: dict[str, Any],
        graph: dict[str, Any],
        timeline: dict[str, Any],
        manifest_map: dict[str, dict[str, Any]],
        person_to_face: dict[str, str],
        face_to_person: dict[str, str],
        device_plan: dict[str, Any],
    ) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        allowed_people_ids = {
            str(node.get("person_id", "")).strip()
            for node in graph.get("nodes", [])
            if str(node.get("person_id", "")).strip()
        }
        # Build person_id -> name mapping from graph
        person_id_to_name = {}
        for node in graph.get("nodes", []):
            pid = str(node.get("person_id", "")).strip()
            name = str(node.get("name", "")).strip()
            if pid and name:
                person_id_to_name[pid] = name
        for event in timeline.get("events", []):
            event_id = str(event.get("event_id", "")).strip()
            if not event_id:
                continue
            kp_count = len(_coerce_str_list(event.get("kp_ids")))
            try:
                total = max(int(event.get("photo_count", 0) or 0), 0)
            except (TypeError, ValueError):
                total = 0
            remaining = max(total - kp_count, 0)
            slots = [slot for slot in event.get("photo_slots", []) if isinstance(slot, dict)]
            slots = sorted(slots, key=lambda item: int(item.get("slot_rank", 0) or 0))
            for idx in range(remaining):
                photo_id = f"{event_id}_{kp_count + idx + 1:03d}"
                manifest_entry = manifest_map.get(photo_id, {})
                raw_slot = slots[idx] if idx < len(slots) else None
                people_plan = self._normalize_event_people_plan(event, raw_slot, allowed_people_ids)
                required_faces = _resolve_people_to_face_ids(people_plan.get("required_people", []), person_to_face)
                optional_faces = _resolve_people_to_face_ids(people_plan.get("optional_people", []), person_to_face)
                visible_face_ids = self._derive_visible_face_ids(
                    required_face_ids=required_faces,
                    optional_face_ids=optional_faces,
                    face_visibility=str(people_plan.get("face_visibility", "medium")),
                    framing_hint=str(people_plan.get("framing_hint", "environment")),
                )
                year_month = str(event.get("year_month", "")).strip() or "2024-01"
                device = _device_for_year_month(device_plan, year_month)
                metadata = dict(manifest_entry.get("metadata") or self._photo_helper._build_metadata_from_fields(
                    year_month,
                    event.get("location", ""),
                    profile,
                    idx,
                    device=device,
                ))
                metadata["device"] = device
                render_layout = manifest_entry.get("render_layout") or self._default_event_layout(event)
                description_input = {
                    "photo_id": photo_id,
                    "photo_role": "event_scene",
                    "year_month": str(event.get("year_month", "")).strip(),
                    "location": str(event.get("location", "")).strip(),
                    "summary": str(event.get("label", "")).strip(),
                    "visible_people": self._visible_people_summary(visible_face_ids),
                    "visual_notes": [
                        f"event type: {event.get('event_type', '')}",
                        f"framing: {people_plan.get('framing_hint', '')}",
                        f"face visibility: {people_plan.get('face_visibility', '')}",
                    ],
                    "visible_text": (
                        ([str(event.get("text_content", "")).strip()] if str(event.get("text_content", "")).strip() else [])
                        + [person_id_to_name[pid] for pid in _coerce_str_list(event.get("participants", [])) if pid in person_id_to_name]
                    ),
                }
                sources.append(
                    {
                        "photo_id": photo_id,
                        "photo_role": "event_scene",
                        "event_id": event_id,
                        "year_month": str(event.get("year_month", "")).strip(),
                        "category": None,
                        "visible_face_ids": visible_face_ids,
                        "visible_text": description_input["visible_text"],
                        "metadata": metadata,
                        "render_layout": render_layout,
                        "image_path": manifest_entry.get("photo_path"),
                        "description_input": description_input,
                        "ground_truth": {
                            "photo_id": photo_id,
                            "photo_role": "event_scene",
                            "source_ref_type": "timeline_event",
                            "source_ref_id": event_id,
                            "event_id": event_id,
                            "event_label": event.get("label", ""),
                            "event_type": event.get("event_type", ""),
                            "participants": _coerce_str_list(event.get("participants")),
                            "year_month": str(event.get("year_month", "")).strip(),
                            "signal_goal": people_plan.get("signal_goal", ""),
                            "face_visibility": people_plan.get("face_visibility", ""),
                            "framing_hint": people_plan.get("framing_hint", ""),
                            "required_people": people_plan.get("required_people", []),
                            "optional_people": people_plan.get("optional_people", []),
                            "visible_face_ids": visible_face_ids,
                            "person_ids_in_frame": [face_to_person.get(face_id, face_id) for face_id in visible_face_ids],
                        },
                    }
                )
        return sources

    def _build_ambient_sources(
        self,
        *,
        profile: dict[str, Any],
        graph: dict[str, Any],
        reasoning: dict[str, Any],
        timeline: dict[str, Any],
        ambient_plan: dict[str, Any],
        manifest_map: dict[str, dict[str, Any]],
        person_to_face: dict[str, str],
        face_to_person: dict[str, str],
        device_plan: dict[str, Any],
    ) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        allowed_people_ids = {
            str(node.get("person_id", "")).strip()
            for node in graph.get("nodes", [])
            if str(node.get("person_id", "")).strip()
        }
        _recurring_summary, recurring_people = _build_recurring_people_summary(profile, graph, timeline)
        for index, raw_item in enumerate(ambient_plan.get("ambient_photos", [])):
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            defaults = _default_ambient_people_plan(item, recurring_people)
            people_plan = _normalize_people_plan(
                item,
                allowed_people_ids=allowed_people_ids,
                default_required_people=list(defaults.get("required_people", [])),
                default_optional_people=list(defaults.get("optional_people", [])),
                default_signal_goal=str(defaults.get("signal_goal", "owner_lifestyle")),
                default_face_visibility=str(defaults.get("face_visibility", "medium")),
                default_framing_hint=str(defaults.get("framing_hint", "environment")),
            )
            amb_id = str(item.get("amb_id", "")).strip() or f"amb_{index + 1:03d}"
            required_faces = _resolve_people_to_face_ids(people_plan.get("required_people", []), person_to_face)
            optional_faces = _resolve_people_to_face_ids(people_plan.get("optional_people", []), person_to_face)
            visible_face_ids = self._derive_visible_face_ids(
                required_face_ids=required_faces,
                optional_face_ids=optional_faces,
                face_visibility=str(people_plan.get("face_visibility", "medium")),
                framing_hint=str(people_plan.get("framing_hint", "environment")),
            )
            manifest_entry = manifest_map.get(amb_id, {})
            year_month = str(item.get("year_month", "")).strip() or "2024-01"
            device = _device_for_year_month(device_plan, year_month)
            location_hint = str(item.get("location_hint", "")).strip()
            metadata = dict(manifest_entry.get("metadata") or self._photo_helper._build_metadata_from_fields(
                year_month,
                location_hint,
                profile,
                0,
                device=device,
                stable_key=f"{profile.get('user_id', '')}:{amb_id}",
            ))
            metadata["device"] = device
            render_layout = manifest_entry.get("render_layout") or self._default_ambient_layout(item)
            # Propagate must_read_text → visible_text for text-bearing ambient categories
            # (screenshot / memory_object / object_detail that carry explicit readable strings).
            # For image-only categories (food, selfie, nature, …) this is an empty list.
            ambient_visible_text = _coerce_str_list(item.get("must_read_text"))
            description_input = {
                "photo_id": amb_id,
                "photo_role": "ambient",
                "year_month": year_month,
                "location": location_hint,
                "summary": str(item.get("description", "")).strip(),
                "visible_people": self._visible_people_summary(visible_face_ids),
                "visual_notes": [
                    f"category: {item.get('category', '')}",
                    f"framing: {people_plan.get('framing_hint', '')}",
                    f"face visibility: {people_plan.get('face_visibility', '')}",
                ],
                "visible_text": ambient_visible_text,
            }
            sources.append(
                {
                    "photo_id": amb_id,
                    "photo_role": "ambient",
                    "event_id": None,
                    "year_month": year_month,
                    "category": item.get("category"),
                    "visible_face_ids": visible_face_ids,
                    "visible_text": ambient_visible_text,
                    "metadata": metadata,
                    "render_layout": render_layout,
                    "image_path": manifest_entry.get("photo_path"),
                    "description_input": description_input,
                    "ground_truth": {
                        "photo_id": amb_id,
                        "photo_role": "ambient",
                        "source_ref_type": "ambient_plan",
                        "source_ref_id": amb_id,
                        "event_id": None,
                        "year_month": year_month,
                        "category": item.get("category", ""),
                        "plan_description": item.get("description", ""),
                        "must_read_text": ambient_visible_text,
                        "location_hint": location_hint,
                        "supports": _coerce_str_list(item.get("supports")),
                        "recurrence_role": str(item.get("recurrence_role", "")),
                        "signal_goal": people_plan.get("signal_goal", ""),
                        "face_visibility": people_plan.get("face_visibility", ""),
                        "framing_hint": people_plan.get("framing_hint", ""),
                        "required_people": people_plan.get("required_people", []),
                        "optional_people": people_plan.get("optional_people", []),
                        "visible_face_ids": visible_face_ids,
                        "person_ids_in_frame": [face_to_person.get(face_id, face_id) for face_id in visible_face_ids],
                    },
                }
            )
        return sources

    def _infer_key_photo_visible_faces(self, kp: dict[str, Any], *, related_face_id: str = "") -> list[str]:
        photo_type = str(kp.get("photo_type", "scene")).strip()
        if photo_type in _TEXT_ONLY_TYPES:
            return []
        visible = _dedupe_str_list(_coerce_str_list(kp.get("required_faces")))
        if photo_type in _SCENE_TYPES and related_face_id and related_face_id not in visible:
            visible.append(related_face_id)
        return _dedupe_str_list(visible)

    def _default_key_photo_layout(self, photo_type: str, text_subtype: str) -> dict[str, Any]:
        if photo_type == "screenshot":
            return _default_render_layout("9:16", "phone_screenshot_portrait")
        if photo_type == "document":
            return _default_render_layout("4:5", "document_portrait")
        if photo_type == "scene_with_text" and text_subtype:
            return _default_render_layout("4:3", "scene_text_hybrid")
        return _default_render_layout("4:3", "scene_landscape")

    def _default_event_layout(self, event: dict[str, Any]) -> dict[str, Any]:
        if str(event.get("event_type", "")).strip() == "text_rich":
            return _default_render_layout("9:16", "phone_screenshot_portrait")
        return _default_render_layout("4:3", "scene_landscape")

    def _default_ambient_layout(self, item: dict[str, Any]) -> dict[str, Any]:
        category = str(item.get("category", "")).strip()
        if category == "screenshot":
            return _default_render_layout("9:16", "phone_screenshot_portrait")
        if category == "selfie":
            return _default_render_layout("4:5", "portrait_selfie")
        return _default_render_layout("4:3", "scene_landscape")

    def _normalize_event_people_plan(
        self,
        event: dict[str, Any],
        raw_slot: dict[str, Any] | None,
        allowed_people_ids: set[str],
    ) -> dict[str, Any]:
        participants = [
            pid for pid in _coerce_str_list(event.get("participants"))
            if pid in allowed_people_ids
        ]
        if raw_slot is not None:
            return _normalize_people_plan(
                raw_slot,
                allowed_people_ids=allowed_people_ids,
                default_required_people=[],
                default_optional_people=[],
                default_signal_goal="text_artifact" if event.get("event_type") == "text_rich" else "owner_lifestyle",
                default_face_visibility="none" if event.get("event_type") == "text_rich" else "medium",
                default_framing_hint="artifact" if event.get("event_type") == "text_rich" else "environment",
            )
        if event.get("event_type") == "text_rich":
            return {
                "required_people": [],
                "optional_people": [],
                "signal_goal": "text_artifact",
                "face_visibility": "none",
                "framing_hint": "artifact",
            }
        if participants:
            return {
                "required_people": ["owner", participants[0]],
                "optional_people": participants[1:3],
                "signal_goal": "co_occurrence",
                "face_visibility": "medium",
                "framing_hint": "pair" if len(participants) == 1 else "group",
            }
        return {
            "required_people": ["owner"],
            "optional_people": [],
            "signal_goal": "owner_lifestyle",
            "face_visibility": "medium",
            "framing_hint": "single_portrait",
        }

    def _derive_visible_face_ids(
        self,
        *,
        required_face_ids: list[str],
        optional_face_ids: list[str],
        face_visibility: str,
        framing_hint: str,
    ) -> list[str]:
        visibility = str(face_visibility or "medium").strip()
        if visibility == "none":
            return []
        faces = _dedupe_str_list(list(required_face_ids))
        if visibility == "weak":
            return faces
        if framing_hint == "group":
            faces.extend(optional_face_ids[:2])
        elif framing_hint == "pair":
            faces.extend(optional_face_ids[:1])
        elif not faces and optional_face_ids:
            faces.extend(optional_face_ids[:1])
        return _dedupe_str_list(faces)

    def _visible_people_summary(self, visible_face_ids: list[str]) -> str:
        count = len(visible_face_ids)
        if count <= 0:
            return "no clearly identifiable face is visible"
        if visible_face_ids == ["owner"]:
            return "the owner is the only clearly identifiable person"
        if count == 1:
            return "one clearly identifiable recurring person is visible"
        if count == 2 and "owner" in visible_face_ids:
            return "the owner and one other clearly identifiable person are visible"
        return f"{count} clearly identifiable people are visible"

    def _extract_text_entities_deterministic(self, sources: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
        """Fast deterministic text-side entity extraction for benchmark export."""
        import re

        stop = {
            "The", "This", "That", "A", "An", "And", "Date", "Time", "Location",
            "Events", "Register", "Confirmed", "Route", "Notes", "Attendees",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            "January", "February", "March", "April", "May", "June", "July", "August",
            "September", "October", "November", "December",
        }
        person_pat = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z][a-z]+['’]s)(?:\s+(?:[A-Z][a-z]+|[A-Z][a-z]+['’]s)){1,3}\b")
        org_terms = ("Association", "Company", "LLC", "Inc", "School", "University", "Clinic", "Hospital", "Club", "Meetup", "Bar", "Office")
        event_terms = ("Ride", "Mixer", "Conference", "Reception", "Birthday", "Brunch", "Dinner", "Party", "Meetup", "Event")
        loc_terms = ("Street", "Avenue", "Road", "Park", "Marina", "Beach", "Center", "Centre", "Causeway", "Miami", "London", "Brooklyn", "New York")

        def classify(surface: str) -> str:
            if any(term in surface for term in org_terms):
                return "organization"
            if any(term in surface for term in event_terms):
                return "event"
            if any(term in surface for term in loc_terms):
                return "location"
            return "person"

        out: dict[str, list[dict[str, str]]] = {}
        for s in sources:
            pid = str(s.get("photo_id", "")).strip()
            if not pid:
                continue
            entities: list[dict[str, str]] = []
            seen: set[tuple[str, str]] = set()
            pairs = [("caption", str(s.get("description") or ""))]
            pairs.extend(("visible_text", str(t)) for t in (s.get("visible_text") or []))
            for source, text in pairs:
                for match in person_pat.finditer(text):
                    surface = match.group(0).strip(" -—:,.()[]")
                    if not surface or surface.split()[0] in stop:
                        continue
                    key = (surface, source)
                    if key in seen:
                        continue
                    seen.add(key)
                    entities.append({"surface": surface, "entity_type": classify(surface), "source": source})
            out[pid] = entities
        return out

    def _extract_text_entities(self, sources: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
        """Text-side NER over agent-visible signals (VLM caption + OCR list).

        This extracts explicitly mentioned text entities only. Visual-side
        person entities are represented separately by face-recognition clusters
        in ``visible_face_ids``; this method must not infer profile facts,
        relations, visual objects, or event context tags.

        Returns
        -------
        dict[photo_id, list[{"surface": str, "source": "caption" | "visible_text", "entity_type": str?}]]
            Photos with no eligible signal are absent from the result. The
            caller should default to an empty list when assembling
            ``album_data.photos``.
        """
        eligible = [
            s for s in sources
            if (s.get("description") or "").strip() or s.get("visible_text")
        ]
        if not eligible:
            return {}

        out: dict[str, list[dict[str, str]]] = {}

        def _one(source: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
            pid = str(source.get("photo_id", "")).strip()
            item = {
                "photo_id": pid,
                "caption": source.get("description", "") or "",
                "visible_text": list(source.get("visible_text") or []),
            }
            prompt = self._text_entities_prompt_tpl.format(
                items_json=json.dumps([item], ensure_ascii=False, indent=2),
            )
            last_exc: Exception | None = None
            for attempt in range(self._text_entity_max_retries + 1):
                try:
                    raw = self._llm.simple(
                        prompt=prompt,
                        system=self.TEXT_ENTITY_SYSTEM,
                        temperature=0.0,
                        max_tokens=4096,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    parsed = _extract_json(raw) or {}
                    photos = parsed.get("photos") or []
                    matched = None
                    for item_out in photos:
                        if isinstance(item_out, dict) and str(item_out.get("photo_id") or "").strip() == pid:
                            matched = item_out
                            break
                    if matched is None:
                        matched = photos[0] if photos and isinstance(photos[0], dict) else {"entities": []}
                    return pid, self._clean_text_entities(matched.get("entities") or [])
                except Exception as exc:
                    last_exc = exc
                    if attempt >= self._text_entity_max_retries:
                        logger.warning("text_entities single-photo extraction failed for %s: %s", pid, last_exc)
            return pid, []

        with ThreadPoolExecutor(max_workers=self._text_entity_max_workers) as pool:
            futures = [pool.submit(_one, source) for source in eligible]
            for fut in as_completed(futures):
                pid, entities = fut.result()
                if pid:
                    out[pid] = entities
        return out

    @staticmethod
    def _clean_text_entities(ents: object) -> list[dict[str, str]]:
        cleaned: list[dict[str, str]] = []
        valid_types = {"person", "organization", "location", "event"}
        for e in ents if isinstance(ents, list) else []:
            if not isinstance(e, dict):
                continue
            surface = str(e.get("surface", "")).strip()
            source_label = str(e.get("source", "")).strip()
            entity_type = str(e.get("entity_type", "person")).strip()
            if entity_type not in valid_types:
                entity_type = "person"
            if surface and source_label in ("caption", "visible_text"):
                cleaned.append({"surface": surface, "entity_type": entity_type, "source": source_label})
        return cleaned

    def _build_atomic_targets_for_fact(self, fact_path: dict[str, Any], fact_id: str) -> list[dict[str, Any]]:
        key_photo_ids = [
            str(kp.get("kp_id") or "").strip()
            for kp in fact_path.get("key_photos", [])
            if str(kp.get("kp_id") or "").strip()
        ]
        if not key_photo_ids:
            return []
        raw_targets = fact_path.get("atomic_targets") if isinstance(fact_path.get("atomic_targets"), list) else []
        records: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_targets, 1):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("target") or item.get("claim") or "").strip()
            if not text:
                continue
            records.append({
                "atomic_id": f"{fact_id}_atom_{idx:03d}",
                "parent_fact_id": fact_id,
                "parent_target": str(fact_path.get("target") or ""),
                "text": text,
                "role": str(item.get("role") or "evaluation_target"),
                "answer_type": str(item.get("answer_type") or "boolean"),
                "supporting_photo_ids": key_photo_ids,
                "evidence_need": str(item.get("evidence_need") or ""),
                "decomposition_rationale": str(item.get("decomposition_rationale") or item.get("rationale") or ""),
                "evidence_sufficiency": "parent_fact_supported",
            })
        if not records:
            target = str(fact_path.get("target") or "").strip()
            if target:
                records.append({
                    "atomic_id": f"{fact_id}_atom_001",
                    "parent_fact_id": fact_id,
                    "parent_target": target,
                    "text": target,
                    "role": "evaluation_target",
                    "answer_type": "boolean",
                    "supporting_photo_ids": key_photo_ids,
                    "evidence_need": "Use the parent fact key evidence chain.",
                    "decomposition_rationale": "Fallback atomic target for older reasoning paths without explicit decomposition.",
                    "evidence_sufficiency": "parent_fact_supported_fallback",
                })
        return records

    # ----------------------------------------------------------------------
    # Agent-visible album_data obfuscation
    # ----------------------------------------------------------------------
    # Rewrites the agent-visible album_data so that GT structure is not
    # leaked through naming. Specifically:
    #   - photo_id        : kp_fact_X_Y / kp_node_X_Y / evt_X_Y / amb_NNN
    #                       → photo_NNNN (4-digit zero-pad, ordered by
    #                       timestamp ascending; ties broken by original id)
    #   - face_id         : 'owner' + face_001..face_NNN
    #                       → face_001..face_(N+1) shuffled by per-user
    #                       deterministic seed; the 'owner' label disappears
    #                       from album_data (agent must self-discover which
    #                       face_id is the owner)
    #   - photo_role      : removed from agent-visible payload (kept in GT
    #                       photo_annotations for evaluation use)
    #   - event_id        : removed from agent-visible payload (agent must
    #                       infer event grouping from timestamp + location +
    #                       face co-occurrence)
    #   - image_path      : rewritten in-string to point to obfuscated id
    #                       (the on-disk PNG file is NOT renamed; evaluator
    #                       uses traceability map to resolve back)
    #   - render_layout   : layout_policy + requested_* removed (these are
    #                       GT design tags, not real image attributes)
    #   - text_verified   : removed (constant True for all photos)
    #   - has_image       : removed (constant True for all photos)
    #   - metadata.gps_*  : kept unchanged (user decision 2026-05-20: real
    #                       albums commonly have semantic location labels;
    #                       this signal is paper-relevant)
    # The traceability dict is returned alongside and stashed into
    # ground_truth.traceability by the caller.
    # ----------------------------------------------------------------------
    def _obfuscate_album_for_agent(
        self,
        *,
        album_photos: list[dict[str, Any]],
        face_catalog: list[dict[str, Any]],
        rng_seed: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Obfuscate album_photos in place. Returns (new_album_photos, traceability).

        ``album_photos`` is expected to already be sorted by timestamp.
        ``face_catalog`` is the GT face catalog (contains 'owner' + face_NNN).
        ``rng_seed`` is per-user deterministic; same user → same mapping.
        """
        # 1. Build photo_id map (timestamp-ordered photo_NNNN).
        photo_id_map: dict[str, str] = {}
        for idx, p in enumerate(album_photos, start=1):
            original_pid = str(p.get("photo_id", ""))
            if original_pid:
                photo_id_map[original_pid] = f"photo_{idx:04d}"

        # 2. Build face_id map. Catalog supplies the universe of face_ids
        #    (including the literal 'owner' label). We assign new sequential
        #    face_NNN labels and shuffle them so 'owner' lands on a random
        #    slot indistinguishable from non-owner face slots.
        original_face_ids: list[str] = []
        for entry in face_catalog or []:
            fid = str(entry.get("face_id", "")).strip()
            if fid and fid not in original_face_ids:
                original_face_ids.append(fid)
        # Defensive fallback: if face_catalog was empty, derive from
        # visible_face_ids appearing in album_photos.
        if not original_face_ids:
            seen: set[str] = set()
            for p in album_photos:
                for fid in p.get("visible_face_ids", []) or []:
                    s = str(fid).strip()
                    if s and s not in seen:
                        seen.add(s)
                        original_face_ids.append(s)
        n_faces = len(original_face_ids)
        new_face_ids = [f"face_{i+1:03d}" for i in range(n_faces)]
        rng = random.Random(rng_seed)
        rng.shuffle(new_face_ids)
        face_id_map: dict[str, str] = dict(zip(original_face_ids, new_face_ids))

        # 3. Apply mappings to album_photos.
        obfuscated_photos: list[dict[str, Any]] = []
        for p in album_photos:
            new_p = dict(p)
            original_pid = str(p.get("photo_id", ""))
            new_pid = photo_id_map.get(original_pid, original_pid)
            new_p["photo_id"] = new_pid

            # visible_face_ids: map each face_id, drop unknowns silently
            old_faces = p.get("visible_face_ids", []) or []
            new_p["visible_face_ids"] = [
                face_id_map.get(str(fid), str(fid)) for fid in old_faces
            ]

            # Drop GT-leaking fields outright
            new_p.pop("photo_role", None)
            new_p.pop("event_id", None)
            new_p.pop("text_verified", None)
            new_p.pop("has_image", None)

            # image_path: rewrite filename component to obfuscated photo_id
            ip = p.get("image_path") or ""
            if ip:
                new_p["image_path"] = re.sub(
                    r"/[^/]+\.png$", f"/{new_pid}.png", str(ip)
                )

            # render_layout: keep only real image attributes
            rl = dict(p.get("render_layout") or {})
            rl.pop("layout_policy", None)
            rl.pop("requested_width", None)
            rl.pop("requested_height", None)
            new_p["render_layout"] = rl

            # metadata: gps_location KEPT per user decision 2026-05-20
            # (no metadata mutation needed here)

            obfuscated_photos.append(new_p)

        traceability: dict[str, Any] = {
            "obfuscation_version": "v1",
            "rng_seed": int(rng_seed),
            "photo_id_map": photo_id_map,
            "face_id_map": face_id_map,
        }
        return obfuscated_photos, traceability

    def _build_ground_truth(
        self,
        *,
        profile: dict[str, Any],
        graph: dict[str, Any],
        reasoning: dict[str, Any],
        timeline: dict[str, Any],
        face_catalog: list[dict[str, Any]],
        photo_annotations: list[dict[str, Any]],
        ambient_plan: dict[str, Any],
        event_map: dict[str, dict[str, Any]],
        verify_status_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        verify_status_map = verify_status_map or {}

        def _evidence_integrity(kp_ids: list[str]) -> str:
            """Compute evidence integrity tag for a set of key_photo IDs.

            - "verified"   : all KPs passed verification (or were skipped as
                             OK) in Step 6.5
            - "partial"    : some KPs passed, some failed
            - "degraded"   : all KPs failed verification (or unverifiable)
            - "unverified" : no Step 6.5 data available for any KP
            """
            if not kp_ids:
                return "unverified"
            statuses = [verify_status_map.get(str(k), "") for k in kp_ids]
            if all(not s for s in statuses):
                return "unverified"
            ok = {"passed", "fixed", "skipped"}
            bad = {"failed", "unverified", "error"}
            ok_count = sum(1 for s in statuses if s in ok)
            bad_count = sum(1 for s in statuses if s in bad)
            if ok_count and not bad_count:
                return "verified"
            if bad_count and not ok_count:
                return "degraded"
            return "partial"

        node_path_map = {
            str(item.get("person_id", "")).strip(): item
            for item in reasoning.get("node_paths", [])
            if str(item.get("person_id", "")).strip()
        }
        facts = []
        for index, fact_path in enumerate(reasoning.get("fact_paths", []), 1):
            fact_id = f"fact_{index:03d}"
            kp_ids = [kp.get("kp_id", "") for kp in fact_path.get("key_photos", []) if kp.get("kp_id")]
            facts.append(
                {
                    "fact_id": fact_id,
                    "target": fact_path.get("target", ""),
                    "observable": bool(fact_path.get("observable", False)),
                    "fact_category": fact_path.get("fact_category", ""),
                    "evaluation_role": fact_path.get("evaluation_role", ""),
                    "evidence_policy": fact_path.get("evidence_policy", {}),
                    "inference_type": (fact_path.get("evidence_policy") or {}).get("inference_type", ""),
                    "difficulty": (fact_path.get("evidence_policy") or {}).get("difficulty", ""),
                    "evidence_types": fact_path.get("evidence_types", []),
                    "reasoning": fact_path.get("reasoning", ""),
                    "key_photo_ids": kp_ids,
                    "atomic_targets": self._build_atomic_targets_for_fact(fact_path, fact_id),
                    "self_check": fact_path.get("self_check", {}),
                    "evidence_integrity": _evidence_integrity(kp_ids),
                }
            )

        people = []
        for node in graph.get("nodes", []):
            person_id = str(node.get("person_id", "")).strip()
            node_path = node_path_map.get(person_id, {})
            person_kp_ids = [kp.get("kp_id", "") for kp in node_path.get("key_photos", []) if kp.get("kp_id")]
            people.append(
                {
                    "person_id": person_id,
                    "face_id": node_path.get("face_id", ""),
                    "name": node.get("name", ""),
                    "relation": node.get("relation", ""),
                    "relation_category": node.get("relation_category", ""),
                    "facts": node.get("facts", []),
                    "identification": node_path.get("identification", ""),
                    "relation_reasoning": node_path.get("relation_reasoning", ""),
                    "key_photo_ids": person_kp_ids,
                    "canonical_mentions": list(node_path.get("canonical_mentions") or []),
                    "evidence_integrity": _evidence_integrity(person_kp_ids),
                }
            )

        events = []
        for event_id, event in sorted(event_map.items()):
            events.append(
                {
                    "event_id": event_id,
                    "year_month": event.get("year_month", ""),
                    "event_type": event.get("event_type", ""),
                    "label": event.get("label", ""),
                    "location": event.get("location", ""),
                    "participants": event.get("participants", []),
                    "photo_count": event.get("photo_count", 0),
                    "kp_ids": event.get("kp_ids", []),
                    "photo_slots": event.get("photo_slots", []),
                }
            )

        # --- cross-modal entity alignment GT (one record per mention) ---
        entity_alignment = _build_entity_alignment(
            profile=profile,
            graph=graph,
            face_catalog=face_catalog,
            photo_annotations=photo_annotations,
            event_map=event_map,
        )

        # Back-reference: annotate each photo with the mention ids it carries
        mention_ids_by_photo: dict[str, list[str]] = {}
        for rec in entity_alignment:
            for pid in rec.get("source_photo_ids") or []:
                mention_ids_by_photo.setdefault(str(pid), []).append(rec["mention_id"])
        for ann in photo_annotations:
            pid = str(ann.get("photo_id") or "")
            if pid in mention_ids_by_photo:
                ann["mention_record_ids"] = list(dict.fromkeys(mention_ids_by_photo[pid]))
            else:
                ann.setdefault("mention_record_ids", [])

        # --- unified persons roster (owner + non-owner) ---
        persons = _build_persons_roster(
            profile=profile,
            graph=graph,
            face_catalog=face_catalog,
            node_path_map=node_path_map,
            entity_alignment=entity_alignment,
            photo_annotations=photo_annotations,
        )

        return {
            "owner_profile": {
                "user_id": profile.get("user_id", ""),
                "name": profile.get("name", ""),
                "facts": profile.get("facts", []),
                "persona_text": profile.get("persona_text", ""),
                "raw_attributes": profile.get("raw_attributes", {}),
            },
            "face_catalog": face_catalog,
            # Phase 2: location_catalog gives the evaluator the canonical
            # mapping from loc_id to a recognizable label (the agent does
            # NOT see this — it must cluster gps strings on its own).
            "location_catalog": list(reasoning.get("_plan", {}).get("location_anchors", [])),
            "facts": facts,
            "people": people,
            "persons": persons,
            "events": events,
            "entity_alignment": entity_alignment,
            "ambient_plan": ambient_plan,
            # Phase 5: structured ambient supports lifted from Step 4 paths.
            # Each support carries target_kind (fact|node), target_ref
            # (fact_id or person_id), support_type enum, and description.
            "ambient_supports": _build_ambient_supports_index(
                fact_paths=reasoning.get("fact_paths", []),
                node_paths=reasoning.get("node_paths", []),
                fact_id_by_target={f["target"]: f["fact_id"] for f in facts if f.get("target") and f.get("fact_id")},
            ),
            "photo_annotations": photo_annotations,
        }

    def _derive_node_alignment_difficulty(self, reasoning: dict[str, Any]) -> dict[str, str]:
        """Recompute ``alignment_difficulty`` per person_id from reasoning paths.

        The Step 4 output's overall ``stats.alignment_difficulty`` is a
        global histogram, not a per-person mapping; the per-path
        classification logic lives in
        ``ReasoningPathGenerator._classify_node_alignment_difficulty``.
        We replay it here so each question can carry the correct tag.
        """
        try:
            from src.step4.reasoning_generator import ReasoningPathGenerator
        except Exception:
            return {}
        out: dict[str, str] = {}
        for path in reasoning.get("node_paths", []) or []:
            pid = str(path.get("person_id") or "").strip()
            if not pid:
                continue
            try:
                out[pid] = ReasoningPathGenerator._classify_node_alignment_difficulty(path)
            except Exception:
                continue
        return out

    def _build_questions(
        self,
        *,
        ground_truth: dict[str, Any],
        owner_name: str,
        node_alignment_difficulty: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Synthesize a top-level Q&A array for direct evaluator consumption.

        Phase 5: HEIR Agent currently has to cross-reference
        ``ground_truth.facts[].atomic_targets[]`` against parent
        ``fact.target`` for the answer, and stitch ``person.canonical_name``
        / ``person.relation_to_owner`` for the node side. This method
        does that synthesis once at export time.

        Each question record includes ``evidence_kp_ids`` so the
        evaluator can score on the exact key_photos the design intended,
        and a stable ``question_id`` for joining against agent outputs.
        """
        node_difficulty = node_alignment_difficulty or {}
        questions: list[dict[str, Any]] = []

        # --- Source: facts + atomic_targets ---
        for fact in ground_truth.get("facts", []) or []:
            fact_id = str(fact.get("fact_id") or "").strip()
            if not fact_id:
                continue
            parent_target = str(fact.get("target") or "").strip()
            fact_kp_ids = list(fact.get("key_photo_ids") or [])
            for atomic in fact.get("atomic_targets") or []:
                if not isinstance(atomic, dict):
                    continue
                role = str(atomic.get("role") or "").strip().lower()
                if role and role != "evaluation_target":
                    # supporting_detail / context atoms are not direct Q&A.
                    continue
                question_text = str(atomic.get("text") or "").strip()
                if not question_text:
                    continue
                evidence = list(atomic.get("supporting_photo_ids") or fact_kp_ids)
                questions.append({
                    "question_id": f"q_{atomic.get('atomic_id') or f'{fact_id}_atom'}",
                    "question_text": question_text,
                    "expected_answer": parent_target,
                    "answer_type": str(atomic.get("answer_type") or "short_text"),
                    "evidence_kp_ids": evidence,
                    "source": "fact",
                    "source_ref": f"fact:{fact_id}",
                })

        # --- Source: persons (identification + relation) ---
        for person in ground_truth.get("persons", []) or []:
            pid = str(person.get("person_id") or "").strip()
            if not pid or pid == "owner":
                continue
            # Phase 5: persons roster uses ``canonical_name`` /
            # ``relation_to_owner`` / ``evidence_photo_ids`` field names
            # (paper-facing PersonRecord schema). Fall back to the legacy
            # ``name`` / ``relation`` / ``key_photo_ids`` if a future
            # caller passes a node_path-style dict.
            name = str(person.get("canonical_name") or person.get("name") or "").strip()
            relation = str(person.get("relation_to_owner") or person.get("relation") or "").strip()
            face_id = str(person.get("face_id") or "").strip()
            # ``evidence_photo_ids`` here is a dict ({face_visible, name_mentions, joint});
            # legacy node_path-style dicts use a flat list under ``key_photo_ids``.
            evidence_blob = person.get("evidence_photo_ids")
            if isinstance(evidence_blob, dict):
                # Pick the joint set first (KPs that bear both face + name);
                # otherwise prefer face-visible KPs since they carry the
                # most evidence weight; fall back to all readable evidence.
                kp_ids = list(evidence_blob.get("joint") or [])
                if not kp_ids:
                    kp_ids = list(evidence_blob.get("face_visible") or evidence_blob.get("name_mentions") or [])
                # Restrict to KP-style ids only (drop ambient/event slots).
                kp_ids = [k for k in kp_ids if str(k).startswith("kp_")] or kp_ids[:8]
            elif isinstance(evidence_blob, list):
                kp_ids = list(evidence_blob)
            else:
                kp_ids = list(person.get("key_photo_ids") or [])
            difficulty = node_difficulty.get(pid)
            if name:
                questions.append({
                    "question_id": f"q_{pid}_identity",
                    "question_text": (
                        f"What is the name of the recurring person identified by face anchor "
                        f"'{face_id}' across the album?"
                    ) if face_id else f"What is the name of person {pid}?",
                    "expected_answer": name,
                    "answer_type": "short_text",
                    "evidence_kp_ids": kp_ids,
                    "source": "node",
                    "source_ref": f"node:{pid}",
                    "alignment_difficulty": difficulty,
                    "subkind": "identity",
                })
            if relation and name:
                questions.append({
                    "question_id": f"q_{pid}_relation",
                    "question_text": (
                        f"What is {name}'s relationship to {owner_name}?"
                        if owner_name else f"What is {name}'s relationship to the album owner?"
                    ),
                    "expected_answer": relation,
                    "answer_type": "category",
                    "evidence_kp_ids": kp_ids,
                    "source": "node",
                    "source_ref": f"node:{pid}",
                    "alignment_difficulty": difficulty,
                    "subkind": "relation",
                })

        return questions

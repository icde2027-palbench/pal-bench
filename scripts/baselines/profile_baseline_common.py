"""Shared utilities for PAL-Bench profile reconstruction baselines.

The scripts in this directory all emit the formal ``predicted_profile.json``
contract used by ``scripts/evidence_chain_dossier/evaluate_formal_batch.py``.
They are intentionally simple and transparent: each baseline differs only in
which public album fields it can see and how it chooses album context.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.evidence_chain_dossier.inventory import build_inventory
from src.agent.evidence_chain_dossier.llm_context import face_context, owner_context, photo_brief
from src.llm import create_llm_for_role
from src.llm.base import LLMMessage
from src.utils.io import load_json, save_json


PROFILE_SYSTEM = """You reconstruct a structured profile from a public synthetic personal album.
Use only the supplied album evidence. Return JSON only.

Output schema:
{
  "owner": {
    "face_id": "face_001 or empty if unavailable",
    "facts": [
      {"text": "atomic owner fact", "evidence_photo_ids": ["photo_0001"], "confidence": 0.0, "reasoning_path": "brief evidence chain"}
    ]
  },
  "persons": [
    {"face_id": "face_002 or empty if unknown", "canonical_name": "Name", "relation_to_owner": "friend", "relation_category": "friend", "evidence_photo_ids": ["photo_0002"], "confidence": 0.0, "reasoning_path": "brief evidence chain"}
  ]
}

Rules:
- Cite only public photo ids that appear in the input.
- Keep owner facts atomic and concrete.
- Do not invent exact emails, phone numbers, street addresses, or private ids.
- For persons, use public face ids only when the input includes face evidence.
- If the evidence is insufficient, omit the claim rather than guessing.
"""


TEXT_ONLY_SYSTEM = PROFILE_SYSTEM + """

Additional text-only restriction:
- You may use captions, visible_text, text_entities, timestamps, and locations.
- You must not use face ids or face co-presence for identity binding.
- Leave person face_id empty unless the input explicitly gives a public face id.
"""


@dataclass
class LLMStats:
    n_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    parse_failures: int = 0
    errors: list[str] = field(default_factory=list)

    def record_response(self, response: Any) -> None:
        self.n_llm_calls += 1
        usage = getattr(response, "usage", None) or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--users", required=True, help="Comma-separated user ids")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--users-root", default="data/full/users")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--llm", default="agent_llm", help="Role from configs/models.yaml")
    parser.add_argument("--max-photos", type=int, default=220)
    parser.add_argument("--photo-chars", type=int, default=260)
    parser.add_argument("--max-owner-facts", type=int, default=40)
    parser.add_argument("--max-persons", type=int, default=24)


def run_batch(
    *,
    args: argparse.Namespace,
    method_id: str,
    run_one: Callable[[str, argparse.Namespace], dict[str, Any]],
    extra_manifest: dict[str, Any] | None = None,
) -> None:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    users = [item.strip() for item in str(args.users).split(",") if item.strip()]
    started = time.time()
    rows: list[dict[str, Any]] = []
    print(f"[{method_id}] users={len(users)} workers={args.workers} llm={args.llm}")
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {executor.submit(run_one, user_id, args): user_id for user_id in users}
        for future in as_completed(futures):
            user_id = futures[future]
            row = future.result()
            rows.append(row)
            status = row.get("status", "unknown")
            print(
                f"  {status.upper()} {user_id}: "
                f"facts={row.get('n_owner_facts', 0)} persons={row.get('n_persons', 0)} "
                f"calls={row.get('n_llm_calls', 0)} parse_failures={row.get('parse_failures', 0)} "
                f"elapsed={float(row.get('elapsed_sec') or 0):.2f}s"
            )
    rows.sort(key=lambda item: str(item.get("user_id") or ""))
    ok = sum(1 for row in rows if row.get("status") == "success")
    manifest = {
        "schema_version": "profile_baseline_batch.v1",
        "method_id": method_id,
        "users_root": str(args.users_root),
        "output_root": str(output_root),
        "llm_role": str(args.llm),
        "n_users": len(users),
        "n_success": ok,
        "elapsed_sec": round(time.time() - started, 3),
        "args": vars(args),
        "rows": rows,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    save_json(manifest, output_root / "batch_manifest.json")
    print(f"[{method_id}] done: {ok}/{len(users)} -> {output_root}")


def run_text_only_user(user_id: str, args: argparse.Namespace) -> dict[str, Any]:
    return _run_llm_profile_user(
        user_id=user_id,
        args=args,
        variant="text_only_profile",
        system=TEXT_ONLY_SYSTEM,
        payload_builder=lambda album, inventory, ns: {
            "user_id": user_id,
            "input_contract": {
                "captions": True,
                "visible_text": True,
                "text_entities": True,
                "face_ids": False,
                "time_location": True,
            },
            "album_summary": album.get("album_summary") or {},
            "text_evidence_photos": _text_rich_photos(
                inventory,
                max_photos=int(ns.max_photos),
                chars=int(ns.photo_chars),
                include_faces=False,
            ),
        },
        allow_empty_face=True,
        owner_face_id="",
    )


def run_multimodal_rag_user(user_id: str, args: argparse.Namespace) -> dict[str, Any]:
    return _run_llm_profile_user(
        user_id=user_id,
        args=args,
        variant="multimodal_rag_profile",
        system=PROFILE_SYSTEM,
        payload_builder=lambda album, inventory, ns: {
            "user_id": user_id,
            "input_contract": {
                "captions": True,
                "visible_text": True,
                "text_entities": True,
                "face_ids": True,
                "time_location": True,
                "retrieval": "field-level BM25 plus face summaries",
            },
            "album_summary": album.get("album_summary") or {},
            "owner_context": owner_context(inventory, max_units=36, chars=int(ns.photo_chars)),
            "retrieved_owner_photos": _text_rich_photos(
                inventory,
                max_photos=min(int(ns.max_photos), 120),
                chars=int(ns.photo_chars),
                include_faces=True,
            ),
            "face_contexts": _ranked_face_contexts(
                inventory,
                max_faces=int(ns.max_persons),
                photos_per_face=6,
                chars=int(ns.photo_chars),
            ),
        },
        owner_face_id=inventory_owner_face,
    )


def run_long_context_user(user_id: str, args: argparse.Namespace) -> dict[str, Any]:
    return _run_llm_profile_user(
        user_id=user_id,
        args=args,
        variant="long_context_mm_llm_profile",
        system=PROFILE_SYSTEM,
        payload_builder=lambda album, inventory, ns: {
            "user_id": user_id,
            "input_contract": {
                "captions": True,
                "visible_text": True,
                "text_entities": True,
                "face_ids": True,
                "time_location": True,
                "context_policy": "compressed whole album sorted by information density",
            },
            "album_summary": album.get("album_summary") or {},
            "faces": album.get("faces") or [],
            "compressed_album": _compressed_album(
                inventory,
                max_photos=int(ns.max_photos),
                chars=int(ns.photo_chars),
            ),
        },
        owner_face_id=inventory_owner_face,
    )


def run_generic_tool_agent_user(user_id: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    users_root = Path(args.users_root)
    output_root = Path(args.output_root)
    album = _load_album(users_root, user_id)
    inventory = build_inventory(album)
    llm = create_llm_for_role(str(args.llm))
    stats = LLMStats()
    out_dir = output_root / user_id
    try:
        planning_payload = {
            "user_id": user_id,
            "album_summary": album.get("album_summary") or {},
            "candidate_owner_context": owner_context(inventory, max_units=18, chars=180),
            "face_ids": sorted(fid for fid in inventory.face_units_by_face if fid != inventory.owner_face_id)[:80],
            "sample_text_photos": _text_rich_photos(inventory, max_photos=40, chars=180, include_faces=True),
        }
        plan = call_json(
            llm,
            system="You are a profile-reconstruction planning agent. Output JSON only.",
            payload={
                "task": "Choose concise search queries for owner facts and person identity/relation evidence.",
                "available_tools": ["search_album_text", "inspect_face_context"],
                "planning_context": planning_payload,
                "output_schema": {
                    "owner_queries": ["query text"],
                    "person_queries": [{"face_id": "face_001", "queries": ["query text"]}],
                },
            },
            stats=stats,
            max_tokens=2048,
        )
        owner_queries = _string_list((plan or {}).get("owner_queries"))[:10]
        person_queries = (plan or {}).get("person_queries") or []
        retrieved_owner = []
        for query in owner_queries:
            retrieved_owner.extend(_search_photos(inventory, query, top_k=8, chars=int(args.photo_chars)))
        retrieved_owner = _dedupe_photo_dicts(retrieved_owner)[: min(120, int(args.max_photos))]
        face_bundles = []
        for row in person_queries[: int(args.max_persons)]:
            if not isinstance(row, dict):
                continue
            face_id = str(row.get("face_id") or "")
            if not face_id or face_id not in inventory.face_units_by_face or face_id == inventory.owner_face_id:
                continue
            evidence = []
            for query in _string_list(row.get("queries"))[:4]:
                evidence.extend(_search_photos(inventory, query, top_k=5, chars=int(args.photo_chars)))
            face_bundles.append(
                {
                    "face": face_context(inventory, face_id, max_photos=5, chars=int(args.photo_chars)),
                    "tool_retrieved_photos": _dedupe_photo_dicts(evidence)[:12],
                }
            )
        if not face_bundles:
            face_bundles = _ranked_face_contexts(
                inventory,
                max_faces=int(args.max_persons),
                photos_per_face=5,
                chars=int(args.photo_chars),
            )
        raw = call_json(
            llm,
            system=PROFILE_SYSTEM,
            payload={
                "user_id": user_id,
                "input_contract": {
                    "style": "generic plan-and-execute tool agent",
                    "tools_used": ["search_album_text", "inspect_face_context"],
                },
                "plan": plan or {},
                "owner_tool_results": retrieved_owner,
                "face_tool_results": face_bundles[: int(args.max_persons)],
            },
            stats=stats,
            max_tokens=8192,
        )
        profile = normalize_profile(
            raw,
            album=album,
            inventory=inventory,
            user_id=user_id,
            variant="generic_tool_agent_profile",
            run_stats=_stats_dict(stats, started, extra={"tool_plan": plan or {}}),
            max_owner_facts=int(args.max_owner_facts),
            max_persons=int(args.max_persons),
        )
        _write_profile(out_dir, profile)
        return _summary_row(user_id, profile, stats, started, "success")
    except Exception as exc:
        stats.errors.append(str(exc))
        profile = empty_profile(
            album=album,
            inventory=inventory,
            user_id=user_id,
            variant="generic_tool_agent_profile",
            run_stats=_stats_dict(stats, started, extra={"fatal_error": str(exc)}),
        )
        _write_profile(out_dir, profile)
        return _summary_row(user_id, profile, stats, started, "error", str(exc))


def run_adapted_lifelog_user(user_id: str, args: argparse.Namespace) -> dict[str, Any]:
    return _run_llm_profile_user(
        user_id=user_id,
        args=args,
        variant="adapted_prior_lifelog_profile",
        system=PROFILE_SYSTEM,
        payload_builder=lambda album, inventory, ns: {
            "user_id": user_id,
            "input_contract": {
                "style": "lifelog-inspired episodic retrieval",
                "captions": True,
                "visible_text": True,
                "text_entities": True,
                "face_ids": True,
                "time_location": True,
            },
            "album_summary": album.get("album_summary") or {},
            "episode_summaries": _episode_summaries(
                inventory,
                max_episodes=int(ns.max_photos),
                chars=int(ns.photo_chars),
            ),
            "face_appearance_summaries": _ranked_face_contexts(
                inventory,
                max_faces=int(ns.max_persons),
                photos_per_face=4,
                chars=int(ns.photo_chars),
            ),
        },
        owner_face_id=inventory_owner_face,
    )


def _run_llm_profile_user(
    *,
    user_id: str,
    args: argparse.Namespace,
    variant: str,
    system: str,
    payload_builder: Callable[[dict[str, Any], Any, argparse.Namespace], dict[str, Any]],
    allow_empty_face: bool = False,
    owner_face_id: Callable[[Any], str] | str | None = None,
) -> dict[str, Any]:
    started = time.time()
    users_root = Path(args.users_root)
    output_root = Path(args.output_root)
    album = _load_album(users_root, user_id)
    inventory = build_inventory(album)
    stats = LLMStats()
    out_dir = output_root / user_id
    try:
        llm = create_llm_for_role(str(args.llm))
        payload = payload_builder(album, inventory, args)
        raw = call_json(llm, system=system, payload=payload, stats=stats, max_tokens=8192)
        owner_face = owner_face_id(inventory) if callable(owner_face_id) else owner_face_id
        profile = normalize_profile(
            raw,
            album=album,
            inventory=inventory,
            user_id=user_id,
            variant=variant,
            run_stats=_stats_dict(stats, started),
            max_owner_facts=int(args.max_owner_facts),
            max_persons=int(args.max_persons),
            allow_empty_face=allow_empty_face,
            owner_face_id=owner_face,
        )
        _write_profile(out_dir, profile)
        return _summary_row(user_id, profile, stats, started, "success")
    except Exception as exc:
        stats.errors.append(str(exc))
        profile = empty_profile(
            album=album,
            inventory=inventory,
            user_id=user_id,
            variant=variant,
            run_stats=_stats_dict(stats, started, extra={"fatal_error": str(exc)}),
            owner_face_id=(owner_face_id(inventory) if callable(owner_face_id) else owner_face_id),
        )
        _write_profile(out_dir, profile)
        return _summary_row(user_id, profile, stats, started, "error", str(exc))


def call_json(
    llm: Any,
    *,
    system: str,
    payload: dict[str, Any],
    stats: LLMStats,
    max_tokens: int,
) -> dict[str, Any] | None:
    prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    for attempt in range(2):
        actual_prompt = prompt
        if attempt:
            actual_prompt += "\n\nReturn only a single valid JSON object matching the requested schema."
        messages = [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=actual_prompt),
        ]
        try:
            try:
                response = llm.chat(
                    messages,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            except Exception:
                response = llm.chat(messages, temperature=0.1, max_tokens=max_tokens)
            stats.record_response(response)
            content = getattr(response, "content", "") or ""
            parsed = extract_json_object(content)
            if isinstance(parsed, dict):
                return parsed
            stats.parse_failures += 1
            preview = " ".join(str(content or "").split())[:600]
            if preview:
                stats.errors.append(f"json_parse_failed attempt={attempt + 1}: {preview}")
        except Exception as exc:
            stats.errors.append(str(exc))
            stats.parse_failures += 1
    return None


def normalize_profile(
    raw: dict[str, Any] | None,
    *,
    album: dict[str, Any],
    inventory: Any,
    user_id: str,
    variant: str,
    run_stats: dict[str, Any],
    max_owner_facts: int,
    max_persons: int,
    allow_empty_face: bool = False,
    owner_face_id: str | None = None,
) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    valid_photos = {str(p.get("photo_id") or "") for p in album.get("photos") or []}
    valid_faces = {str(f.get("face_id") or "") for f in album.get("faces") or []}
    owner = raw.get("owner") if isinstance(raw.get("owner"), dict) else {}
    facts = []
    for item in owner.get("facts") or []:
        if not isinstance(item, dict):
            continue
        text = _clean_sentence(item.get("text"))
        if not text or _private_or_contact_like(text):
            continue
        facts.append(
            {
                "text": text,
                "evidence_photo_ids": _valid_photo_ids(item.get("evidence_photo_ids"), valid_photos)[:5],
                "confidence": _bounded_float(item.get("confidence"), 0.55),
                "reasoning_path": _truncate(str(item.get("reasoning_path") or ""), 600)
                or "Baseline extracted this owner fact from the cited public album evidence.",
            }
        )
    facts = _dedupe_claim_rows(facts, "text")[:max_owner_facts]

    persons = []
    for item in raw.get("persons") or []:
        if not isinstance(item, dict):
            continue
        face_id = str(item.get("face_id") or "").strip()
        if face_id and face_id not in valid_faces:
            face_id = ""
        if not allow_empty_face and not face_id:
            continue
        name = _clean_name(item.get("canonical_name"))
        relation = _clean_relation(item.get("relation_to_owner"))
        category = _clean_category(item.get("relation_category") or relation)
        if not name and not relation and not category:
            continue
        persons.append(
            {
                "face_id": face_id,
                "canonical_name": name,
                "relation_to_owner": relation,
                "relation_category": category,
                "evidence_photo_ids": _valid_photo_ids(item.get("evidence_photo_ids"), valid_photos)[:8],
                "confidence": _bounded_float(item.get("confidence"), 0.45),
                "reasoning_path": _truncate(str(item.get("reasoning_path") or ""), 700)
                or "Baseline inferred this person row from the cited public album evidence.",
            }
        )
    persons = _dedupe_person_rows(persons)[:max_persons]

    if owner_face_id is None:
        owner_face_id = str(owner.get("face_id") or inventory.owner_face_id or "")
    if owner_face_id and owner_face_id not in valid_faces:
        owner_face_id = ""
    run_stats.setdefault("n_owner_facts", len(facts))
    run_stats.setdefault("n_persons", len(persons))
    return {
        "schema_version": "predicted_profile.v2",
        "user_id": user_id,
        "framework": "profile_baseline",
        "variant": variant,
        "owner": {
            "face_id": owner_face_id or "",
            "facts": facts,
        },
        "persons": persons,
        "run_stats": run_stats,
    }


def empty_profile(
    *,
    album: dict[str, Any],
    inventory: Any,
    user_id: str,
    variant: str,
    run_stats: dict[str, Any],
    owner_face_id: str | None = None,
) -> dict[str, Any]:
    return normalize_profile(
        None,
        album=album,
        inventory=inventory,
        user_id=user_id,
        variant=variant,
        run_stats=run_stats,
        max_owner_facts=0,
        max_persons=0,
        owner_face_id=owner_face_id,
    )


def inventory_owner_face(inventory: Any) -> str:
    return str(getattr(inventory, "owner_face_id", "") or "")


def _stats_dict(stats: LLMStats, started: float, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {
        "n_llm_calls": int(stats.n_llm_calls),
        "budget_used": int(stats.n_llm_calls),
        "prompt_tokens": int(stats.prompt_tokens),
        "completion_tokens": int(stats.completion_tokens),
        "total_tokens": int(stats.total_tokens),
        "parse_failures": int(stats.parse_failures),
        "total_time_s": round(time.time() - started, 3),
        "elapsed_sec": round(time.time() - started, 3),
        "errors": stats.errors[:20],
    }
    if extra:
        out.update(extra)
    return out


def _summary_row(
    user_id: str,
    profile: dict[str, Any],
    stats: LLMStats,
    started: float,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    row = {
        "user_id": user_id,
        "status": status,
        "elapsed_sec": round(time.time() - started, 3),
        "n_owner_facts": len((profile.get("owner") or {}).get("facts") or []),
        "n_persons": len(profile.get("persons") or []),
        "n_llm_calls": stats.n_llm_calls,
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "parse_failures": stats.parse_failures,
    }
    if error:
        row["error"] = error
    return row


def _write_profile(out_dir: Path, profile: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(profile, out_dir / "predicted_profile.json")
    save_json(profile.get("run_stats") or {}, out_dir / "run_stats.json")


def _load_album(users_root: Path, user_id: str) -> dict[str, Any]:
    album_path = users_root / user_id / f"{user_id}_agent_album.json"
    album = load_json(album_path)
    if not isinstance(album, dict):
        raise ValueError(f"Expected album object at {album_path}")
    return album


def _text_rich_photos(
    inventory: Any,
    *,
    max_photos: int,
    chars: int,
    include_faces: bool,
) -> list[dict[str, Any]]:
    scored = []
    for photo in inventory.photos:
        photo_id = str(photo.get("photo_id") or "")
        text_items = [str(t) for t in photo.get("visible_text") or [] if str(t).strip()]
        entities = [e for e in photo.get("text_entities") or [] if isinstance(e, dict)]
        caption = str(photo.get("caption") or "")
        faces = [str(fid) for fid in photo.get("visible_face_ids") or []]
        score = 0.02 * len(caption) + 1.5 * len(text_items) + 1.0 * len(entities)
        if faces:
            score += 0.6
        if inventory.owner_face_id and inventory.owner_face_id in faces:
            score += 1.0
        if score <= 0:
            continue
        brief = photo_brief(inventory, photo_id, chars=chars)
        if not include_faces:
            brief.pop("faces", None)
        scored.append((score, str(photo.get("timestamp") or ""), photo_id, brief))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    return [row[3] for row in scored[:max_photos]]


def _compressed_album(inventory: Any, *, max_photos: int, chars: int) -> list[dict[str, Any]]:
    scored = []
    for photo in inventory.photos:
        photo_id = str(photo.get("photo_id") or "")
        text_items = [str(t) for t in photo.get("visible_text") or [] if str(t).strip()]
        faces = [str(fid) for fid in photo.get("visible_face_ids") or []]
        caption = str(photo.get("caption") or "")
        score = 0.01 * len(caption) + 1.25 * len(text_items) + 0.7 * len(faces)
        if inventory.owner_face_id and inventory.owner_face_id in faces:
            score += 0.8
        scored.append((score, str(photo.get("timestamp") or ""), photo_id))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    selected = sorted(scored[:max_photos], key=lambda row: (row[1], row[2]))
    return [photo_brief(inventory, row[2], chars=chars) for row in selected]


def _ranked_face_contexts(
    inventory: Any,
    *,
    max_faces: int,
    photos_per_face: int,
    chars: int,
) -> list[dict[str, Any]]:
    ranked = []
    for face_id, units in inventory.face_units_by_face.items():
        if face_id == inventory.owner_face_id:
            continue
        text_hits = sum(1 for unit in units if unit.visible_text)
        owner_hits = sum(1 for unit in units if unit.owner_present)
        score = len(units) + 2 * text_hits + 2 * owner_hits
        ranked.append((score, face_id))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [
        face_context(inventory, face_id, max_photos=photos_per_face, chars=chars)
        for _, face_id in ranked[:max_faces]
    ]


def _episode_summaries(inventory: Any, *, max_episodes: int, chars: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for photo in inventory.photos:
        metadata = photo.get("metadata") or {}
        key = (str(photo.get("year_month") or ""), str(metadata.get("gps_city") or ""))
        buckets[key].append(photo)
    episodes = []
    for (year_month, city), photos in buckets.items():
        counter = Counter()
        candidate_ids = []
        for photo in photos:
            pid = str(photo.get("photo_id") or "")
            text = " ".join(
                [
                    str(photo.get("caption") or ""),
                    *(str(t) for t in photo.get("visible_text") or []),
                    *(str((e or {}).get("surface") or "") for e in photo.get("text_entities") or [] if isinstance(e, dict)),
                ]
            )
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower()):
                counter[token] += 1
            score = len(photo.get("visible_text") or []) + len(photo.get("visible_face_ids") or [])
            if score:
                candidate_ids.append((score, pid))
        candidate_ids.sort(key=lambda row: (-row[0], row[1]))
        evidence = [photo_brief(inventory, pid, chars=chars) for _, pid in candidate_ids[:8]]
        episodes.append(
            {
                "year_month": year_month,
                "gps_city": city,
                "n_photos": len(photos),
                "top_terms": [term for term, _ in counter.most_common(18)],
                "representative_photos": evidence,
            }
        )
    episodes.sort(key=lambda row: (-row["n_photos"], row["year_month"], row["gps_city"]))
    return episodes[:max_episodes]


def _search_photos(inventory: Any, query: str, *, top_k: int, chars: int) -> list[dict[str, Any]]:
    query_terms = _terms(query)
    if not query_terms:
        return []
    scored = []
    for photo in inventory.photos:
        text = " ".join(
            [
                str(photo.get("caption") or ""),
                *(str(t) for t in photo.get("visible_text") or []),
                *(str((e or {}).get("surface") or "") for e in photo.get("text_entities") or [] if isinstance(e, dict)),
                *(str(fid) for fid in photo.get("visible_face_ids") or []),
                str((photo.get("metadata") or {}).get("gps_city") or ""),
                str((photo.get("metadata") or {}).get("gps_location") or ""),
            ]
        )
        terms = _terms(text)
        overlap = len(query_terms & terms)
        if overlap <= 0:
            continue
        score = overlap / math.sqrt(max(1, len(terms)))
        scored.append((score, str(photo.get("timestamp") or ""), str(photo.get("photo_id") or "")))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    return [photo_brief(inventory, pid, chars=chars) for _, _, pid in scored[:top_k]]


def _dedupe_photo_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        pid = str(row.get("photo_id") or "")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(row)
    return out


def _terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", str(text or ""))
        if token.lower() not in {"the", "and", "for", "with", "from", "this", "that"}
    }


def extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        try:
            parsed = json.loads(fence.group(1).strip())
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _clean_sentence(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip(" -")
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text[:260]


def _clean_name(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip(" ,.;:-")
    return text[:80]


def _clean_relation(value: Any) -> str:
    text = " ".join(str(value or "").lower().split()).strip(" ,.;:-")
    return text[:80]


def _clean_category(value: Any) -> str:
    text = _clean_relation(value)
    mapping = {
        "coworker": "colleague",
        "co-worker": "colleague",
        "work colleague": "colleague",
        "relative": "family",
        "partner": "family",
        "spouse": "family",
    }
    return mapping.get(text, text)


def _private_or_contact_like(text: str) -> bool:
    lower = text.lower()
    if re.search(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", lower):
        return True
    if re.search(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", lower):
        return True
    if re.search(r"\b\d{1,5}\s+[a-z0-9 .'-]+\s+(street|st|road|rd|avenue|ave|drive|dr|lane|ln|apartment|apt)\b", lower):
        return True
    return False


def _valid_photo_ids(value: Any, valid_photos: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        pid = str(item or "").strip()
        if pid in valid_photos and pid not in out:
            out.append(pid)
    return out


def _bounded_float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = fallback
    return round(max(0.0, min(1.0, number)), 3)


def _truncate(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _dedupe_claim_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        sig = re.sub(r"[^a-z0-9]+", " ", str(row.get(key) or "").lower()).strip()
        if not sig or sig in seen:
            continue
        seen.add(sig)
        out.append(row)
    return out


def _dedupe_person_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        sig = (str(row.get("face_id") or ""), re.sub(r"[^a-z0-9]+", " ", str(row.get("canonical_name") or "").lower()).strip())
        if sig in seen:
            continue
        seen.add(sig)
        out.append(row)
    return out


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def write_method_contract(output_root: str | Path, contract: dict[str, Any]) -> None:
    save_json({"schema_version": "baseline_contract.v1", **contract}, Path(output_root) / "method_contract.json")

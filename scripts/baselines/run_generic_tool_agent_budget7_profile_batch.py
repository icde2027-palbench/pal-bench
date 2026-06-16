#!/usr/bin/env python3
"""Run a compute-scaled generic tool-use profile baseline.

This diagnostic gives the generic tool-use baseline a PAL-TRACE-like
reconstruction-call budget while keeping the agent generic: intermediate calls
produce non-authoritative tool notes, and the final call composes the profile
without a frozen identity table.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.baselines.profile_baseline_common import (
    LLMStats,
    PROFILE_SYSTEM,
    _dedupe_photo_dicts,
    _episode_summaries,
    _load_album,
    _ranked_face_contexts,
    _search_photos,
    _stats_dict,
    _string_list,
    _summary_row,
    _text_rich_photos,
    _write_profile,
    add_common_args,
    call_json,
    empty_profile,
    inventory_owner_face,
    normalize_profile,
    run_batch,
    write_method_contract,
)
from src.agent.evidence_chain_dossier.inventory import build_inventory
from src.agent.evidence_chain_dossier.llm_context import face_context, owner_context
from src.llm import create_llm_for_role
from src.utils.io import save_json


TOOL_NOTE_SYSTEM = """You are a generic plan-and-execute tool-use agent for profile reconstruction.
Use only the supplied public album tool results. Return JSON only.

Constraints:
- Do not create or freeze a canonical identity table.
- Treat face ids as public observations, not authoritative identities.
- Keep notes concise and cite public photo ids.
- If evidence is weak or ambiguous, say so explicitly.
"""


def run_generic_tool_agent_budget7_user(user_id: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    users_root = Path(args.users_root)
    output_root = Path(args.output_root)
    album = _load_album(users_root, user_id)
    inventory = build_inventory(album)
    llm = create_llm_for_role(str(args.llm))
    stats = LLMStats()
    out_dir = output_root / user_id

    try:
        plan = call_json(
            llm,
            system="You are a profile-reconstruction planning agent. Output JSON only.",
            payload={
                "task": "Plan generic tool calls for owner facts and social identity/relation evidence.",
                "available_tools": [
                    "search_album_text",
                    "inspect_face_context",
                    "summarize_temporal_context",
                    "select_evidence_support",
                ],
                "planning_context": {
                    "user_id": user_id,
                    "album_summary": album.get("album_summary") or {},
                    "candidate_owner_context": owner_context(inventory, max_units=18, chars=180),
                    "face_ids": sorted(fid for fid in inventory.face_units_by_face if fid != inventory.owner_face_id)[:80],
                    "sample_text_photos": _text_rich_photos(
                        inventory,
                        max_photos=40,
                        chars=180,
                        include_faces=True,
                    ),
                },
                "output_schema": {
                    "owner_queries": ["query text"],
                    "identity_queries": ["query text"],
                    "person_focus": [{"face_id": "face_001", "queries": ["query text"]}],
                    "temporal_queries": ["query text"],
                },
            },
            stats=stats,
            max_tokens=2048,
        )

        owner_queries = _string_list((plan or {}).get("owner_queries"))[:12]
        identity_queries = _string_list((plan or {}).get("identity_queries"))[:12]
        temporal_queries = _string_list((plan or {}).get("temporal_queries"))[:8]
        person_focus = [row for row in ((plan or {}).get("person_focus") or []) if isinstance(row, dict)]

        owner_tool_results = _retrieve_queries(inventory, owner_queries, top_k=8, chars=int(args.photo_chars))[
            : min(100, int(args.max_photos))
        ]
        identity_text_results = _retrieve_queries(inventory, identity_queries, top_k=8, chars=int(args.photo_chars))[
            : min(80, int(args.max_photos))
        ]
        temporal_text_results = _retrieve_queries(inventory, temporal_queries, top_k=8, chars=int(args.photo_chars))[
            : min(64, int(args.max_photos))
        ]
        face_tool_results = _face_tool_results(inventory, person_focus, args)
        if not face_tool_results:
            face_tool_results = _ranked_face_contexts(
                inventory,
                max_faces=int(args.max_persons),
                photos_per_face=5,
                chars=int(args.photo_chars),
            )

        owner_notes = call_json(
            llm,
            system=TOOL_NOTE_SYSTEM,
            payload={
                "tool_call": "search_album_text for owner facts",
                "user_id": user_id,
                "queries": owner_queries,
                "owner_context": owner_context(inventory, max_units=32, chars=int(args.photo_chars)),
                "tool_results": owner_tool_results,
                "output_schema": {
                    "owner_fact_notes": [
                        {
                            "claim_hint": "candidate atomic owner fact",
                            "photo_ids": ["photo_0001"],
                            "confidence": 0.0,
                            "uncertainty": "why this could be wrong",
                        }
                    ],
                    "missing_evidence": ["short caveat"],
                },
            },
            stats=stats,
            max_tokens=int(args.tool_max_tokens),
        )

        identity_notes = call_json(
            llm,
            system=TOOL_NOTE_SYSTEM,
            payload={
                "tool_call": "inspect_face_context plus text search for identity clues",
                "user_id": user_id,
                "queries": identity_queries,
                "identity_text_results": identity_text_results,
                "face_tool_results": face_tool_results[: int(args.max_persons)],
                "output_schema": {
                    "social_evidence_notes": [
                        {
                            "face_id_or_empty": "face_001",
                            "name_hint": "possible name or empty",
                            "photo_ids": ["photo_0002"],
                            "confidence": 0.0,
                            "uncertainty": "why this could be wrong",
                        }
                    ],
                    "ambiguous_faces": ["face_001"],
                },
            },
            stats=stats,
            max_tokens=int(args.tool_max_tokens),
        )

        relation_notes = call_json(
            llm,
            system=TOOL_NOTE_SYSTEM,
            payload={
                "tool_call": "inspect_face_context for relation and category clues",
                "user_id": user_id,
                "face_tool_results": face_tool_results[: int(args.max_persons)],
                "episode_context": _episode_summaries(
                    inventory,
                    max_episodes=min(30, int(args.max_photos)),
                    chars=int(args.photo_chars),
                ),
                "output_schema": {
                    "relation_notes": [
                        {
                            "face_id_or_empty": "face_001",
                            "relation_or_category_hint": "friend/family/colleague/etc",
                            "photo_ids": ["photo_0003"],
                            "confidence": 0.0,
                            "uncertainty": "why this could be wrong",
                        }
                    ],
                    "context_overreach_risks": ["short caveat"],
                },
            },
            stats=stats,
            max_tokens=int(args.tool_max_tokens),
        )

        temporal_notes = call_json(
            llm,
            system=TOOL_NOTE_SYSTEM,
            payload={
                "tool_call": "summarize_temporal_context",
                "user_id": user_id,
                "queries": temporal_queries,
                "temporal_text_results": temporal_text_results,
                "episode_context": _episode_summaries(
                    inventory,
                    max_episodes=min(42, int(args.max_photos)),
                    chars=int(args.photo_chars),
                ),
                "output_schema": {
                    "temporal_notes": [
                        {
                            "claim_or_identity_hint": "candidate cross-month pattern",
                            "photo_ids": ["photo_0004"],
                            "confidence": 0.0,
                            "uncertainty": "why this could be wrong",
                        }
                    ],
                    "temporal_shortcut_risks": ["short caveat"],
                },
            },
            stats=stats,
            max_tokens=int(args.tool_max_tokens),
        )

        evidence_notes = call_json(
            llm,
            system=TOOL_NOTE_SYSTEM,
            payload={
                "tool_call": "select_evidence_support",
                "user_id": user_id,
                "non_authoritative_notes": {
                    "owner": _compact_list((owner_notes or {}).get("owner_fact_notes"), 60),
                    "identity": _compact_list((identity_notes or {}).get("social_evidence_notes"), 80),
                    "relation": _compact_list((relation_notes or {}).get("relation_notes"), 80),
                    "temporal": _compact_list((temporal_notes or {}).get("temporal_notes"), 60),
                },
                "instruction": "Select the strongest candidate claims and public photo ids for final composition, without resolving ambiguities by guessing.",
                "output_schema": {
                    "owner_candidates": [
                        {
                            "claim_hint": "candidate owner fact",
                            "photo_ids": ["photo_0001"],
                            "confidence": 0.0,
                            "support_comment": "why cited evidence supports it",
                        }
                    ],
                    "person_candidates": [
                        {
                            "face_id_or_empty": "face_001",
                            "name_hint": "possible name",
                            "relation_hint": "possible relation",
                            "category_hint": "friend/family/colleague/etc",
                            "photo_ids": ["photo_0002"],
                            "confidence": 0.0,
                            "support_comment": "why cited evidence supports it",
                        }
                    ],
                    "drop_reasons": ["short caveat"],
                },
            },
            stats=stats,
            max_tokens=int(args.tool_max_tokens),
        )

        final = call_json(
            llm,
            system=PROFILE_SYSTEM
            + "\n\nYou are the final composer for a generic 7-call tool-use baseline. "
            "Use the non-authoritative tool notes and cited public photo ids only. "
            "Do not assume a frozen identity table; omit ambiguous claims rather than guessing.",
            payload={
                "user_id": user_id,
                "input_contract": {
                    "style": "generic plan-and-execute tool agent with fixed 7-call budget",
                    "tools_used": [
                        "search_album_text",
                        "inspect_face_context",
                        "summarize_temporal_context",
                        "select_evidence_support",
                    ],
                    "no_pal_trace_identity_freeze": True,
                },
                "plan": plan or {},
                "evidence_notes": evidence_notes or {},
                "tool_notes": {
                    "owner": _compact_list((owner_notes or {}).get("owner_fact_notes"), 50),
                    "identity": _compact_list((identity_notes or {}).get("social_evidence_notes"), 60),
                    "relation": _compact_list((relation_notes or {}).get("relation_notes"), 60),
                    "temporal": _compact_list((temporal_notes or {}).get("temporal_notes"), 50),
                },
                "final_instruction": "Emit the formal predicted_profile JSON using only claims supported by cited public photo ids.",
            },
            stats=stats,
            max_tokens=8192,
        )

        profile = normalize_profile(
            final,
            album=album,
            inventory=inventory,
            user_id=user_id,
            variant="generic_tool_agent_budget7_profile",
            run_stats=_stats_dict(
                stats,
                started,
                extra={
                    "tool_budget_design": "planning + owner search notes + identity notes + relation notes + temporal notes + evidence selection + final composition",
                    "target_reconstruction_calls": 7,
                    "max_photos": int(args.max_photos),
                    "photo_chars": int(args.photo_chars),
                    "tool_plan": plan or {},
                },
            ),
            max_owner_facts=int(args.max_owner_facts),
            max_persons=int(args.max_persons),
            owner_face_id=inventory_owner_face(inventory),
        )
        _write_profile(out_dir, profile)
        save_json(
            {
                "plan": plan or {},
                "owner_notes": owner_notes or {},
                "identity_notes": identity_notes or {},
                "relation_notes": relation_notes or {},
                "temporal_notes": temporal_notes or {},
                "evidence_notes": evidence_notes or {},
            },
            out_dir / "tool_trace_notes.json",
        )
        return _summary_row(user_id, profile, stats, started, "success")
    except Exception as exc:
        stats.errors.append(str(exc))
        profile = empty_profile(
            album=album,
            inventory=inventory,
            user_id=user_id,
            variant="generic_tool_agent_budget7_profile",
            run_stats=_stats_dict(stats, started, extra={"fatal_error": str(exc)}),
            owner_face_id=inventory_owner_face(inventory),
        )
        _write_profile(out_dir, profile)
        return _summary_row(user_id, profile, stats, started, "error", str(exc))


def _retrieve_queries(inventory: Any, queries: list[str], *, top_k: int, chars: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        rows.extend(_search_photos(inventory, query, top_k=top_k, chars=chars))
    return _dedupe_photo_dicts(rows)


def _face_tool_results(inventory: Any, person_focus: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in person_focus[: int(args.max_persons)]:
        face_id = str(row.get("face_id") or "")
        if not face_id or face_id in seen or face_id == inventory.owner_face_id:
            continue
        if face_id not in inventory.face_units_by_face:
            continue
        evidence: list[dict[str, Any]] = []
        for query in _string_list(row.get("queries"))[:4]:
            evidence.extend(_search_photos(inventory, query, top_k=5, chars=int(args.photo_chars)))
        bundles.append(
            {
                "face": face_context(inventory, face_id, max_photos=6, chars=int(args.photo_chars)),
                "tool_retrieved_photos": _dedupe_photo_dicts(evidence)[:14],
            }
        )
        seen.add(face_id)
    if len(bundles) < int(args.max_persons):
        for ctx in _ranked_face_contexts(
            inventory,
            max_faces=int(args.max_persons),
            photos_per_face=5,
            chars=int(args.photo_chars),
        ):
            face_id = str(ctx.get("face_id") or "")
            if face_id and face_id not in seen:
                bundles.append(ctx)
                seen.add(face_id)
            if len(bundles) >= int(args.max_persons):
                break
    return bundles[: int(args.max_persons)]


def _compact_list(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="PAL-Bench generic tool-use 7-call diagnostic")
    add_common_args(parser)
    parser.add_argument("--tool-max-tokens", type=int, default=4096)
    parser.set_defaults(max_photos=160, photo_chars=220, max_owner_facts=48, max_persons=30)
    args = parser.parse_args()
    write_method_contract(
        args.output_root,
        {
            "method_id": "generic_tool_agent_budget7",
            "visible_fields": ["caption", "visible_text", "text_entities", "visible_face_ids", "timestamp", "location"],
            "tools": [
                "search_album_text",
                "inspect_face_context",
                "summarize_temporal_context",
                "select_evidence_support",
            ],
            "stopping_rule": "fixed 7 reconstruction calls: one plan, five tool-note calls, one final composition call",
            "llm_policy": "generic plan-and-execute without PAL-TRACE identity anchoring or frozen identity table",
            "target_reconstruction_calls": 7,
            "output_schema": "predicted_profile.v2",
        },
    )
    run_batch(args=args, method_id="generic_tool_agent_budget7", run_one=run_generic_tool_agent_budget7_user)


if __name__ == "__main__":
    main()

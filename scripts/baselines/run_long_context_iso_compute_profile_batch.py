#!/usr/bin/env python3
"""Run a compute-scaled long-context profile baseline.

This baseline spends a PAL-TRACE-like reconstruction budget without using the
PAL-TRACE identity-freeze mechanism. It repeatedly reads the same public album
through flat evidence-summary prompts, then composes a final profile from those
summaries. Intermediate calls must not emit a durable persons table.
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
    _compressed_album,
    _load_album,
    _stats_dict,
    _summary_row,
    _write_profile,
    add_common_args,
    call_json,
    inventory_owner_face,
    normalize_profile,
    run_batch,
    write_method_contract,
)
from src.agent.evidence_chain_dossier.inventory import build_inventory
from src.llm import create_llm_for_role
from src.utils.io import save_json


ISO_SUMMARY_SYSTEM = """You are an evidence summarizer for a public synthetic personal album.
Use only the supplied public records. Return JSON only.

Important constraints:
- Do not produce a final profile.
- Do not build or freeze a canonical persons table.
- Treat face ids as public observations, not as authoritative identities.
- Produce flat evidence notes with photo ids so a later composer can decide what to use.
- If evidence is weak or ambiguous, say so explicitly.
"""


PASS_SPECS: list[dict[str, str]] = [
    {
        "pass_id": "owner_facts",
        "focus": "Find concrete owner facts, habits, work/school clues, hobbies, recurring routines, and family/home context.",
    },
    {
        "pass_id": "social_name_clues",
        "focus": "Find names, nicknames, face ids, screenshots, cards, badges, and any weak name-to-face clues.",
    },
    {
        "pass_id": "relations_and_categories",
        "focus": "Find relation/category clues for recurring people: family, friend, colleague, classmate, neighbor, partner, or other.",
    },
    {
        "pass_id": "temporal_patterns",
        "focus": "Find cross-month patterns that change the interpretation of facts, people, places, and routines.",
    },
    {
        "pass_id": "ocr_entities",
        "focus": "Audit visible_text and text_entities for names, organizations, locations, events, dates, and owner-related mentions.",
    },
    {
        "pass_id": "evidence_support",
        "focus": "Identify the strongest public photo ids that would support likely owner facts and social-person claims.",
    },
]


def run_long_context_iso_compute_user(user_id: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    users_root = Path(args.users_root)
    output_root = Path(args.output_root)
    album = _load_album(users_root, user_id)
    inventory = build_inventory(album)
    llm = create_llm_for_role(str(args.llm))
    stats = LLMStats()
    out_dir = output_root / user_id

    try:
        album_context = _compressed_album(
            inventory,
            max_photos=int(args.max_photos),
            chars=int(args.photo_chars),
        )
        stage_summaries: list[dict[str, Any]] = []
        for spec in PASS_SPECS[: int(args.summary_passes)]:
            raw = call_json(
                llm,
                system=ISO_SUMMARY_SYSTEM,
                payload={
                    "user_id": user_id,
                    "baseline": "long_context_iso_compute",
                    "input_contract": {
                        "visible_fields": [
                            "caption",
                            "visible_text",
                            "text_entities",
                            "visible_face_ids",
                            "timestamp",
                            "location",
                        ],
                        "intermediate_state": "flat evidence notes only; no frozen identity table",
                    },
                    "pass": spec,
                    "album_summary": album.get("album_summary") or {},
                    "public_album_records": album_context,
                    "output_schema": {
                        "pass_id": spec["pass_id"],
                        "owner_fact_notes": [
                            {
                                "claim_hint": "short candidate owner fact",
                                "photo_ids": ["photo_0001"],
                                "confidence": 0.0,
                                "uncertainty": "why this may be wrong",
                            }
                        ],
                        "social_evidence_notes": [
                            {
                                "face_id_or_empty": "face_001",
                                "name_or_relation_hint": "short clue",
                                "photo_ids": ["photo_0002"],
                                "confidence": 0.0,
                                "uncertainty": "why this may be wrong",
                            }
                        ],
                        "global_uncertainties": ["short caveat"],
                    },
                },
                stats=stats,
                max_tokens=int(args.summary_max_tokens),
            )
            stage_summaries.append(_compact_stage_summary(spec["pass_id"], raw))

        final = call_json(
            llm,
            system=PROFILE_SYSTEM
            + "\n\nYou are composing the final profile for a compute-scaled long-context baseline. "
            "Use the flat stage summaries as non-authoritative notes. Do not assume any identity table was frozen.",
            payload={
                "user_id": user_id,
                "input_contract": {
                    "style": "iso-compute multi-pass long-context summarization",
                    "summary_passes": len(stage_summaries),
                    "no_pal_trace_identity_freeze": True,
                },
                "album_summary": album.get("album_summary") or {},
                "owner_face_id_hint": inventory_owner_face(inventory),
                "stage_summaries": stage_summaries,
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
            variant="long_context_iso_compute_profile",
            run_stats=_stats_dict(
                stats,
                started,
                extra={
                    "summary_passes": len(stage_summaries),
                    "max_photos": int(args.max_photos),
                    "photo_chars": int(args.photo_chars),
                    "stage_summaries": stage_summaries,
                    "iso_compute_design": "flat multi-pass album summaries plus final composition; no frozen identity table",
                },
            ),
            max_owner_facts=int(args.max_owner_facts),
            max_persons=int(args.max_persons),
            owner_face_id=inventory_owner_face(inventory),
        )
        _write_profile(out_dir, profile)
        save_json(stage_summaries, out_dir / "stage_summaries.json")
        return _summary_row(user_id, profile, stats, started, "success")
    except Exception as exc:
        stats.errors.append(str(exc))
        profile = normalize_profile(
            None,
            album=album,
            inventory=inventory,
            user_id=user_id,
            variant="long_context_iso_compute_profile",
            run_stats=_stats_dict(stats, started, extra={"fatal_error": str(exc)}),
            max_owner_facts=0,
            max_persons=0,
            owner_face_id=inventory_owner_face(inventory),
        )
        _write_profile(out_dir, profile)
        return _summary_row(user_id, profile, stats, started, "error", str(exc))


def _compact_stage_summary(pass_id: str, raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "pass_id": pass_id,
        "owner_fact_notes": _limit_list(raw.get("owner_fact_notes"), 80),
        "social_evidence_notes": _limit_list(raw.get("social_evidence_notes"), 100),
        "global_uncertainties": _limit_list(raw.get("global_uncertainties"), 20),
    }


def _limit_list(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="PAL-Bench iso-compute long-context profile baseline")
    add_common_args(parser)
    parser.add_argument("--summary-passes", type=int, default=6)
    parser.add_argument("--summary-max-tokens", type=int, default=8192)
    parser.set_defaults(max_photos=720, photo_chars=420, max_owner_facts=48, max_persons=30)
    args = parser.parse_args()
    write_method_contract(
        args.output_root,
        {
            "method_id": "long_context_iso_compute",
            "visible_fields": ["caption", "visible_text", "text_entities", "visible_face_ids", "timestamp", "location"],
            "context_strategy": "six flat multi-pass whole-album summaries plus final composition",
            "llm_policy": "matched-compute long-context baseline without PAL-TRACE identity anchoring",
            "summary_passes": int(args.summary_passes),
            "stopping_rule": "fixed summary-pass budget plus one final profile-composition call",
            "output_schema": "predicted_profile.v2",
        },
    )
    run_batch(args=args, method_id="long_context_iso_compute", run_one=run_long_context_iso_compute_user)


if __name__ == "__main__":
    main()

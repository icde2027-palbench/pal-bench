#!/usr/bin/env python3
"""Run field-only modality ablations for selected ECD/PAL-TRACE methods."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import load_json, save_json


STABLE_CONFIG = "configs/evidence_chain_dossier/llm_hard_block_adaptive_frozen_aug_core.yaml"
OWNER_CONFIG = "configs/evidence_chain_dossier/llm_hard_block_adaptive_frozen_aug_odm_core.yaml"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _mask_album(album: dict[str, Any], setting: str) -> dict[str, Any]:
    masked = dict(album)
    photos = []
    for photo in album.get("photos") or []:
        row = dict(photo)
        metadata = dict(row.get("metadata") or {})
        if setting == "text_off":
            row["caption"] = ""
            row["visible_text"] = []
            row["text_entities"] = []
        elif setting == "face_off":
            row["visible_face_ids"] = []
        elif setting == "metadata_off":
            row["timestamp"] = ""
            row["year_month"] = ""
            metadata["gps_city"] = ""
            metadata["gps_location"] = ""
        elif setting == "visual_off":
            row["visible_face_ids"] = []
        elif setting != "full":
            raise ValueError(f"Unknown input setting: {setting}")
        row["metadata"] = metadata
        photos.append(row)
    masked["photos"] = photos
    if setting == "face_off" or setting == "visual_off":
        masked["faces"] = []
    summary = dict(masked.get("album_summary") or {})
    summary["modality_ablation_setting"] = setting
    summary["field_only_mask"] = True
    masked["album_summary"] = summary
    return masked


def _write_masked_users(users: list[str], users_root: Path, output_root: Path, setting: str) -> Path:
    masked_root = output_root / "_masked_users" / setting
    for user_id in users:
        src_dir = users_root / user_id
        dst_dir = masked_root / user_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        album = load_json(src_dir / f"{user_id}_agent_album.json")
        save_json(_mask_album(album, setting), dst_dir / f"{user_id}_agent_album.json")
        # Evaluation GT is not used by the runner but keeping a copy makes the
        # masked root self-describing for artifact inspection.
        gt_path = src_dir / f"{user_id}_eval_gt.json"
        if gt_path.exists():
            save_json(load_json(gt_path), dst_dir / f"{user_id}_eval_gt.json")
    return masked_root


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run field-only modality ablation")
    parser.add_argument("--users", required=True)
    parser.add_argument("--users-root", default="data/full/users")
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "no_llm_heuristic",
            "stable_identity_baseline",
            "fresh_owner_mining_rerun",
            "pal_trace",
            "text_only_profile",
            "multimodal_rag",
            "long_context_mm_llm",
            "generic_tool_agent",
        ],
    )
    parser.add_argument("--input-setting", required=True, choices=["full", "text_off", "face_off", "metadata_off", "visual_off"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--llm", default="agent_llm")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-photos", type=int, default=None)
    parser.add_argument("--photo-chars", type=int, default=None)
    parser.add_argument("--max-owner-facts", type=int, default=None)
    parser.add_argument("--max-persons", type=int, default=None)
    args = parser.parse_args()

    users = _split_csv(args.users)
    users_root = Path(args.users_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    masked_root = users_root if args.input_setting == "full" else _write_masked_users(users, users_root, output_root, args.input_setting)
    users_arg = ",".join(users)
    py = sys.executable

    baseline_scripts = {
        "text_only_profile": "scripts/baselines/run_text_only_profile_batch.py",
        "multimodal_rag": "scripts/baselines/run_multimodal_rag_profile_batch.py",
        "long_context_mm_llm": "scripts/baselines/run_long_context_profile_batch.py",
        "generic_tool_agent": "scripts/baselines/run_generic_tool_agent_profile_batch.py",
    }

    if args.method in baseline_scripts:
        cmd = [
            py,
            baseline_scripts[args.method],
            "--users",
            users_arg,
            "--workers",
            str(args.workers),
            "--users-root",
            str(masked_root),
            "--output-root",
            str(output_root),
            "--llm",
            args.llm,
        ]
        for flag, value in (
            ("--max-photos", args.max_photos),
            ("--photo-chars", args.photo_chars),
            ("--max-owner-facts", args.max_owner_facts),
            ("--max-persons", args.max_persons),
        ):
            if value is not None:
                cmd.extend([flag, str(value)])
        _run(cmd)
    elif args.method == "pal_trace":
        stable_root = output_root / "_stable_identity"
        owner_root = output_root / "_owner_mining"
        _run(
            [
                py,
                "scripts/evidence_chain_dossier/run_batch.py",
                "--users",
                users_arg,
                "--workers",
                str(args.workers),
                "--users-root",
                str(masked_root),
                "--output-root",
                str(stable_root),
                "--config",
                STABLE_CONFIG,
                "--use-llm",
                "--llm",
                args.llm,
            ]
        )
        _run(
            [
                py,
                "scripts/evidence_chain_dossier/run_batch.py",
                "--users",
                users_arg,
                "--workers",
                str(args.workers),
                "--users-root",
                str(masked_root),
                "--output-root",
                str(owner_root),
                "--config",
                OWNER_CONFIG,
                "--use-llm",
                "--llm",
                args.llm,
            ]
        )
        _run(
            [
                py,
                "scripts/evidence_chain_dossier/compose_frozen_identity_run.py",
                "--stable-root",
                str(stable_root),
                "--owner-root",
                str(owner_root),
                "--output-root",
                str(output_root),
                "--users-root",
                str(masked_root),
                "--users",
                users_arg,
                "--owner-extra-llm-calls",
                "1",
            ]
        )
    else:
        config = {
            "no_llm_heuristic": STABLE_CONFIG,
            "stable_identity_baseline": STABLE_CONFIG,
            "fresh_owner_mining_rerun": OWNER_CONFIG,
        }[args.method]
        cmd = [
            py,
            "scripts/evidence_chain_dossier/run_batch.py",
            "--users",
            users_arg,
            "--workers",
            str(args.workers),
            "--users-root",
            str(masked_root),
            "--output-root",
            str(output_root),
            "--config",
            config,
        ]
        if args.method != "no_llm_heuristic":
            cmd.extend(["--use-llm", "--llm", args.llm])
        else:
            cmd.append("--force-no-llm")
        _run(cmd)

    save_json(
        {
            "schema_version": "modality_ablation_run.v1",
            "method": args.method,
            "input_setting": args.input_setting,
            "field_only_mask": True,
            "users": users,
            "source_users_root": str(users_root),
            "masked_users_root": str(masked_root),
            "output_root": str(output_root),
        },
        output_root / "modality_ablation_manifest.json",
    )
    print(f"[modality] {args.method}/{args.input_setting} -> {output_root}")


if __name__ == "__main__":
    main()

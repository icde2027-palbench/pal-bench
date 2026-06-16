#!/usr/bin/env python3
"""Run the caption/OCR-only profile reconstruction baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.baselines.profile_baseline_common import (
    add_common_args,
    run_batch,
    run_text_only_user,
    write_method_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="PAL-Bench text-only profile baseline")
    add_common_args(parser)
    args = parser.parse_args()
    write_method_contract(
        args.output_root,
        {
            "method_id": "text_only_profile",
            "visible_fields": ["caption", "visible_text", "text_entities", "timestamp", "location"],
            "hidden_fields": ["visible_face_ids", "face summaries", "raw images"],
            "retrieval_strategy": "rank text-rich public photos by OCR/entity/caption density",
            "llm_policy": "single bounded JSON profile extraction call per user",
            "output_schema": "predicted_profile.v2",
        },
    )
    run_batch(args=args, method_id="text_only_profile", run_one=run_text_only_user)


if __name__ == "__main__":
    main()

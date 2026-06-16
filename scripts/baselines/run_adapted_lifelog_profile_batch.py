#!/usr/bin/env python3
"""Run a conservative lifelog/Memex-style episodic retrieval baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.baselines.profile_baseline_common import (
    add_common_args,
    run_adapted_lifelog_user,
    run_batch,
    write_method_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="PAL-Bench adapted lifelog profile baseline")
    add_common_args(parser)
    parser.set_defaults(max_photos=160, photo_chars=220)
    args = parser.parse_args()
    write_method_contract(
        args.output_root,
        {
            "method_id": "adapted_prior_lifelog",
            "visible_fields": ["caption", "visible_text", "text_entities", "visible_face_ids", "timestamp", "location"],
            "adaptation_contract": "episodes are month-city buckets with representative public photos and face summaries",
            "dropped_prior_assumptions": ["explicit user queries", "manual lifelog tags", "known target list"],
            "llm_policy": "single profile extraction call over episodic summaries",
            "output_schema": "predicted_profile.v2",
        },
    )
    run_batch(args=args, method_id="adapted_prior_lifelog", run_one=run_adapted_lifelog_user)


if __name__ == "__main__":
    main()

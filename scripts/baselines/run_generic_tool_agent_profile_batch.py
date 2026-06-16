#!/usr/bin/env python3
"""Run a generic plan-and-execute tool-use profile baseline."""

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
    run_generic_tool_agent_user,
    write_method_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="PAL-Bench generic tool-use agent baseline")
    add_common_args(parser)
    parser.set_defaults(max_photos=180, photo_chars=240)
    args = parser.parse_args()
    write_method_contract(
        args.output_root,
        {
            "method_id": "generic_tool_agent",
            "visible_fields": ["caption", "visible_text", "text_entities", "visible_face_ids", "timestamp", "location"],
            "tools": ["search_album_text", "inspect_face_context"],
            "stopping_rule": "one planning call plus one final profile-composition call",
            "llm_policy": "generic plan-and-execute without PAL-TRACE identity anchoring",
            "output_schema": "predicted_profile.v2",
        },
    )
    run_batch(args=args, method_id="generic_tool_agent", run_one=run_generic_tool_agent_user)


if __name__ == "__main__":
    main()

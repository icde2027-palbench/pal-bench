#!/usr/bin/env python3
"""Run the compressed long-context multimodal LLM profile baseline."""

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
    run_long_context_user,
    write_method_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="PAL-Bench long-context profile baseline")
    add_common_args(parser)
    parser.set_defaults(max_photos=420, photo_chars=220)
    args = parser.parse_args()
    write_method_contract(
        args.output_root,
        {
            "method_id": "long_context_mm_llm",
            "visible_fields": ["caption", "visible_text", "text_entities", "visible_face_ids", "timestamp", "location"],
            "context_strategy": "compressed information-dense album dump sorted back to timeline order",
            "llm_policy": "single full-profile JSON extraction call per user",
            "output_schema": "predicted_profile.v2",
        },
    )
    run_batch(args=args, method_id="long_context_mm_llm", run_one=run_long_context_user)


if __name__ == "__main__":
    main()

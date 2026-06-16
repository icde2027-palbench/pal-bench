#!/usr/bin/env python3
"""Run a generic multimodal RAG profile reconstruction baseline."""

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
    run_multimodal_rag_user,
    write_method_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="PAL-Bench multimodal RAG profile baseline")
    add_common_args(parser)
    args = parser.parse_args()
    write_method_contract(
        args.output_root,
        {
            "method_id": "multimodal_rag",
            "visible_fields": ["caption", "visible_text", "text_entities", "visible_face_ids", "timestamp", "location"],
            "retrieval_strategy": "field-level ranking of text-rich photos plus per-face evidence summaries",
            "ranking_features": ["OCR/entity density", "owner-face presence", "face frequency", "owner co-presence"],
            "llm_policy": "single bounded JSON profile extraction call per user",
            "output_schema": "predicted_profile.v2",
        },
    )
    run_batch(args=args, method_id="multimodal_rag", run_one=run_multimodal_rag_user)


if __name__ == "__main__":
    main()

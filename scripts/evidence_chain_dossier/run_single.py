#!/usr/bin/env python3
"""Run Evidence-Chain Dossier on a single user."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.evidence_chain_dossier.runner import run_evidence_chain_dossier


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Evidence-Chain Dossier on one user")
    parser.add_argument("--user", required=True, help="User ID, e.g. user_0000")
    parser.add_argument("--users-root", default="data/full/users")
    parser.add_argument("--output-root", default="outputs/evidence_chain_dossier_runs")
    parser.add_argument("--config", default=None)
    parser.add_argument("--use-llm", action="store_true", help="Enable ECD-L LLM rerank/global/readout passes")
    parser.add_argument("--force-no-llm", action="store_true", help="Disable all LLM-backed config phases")
    parser.add_argument("--llm", default="agent_llm", help="LLM role from configs/models.yaml")
    args = parser.parse_args()

    user_id = args.user
    users_root = Path(args.users_root)
    output_root = Path(args.output_root)
    agent_album_path = users_root / user_id / f"{user_id}_agent_album.json"
    if not agent_album_path.exists():
        print(f"ERROR: Agent album not found: {agent_album_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = output_root / user_id
    print(f"[ECD] Running on {user_id}")
    print(f"  Album: {agent_album_path}")
    print(f"  Output: {output_dir}")
    llm_client = None
    if args.use_llm:
        from src.llm import create_llm_for_role

        print(f"  LLM: {args.llm}")
        llm_client = create_llm_for_role(args.llm)
    t0 = time.time()
    profile = run_evidence_chain_dossier(
        agent_album_path=str(agent_album_path),
        output_dir=str(output_dir),
        config_path=args.config,
        llm_client=llm_client,
        use_llm=args.use_llm,
        force_no_llm=args.force_no_llm,
    )
    elapsed = time.time() - t0
    persons = profile.get("persons") or []
    named = sum(1 for row in persons if row.get("canonical_name"))
    facts = len((profile.get("owner") or {}).get("facts") or [])
    print(f"\n[ECD] Done in {elapsed:.2f}s")
    print(f"  Persons: {len(persons)} total, {named} named")
    print(f"  Owner facts: {facts}")
    print(f"  LLM calls: {profile.get('run_stats', {}).get('n_llm_calls', 0)}")


if __name__ == "__main__":
    main()

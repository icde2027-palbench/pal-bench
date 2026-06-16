#!/usr/bin/env python3
"""Run Evidence-Chain Dossier on multiple users."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.evidence_chain_dossier.runner import run_evidence_chain_dossier


def run_one(
    user_id: str,
    users_root: Path,
    output_root: Path,
    config: str | None,
    use_llm: bool,
    force_no_llm: bool,
    llm_role: str,
) -> dict:
    album = users_root / user_id / f"{user_id}_agent_album.json"
    out_dir = output_root / user_id
    t0 = time.time()
    try:
        llm_client = None
        if use_llm:
            from src.llm import create_llm_for_role

            llm_client = create_llm_for_role(llm_role)
        profile = run_evidence_chain_dossier(
            str(album),
            str(out_dir),
            config_path=config,
            llm_client=llm_client,
            use_llm=use_llm,
            force_no_llm=force_no_llm,
        )
        persons = profile.get("persons") or []
        return {
            "user_id": user_id,
            "status": "success",
            "elapsed": time.time() - t0,
            "n_persons": len(persons),
            "n_named": sum(1 for row in persons if row.get("canonical_name")),
            "n_owner_facts": len((profile.get("owner") or {}).get("facts") or []),
            "n_llm_calls": profile.get("run_stats", {}).get("n_llm_calls", 0),
        }
    except Exception as exc:
        return {
            "user_id": user_id,
            "status": f"error: {exc}",
            "elapsed": time.time() - t0,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ECD on multiple users")
    parser.add_argument("--users", required=True, help="Comma-separated user IDs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--users-root", default="data/full/users")
    parser.add_argument("--output-root", default="outputs/evidence_chain_dossier_runs")
    parser.add_argument("--config", default=None)
    parser.add_argument("--use-llm", action="store_true", help="Enable ECD-L LLM passes")
    parser.add_argument("--force-no-llm", action="store_true", help="Disable all LLM-backed config phases")
    parser.add_argument("--llm", default="agent_llm", help="LLM role from configs/models.yaml")
    args = parser.parse_args()

    user_ids = [u.strip() for u in args.users.split(",") if u.strip()]
    users_root = Path(args.users_root)
    output_root = Path(args.output_root)
    print(f"[ECD Batch] {len(user_ids)} users, workers={args.workers}, llm={args.llm if args.use_llm else 'off'}")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one,
                uid,
                users_root,
                output_root,
                args.config,
                args.use_llm,
                args.force_no_llm,
                args.llm,
            ): uid
            for uid in user_ids
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["status"] == "success":
                print(
                    f"  OK {result['user_id']}: {result['n_named']}/{result['n_persons']} named, "
                    f"{result['n_owner_facts']} facts, {result['n_llm_calls']} calls, {result['elapsed']:.2f}s"
                )
            else:
                print(f"  ERR {result['user_id']}: {result['status']} ({result['elapsed']:.2f}s)")
    n_ok = sum(1 for row in results if row["status"] == "success")
    print(f"\n[ECD Batch] Done: {n_ok}/{len(results)} succeeded in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()

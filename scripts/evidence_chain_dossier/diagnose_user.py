#!/usr/bin/env python3
"""Inspect ECD evidence chains for one user, face, or name."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.evidence_chain_dossier.candidates import build_identity_hypotheses
from src.agent.evidence_chain_dossier.config import ECDConfig
from src.agent.evidence_chain_dossier.inventory import build_inventory
from src.agent.evidence_chain_dossier.name_cluster import build_name_clusters
from src.agent.evidence_chain_dossier.resolver import resolve_identities
from src.agent.evidence_chain_dossier.text import normalize_name
from src.utils.io import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose ECD evidence for a user")
    parser.add_argument("--user", required=True)
    parser.add_argument("--face", default=None, help="Optional face_id to inspect")
    parser.add_argument("--name", default=None, help="Optional name or substring to inspect")
    parser.add_argument("--users-root", default="data/full/users")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    album_path = Path(args.users_root) / args.user / f"{args.user}_agent_album.json"
    if not album_path.exists():
        print(f"ERROR: Missing album: {album_path}", file=sys.stderr)
        sys.exit(1)

    album = load_json(album_path)
    inventory = build_inventory(album)
    clusters = build_name_clusters(inventory)
    hypotheses, diagnostics = build_identity_hypotheses(inventory, clusters, max_per_face=20)
    resolved, resolver_diag = resolve_identities(inventory, clusters, hypotheses, ECDConfig())

    name_query = normalize_name(args.name or "").lower()
    selected_clusters = [
        c for c in clusters
        if not name_query or name_query in c.primary_surface.lower() or any(name_query in s.lower() for s in c.surfaces)
    ]
    selected_hyps = [
        h for h in hypotheses
        if (not args.face or h.face_id == args.face)
        and (not selected_clusters or h.name_cluster_id in {c.cluster_id for c in selected_clusters})
    ]
    selected_hyps.sort(key=lambda h: -h.score)

    result = {
        "user_id": args.user,
        "owner": {
            "face_id": inventory.owner_face_id,
            "name": inventory.owner_name,
            "name_candidates": inventory.owner_name_candidates[:10],
        },
        "cluster_matches": [c.to_dict() for c in selected_clusters[:20]],
        "hypotheses": [h.to_dict() for h in selected_hyps[:50]],
        "resolved": [r.to_profile_row() for r in resolved if not args.face or r.face_id == args.face],
        "topk_by_face": diagnostics,
        "resolver": resolver_diag,
    }
    if args.output:
        save_json(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

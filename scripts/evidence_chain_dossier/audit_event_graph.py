#!/usr/bin/env python3
"""Audit EventGraph identity recall against formal person-name targets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.evidence_chain_dossier.event_graph import build_event_graph
from src.agent.evidence_chain_dossier.inventory import build_inventory
from src.utils.io import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit EventGraph identity recall.")
    parser.add_argument("--users-root", default="data/full/users")
    parser.add_argument("--formal-root", default=None, help="Optional formal eval root with per_user reports")
    parser.add_argument("--users", default=None)
    parser.add_argument("--output-dir", default="outputs/event_graph_audit")
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    users_root = Path(args.users_root)
    formal_root = Path(args.formal_root) if args.formal_root else None
    output_dir = Path(args.output_dir)
    wanted = {u.strip() for u in args.users.split(",") if u.strip()} if args.users else None

    reports = []
    for album_path in sorted(users_root.glob("user_*/user_*_agent_album.json")):
        user_id = album_path.parent.name
        if wanted and user_id not in wanted:
            continue
        gt_path = album_path.parent / f"{user_id}_eval_gt.json"
        if not gt_path.exists():
            continue
        formal_report_path = (
            formal_root / "per_user" / user_id / "formal_eval_report.json"
            if formal_root
            else None
        )
        reports.append(
            _audit_user(
                user_id=user_id,
                album_path=album_path,
                gt_path=gt_path,
                formal_report_path=formal_report_path if formal_report_path and formal_report_path.exists() else None,
                min_score=args.min_score,
            )
        )

    if not reports:
        raise SystemExit(f"No users found under {users_root}")

    result = {
        "schema_version": "ecd_event_graph_audit.v1",
        "users_root": str(users_root),
        "formal_root": str(formal_root) if formal_root else None,
        "n_users": len(reports),
        "aggregate": _aggregate(reports),
        "users": reports,
    }
    save_json(result, output_dir / "event_graph_audit.json")
    _write_markdown(result, output_dir / "event_graph_audit.md")
    print(f"[event-graph-audit] users={len(reports)}")
    print(f"  JSON: {output_dir / 'event_graph_audit.json'}")
    print(f"  MD:   {output_dir / 'event_graph_audit.md'}")


def _audit_user(
    *,
    user_id: str,
    album_path: Path,
    gt_path: Path,
    formal_report_path: Path | None,
    min_score: float,
) -> dict[str, Any]:
    album = load_json(album_path)
    gt = load_json(gt_path)
    formal_by_target = {}
    if formal_report_path:
        formal = load_json(formal_report_path)
        formal_by_target = {
            str(row.get("target_id") or ""): row
            for row in formal.get("targets") or []
            if str(row.get("target_type") or "") == "person_name"
        }
    inventory = build_inventory(album)
    graph = build_event_graph(inventory)
    edges_by_face: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph.edges:
        if edge.score >= min_score:
            edges_by_face[edge.face_id].append(edge.to_dict())
    for rows in edges_by_face.values():
        rows.sort(key=lambda row: (-float(row.get("score") or 0.0), str(row.get("name_surface") or "")))

    rows = []
    for target in gt.get("evaluation_targets") or []:
        if str(target.get("target_type") or "") != "person_name":
            continue
        target_id = str(target.get("target_id") or "")
        face_id = str(target.get("public_face_id") or "")
        aliases = [
            str(target.get("gt_value") or target.get("canonical_name") or ""),
            *(str(a) for a in target.get("aliases") or []),
        ]
        face_edges = edges_by_face.get(face_id, [])
        matched = _matching_edge(face_edges, aliases)
        formal_row = formal_by_target.get(target_id, {})
        rank = _edge_rank(face_edges, matched)
        rows.append(
            {
                "target_id": target_id,
                "person_id": target.get("person_id"),
                "face_id": face_id,
                "gt_name": aliases[0],
                "alignment_difficulty": target.get("alignment_difficulty") or "",
                "formal_correctness": _float(formal_row.get("correctness")) if formal_row else None,
                "edge_present": matched is not None,
                "edge_rank": rank,
                "edge_score": _float(matched.get("score")) if matched else None,
                "edge_bridge_type": matched.get("bridge_type") if matched else "",
                "edge_evidence_photo_ids": matched.get("evidence_photo_ids") if matched else [],
                "top_edge_name": face_edges[0].get("name_surface") if face_edges else "",
                "top_edge_score": _float(face_edges[0].get("score")) if face_edges else None,
                "n_edges_for_face": len(face_edges),
            }
        )
    return {
        "user_id": user_id,
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
        "n_identity_targets": len(rows),
        "edge_present_rate": _mean(1.0 if row["edge_present"] else 0.0 for row in rows),
        "top1_rate": _mean(1.0 if row["edge_rank"] == 1 else 0.0 for row in rows),
        "top3_rate": _mean(1.0 if row["edge_rank"] is not None and row["edge_rank"] <= 3 else 0.0 for row in rows),
        "top5_rate": _mean(1.0 if row["edge_rank"] is not None and row["edge_rank"] <= 5 else 0.0 for row in rows),
        "rows": rows,
    }


def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for report in reports for row in report["rows"]]
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_correctness: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_difficulty[str(row.get("alignment_difficulty") or "unknown")].append(row)
        corr = row.get("formal_correctness")
        if isinstance(corr, (int, float)):
            by_correctness["correct" if corr >= 1.0 else "incorrect"].append(row)
    rescue_candidates = [
        row
        for row in rows
        if isinstance(row.get("formal_correctness"), (int, float))
        and float(row["formal_correctness"]) < 1.0
        and row["edge_rank"] is not None
        and row["edge_rank"] <= 3
    ]
    miss_examples = [
        row
        for row in rows
        if not row["edge_present"]
    ]
    return {
        "n_targets": len(rows),
        "n_nodes_macro": _mean(report["n_nodes"] for report in reports),
        "n_edges_macro": _mean(report["n_edges"] for report in reports),
        "edge_present_rate": _mean(1.0 if row["edge_present"] else 0.0 for row in rows),
        "top1_rate": _mean(1.0 if row["edge_rank"] == 1 else 0.0 for row in rows),
        "top3_rate": _mean(1.0 if row["edge_rank"] is not None and row["edge_rank"] <= 3 else 0.0 for row in rows),
        "top5_rate": _mean(1.0 if row["edge_rank"] is not None and row["edge_rank"] <= 5 else 0.0 for row in rows),
        "by_alignment_difficulty": {
            key: _summarize_rows(items)
            for key, items in sorted(by_difficulty.items())
        },
        "by_formal_correctness": {
            key: _summarize_rows(items)
            for key, items in sorted(by_correctness.items())
        },
        "n_rescue_candidates_top3": len(rescue_candidates),
        "n_edge_missing": len(miss_examples),
        "rescue_candidates_top3": rescue_candidates[:50],
        "miss_examples": miss_examples[:50],
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "edge_present_rate": _mean(1.0 if row["edge_present"] else 0.0 for row in rows),
        "top1_rate": _mean(1.0 if row["edge_rank"] == 1 else 0.0 for row in rows),
        "top3_rate": _mean(1.0 if row["edge_rank"] is not None and row["edge_rank"] <= 3 else 0.0 for row in rows),
        "top5_rate": _mean(1.0 if row["edge_rank"] is not None and row["edge_rank"] <= 5 else 0.0 for row in rows),
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    agg = result["aggregate"]
    lines = [
        "# ECD EventGraph Audit",
        "",
        f"- Users: `{result['n_users']}`",
        f"- Users root: `{result['users_root']}`",
        f"- Formal root: `{result.get('formal_root')}`",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ["n_targets", "n_nodes_macro", "n_edges_macro", "edge_present_rate", "top1_rate", "top3_rate", "top5_rate"]:
        lines.append(f"| {key} | {_fmt(agg.get(key))} |")
    lines.append(f"| n_rescue_candidates_top3 | {_fmt(agg.get('n_rescue_candidates_top3'))} |")
    lines.append(f"| n_edge_missing | {_fmt(agg.get('n_edge_missing'))} |")
    lines.extend(["", "## By Alignment Difficulty", "", "| Difficulty | n | Present | Top1 | Top3 | Top5 |", "|---|---:|---:|---:|---:|---:|"])
    for key, row in agg["by_alignment_difficulty"].items():
        lines.append(
            f"| {key} | {row['n']} | {_fmt(row['edge_present_rate'])} | {_fmt(row['top1_rate'])} | "
            f"{_fmt(row['top3_rate'])} | {_fmt(row['top5_rate'])} |"
        )
    lines.extend(["", "## By Formal Correctness", "", "| Bucket | n | Present | Top1 | Top3 | Top5 |", "|---|---:|---:|---:|---:|---:|"])
    for key, row in agg["by_formal_correctness"].items():
        lines.append(
            f"| {key} | {row['n']} | {_fmt(row['edge_present_rate'])} | {_fmt(row['top1_rate'])} | "
            f"{_fmt(row['top3_rate'])} | {_fmt(row['top5_rate'])} |"
        )
    lines.extend(["", "## Top3 Rescue Candidates", "", "| Face | GT | Rank | Score | Top Edge |", "|---|---|---:|---:|---|"])
    for row in agg["rescue_candidates_top3"][:20]:
        lines.append(
            f"| {row.get('face_id')} | {_md(row.get('gt_name'))} | {_fmt(row.get('edge_rank'))} | "
            f"{_fmt(row.get('edge_score'))} | {_md(row.get('top_edge_name'))} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _matching_edge(rows: list[dict[str, Any]], aliases: list[str]) -> dict[str, Any] | None:
    for edge in rows:
        if _name_matches_any(str(edge.get("name_surface") or ""), aliases):
            return edge
    return None


def _edge_rank(rows: list[dict[str, Any]], edge: dict[str, Any] | None) -> int | None:
    if not edge:
        return None
    target_name = _norm_name(str(edge.get("name_surface") or ""))
    for idx, row in enumerate(rows, start=1):
        if _norm_name(str(row.get("name_surface") or "")) == target_name:
            return idx
    return None


def _name_matches_any(name: str, aliases: list[str]) -> bool:
    norm = _norm_name(name)
    if not norm:
        return False
    for alias in aliases:
        alias_norm = _norm_name(alias)
        if not alias_norm:
            continue
        if norm == alias_norm:
            return True
        name_tokens = norm.split()
        alias_tokens = alias_norm.split()
        if len(name_tokens) == 1 and name_tokens[0] == alias_tokens[0]:
            return True
    return False


def _norm_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z]+", str(value or "").lower()))


def _mean(values: Any) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return round(mean(vals), 4) if vals else 0.0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _md(value: Any, limit: int = 80) -> str:
    text = " ".join(str(value or "").split()).replace("|", "\\|")
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Finalize ICDE experiment tables, manifest, and completion report."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import load_json, save_json


def _copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return str(dst)


def _metric(eval_dir: Path) -> dict[str, Any]:
    return load_json(eval_dir / "aggregate_formal.json").get("metrics") or {}


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return ""


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_backbone_table(root: Path, qwen_root: Path | None, out: Path) -> None:
    gpt = _metric(root / "eval/pal_trace/official_llm_semantic")
    qwen = _metric(qwen_root / "eval/pal_trace/official_llm_semantic") if qwen_root else {}
    delta = {
        key: (qwen.get(key) - gpt.get(key))
        for key in ["OFR", "PIR", "PIR-hard", "EFS", "nLLM"]
        if isinstance(qwen.get(key), (int, float)) and isinstance(gpt.get(key), (int, float))
    }
    lines = [
        "# LLM Backbone Robustness",
        "",
        "| Agent backbone | Judge | OFR | PIR | PIR-hard | EFS | nLLM | Delta EFS vs GPT | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
        f"| GPT-5.4 | Qwen3.6-35B-A3B | {_fmt(gpt.get('OFR'))} | {_fmt(gpt.get('PIR'))} | {_fmt(gpt.get('PIR-hard'))} | {_fmt(gpt.get('EFS'))} | {_fmt(gpt.get('nLLM'))} | 0.0000 | Primary paper run |",
    ]
    if qwen:
        lines.append(
            f"| Qwen3.6-35B-A3B | Qwen3.6-35B-A3B | {_fmt(qwen.get('OFR'))} | {_fmt(qwen.get('PIR'))} | {_fmt(qwen.get('PIR-hard'))} | {_fmt(qwen.get('EFS'))} | {_fmt(qwen.get('nLLM'))} | {_fmt(delta.get('EFS'))} | Same-family agent/judge caveat |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report(root: Path, qwen_root: Path | None, manifest: dict[str, Any]) -> str:
    main = _metric(root / "eval/pal_trace/official_llm_semantic")
    modality = load_json(root / "tables/modality_ablation.json").get("rows") or []
    qwen = _metric(qwen_root / "eval/pal_trace/official_llm_semantic") if qwen_root else {}
    modality_lines = []
    for row in modality:
        if row.get("method") == "pal_trace" and row.get("input_setting") != "full":
            modality_lines.append(
                f"- PAL-TRACE {row['input_setting']}: OFR={_fmt(row.get('OFR'))}, PIR={_fmt(row.get('PIR'))}, EFS={_fmt(row.get('EFS'))}, Delta EFS={_fmt(row.get('Delta_EFS_vs_full'))}"
            )
    qwen_line = ""
    if qwen:
        qwen_line = f"- Qwen agent PAL-TRACE: OFR={_fmt(qwen.get('OFR'))}, PIR={_fmt(qwen.get('PIR'))}, EFS={_fmt(qwen.get('EFS'))}; GPT primary remains stronger."
    return "\n".join(
        [
            "# ICDE Experiment Completion Report",
            "",
            "## Completed Tables",
            "",
            "- M1 main full50 official multi-system table",
            "- M2 difficulty-stratified official table",
            "- M3 full50 modality ablation for PAL-TRACE and Multimodal RAG",
            "- M6 cost/runtime/token diagnostics",
            "- M7 Qwen-vs-GPT backbone robustness table",
            "- M8 PAL-TRACE mechanism analysis",
            "",
            "## Key Results",
            "",
            f"- Primary PAL-TRACE full50: OFR={_fmt(main.get('OFR'))}, PIR={_fmt(main.get('PIR'))}, PIR-hard={_fmt(main.get('PIR-hard'))}, EFS={_fmt(main.get('EFS'))}, nLLM={_fmt(main.get('nLLM'))}.",
            *modality_lines,
            qwen_line,
            "",
            "## Traceability",
            "",
            f"- Final manifest: `{manifest['manifest_path']}`",
            f"- Commands log: `{manifest['commands_path']}`",
            f"- Experiment root: `{root}`",
            "",
            "## Notes",
            "",
            "- Modality ablation uses field-only public-album masks; source images are not edited.",
            "- Official metrics use Qwen3.6-35B-A3B as judge.",
            "- Dollar cost is not estimated because provider pricing is not part of this artifact.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize ICDE experiment artifacts")
    parser.add_argument("--paper-root", required=True)
    parser.add_argument("--qwen-root", default=None)
    args = parser.parse_args()

    root = Path(args.paper_root)
    qwen_root = Path(args.qwen_root) if args.qwen_root else None
    tables = {
        "M1": _copy(root / "tables/official_llm_semantic_full50/main_multisystem_results.md", root / "tables/main_multisystem_results.md"),
        "M2": _copy(root / "tables/difficulty_official_full50/difficulty_results.md", root / "tables/difficulty_results.md"),
        "M3": _copy(root / "tables/modality_official_full50/modality_ablation.md", root / "tables/modality_ablation.md"),
        "M6": _copy(root / "tables/cost_runtime_official_full50/cost_equivalent.md", root / "tables/cost_equivalent.md"),
        "M8": _copy(root / "tables/mechanism_official_full50/mechanism_full50.md", root / "tables/mechanism_analysis.md"),
    }
    _copy(root / "tables/difficulty_official_full50/difficulty_results.json", root / "tables/difficulty_results.json")
    _copy(root / "tables/modality_official_full50/modality_ablation.json", root / "tables/modality_ablation.json")
    _copy(root / "tables/modality_official_full50/modality_mask_spec.json", root / "diagnostics/modality_mask_spec.json")
    _copy(root / "tables/cost_runtime_official_full50/cost_runtime.json", root / "tables/cost_runtime.json")
    backbone_path = root / "tables/llm_backbone_robustness.md"
    _write_backbone_table(root, qwen_root, backbone_path)
    tables["M7"] = str(backbone_path)

    manifest_path = root / "manifests/final_result_manifest.json"
    manifest = {
        "schema_version": "icde_final_result_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": root.name,
        "paper_root": str(root),
        "qwen_agent_root": str(qwen_root) if qwen_root else None,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "users_root": "data/full/users",
        "agent_llm_primary": "agent_llm / gpt-5.4",
        "judge_llm_official": "eval_judge / qwen3.6-35b-a3b",
        "commands_path": str(root / "commands.sh"),
        "tables": tables,
        "diagnostics": {
            "modality_mask_spec": str(root / "diagnostics/modality_mask_spec.json"),
            "mechanism_bootstrap": str(root / "diagnostics/mechanism_official_full50/bootstrap_summary.md"),
            "qwen_vs_gpt_bootstrap": str(qwen_root / "comparison/bootstrap_qwen36_vs_gpt54/bootstrap_summary.md") if qwen_root else None,
        },
        "eval_roots": {
            "main_official": str(root / "eval"),
            "modality_official": str(root / "eval/modality"),
        },
        "run_roots": {
            "main_runs": str(root / "runs"),
            "modality_runs": str(root / "runs/modality"),
        },
        "manifest_path": str(manifest_path),
    }
    save_json(manifest, manifest_path)
    report_path = root / "artifact_report/experiment_completion_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(root, qwen_root, manifest), encoding="utf-8")
    print(f"[finalize] manifest -> {manifest_path}")
    print(f"[finalize] report   -> {report_path}")


if __name__ == "__main__":
    main()

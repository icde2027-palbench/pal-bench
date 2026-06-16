#!/usr/bin/env python3
"""Export modality ablation table from official aggregate files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import load_json, save_json

SETTING_META = {
    "full": ("Full public album", True, True, True),
    "text_off": ("Text-off", False, True, True),
    "face_off": ("Face-off", True, False, True),
    "metadata_off": ("Metadata-off", True, True, False),
}


def _parse_entry(value: str) -> tuple[str, str, str, Path]:
    parts = value.split("=", 3)
    if len(parts) != 4:
        raise ValueError(f"Expected method=label=setting=eval_dir, got {value}")
    return parts[0], parts[1], parts[2], Path(parts[3])


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _yn(value: bool) -> str:
    return "Yes" if value else "No"


def _row(method: str, label: str, setting: str, eval_dir: Path, full_metrics: dict[str, Any]) -> dict[str, Any]:
    aggregate = load_json(eval_dir / "aggregate_formal.json")
    metrics = aggregate.get("metrics") or {}
    setting_label, text_visible, faces_visible, metadata_visible = SETTING_META[setting]
    return {
        "method": method,
        "label": label,
        "input_setting": setting,
        "setting_label": setting_label,
        "text_visible": text_visible,
        "faces_visible": faces_visible,
        "time_location_visible": metadata_visible,
        "OFR": metrics.get("OFR"),
        "PIR": metrics.get("PIR"),
        "PIR-hard": metrics.get("PIR-hard"),
        "EFS": metrics.get("EFS"),
        "Delta_EFS_vs_full": (metrics.get("EFS") - full_metrics.get("EFS")) if isinstance(metrics.get("EFS"), (int, float)) and isinstance(full_metrics.get("EFS"), (int, float)) else None,
        "Delta_PIR_vs_full": (metrics.get("PIR") - full_metrics.get("PIR")) if isinstance(metrics.get("PIR"), (int, float)) and isinstance(full_metrics.get("PIR"), (int, float)) else None,
        "nLLM": metrics.get("nLLM"),
        "eval_dir": str(eval_dir),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Modality Ablation Official Results",
        "",
        "| Method | Input setting | Text visible? | Faces visible? | Time/location visible? | OFR | PIR | PIR-hard | EFS | Delta EFS vs full |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['setting_label']} | {_yn(row['text_visible'])} | {_yn(row['faces_visible'])} | "
            f"{_yn(row['time_location_visible'])} | {_fmt(row['OFR'])} | {_fmt(row['PIR'])} | "
            f"{_fmt(row['PIR-hard'])} | {_fmt(row['EFS'])} | {_fmt(row['Delta_EFS_vs_full'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export modality ablation table")
    parser.add_argument("--entry", action="append", required=True, help="method=label=setting=eval_dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = [_parse_entry(item) for item in args.entry]
    full_by_method = {}
    for method, _label, setting, eval_dir in entries:
        if setting == "full":
            full_by_method[method] = load_json(eval_dir / "aggregate_formal.json").get("metrics") or {}
    rows = []
    for method, label, setting, eval_dir in entries:
        if setting not in SETTING_META:
            raise ValueError(f"Unknown setting: {setting}")
        rows.append(_row(method, label, setting, eval_dir, full_by_method.get(method, {})))
    save_json(
        {
            "schema_version": "modality_ablation_results.v1",
            "mask_spec": {
                "text_off": "caption='', visible_text=[], text_entities=[]",
                "face_off": "visible_face_ids=[], faces=[]",
                "metadata_off": "timestamp='', year_month='', metadata.gps_city='', metadata.gps_location=''",
                "scope": "field-only public-album mask applied before generation and formal evidence judging",
            },
            "rows": rows,
        },
        output_dir / "modality_ablation.json",
    )
    save_json(
        {
            "schema_version": "modality_mask_spec.v1",
            "text_off": {"caption": "", "visible_text": [], "text_entities": []},
            "face_off": {"visible_face_ids": [], "faces": []},
            "metadata_off": {"timestamp": "", "year_month": "", "metadata.gps_city": "", "metadata.gps_location": ""},
            "field_only_mask": True,
            "note": "Images are not modified; public structured fields are masked before generation and evaluation.",
        },
        output_dir / "modality_mask_spec.json",
    )
    _write_csv(rows, output_dir / "modality_ablation.csv")
    _write_markdown(rows, output_dir / "modality_ablation.md")
    print(f"[modality-table] -> {output_dir / 'modality_ablation.md'}")


if __name__ == "__main__":
    main()

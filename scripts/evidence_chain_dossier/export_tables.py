#!/usr/bin/env python3
"""Generate Markdown experiment tables from formal eval outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import load_json, save_json

MAIN_COLUMNS = ["OFR", "PIR", "PIR-hard", "PRR-ID", "EFS", "ECE", "nLLM"]


def _parse_eval(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    return name, Path(path)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "NA"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Markdown tables")
    parser.add_argument("--eval", action="append", required=True, help="method=path/to/aggregate_formal.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in args.eval:
        method, path = _parse_eval(spec)
        data = load_json(path)
        rows.append({"Method": method, **(data.get("metrics") or {})})

    lines = ["| Method | " + " | ".join(MAIN_COLUMNS) + " |"]
    lines.append("|---|" + "|".join("---:" for _ in MAIN_COLUMNS) + "|")
    for row in rows:
        lines.append("| " + str(row["Method"]) + " | " + " | ".join(_fmt(row.get(col)) for col in MAIN_COLUMNS) + " |")
    table = "\n".join(lines) + "\n"
    (output_dir / "main_multisystem_results.md").write_text(table, encoding="utf-8")
    save_json({"schema_version": "table_export.v1", "rows": rows}, output_dir / "table_export.json")
    print(f"[tables] -> {output_dir / 'main_multisystem_results.md'}")


if __name__ == "__main__":
    main()

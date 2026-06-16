#!/usr/bin/env python3
"""Export compact diagnostics CSV/JSON from formal eval aggregates."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import load_json, save_json


def _parse_eval(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export formal eval diagnostics")
    parser.add_argument("--eval", action="append", required=True, help="method=path/to/aggregate_formal.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_user_rows = []
    summary_rows = []
    for spec in args.eval:
        method, path = _parse_eval(spec)
        data = load_json(path)
        summary_rows.append(
            {
                "method": method,
                "n_users": data.get("n_users"),
                "judge_mode": data.get("judge_mode"),
                "evidence_judge_mode": data.get("evidence_judge_mode"),
                **{f"metric_{k}": v for k, v in (data.get("metrics") or {}).items()},
                **{f"diag_{k}": v for k, v in (data.get("diagnostics") or {}).items() if not isinstance(v, dict)},
            }
        )
        for row in data.get("per_user") or []:
            flat = {"method": method, "user_id": row.get("user_id")}
            flat.update({f"metric_{k}": v for k, v in (row.get("metrics") or {}).items()})
            flat.update({f"diag_{k}": v for k, v in (row.get("diagnostics") or {}).items() if not isinstance(v, dict)})
            per_user_rows.append(flat)

    _write_csv(output_dir / "summary_metrics.csv", summary_rows)
    _write_csv(output_dir / "per_user_metrics.csv", per_user_rows)
    save_json(
        {
            "schema_version": "formal_diagnostics_export.v1",
            "summary_rows": summary_rows,
            "per_user_rows": per_user_rows,
        },
        output_dir / "diagnostics_export.json",
    )
    print(f"[diagnostics] -> {output_dir}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

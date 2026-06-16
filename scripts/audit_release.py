#!/usr/bin/env python3
"""Lightweight release audit for PAL-Bench."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "outputs",
    "data/full",
    "data/images",
}

PATTERNS = {
    "mac_absolute_path": re.compile(r"/Users/[A-Za-z0-9_.-]+/"),
    "github_pat": re.compile("github" + r"_pat_[A-Za-z0-9_]+"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    "hardcoded_bearer": re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    "hardcoded_authorization": re.compile(r"Authorization['\"]?\s*[:=]\s*['\"][A-Za-z0-9._~+/=-]{20,}"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit release files for common leaks")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in DEFAULT_EXCLUDES for part in rel.parts):
            continue
        if path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{name}: {rel}")

    audit_path = root / "data/sample/users/user_0000/user_0000_export_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("passed") is not True:
            findings.append(f"sample_export_audit_failed: {audit_path.relative_to(root)}")

    if findings:
        print("Release audit failed:", file=sys.stderr)
        for item in findings:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("Release audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create a PAL-Bench users-root with cross-user swapped captions.

The derived root preserves each target user's IDs, timestamps, metadata, OCR,
faces, and evaluator ground truth. It replaces only the public caption channel
and caption-sourced text entities with donor-user content. This supports a
shortcut probe for whether caption phrasing acts as an unintended oracle.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_json, save_json


def _user_ids(users_root: Path, users: str | None) -> list[str]:
    if users:
        return [item.strip() for item in users.split(",") if item.strip()]
    return sorted(path.name for path in users_root.glob("user_*") if path.is_dir())


def _load_album(users_root: Path, user_id: str) -> dict[str, Any]:
    path = users_root / user_id / f"{user_id}_agent_album.json"
    album = load_json(path)
    if not isinstance(album, dict):
        raise ValueError(f"Expected album object at {path}")
    return album


def _photo_bucket(photo: dict[str, Any]) -> tuple[str, str, str]:
    timestamp = str(photo.get("timestamp") or photo.get("year_month") or "")
    month = timestamp[5:7] if len(timestamp) >= 7 else "00"
    text_count = len(photo.get("visible_text") or [])
    face_count = len(photo.get("visible_face_ids") or [])
    text_bucket = "t0" if text_count == 0 else "t1" if text_count <= 3 else "t2"
    face_bucket = "f1" if face_count else "f0"
    return month, text_bucket, face_bucket


def _caption_entities(photo: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for ent in photo.get("text_entities") or []:
        if not isinstance(ent, dict):
            continue
        if str(ent.get("source") or "").lower() == "caption":
            out.append(dict(ent))
    return out


def _non_caption_entities(photo: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for ent in photo.get("text_entities") or []:
        if not isinstance(ent, dict):
            continue
        if str(ent.get("source") or "").lower() != "caption":
            out.append(dict(ent))
    return out


def _make_donor_index(donor_album: dict[str, Any]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for photo in donor_album.get("photos") or []:
        if isinstance(photo, dict):
            index[_photo_bucket(photo)].append(photo)
    for rows in index.values():
        rows.sort(key=lambda p: str(p.get("photo_id") or ""))
    return index


def _choose_donor_photo(
    *,
    target_photo: dict[str, Any],
    donor_album: dict[str, Any],
    donor_index: dict[tuple[str, str, str], list[dict[str, Any]]],
    ordinal: int,
) -> tuple[dict[str, Any], str]:
    key = _photo_bucket(target_photo)
    candidates = donor_index.get(key)
    match_level = "month_text_face"
    if not candidates:
        month = key[0]
        candidates = [
            p for p in donor_album.get("photos") or []
            if isinstance(p, dict) and _photo_bucket(p)[0] == month
        ]
        match_level = "month"
    if not candidates:
        candidates = [p for p in donor_album.get("photos") or [] if isinstance(p, dict)]
        match_level = "fallback_all"
    if not candidates:
        return {}, "missing"
    return candidates[ordinal % len(candidates)], match_level


def build_caption_swap_root(users_root: Path, output_root: Path, users: list[str], donor_shift: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    albums = {user_id: _load_album(users_root, user_id) for user_id in users}
    manifest_rows = []
    for idx, user_id in enumerate(users):
        donor_id = users[(idx + donor_shift) % len(users)]
        if donor_id == user_id:
            donor_id = users[(idx + 1) % len(users)]
        target_album = json.loads(json.dumps(albums[user_id]))
        donor_album = albums[donor_id]
        donor_index = _make_donor_index(donor_album)
        photo_rows = []
        for ordinal, photo in enumerate(target_album.get("photos") or []):
            if not isinstance(photo, dict):
                continue
            donor_photo, match_level = _choose_donor_photo(
                target_photo=photo,
                donor_album=donor_album,
                donor_index=donor_index,
                ordinal=ordinal,
            )
            old_caption = str(photo.get("caption") or "")
            photo["caption"] = str(donor_photo.get("caption") or "") if donor_photo else ""
            photo["text_entities"] = _non_caption_entities(photo) + _caption_entities(donor_photo)
            photo_rows.append(
                {
                    "photo_id": str(photo.get("photo_id") or ""),
                    "donor_user_id": donor_id,
                    "donor_photo_id": str(donor_photo.get("photo_id") or "") if donor_photo else "",
                    "match_level": match_level,
                    "old_caption_chars": len(old_caption),
                    "new_caption_chars": len(str(photo.get("caption") or "")),
                }
            )
        target_album.setdefault("derivation", {})
        target_album["derivation"] = {
            "schema_version": "caption_swap.v1",
            "source_users_root": str(users_root),
            "target_user_id": user_id,
            "donor_user_id": donor_id,
            "caption_source": "cross-user swapped by month/text/face richness bucket",
            "caption_entities": "caption-sourced entities replaced by donor caption-sourced entities",
        }
        out_dir = output_root / user_id
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(target_album, out_dir / f"{user_id}_agent_album.json")
        src_gt = users_root / user_id / f"{user_id}_eval_gt.json"
        dst_gt = out_dir / f"{user_id}_eval_gt.json"
        if src_gt.exists():
            shutil.copyfile(src_gt, dst_gt)
        manifest_rows.append(
            {
                "user_id": user_id,
                "donor_user_id": donor_id,
                "n_photos": len(photo_rows),
                "match_counts": _count_levels(photo_rows),
                "photos": photo_rows,
            }
        )
    save_json(
        {
            "schema_version": "caption_swap_manifest.v1",
            "users_root": str(users_root),
            "output_root": str(output_root),
            "donor_shift": donor_shift,
            "n_users": len(users),
            "rows": manifest_rows,
        },
        output_root / "caption_swap_manifest.json",
    )


def _count_levels(photo_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in photo_rows:
        key = str(row.get("match_level") or "")
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Create cross-user caption-swap PAL-Bench users root")
    parser.add_argument("--users-root", default="data/full/users")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--users", default=None)
    parser.add_argument("--donor-shift", type=int, default=17)
    args = parser.parse_args()

    users_root = Path(args.users_root)
    output_root = Path(args.output_root)
    users = _user_ids(users_root, args.users)
    if len(users) < 2:
        raise SystemExit("Need at least two users for caption swap")
    build_caption_swap_root(users_root, output_root, users, int(args.donor_shift))
    print(f"[caption-swap] users={len(users)} -> {output_root}")


if __name__ == "__main__":
    main()

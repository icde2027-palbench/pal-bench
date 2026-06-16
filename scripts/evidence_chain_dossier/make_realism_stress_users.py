#!/usr/bin/env python3
"""Create field-level realism-stress PAL-Bench user roots.

The derived roots preserve user ids, photo ids, face-id namespace, and evaluator
ground truth. They perturb only agent-visible public album fields to approximate
real-album messiness such as shorter captions, missed OCR, sparse location
metadata, and missed face detections.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import load_json, save_json


SETTING_DESCRIPTIONS = {
    "caption_sparse": "Truncate perception captions and retain only caption entities still present.",
    "ocr_dropout_35": "Drop visible OCR text and visible-text entities from 35% of OCR-bearing photos.",
    "location_sparse_40": "Blank city/location metadata for 40% of geotagged photos while preserving time.",
    "face_miss_15": "Drop 15% of non-owner face appearances and 3% of owner appearances.",
    "combined_mild": "Apply caption truncation, 25% OCR dropout, 30% location sparsity, 10% non-owner face misses, and 2% owner face misses.",
}


def _split_users(users_root: Path, users: str | None) -> list[str]:
    if users:
        return [item.strip() for item in users.split(",") if item.strip()]
    return sorted(path.name for path in users_root.glob("user_*") if path.is_dir())


def _hash_unit(seed: int, *parts: str) -> float:
    joined = "::".join([str(seed), *parts])
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _word_truncate(text: str, max_words: int) -> str:
    words = str(text or "").split()
    if len(words) <= max_words:
        return str(text or "")
    return " ".join(words[:max_words]).rstrip(" ,;:") + "."


def _entity_source(entity: dict[str, Any]) -> str:
    return str(entity.get("source") or "").strip().lower()


def _caption_entities_present(entities: list[dict[str, Any]], caption: str) -> list[dict[str, Any]]:
    lower_caption = caption.lower()
    kept = []
    for entity in entities:
        if _entity_source(entity) != "caption":
            kept.append(copy.deepcopy(entity))
            continue
        surface = str(entity.get("surface") or "").strip()
        if surface and surface.lower() in lower_caption:
            kept.append(copy.deepcopy(entity))
    return kept


def _drop_visible_text(photo: dict[str, Any]) -> None:
    photo["visible_text"] = []
    photo["text_entities"] = [
        copy.deepcopy(entity)
        for entity in (photo.get("text_entities") or [])
        if _entity_source(entity) != "visible_text"
    ]


def _owner_face_id(album: dict[str, Any]) -> str:
    faces = []
    for row in album.get("faces") or []:
        face_id = str(row.get("face_id") or "")
        if not face_id:
            continue
        try:
            count = int(row.get("n_appearances") or 0)
        except (TypeError, ValueError):
            count = 0
        faces.append((count, face_id))
    if faces:
        faces.sort(key=lambda item: (-item[0], item[1]))
        return faces[0][1]
    counts = Counter()
    for photo in album.get("photos") or []:
        for face_id in photo.get("visible_face_ids") or []:
            counts[str(face_id)] += 1
    return counts.most_common(1)[0][0] if counts else ""


def _drop_faces(photo: dict[str, Any], *, user_id: str, setting: str, seed: int, owner_face: str, p_non_owner: float, p_owner: float) -> int:
    kept = []
    dropped = 0
    for face_id in [str(fid) for fid in (photo.get("visible_face_ids") or [])]:
        probability = p_owner if face_id == owner_face else p_non_owner
        value = _hash_unit(seed, setting, user_id, str(photo.get("photo_id") or ""), face_id, "face")
        if value < probability:
            dropped += 1
        else:
            kept.append(face_id)
    photo["visible_face_ids"] = kept
    return dropped


def _blank_location(photo: dict[str, Any]) -> bool:
    metadata = dict(photo.get("metadata") or {})
    had_location = bool(metadata.get("gps_city") or metadata.get("gps_location"))
    metadata["gps_city"] = ""
    metadata["gps_location"] = ""
    photo["metadata"] = metadata
    return had_location


def _rebuild_faces(album: dict[str, Any]) -> None:
    sightings: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for photo in album.get("photos") or []:
        month = str(photo.get("year_month") or "")
        for face_id in photo.get("visible_face_ids") or []:
            sightings[str(face_id)].append((month, str(photo.get("photo_id") or "")))
    rows = []
    for face_id, seen in sorted(sightings.items()):
        months = sorted(month for month, _ in seen if month)
        rows.append(
            {
                "face_id": face_id,
                "n_appearances": len(seen),
                "first_seen": months[0] if months else "",
                "last_seen": months[-1] if months else "",
            }
        )
    album["faces"] = rows


def _stats(album: dict[str, Any]) -> dict[str, int]:
    photos = list(album.get("photos") or [])
    return {
        "n_photos": len(photos),
        "caption_chars": sum(len(str(p.get("caption") or "")) for p in photos),
        "visible_text_items": sum(len(p.get("visible_text") or []) for p in photos),
        "text_entities": sum(len(p.get("text_entities") or []) for p in photos),
        "face_appearances": sum(len(p.get("visible_face_ids") or []) for p in photos),
        "photos_with_location": sum(
            1
            for p in photos
            if (p.get("metadata") or {}).get("gps_city") or (p.get("metadata") or {}).get("gps_location")
        ),
        "photos_with_faces": sum(1 for p in photos if p.get("visible_face_ids")),
        "photos_with_ocr": sum(1 for p in photos if p.get("visible_text")),
    }


def _apply_setting(album: dict[str, Any], *, user_id: str, setting: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    stressed = copy.deepcopy(album)
    owner_face = _owner_face_id(stressed)
    counters: Counter[str] = Counter()
    before = _stats(stressed)

    for photo in stressed.get("photos") or []:
        photo_id = str(photo.get("photo_id") or "")
        if setting in {"caption_sparse", "combined_mild"}:
            max_words = 28 if setting == "caption_sparse" else 36
            old_caption = str(photo.get("caption") or "")
            new_caption = _word_truncate(old_caption, max_words)
            if new_caption != old_caption:
                counters["captions_truncated"] += 1
            photo["caption"] = new_caption
            photo["text_entities"] = _caption_entities_present(photo.get("text_entities") or [], new_caption)

        if setting in {"ocr_dropout_35", "combined_mild"}:
            probability = 0.35 if setting == "ocr_dropout_35" else 0.25
            if photo.get("visible_text") and _hash_unit(seed, setting, user_id, photo_id, "ocr") < probability:
                counters["ocr_photos_dropped"] += 1
                counters["ocr_items_dropped"] += len(photo.get("visible_text") or [])
                _drop_visible_text(photo)

        if setting in {"location_sparse_40", "combined_mild"}:
            probability = 0.40 if setting == "location_sparse_40" else 0.30
            if _hash_unit(seed, setting, user_id, photo_id, "location") < probability:
                if _blank_location(photo):
                    counters["location_photos_blank"] += 1

        if setting in {"face_miss_15", "combined_mild"}:
            if setting == "face_miss_15":
                dropped = _drop_faces(
                    photo,
                    user_id=user_id,
                    setting=setting,
                    seed=seed,
                    owner_face=owner_face,
                    p_non_owner=0.15,
                    p_owner=0.03,
                )
            else:
                dropped = _drop_faces(
                    photo,
                    user_id=user_id,
                    setting=setting,
                    seed=seed,
                    owner_face=owner_face,
                    p_non_owner=0.10,
                    p_owner=0.02,
                )
            counters["face_appearances_dropped"] += dropped

    _rebuild_faces(stressed)
    after = _stats(stressed)
    return stressed, {
        "user_id": user_id,
        "setting": setting,
        "owner_face_id_estimate": owner_face,
        "derivation": {
            "schema_version": "realism_stress.v1",
            "source_user_id": user_id,
            "setting": setting,
            "seed": seed,
            "description": SETTING_DESCRIPTIONS[setting],
            "preserves_eval_gt": True,
            "perturbs_public_fields_only": True,
            "agent_visible_stress_metadata": False,
        },
        "before": before,
        "after": after,
        "changes": dict(counters),
    }


def build_stress_roots(users_root: Path, output_root: Path, users: list[str], settings: list[str], seed: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for setting in settings:
        if setting not in SETTING_DESCRIPTIONS:
            raise ValueError(f"Unknown stress setting: {setting}")
        setting_root = output_root / setting
        setting_root.mkdir(parents=True, exist_ok=True)
        setting_rows = []
        for user_id in users:
            src_dir = users_root / user_id
            album_path = src_dir / f"{user_id}_agent_album.json"
            gt_path = src_dir / f"{user_id}_eval_gt.json"
            album = load_json(album_path)
            if not isinstance(album, dict):
                raise ValueError(f"Expected album object at {album_path}")
            stressed, row = _apply_setting(album, user_id=user_id, setting=setting, seed=seed)
            dst_dir = setting_root / user_id
            dst_dir.mkdir(parents=True, exist_ok=True)
            save_json(stressed, dst_dir / f"{user_id}_agent_album.json")
            if gt_path.exists():
                shutil.copyfile(gt_path, dst_dir / f"{user_id}_eval_gt.json")
            setting_rows.append(row)
            all_rows.append(row)
        save_json(
            {
                "schema_version": "realism_stress_setting_manifest.v1",
                "setting": setting,
                "description": SETTING_DESCRIPTIONS[setting],
                "seed": seed,
                "source_users_root": str(users_root),
                "output_users_root": str(setting_root),
                "users": users,
                "aggregate": _aggregate_rows(setting_rows),
                "rows": setting_rows,
            },
            setting_root / "realism_stress_manifest.json",
        )
    save_json(
        {
            "schema_version": "realism_stress_manifest.v1",
            "seed": seed,
            "source_users_root": str(users_root),
            "output_root": str(output_root),
            "settings": [
                {"setting": setting, "description": SETTING_DESCRIPTIONS[setting], "users_root": str(output_root / setting)}
                for setting in settings
            ],
            "users": users,
            "aggregate_by_setting": {
                setting: _aggregate_rows([row for row in all_rows if row["setting"] == setting])
                for setting in settings
            },
        },
        output_root / "realism_stress_manifest.json",
    )


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    before_totals: Counter[str] = Counter()
    after_totals: Counter[str] = Counter()
    change_totals: Counter[str] = Counter()
    for row in rows:
        before_totals.update(row.get("before") or {})
        after_totals.update(row.get("after") or {})
        change_totals.update(row.get("changes") or {})
    result = {
        "n_users": len(rows),
        "before": dict(before_totals),
        "after": dict(after_totals),
        "changes": dict(change_totals),
        "retention": {},
    }
    for key, before_value in before_totals.items():
        after_value = after_totals.get(key, 0)
        result["retention"][key] = round(after_value / before_value, 4) if before_value else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create realism-stress PAL-Bench users roots.")
    parser.add_argument("--users-root", default="data/full/users")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--users", default=None)
    parser.add_argument(
        "--settings",
        default="caption_sparse,ocr_dropout_35,location_sparse_40,face_miss_15,combined_mild",
        help="Comma-separated stress settings.",
    )
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    users_root = Path(args.users_root)
    output_root = Path(args.output_root)
    users = _split_users(users_root, args.users)
    settings = [item.strip() for item in args.settings.split(",") if item.strip()]
    if not users:
        raise SystemExit("No users selected.")
    if not settings:
        raise SystemExit("No stress settings selected.")
    build_stress_roots(users_root, output_root, users, settings, int(args.seed))
    print(f"[realism-stress] users={len(users)} settings={len(settings)} -> {output_root}")


if __name__ == "__main__":
    main()

"""Small Step-6 compatibility layer for public benchmark export.

The original image rendering pipeline is intentionally not part of the default
runtime path for the paper results. Benchmark export still needs a handful of
deterministic metadata and manifest helpers, which are provided here.
"""

from __future__ import annotations

import hashlib
import random
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

DEVICE_POOL = [
    "iPhone 13",
    "iPhone 14",
    "iPhone 15",
    "Samsung Galaxy S23",
    "Samsung Galaxy S24",
    "Google Pixel 7",
    "Google Pixel 8",
]

_RENDER_DIMS_BY_ASPECT_RATIO = {
    "1:1": (1024, 1024),
    "2:3": (896, 1344),
    "3:2": (1344, 896),
    "3:4": (960, 1280),
    "4:3": (1152, 864),
    "4:5": (1024, 1280),
    "5:4": (1280, 1024),
    "9:16": (768, 1376),
    "16:9": (1376, 768),
    "21:9": (1536, 656),
}


class ImageBackend(ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    @abstractmethod
    def generate(
        self,
        prompt: str,
        ref_image_paths: list[Path] | None = None,
        width: int = 1024,
        height: int = 1024,
        aspect_ratio: str | None = None,
        seed: int = -1,
    ) -> bytes:
        raise NotImplementedError


class PhotoSignalGenerator:
    def __init__(
        self,
        llm: Any,
        backend: ImageBackend,
        max_workers: int = 8,
        rng: random.Random | None = None,
        target_album_photo_min: int = 500,
        target_album_photo_max: int = 1000,
    ) -> None:
        self._llm = llm
        self._backend = backend
        self._max_workers = max_workers
        self._rng = rng or random.Random(0)
        self._target_album_photo_min = target_album_photo_min
        self._target_album_photo_max = target_album_photo_max

    def _build_metadata_from_fields(
        self,
        year_month: str,
        location: str,
        profile: dict[str, Any],
        idx: int,
        device: str | None = None,
        stable_key: str | None = None,
        *,
        location_anchor: dict[str, Any] | None = None,
        expected_metadata: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        key = stable_key or f"{profile.get('user_id', 'user')}:{year_month}:{location}:{idx}"
        timestamp = _stable_timestamp(year_month or "2024-01", key, idx)
        if location_anchor:
            city = str(
                location_anchor.get("city")
                or profile.get("raw_attributes", {}).get("city", "")
            ).strip()
            parts = [
                str(location_anchor.get(field) or "").strip()
                for field in ("label", "neighborhood", "address_text")
            ]
            parts = [part for part in parts if part]
            if city:
                parts.append(city)
            return {
                "timestamp": timestamp,
                "gps_city": city,
                "gps_location": ", ".join(parts),
                "device": device or DEVICE_POOL[0],
            }
        city, loc = _parse_location(
            location,
            default_city=str(profile.get("raw_attributes", {}).get("city", "") or ""),
        )
        return {
            "timestamp": timestamp,
            "gps_city": city,
            "gps_location": loc,
            "device": device or DEVICE_POOL[0],
        }


def sample_album_photo_target(
    window_months: int,
    *,
    key: str,
    min_total: int = 500,
    max_total: int = 1000,
) -> int:
    lower = max(int(min_total), 1)
    upper = max(int(max_total), lower)
    span = upper - lower
    months = max(int(window_months or 1), 1)
    baseline = lower + span * min(months, 24) / 24.0
    seed = int(hashlib.sha256(f"{key}#album_total".encode("utf-8")).hexdigest()[:16], 16)
    jitter = random.Random(seed).gauss(0.0, max(span * 0.12, 1.0))
    return max(lower, min(upper, int(round(baseline + jitter))))


def _build_user_device_plan(profile: dict[str, Any], timeline: dict[str, Any] | None = None) -> dict[str, Any]:
    key = str(profile.get("user_id") or profile.get("name") or "user")
    return {"family": "generic", "primary_device": DEVICE_POOL[_stable_int_from_key(key, len(DEVICE_POOL))], "upgrades": []}


def _device_for_year_month(device_plan: dict[str, Any], year_month: str) -> str:
    selected = str(device_plan.get("primary_device") or DEVICE_POOL[0])
    ym_ord = _year_month_to_ordinal(year_month)
    for upgrade in device_plan.get("upgrades", []) or []:
        if not isinstance(upgrade, dict):
            continue
        if _year_month_to_ordinal(str(upgrade.get("from_month") or "")) <= ym_ord:
            selected = str(upgrade.get("device") or selected)
    return selected


def _default_dimensions_for_aspect_ratio(aspect_ratio: str) -> tuple[int, int]:
    return _RENDER_DIMS_BY_ASPECT_RATIO.get(aspect_ratio, _RENDER_DIMS_BY_ASPECT_RATIO["4:3"])


def _build_recurring_people_summary(
    profile: dict[str, Any],
    graph: dict[str, Any],
    timeline: dict[str, Any],
) -> tuple[str, list[str]]:
    counts = _event_participant_counts(timeline)
    nodes = [
        str(node.get("person_id") or "").strip()
        for node in graph.get("nodes", [])
        if str(node.get("person_id") or "").strip()
    ]
    recurring = sorted(nodes, key=lambda pid: (-counts.get(pid, 0), pid))[:6]
    owner_name = str(profile.get("name") or "album owner")
    lines = [f"- owner = {owner_name}"]
    for pid in recurring:
        lines.append(f"- {pid}")
    return "\n".join(lines), recurring


def _default_ambient_people_plan(item: dict[str, Any], recurring_people: list[str]) -> dict[str, Any]:
    category = str(item.get("category") or "").strip()
    description = str(item.get("description") or "").lower()
    if category == "screenshot":
        return {
            "required_people": [],
            "optional_people": [],
            "signal_goal": "text_artifact",
            "face_visibility": "none",
            "framing_hint": "artifact",
        }
    if category == "selfie" or "selfie" in description:
        return {
            "required_people": ["owner"],
            "optional_people": recurring_people[:1],
            "signal_goal": "face_cluster",
            "face_visibility": "clear",
            "framing_hint": "single_portrait",
        }
    return {
        "required_people": ["owner"],
        "optional_people": recurring_people[:2],
        "signal_goal": "owner_lifestyle",
        "face_visibility": "medium",
        "framing_hint": "environment",
    }


def _normalize_people_plan(
    raw_plan: dict[str, Any] | None,
    *,
    allowed_people_ids: set[str],
    default_required_people: list[str],
    default_optional_people: list[str],
    default_signal_goal: str,
    default_face_visibility: str,
    default_framing_hint: str,
) -> dict[str, Any]:
    plan = raw_plan or {}
    required = _normalize_people_ids(
        plan.get("required_people", default_required_people),
        allowed_people_ids,
    )
    optional = [
        pid
        for pid in _normalize_people_ids(plan.get("optional_people", default_optional_people), allowed_people_ids)
        if pid not in set(required)
    ]
    return {
        "required_people": required,
        "optional_people": optional,
        "signal_goal": str(plan.get("signal_goal") or default_signal_goal),
        "face_visibility": str(plan.get("face_visibility") or default_face_visibility),
        "framing_hint": str(plan.get("framing_hint") or default_framing_hint),
    }


def _resolve_people_to_face_ids(people: list[str], pid_to_face: dict[str, str]) -> list[str]:
    out: list[str] = []
    for person_id in people:
        if person_id == "owner":
            out.append("owner")
        elif person_id in pid_to_face:
            out.append(pid_to_face[person_id])
    return _dedupe(out)


def _normalize_people_ids(values: object, allowed_people_ids: set[str]) -> list[str]:
    if isinstance(values, str):
        raw_values: list[object] = [values]
    elif isinstance(values, list):
        raw_values = values
    else:
        raw_values = []
    out: list[str] = []
    for value in raw_values:
        person_id = str(value).strip()
        if person_id == "owner" or person_id in allowed_people_ids:
            out.append(person_id)
    return _dedupe(out)


def _event_participant_counts(timeline: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in timeline.get("events", []) or []:
        participants = event.get("participants") or []
        if isinstance(participants, str):
            participants = [participants]
        for value in participants if isinstance(participants, list) else []:
            pid = str(value).strip()
            if pid:
                counts[pid] = counts.get(pid, 0) + 1
    return counts


def _parse_location(location: str, *, default_city: str = "") -> tuple[str, str]:
    text = re.sub(r"\s+", " ", str(location or "")).strip()
    if not text:
        return default_city, default_city
    parts = [part.strip() for part in text.split(",") if part.strip()]
    city = parts[-1] if len(parts) >= 2 else default_city
    return city, text


def _stable_timestamp(year_month: str, key: str, offset_index: int = 0) -> str:
    try:
        year = int(str(year_month)[:4])
        month = int(str(year_month)[5:7])
    except (TypeError, ValueError):
        year, month = 2024, 1
    seed = _stable_int_from_key(f"{key}:timestamp", 28 * 24 * 60)
    day = seed // (24 * 60) + 1
    minute_of_day = seed % (24 * 60)
    dt = datetime(year, month, min(day, 28), minute_of_day // 60, minute_of_day % 60)
    return (dt + timedelta(minutes=offset_index)).isoformat()


def _year_month_to_ordinal(year_month: str) -> int:
    try:
        return int(str(year_month)[:4]) * 12 + int(str(year_month)[5:7])
    except (TypeError, ValueError):
        return 0


def _stable_int_from_key(key: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out

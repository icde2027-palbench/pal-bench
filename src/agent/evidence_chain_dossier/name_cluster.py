"""Name clustering for Evidence-Chain Dossier."""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher

from .schemas import EvidenceInventory, NameCluster, TextEvidenceUnit
from .text import dedupe, first_name, is_full_name, last_name, normalize_name


def build_name_clusters(inventory: EvidenceInventory) -> list[NameCluster]:
    """Cluster observed person surfaces without deleting surname relatives."""
    surface_units: dict[str, list[TextEvidenceUnit]] = defaultdict(list)
    surface_counts: Counter[str] = Counter()
    relation_by_surface: dict[str, list[str]] = defaultdict(list)
    activity_by_surface: dict[str, list[str]] = defaultdict(list)
    venue_by_surface: dict[str, list[str]] = defaultdict(list)

    for unit in inventory.text_units:
        for surface in unit.person_surfaces:
            norm = normalize_name(surface)
            if not norm:
                continue
            surface_units[norm].append(unit)
            surface_counts[norm] += 1
            relation_by_surface[norm].extend(unit.relation_terms)
            activity_by_surface[norm].extend(unit.activity_terms)
            venue_by_surface[norm].extend(unit.venue_terms)

    full_by_first: dict[str, list[str]] = defaultdict(list)
    short_by_first: dict[str, list[str]] = defaultdict(list)
    for surface in surface_counts:
        fn = first_name(surface)
        if not fn:
            continue
        if is_full_name(surface):
            full_by_first[fn.lower()].append(surface)
        else:
            short_by_first[fn.lower()].append(surface)

    groups: list[list[str]] = []
    assigned: set[str] = set()
    fuzzy_short_to_full = _fuzzy_short_map(short_by_first, full_by_first)

    for fn_key, fulls in sorted(full_by_first.items()):
        fulls = sorted(fulls, key=lambda s: (-surface_counts[s], -len(s), s))
        # Multiple last names with the same first are kept separate unless they
        # are obvious initial/abbreviation variants.
        for full in fulls:
            group = [full]
            assigned.add(full)
            full_last = last_name(full)
            for other in fulls:
                if other == full or other in assigned:
                    continue
                other_last = last_name(other)
                if _compatible_last(full_last, other_last):
                    group.append(other)
                    assigned.add(other)
            shorts = short_by_first.get(fn_key, [])
            if len(fulls) == 1:
                group.extend(shorts)
                assigned.update(shorts)
            for short, mapped_full in fuzzy_short_to_full.items():
                if mapped_full == full and short not in assigned:
                    group.append(short)
                    assigned.add(short)
            groups.append(dedupe(group))

    for surface in sorted(surface_counts):
        if surface not in assigned:
            groups.append([surface])
            assigned.add(surface)

    clusters: list[NameCluster] = []
    for idx, surfaces in enumerate(groups, start=1):
        primary = max(surfaces, key=lambda s: (is_full_name(s), surface_counts[s], len(s)))
        units = []
        for surface in surfaces:
            units.extend(surface_units.get(surface, []))
        mention_photo_ids = dedupe(u.photo_id for u in units)
        text_unit_ids = dedupe(u.unit_id for u in units)
        last_candidates: Counter[str] = Counter()
        for surface in surfaces:
            ln = last_name(surface)
            if ln:
                last_candidates[ln] += surface_counts[surface]
        relation_terms = []
        activity_terms = []
        venue_terms = []
        for surface in surfaces:
            relation_terms.extend(relation_by_surface.get(surface, []))
            activity_terms.extend(activity_by_surface.get(surface, []))
            venue_terms.extend(venue_by_surface.get(surface, []))
        flags = []
        if not is_full_name(primary):
            flags.append("first_name_only")
        if len([s for s in surfaces if is_full_name(s)]) > 1:
            flags.append("multi_surface")
        clusters.append(
            NameCluster(
                cluster_id=f"name_{idx:04d}",
                surfaces=dedupe(surfaces),
                primary_surface=primary,
                first_name=first_name(primary),
                last_name_candidates=dict(last_candidates),
                mention_photo_ids=mention_photo_ids,
                text_unit_ids=text_unit_ids,
                relation_terms=dedupe(relation_terms),
                activity_terms=dedupe(activity_terms),
                venue_terms=dedupe(venue_terms),
                quality_flags=flags,
            )
        )
    clusters.sort(key=lambda c: (-len(c.mention_photo_ids), c.primary_surface))
    return clusters


def _compatible_last(a: str, b: str) -> bool:
    if not a or not b:
        return True
    if a == b:
        return True
    if len(a) == 1 and b.startswith(a):
        return True
    if len(b) == 1 and a.startswith(b):
        return True
    return False


def _fuzzy_short_map(
    short_by_first: dict[str, list[str]],
    full_by_first: dict[str, list[str]],
) -> dict[str, str]:
    all_fulls = [full for fulls in full_by_first.values() for full in fulls]
    out: dict[str, str] = {}
    for shorts in short_by_first.values():
        for short in shorts:
            s_first = first_name(short)
            if len(s_first) < 4:
                continue
            matches = []
            for full in all_fulls:
                f_first = first_name(full)
                if len(f_first) < 4:
                    continue
                ratio = SequenceMatcher(None, s_first.lower(), f_first.lower()).ratio()
                if ratio >= 0.82:
                    matches.append((ratio, full))
            matches.sort(reverse=True)
            if len(matches) == 1 or (matches and (len(matches) < 2 or matches[0][0] - matches[1][0] >= 0.08)):
                out[short] = matches[0][1]
    return out

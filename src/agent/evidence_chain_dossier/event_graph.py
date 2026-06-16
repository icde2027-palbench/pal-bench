"""Event graph utilities for bridge-aware identity assignment.

The graph is a public-evidence abstraction over an album: photos are grouped
into coarse event nodes, and name-face evidence edges are derived from direct
same-event anchors or cross-photo event bridges.  The module is deterministic
and intentionally LLM-free; later BEA passes can use these edges as compact
LLM input or as a pre-filter for global assignment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .event_bridge import photo_event_features, score_event_bridge_from_features, social_channels_from_terms
from .schemas import EvidenceInventory
from .text import dedupe, is_full_name, norm_key, normalize_name


@dataclass(slots=True)
class EventNode:
    event_id: str
    event_key: str
    year_month: str
    gps_city: str
    location_key: str
    primary_channel: str
    photo_ids: list[str] = field(default_factory=list)
    text_photo_ids: list[str] = field(default_factory=list)
    face_photo_ids: list[str] = field(default_factory=list)
    face_ids: list[str] = field(default_factory=list)
    name_surfaces: list[str] = field(default_factory=list)
    relation_terms: list[str] = field(default_factory=list)
    activity_terms: list[str] = field(default_factory=list)
    venue_terms: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    owner_present: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EventIdentityEdge:
    face_id: str
    name_surface: str
    score: float
    bridge_type: str
    event_ids: list[str]
    evidence_photo_ids: list[str]
    text_photo_ids: list[str]
    face_photo_ids: list[str]
    shared_channels: list[str] = field(default_factory=list)
    shared_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["score"] = round(float(self.score), 4)
        return data


@dataclass(slots=True)
class EventGraph:
    user_id: str
    owner_face_id: str
    nodes: list[EventNode]
    edges: list[EventIdentityEdge]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ecd_event_graph.v1",
            "user_id": self.user_id,
            "owner_face_id": self.owner_face_id,
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


def build_event_graph(
    inventory: EvidenceInventory,
    *,
    max_cross_edges_per_name: int = 12,
    min_cross_bridge_score: float = 0.34,
) -> EventGraph:
    """Build event nodes and deterministic face-name bridge edges."""
    nodes = _build_event_nodes(inventory)
    edges = _build_event_identity_edges(
        inventory,
        nodes,
        max_cross_edges_per_name=max_cross_edges_per_name,
        min_cross_bridge_score=min_cross_bridge_score,
    )
    return EventGraph(
        user_id=str(inventory.album.get("user_id") or ""),
        owner_face_id=inventory.owner_face_id,
        nodes=nodes,
        edges=edges,
    )


def _build_event_nodes(inventory: EvidenceInventory) -> list[EventNode]:
    buckets: dict[str, EventNode] = {}
    for photo in inventory.photos:
        photo_id = str(photo.get("photo_id") or "")
        if not photo_id:
            continue
        metadata = photo.get("metadata") or {}
        year_month = str(photo.get("year_month") or "")
        city = str(metadata.get("gps_city") or "")
        location = _location_key(str(metadata.get("gps_location") or ""))
        features = photo_event_features(photo)
        channels = [str(c) for c in (features.get("channels") or [])]
        primary_channel = channels[0] if channels else "general"
        event_key = "|".join([year_month, norm_key(city), location, primary_channel])
        node = buckets.get(event_key)
        if node is None:
            node = EventNode(
                event_id=f"event_{len(buckets):04d}",
                event_key=event_key,
                year_month=year_month,
                gps_city=city,
                location_key=location,
                primary_channel=primary_channel,
            )
            buckets[event_key] = node
        node.photo_ids.append(photo_id)
        face_ids = [str(fid) for fid in (photo.get("visible_face_ids") or [])]
        if face_ids:
            node.face_photo_ids.append(photo_id)
            node.face_ids.extend(face_ids)
        if _photo_has_text_anchor(photo):
            node.text_photo_ids.append(photo_id)
        names, relation_terms, activity_terms, venue_terms = _photo_text_terms(inventory, photo_id, photo)
        node.name_surfaces.extend(names)
        node.relation_terms.extend(relation_terms)
        node.activity_terms.extend(activity_terms)
        node.venue_terms.extend(venue_terms)
        node.channels.extend(channels)
        node.terms.extend(str(t) for t in (features.get("terms") or []))
        node.owner_present = node.owner_present or inventory.owner_face_id in face_ids

    nodes = []
    for node in buckets.values():
        node.photo_ids = dedupe(node.photo_ids)
        node.text_photo_ids = dedupe(node.text_photo_ids)
        node.face_photo_ids = dedupe(node.face_photo_ids)
        node.face_ids = dedupe(node.face_ids)
        node.name_surfaces = dedupe(normalize_name(name) for name in node.name_surfaces if normalize_name(name))
        node.relation_terms = dedupe(node.relation_terms)
        node.activity_terms = dedupe(node.activity_terms)
        node.venue_terms = dedupe(node.venue_terms)
        node.channels = dedupe(node.channels)
        node.terms = dedupe(node.terms)
        nodes.append(node)
    nodes.sort(key=lambda n: (n.year_month, n.gps_city, n.location_key, n.primary_channel, n.event_id))
    return nodes


def _build_event_identity_edges(
    inventory: EvidenceInventory,
    nodes: list[EventNode],
    *,
    max_cross_edges_per_name: int,
    min_cross_bridge_score: float,
) -> list[EventIdentityEdge]:
    best: dict[tuple[str, str], EventIdentityEdge] = {}

    for node in nodes:
        if not node.face_ids or not node.name_surfaces:
            continue
        for face_id in node.face_ids:
            if face_id == inventory.owner_face_id:
                continue
            for name in node.name_surfaces:
                score = _direct_edge_score(node, name)
                edge = EventIdentityEdge(
                    face_id=face_id,
                    name_surface=name,
                    score=score,
                    bridge_type="same_event",
                    event_ids=[node.event_id],
                    evidence_photo_ids=dedupe(node.text_photo_ids + node.face_photo_ids)[:8],
                    text_photo_ids=node.text_photo_ids[:6],
                    face_photo_ids=node.face_photo_ids[:6],
                    shared_channels=node.channels[:8],
                    shared_terms=node.terms[:8],
                )
                _keep_best_edge(best, edge)

    text_nodes = [node for node in nodes if node.name_surfaces and node.text_photo_ids]
    face_nodes = [node for node in nodes if node.face_ids and node.face_photo_ids]
    for text_node in text_nodes:
        scored_face_nodes = []
        for face_node in face_nodes:
            if text_node.event_id == face_node.event_id:
                continue
            bridge = _score_node_bridge(inventory, text_node, face_node)
            bridge_score = float(bridge.get("score") or 0.0)
            if bridge_score < min_cross_bridge_score:
                continue
            scored_face_nodes.append((bridge_score, bridge, face_node))
        scored_face_nodes.sort(key=lambda row: (-row[0], row[2].event_id))
        for bridge_score, bridge, face_node in scored_face_nodes[:max_cross_edges_per_name]:
            for face_id in face_node.face_ids:
                if face_id == inventory.owner_face_id:
                    continue
                for name in text_node.name_surfaces:
                    edge = EventIdentityEdge(
                        face_id=face_id,
                        name_surface=name,
                        score=_cross_edge_score(text_node, face_node, name, bridge_score),
                        bridge_type=str(bridge.get("bridge_type") or "cross_event"),
                        event_ids=[text_node.event_id, face_node.event_id],
                        evidence_photo_ids=dedupe(text_node.text_photo_ids + face_node.face_photo_ids)[:8],
                        text_photo_ids=text_node.text_photo_ids[:6],
                        face_photo_ids=face_node.face_photo_ids[:6],
                        shared_channels=[str(c) for c in (bridge.get("shared_channels") or [])[:8]],
                        shared_terms=[str(t) for t in (bridge.get("shared_keywords") or [])[:8]],
                    )
                    _keep_best_edge(best, edge)

    edges = sorted(best.values(), key=lambda e: (-e.score, e.face_id, e.name_surface))
    return edges


def _score_node_bridge(inventory: EvidenceInventory, text_node: EventNode, face_node: EventNode) -> dict[str, Any]:
    if text_node.year_month and text_node.year_month == face_node.year_month:
        if text_node.location_key and text_node.location_key == face_node.location_key:
            base = 0.58
            bridge_type = "same_location_month"
        elif norm_key(text_node.gps_city) and norm_key(text_node.gps_city) == norm_key(face_node.gps_city):
            base = 0.42
            bridge_type = "same_city_month"
        else:
            base = 0.0
            bridge_type = "cross_event"
    else:
        base = 0.0
        bridge_type = "cross_event"

    shared_channels = sorted(set(text_node.channels) & set(face_node.channels))
    shared_terms = sorted(set(text_node.terms) & set(face_node.terms))
    if shared_channels:
        base += min(0.22, 0.10 * len(shared_channels))
    if shared_terms:
        base += min(0.12, 0.04 * len(shared_terms))
    if text_node.owner_present or face_node.owner_present:
        base += 0.03

    # Fall back to photo-level event bridge for semantically similar nodes that
    # do not share a clean location/month key.
    if text_node.text_photo_ids and face_node.face_photo_ids:
        text_photo = inventory.photo_lookup.get(text_node.text_photo_ids[0]) or {}
        face_photo = inventory.photo_lookup.get(face_node.face_photo_ids[0]) or {}
        photo_bridge = score_event_bridge_from_features(
            photo_event_features(text_photo),
            photo_event_features(face_photo),
            text_photo_has_faces=bool(text_photo.get("visible_face_ids")),
            face_photo_face_count=len(face_photo.get("visible_face_ids") or []),
            text_photo_person_entities=_person_entity_count(text_photo),
        )
        if float(photo_bridge.get("score") or 0.0) > base:
            base = float(photo_bridge.get("score") or 0.0)
            bridge_type = "semantic_event"
            shared_channels = [str(c) for c in (photo_bridge.get("shared_channels") or [])]
            shared_terms = [str(t) for t in (photo_bridge.get("shared_keywords") or [])]

    return {
        "score": round(max(0.0, min(1.0, base)), 4),
        "bridge_type": bridge_type,
        "shared_channels": shared_channels,
        "shared_keywords": shared_terms,
    }


def _direct_edge_score(node: EventNode, name: str) -> float:
    score = 0.58
    if node.owner_present:
        score += 0.06
    if node.relation_terms:
        score += 0.06
    if node.primary_channel != "general":
        score += 0.05
    if is_full_name(name):
        score += 0.08
    if len(node.face_ids) <= 3:
        score += 0.04
    else:
        score -= min(0.10, 0.01 * (len(node.face_ids) - 3))
    return max(0.0, min(1.0, score))


def _cross_edge_score(text_node: EventNode, face_node: EventNode, name: str, bridge_score: float) -> float:
    score = 0.30 + bridge_score * 0.55
    if text_node.owner_present or face_node.owner_present:
        score += 0.04
    if set(text_node.channels) & set(face_node.channels):
        score += 0.04
    if is_full_name(name):
        score += 0.04
    if len(face_node.face_ids) > 4:
        score -= min(0.08, 0.01 * (len(face_node.face_ids) - 4))
    return max(0.0, min(1.0, score))


def _keep_best_edge(best: dict[tuple[str, str], EventIdentityEdge], edge: EventIdentityEdge) -> None:
    key = (edge.face_id, normalize_name(edge.name_surface).lower())
    current = best.get(key)
    if current is None or edge.score > current.score:
        best[key] = edge


def _photo_text_terms(
    inventory: EvidenceInventory,
    photo_id: str,
    photo: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[str]]:
    names = []
    relation_terms = []
    activity_terms = []
    venue_terms = []
    for unit in inventory.text_units:
        if unit.photo_id == photo_id:
            names.extend(unit.person_surfaces)
            relation_terms.extend(unit.relation_terms)
            activity_terms.extend(unit.activity_terms)
            venue_terms.extend(unit.venue_terms)
    for unit in inventory.face_units:
        if unit.photo_id == photo_id:
            relation_terms.extend(unit.relation_terms)
            activity_terms.extend(unit.activity_terms)
            venue_terms.extend(unit.venue_terms)
    channels = social_channels_from_terms(relation_terms + activity_terms + venue_terms)
    if channels:
        activity_terms.extend(sorted(channels))
    return dedupe(names), dedupe(relation_terms), dedupe(activity_terms), dedupe(venue_terms)


def _photo_has_text_anchor(photo: dict[str, Any]) -> bool:
    return bool(photo.get("visible_text") or photo.get("text_entities") or photo.get("caption"))


def _location_key(value: str) -> str:
    normalized = norm_key(value)
    if not normalized:
        return ""
    parts = [part.strip() for part in re_split_location(normalized) if part.strip()]
    return parts[-1] if parts else normalized


def re_split_location(value: str) -> list[str]:
    return value.replace("—", "-").replace(" - ", "-").split("-")


def _person_entity_count(photo: dict[str, Any]) -> int:
    return sum(
        1
        for ent in (photo.get("text_entities") or [])
        if str(ent.get("entity_type") or "").lower() == "person"
    )

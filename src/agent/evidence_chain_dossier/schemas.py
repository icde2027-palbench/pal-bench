"""Dataclasses shared by Evidence-Chain Dossier phases."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TextEvidenceUnit:
    unit_id: str
    photo_id: str
    year_month: str
    timestamp: str
    gps_city: str
    gps_location: str
    caption: str
    visible_text: list[str]
    person_surfaces: list[str]
    relation_terms: list[str]
    activity_terms: list[str]
    venue_terms: list[str]
    organization_terms: list[str]
    owner_reference_score: float
    has_face: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FaceEvidenceUnit:
    face_id: str
    photo_id: str
    year_month: str
    timestamp: str
    gps_city: str
    gps_location: str
    caption: str
    visible_text: list[str]
    co_faces: list[str]
    owner_present: bool
    relation_terms: list[str]
    activity_terms: list[str]
    venue_terms: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NameCluster:
    cluster_id: str
    surfaces: list[str]
    primary_surface: str
    first_name: str
    last_name_candidates: dict[str, int]
    mention_photo_ids: list[str]
    text_unit_ids: list[str]
    relation_terms: list[str]
    activity_terms: list[str]
    venue_terms: list[str]
    quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidencePacket:
    text_photo_ids: list[str] = field(default_factory=list)
    face_photo_ids: list[str] = field(default_factory=list)
    same_photo_ids: list[str] = field(default_factory=list)
    bridge_photo_pairs: list[dict[str, Any]] = field(default_factory=list)
    relation_clues: list[str] = field(default_factory=list)
    activity_clues: list[str] = field(default_factory=list)
    venue_clues: list[str] = field(default_factory=list)
    coface_clues: list[str] = field(default_factory=list)
    negative_clues: list[str] = field(default_factory=list)
    narrative_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IdentityHypothesis:
    face_id: str
    name_cluster_id: str
    observed_surface: str
    canonical_name_candidate: str
    score: float
    margin: float
    signal_breakdown: dict[str, float]
    evidence_packet: EvidencePacket
    status: str = "candidate"
    llm_decision: str = ""
    llm_confidence: float | None = None
    llm_canonical_name: str = ""
    llm_relation_to_owner: str = ""
    llm_relation_category: str = ""
    llm_reasoning_path: str = ""
    llm_failure_mode: str = ""
    llm_evidence_photo_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["score"] = round(float(self.score), 4)
        data["margin"] = round(float(self.margin), 4)
        data["signal_breakdown"] = {
            k: round(float(v), 4) for k, v in self.signal_breakdown.items()
        }
        if self.llm_confidence is not None:
            data["llm_confidence"] = round(float(self.llm_confidence), 4)
        return data


@dataclass(slots=True)
class ResolvedIdentity:
    face_id: str
    canonical_name: str
    observed_surface: str
    confidence: float
    evidence_packet: EvidencePacket
    name_source: str
    relation_to_owner: str
    relation_category: str
    reasoning_path: str
    score: float

    def to_profile_row(self, max_evidence: int = 10) -> dict[str, Any]:
        evidence_ids = []
        for pid in (
            self.evidence_packet.same_photo_ids
            + self.evidence_packet.text_photo_ids
            + self.evidence_packet.face_photo_ids
        ):
            if pid and pid not in evidence_ids:
                evidence_ids.append(pid)
        return {
            "face_id": self.face_id,
            "canonical_name": self.canonical_name,
            "relation_to_owner": self.relation_to_owner,
            "relation_category": self.relation_category,
            "evidence_photo_ids": evidence_ids[:max_evidence],
            "confidence": round(max(0.0, min(1.0, self.confidence)), 3),
            "reasoning_path": self.reasoning_path,
        }


@dataclass(slots=True)
class EvidenceInventory:
    album: dict[str, Any]
    photos: list[dict[str, Any]]
    photo_lookup: dict[str, dict[str, Any]]
    text_units: list[TextEvidenceUnit]
    text_units_by_id: dict[str, TextEvidenceUnit]
    face_units: list[FaceEvidenceUnit]
    face_units_by_face: dict[str, list[FaceEvidenceUnit]]
    face_photo_ids: dict[str, list[str]]
    owner_face_id: str
    owner_name: str
    owner_name_candidates: list[dict[str, Any]]
    owner_last_name: str
    face_appearance_counts: dict[str, int]

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "user_id": self.album.get("user_id"),
            "n_photos": len(self.photos),
            "n_text_units": len(self.text_units),
            "n_face_units": len(self.face_units),
            "n_faces": len(self.face_units_by_face),
            "owner_face_id": self.owner_face_id,
            "owner_name": self.owner_name,
            "owner_last_name": self.owner_last_name,
            "owner_name_candidates": self.owner_name_candidates[:10],
            "face_appearance_counts": self.face_appearance_counts,
        }

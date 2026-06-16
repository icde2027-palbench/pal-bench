"""Configuration for Evidence-Chain Dossier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ECDConfig:
    """Small deterministic config for the first runnable ECD prototype."""

    max_candidates_per_face: int = 12
    max_evidence_photos_per_person: int = 10
    max_owner_facts: int = 18
    min_identity_score: float = 0.34
    min_identity_margin: float = 0.03
    allow_first_name_predictions: bool = True
    infer_family_last_name: bool = True
    use_evidence_recall_broadener: bool = False
    use_owner_evidence_planner: bool = True
    use_owner_fact_census: bool = False
    owner_census_min_cards_per_source: int = 2
    use_frozen_evidence_augmentation: bool = False
    owner_addendum_max_new_facts: int = 40
    owner_addendum_max_total_facts: int = 58
    owner_addendum_min_score: float = 4.0
    use_owner_detail_miner: bool = False
    owner_detail_miner_max_calls: int = 0
    owner_detail_miner_photos_per_call: int = 32
    owner_detail_miner_max_new_facts: int = 12
    owner_detail_miner_min_confidence: float = 0.62
    use_evidence_citation_optimizer: bool = False
    evidence_optimizer_owner_max_photos: int = 5
    evidence_optimizer_person_max_photos: int = 10
    evidence_optimizer_min_score: float = 1.25
    use_relation_evidence_repair: bool = True
    use_person_evidence_routing: bool = False
    use_person_evidence_certificates: bool = False
    person_evidence_certificate_max_chars: int = 1450
    recall_max_candidates_per_face: int = 20
    recall_min_event_bridge_score: float = 0.30
    recall_event_bonus_weight: float = 0.18
    recall_text_only_bonus_weight: float = 0.08
    recall_role_bonus_weight: float = 0.08
    use_llm_reranker: bool = False
    use_llm_batch_adjudicator: bool = False
    use_llm_joint_assignment: bool = False
    use_bridge_event_assignment: bool = False
    bea_max_calls: int = 0
    bea_faces_per_call: int = 4
    bea_review_max_faces: int = 12
    bea_max_candidate_options_per_face: int = 6
    bea_top_event_edges_per_face: int = 4
    bea_local_candidates_per_face: int = 3
    bea_min_edge_score: float = 0.56
    bea_min_apply_edge_score: float = 0.62
    bea_accept_min_confidence: float = 0.66
    bea_skip_direct_margin: float = 0.16
    bea_override_direct_min_confidence: float = 0.88
    bea_override_direct_min_edge_score: float = 0.84
    bea_max_cross_edges_per_name: int = 10
    use_llm_joint_photo_scenes: bool = True
    use_llm_joint_pair_swaps: bool = True
    use_llm_joint_name_conflicts: bool = True
    use_llm_joint_coface_ambiguity: bool = True
    use_llm_global_consistency: bool = False
    use_llm_readout: bool = False
    max_llm_calls: int = 40
    llm_top_candidates_per_face: int = 4
    llm_recall_top_candidates_per_face: int = 8
    llm_max_face_reviews: int = 40
    llm_max_joint_groups: int = 6
    llm_joint_top_candidates_per_face: int = 5
    llm_joint_recall_top_candidates_per_face: int = 8
    llm_joint_max_faces_per_group: int = 7
    llm_joint_max_scene_faces: int = 4
    llm_joint_max_names_per_group: int = 12
    use_hard_identity_blocks: bool = False
    llm_hard_max_blocks: int = 3
    llm_hard_top_candidates_per_face: int = 7
    llm_hard_max_faces_per_block: int = 4
    llm_hard_max_names_per_block: int = 10
    llm_hard_min_face_channel_score: float = 1.20
    llm_hard_accept_min_confidence: float = 0.68
    llm_hard_require_channel_support: bool = True
    llm_hard_guard_duplicate_clusters: bool = True
    use_llm_adaptive_identity_repair: bool = False
    llm_adaptive_repair_max_groups: int = 0
    llm_adaptive_repair_top_faces: int = 8
    llm_adaptive_repair_min_risk: float = 0.38
    llm_adaptive_repair_accept_min_confidence: float = 0.58
    llm_adaptive_repair_min_bridge_score: float = 0.36
    llm_adaptive_repair_min_recall_priority: float = 0.30
    llm_adaptive_repair_window_accept_min_confidence: float = 0.68
    llm_adaptive_repair_window_max_candidate_rank: int = 2
    llm_batch_max_faces: int = 12
    llm_batch_faces_per_call: int = 4
    llm_batch_top_candidates_per_face: int = 5
    llm_batch_max_calls: int = 3
    llm_batch_identity_mode: str = "score"
    llm_accept_min_confidence: float = 0.40
    llm_owner_fact_budget: int = 14
    owner_evidence_max_cards: int = 28
    llm_photo_snippet_chars: int = 360

    @classmethod
    def from_path(cls, path: str | Path | None) -> "ECDConfig":
        """Load optional JSON/YAML config, falling back to defaults.

        The implementation avoids adding a hard dependency on PyYAML. If YAML is
        unavailable, users can pass JSON or rely on defaults.
        """
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ECD config not found: {p}")
        text = p.read_text("utf-8")
        data: dict[str, Any]
        if p.suffix.lower() == ".json":
            import json

            data = json.loads(text)
        else:
            try:
                import yaml  # type: ignore

                data = yaml.safe_load(text) or {}
            except Exception as exc:  # pragma: no cover - depends on optional dep
                raise RuntimeError(
                    "YAML config requires PyYAML; use JSON or install PyYAML"
                ) from exc
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

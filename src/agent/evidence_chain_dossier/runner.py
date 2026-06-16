"""Runner for the Evidence-Chain Dossier agent."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.utils.io import load_json, save_json

from .bridge_event_assignment import assign_bridge_events
from .candidates import build_identity_hypotheses
from .config import ECDConfig
from .evidence_citation_optimizer import optimize_evidence_citations
from .frozen_augmentation import augment_frozen_evidence_profile
from .global_consistency import apply_global_consistency
from .inventory import build_inventory
from .llm_batch_adjudicator import adjudicate_identity_batches
from .llm_readout import enhance_profile_readout
from .llm_reranker import rerank_identity_hypotheses
from .name_cluster import build_name_clusters
from .owner_detail_miner import mine_owner_details
from .owner_evidence_planner import build_owner_evidence_plan
from .readout import build_profile
from .resolver import resolve_identities
from .scene_assignment import assign_identity_jointly

logger = logging.getLogger(__name__)


def run_evidence_chain_dossier(
    agent_album_path: str,
    output_dir: str,
    config_path: str | None = None,
    llm_client: Any = None,
    use_llm: bool = False,
    force_no_llm: bool = False,
) -> dict[str, Any]:
    """Run Evidence-Chain Dossier on one agent album."""
    t_start = time.time()
    config = ECDConfig.from_path(config_path)
    if force_no_llm:
        _disable_llm(config)
    if use_llm and not _llm_enabled(config):
        config.use_llm_reranker = True
        config.use_llm_joint_assignment = True
        config.use_llm_global_consistency = True
        config.use_llm_readout = True
    llm = _init_llm(llm_client, config) if _llm_enabled(config) else None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    album = load_json(agent_album_path)
    if not isinstance(album, dict):
        raise ValueError(f"Expected album JSON object at {agent_album_path}")

    t_phase = time.time()
    inventory = build_inventory(album)
    inventory_time = time.time() - t_phase
    logger.info("ECD inventory: %s", inventory.to_diagnostics())

    t_phase = time.time()
    clusters = build_name_clusters(inventory)
    cluster_time = time.time() - t_phase
    logger.info("ECD clusters: %d", len(clusters))

    t_phase = time.time()
    hypotheses, candidate_diagnostics = build_identity_hypotheses(
        inventory,
        clusters,
        max_per_face=config.max_candidates_per_face,
        config=config,
    )
    candidate_time = time.time() - t_phase
    logger.info("ECD hypotheses: %d", len(hypotheses))

    llm_rerank_diagnostics: dict[str, Any] = {"enabled": False}
    t_phase = time.time()
    if config.use_llm_reranker:
        hypotheses, llm_rerank_diagnostics = rerank_identity_hypotheses(
            inventory=inventory,
            clusters=clusters,
            hypotheses=hypotheses,
            config=config,
            llm=llm,
        )
    llm_rerank_time = time.time() - t_phase

    llm_batch_adjudicator_diagnostics: dict[str, Any] = {"enabled": False}
    t_phase = time.time()
    if config.use_llm_batch_adjudicator:
        hypotheses, llm_batch_adjudicator_diagnostics = adjudicate_identity_batches(
            inventory=inventory,
            clusters=clusters,
            hypotheses=hypotheses,
            config=config,
            llm=llm,
        )
    llm_batch_adjudicator_time = time.time() - t_phase

    llm_joint_assignment_diagnostics: dict[str, Any] = {"enabled": False}
    t_phase = time.time()
    if config.use_llm_joint_assignment:
        hypotheses, llm_joint_assignment_diagnostics = assign_identity_jointly(
            inventory=inventory,
            clusters=clusters,
            hypotheses=hypotheses,
            config=config,
            llm=llm,
        )
    llm_joint_assignment_time = time.time() - t_phase

    bridge_event_assignment_diagnostics: dict[str, Any] = {"enabled": False}
    t_phase = time.time()
    if config.use_bridge_event_assignment:
        hypotheses, bridge_event_assignment_diagnostics = assign_bridge_events(
            inventory=inventory,
            clusters=clusters,
            hypotheses=hypotheses,
            config=config,
            llm=llm,
        )
    bridge_event_assignment_time = time.time() - t_phase

    t_phase = time.time()
    resolved, resolver_diagnostics = resolve_identities(inventory, clusters, hypotheses, config)
    resolve_time = time.time() - t_phase
    logger.info("ECD resolved: %d", len(resolved))

    global_consistency_diagnostics: dict[str, Any] = {"enabled": False}
    t_phase = time.time()
    if config.use_llm_global_consistency:
        resolved, global_consistency_diagnostics = apply_global_consistency(
            inventory=inventory,
            clusters=clusters,
            hypotheses=hypotheses,
            resolved=resolved,
            config=config,
            llm=llm,
        )
    global_consistency_time = time.time() - t_phase

    run_stats = {
        "n_llm_calls": _llm_calls(llm),
        "n_photos": len(inventory.photos),
        "n_faces": len(inventory.face_units_by_face),
        "n_name_clusters": len(clusters),
        "n_identity_hypotheses": len(hypotheses),
        "n_resolved_identities": sum(1 for row in resolved if row.canonical_name),
        "CEHR": _cehr(inventory, resolved),
    }

    owner_evidence_plan = build_owner_evidence_plan(
        inventory,
        max_cards=config.owner_evidence_max_cards,
        balanced=config.use_owner_fact_census,
        min_cards_per_source=config.owner_census_min_cards_per_source,
    )

    t_phase = time.time()
    profile = build_profile(inventory, resolved, config, run_stats)
    llm_readout_diagnostics: dict[str, Any] = {"enabled": False}
    if config.use_llm_readout:
        profile, llm_readout_diagnostics = enhance_profile_readout(
            profile=profile,
            inventory=inventory,
            resolved=resolved,
            config=config,
            llm=llm,
        )
        profile["run_stats"]["n_llm_calls"] = _llm_calls(llm)
    owner_detail_miner_diagnostics: dict[str, Any] = {"enabled": False}
    t_owner_detail = time.time()
    if config.use_owner_detail_miner:
        profile, owner_detail_miner_diagnostics = mine_owner_details(
            profile=profile,
            inventory=inventory,
            config=config,
            llm=llm,
        )
        profile["run_stats"]["n_llm_calls"] = _llm_calls(llm)
    owner_detail_miner_time = time.time() - t_owner_detail
    frozen_augmentation_diagnostics: dict[str, Any] = {"enabled": False}
    if config.use_frozen_evidence_augmentation:
        profile, frozen_augmentation_diagnostics = augment_frozen_evidence_profile(
            profile=profile,
            inventory=inventory,
            resolved=resolved,
            config=config,
        )
    evidence_citation_optimizer_diagnostics: dict[str, Any] = {"enabled": False}
    t_evidence_optimizer = time.time()
    if config.use_evidence_citation_optimizer:
        profile, evidence_citation_optimizer_diagnostics = optimize_evidence_citations(
            profile=profile,
            inventory=inventory,
            resolved=resolved,
            config=config,
        )
    evidence_citation_optimizer_time = time.time() - t_evidence_optimizer
    readout_time = time.time() - t_phase
    total_time = time.time() - t_start
    usage_summary = _llm_usage(llm)

    profile["run_stats"].update(
        {
            "total_time_s": round(total_time, 3),
            "inventory_time_s": round(inventory_time, 3),
            "cluster_time_s": round(cluster_time, 3),
            "candidate_time_s": round(candidate_time, 3),
            "llm_rerank_time_s": round(llm_rerank_time, 3),
            "llm_batch_adjudicator_time_s": round(llm_batch_adjudicator_time, 3),
            "llm_joint_assignment_time_s": round(llm_joint_assignment_time, 3),
            "bridge_event_assignment_time_s": round(bridge_event_assignment_time, 3),
            "resolve_time_s": round(resolve_time, 3),
            "global_consistency_time_s": round(global_consistency_time, 3),
            "owner_detail_miner_time_s": round(owner_detail_miner_time, 3),
            "evidence_citation_optimizer_time_s": round(evidence_citation_optimizer_time, 3),
            "readout_time_s": round(readout_time, 3),
            "budget_used": _llm_calls(llm),
            "budget_max": config.max_llm_calls,
            "parse_failures": _llm_parse_failures(llm),
            **_token_run_stats(usage_summary),
        }
    )

    save_json(profile, output_path / "predicted_profile.json")
    save_json(profile["run_stats"], output_path / "run_stats.json")
    save_json(inventory.to_diagnostics(), output_path / "inventory_diagnostics.json")
    save_json([c.to_dict() for c in clusters], output_path / "name_clusters.json")
    save_json([h.to_dict() for h in hypotheses], output_path / "identity_hypotheses.json")
    save_json(
        {
            "candidate_topk_by_face": candidate_diagnostics,
            "llm_reranker": llm_rerank_diagnostics,
            "llm_batch_adjudicator": llm_batch_adjudicator_diagnostics,
            "llm_joint_assignment": llm_joint_assignment_diagnostics,
            "bridge_event_assignment": bridge_event_assignment_diagnostics,
            "resolver": resolver_diagnostics,
            "llm_global_consistency": global_consistency_diagnostics,
            "owner_evidence_plan": owner_evidence_plan,
            "llm_readout": llm_readout_diagnostics,
            "owner_detail_miner": owner_detail_miner_diagnostics,
            "frozen_evidence_augmentation": frozen_augmentation_diagnostics,
            "evidence_citation_optimizer": evidence_citation_optimizer_diagnostics,
            "llm_usage": usage_summary,
        },
        output_path / "diagnostics.json",
    )
    return profile


def _llm_enabled(config: ECDConfig) -> bool:
    return bool(
        config.use_llm_reranker
        or config.use_llm_batch_adjudicator
        or config.use_llm_joint_assignment
        or config.use_bridge_event_assignment
        or config.use_llm_global_consistency
        or config.use_llm_readout
        or config.use_owner_detail_miner
    )


def _disable_llm(config: ECDConfig) -> None:
    config.use_llm_reranker = False
    config.use_llm_batch_adjudicator = False
    config.use_llm_joint_assignment = False
    config.use_bridge_event_assignment = False
    config.use_llm_global_consistency = False
    config.use_llm_readout = False
    config.use_owner_detail_miner = False


def _init_llm(llm_client: Any, config: ECDConfig):
    from src.agent.evidence_chain_dossier.llm_interface import DossierLLM

    return DossierLLM(llm=llm_client, max_calls=config.max_llm_calls)


def _llm_calls(llm: Any) -> int:
    if llm is None:
        return 0
    return int(getattr(getattr(llm, "budget", None), "used", 0) or 0)


def _llm_parse_failures(llm: Any) -> int:
    if llm is None:
        return 0
    return int(getattr(getattr(llm, "budget", None), "parse_failures", 0) or 0)


def _llm_usage(llm: Any) -> dict[str, Any]:
    if llm is None or not hasattr(llm, "usage_summary"):
        return {}
    summary = llm.usage_summary()
    return summary if isinstance(summary, dict) else {}


def _token_run_stats(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary:
        return {}
    fields: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "calls_with_usage",
        "calls_without_usage",
        "token_logging_coverage",
        "usage_by_stage",
    ):
        if key in summary:
            fields[key] = summary[key]
    return fields


def _cehr(inventory, resolved) -> float:
    cited = set()
    for row in resolved:
        cited.update(row.evidence_packet.same_photo_ids)
        cited.update(row.evidence_packet.text_photo_ids)
        cited.update(row.evidence_packet.face_photo_ids)
    cited.update(f.get("evidence_photo_ids", [])[0] for f in [] if f)
    return round(len(cited) / max(1, len(inventory.photos)), 4)

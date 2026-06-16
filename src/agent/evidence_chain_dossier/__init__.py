"""Evidence-Chain Dossier agent.

This package implements the new public-only, evidence-chain-first agent
prototype. It is intentionally separate from ``src.agent.dossier`` so that
experiments can compare the frameworks without inheriting old phase state.
"""

from .runner import run_evidence_chain_dossier

__all__ = ["run_evidence_chain_dossier"]

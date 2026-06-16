from __future__ import annotations

from typing import Any


class ReasoningPathGenerator:
    @staticmethod
    def _classify_node_alignment_difficulty(path: dict[str, Any]) -> str:
        key_photos = path.get("key_photos") or []
        relation = str(path.get("relation") or "").lower()
        identification = str(path.get("identification") or "").lower()
        relation_reasoning = str(path.get("relation_reasoning") or "").lower()
        text = " ".join([relation, identification, relation_reasoning])
        if len(key_photos) >= 3 or any(token in text for token in ("ambiguous", "indirect", "nickname")):
            return "hard"
        if len(key_photos) >= 2 or any(token in text for token in ("recurring", "repeated", "co-occurrence")):
            return "medium"
        return "easy"

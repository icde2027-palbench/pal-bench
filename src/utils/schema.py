"""Schema versioning system for cache management.

This module provides a centralized registry for schema versions across
the pipeline, enabling automatic cache invalidation on version changes
and migration tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import logging

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Types of cacheable components in the photo generation pipeline."""
    AMBIENT_PLAN = "ambient_plan"
    AMBIENT_CHUNK = "ambient_plan_chunk"
    KEY_EVIDENCE_PROMPT = "key_evidence_prompt"
    EVENT_SCENE_PROMPT = "event_scene_prompt"
    AMBIENT_PHOTO_PROMPT = "ambient_photo_prompt"


@dataclass(frozen=True)
class SchemaVersion:
    """Immutable schema version descriptor.
    
    Attributes:
        component: Type of component (e.g., 'ambient_plan')
        version: Semantic version string (e.g., 'v2_chunked')
        introduced_date: When this version was introduced
        description: Human-readable description of this version
    """
    component: str
    version: str
    introduced_date: str = field(default_factory=lambda: datetime.now().isoformat()[:10])
    description: str = ""
    
    def __str__(self) -> str:
        """Return versioned component string for cache keys."""
        # Version strings already include 'v', so no need to add it
        return f"{self.component}_{self.version}"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "component": self.component,
            "version": self.version,
            "introduced_date": self.introduced_date,
            "description": self.description,
        }


class SchemaRegistry:
    """Central registry for all schema versions in the pipeline.
    
    This registry provides:
    - A single source of truth for component versions
    - Automatic version migration detection
    - Backward compatibility tracking
    - Cache invalidation hints
    """
    
    # Default schema versions (frozen at module load time)
    _DEFAULTS = {
        ComponentType.AMBIENT_PLAN: SchemaVersion(
            component="ambient_plan",
            version="v2_chunked",
            introduced_date="2026-05-10",
            description="Two-phase ambient photo plan with chunked LLM generation",
        ),
        ComponentType.AMBIENT_CHUNK: SchemaVersion(
            component="ambient_plan_chunk",
            version="v2_chunked",
            introduced_date="2026-05-10",
            description="Chunked ambient photo generation with normalization",
        ),
        ComponentType.KEY_EVIDENCE_PROMPT: SchemaVersion(
            component="key_evidence_prompt",
            version="v1",
            introduced_date="2026-05-01",
            description="Key evidence photo prompt generation",
        ),
        ComponentType.EVENT_SCENE_PROMPT: SchemaVersion(
            component="event_scene_prompt",
            version="v1",
            introduced_date="2026-05-01",
            description="Event narrative photo prompt generation",
        ),
        ComponentType.AMBIENT_PHOTO_PROMPT: SchemaVersion(
            component="ambient_photo_prompt",
            version="v1",
            introduced_date="2026-05-01",
            description="Individual ambient photo prompt generation",
        ),
    }
    
    def __init__(self):
        """Initialize registry with default versions."""
        self._versions: dict[ComponentType | str, SchemaVersion] = {}
        # Register defaults
        for comp_type, version in self._DEFAULTS.items():
            self._versions[comp_type] = version
            # Also register by string key for convenience
            self._versions[comp_type.value] = version
    
    def get(self, component: ComponentType | str) -> SchemaVersion:
        """Get schema version for a component.
        
        Parameters:
            component: Component type (enum or string)
            
        Returns:
            SchemaVersion object
            
        Raises:
            KeyError: If component not registered
        """
        if component in self._versions:
            return self._versions[component]
        raise KeyError(f"Unknown component: {component}")
    
    def register(self, component: ComponentType | str, version: SchemaVersion) -> None:
        """Register or update a schema version.
        
        Parameters:
            component: Component type
            version: SchemaVersion object
            
        Raises:
            ValueError: If version string is invalid
        """
        if not version.version or not version.component:
            raise ValueError("Version and component must be non-empty")
        
        # Update both enum and string keys
        if isinstance(component, ComponentType):
            self._versions[component] = version
            self._versions[component.value] = version
        else:
            self._versions[component] = version
        
        logger.info(
            "Registered schema version: %s@%s (introduced: %s)",
            version.component,
            version.version,
            version.introduced_date,
        )
    
    def check_migration(
        self,
        component: ComponentType | str,
        stored_version: str,
    ) -> tuple[bool, str]:
        """Check if a stored version matches the current version.
        
        Parameters:
            component: Component type
            stored_version: Version string from cache/metadata
            
        Returns:
            (matches: bool, message: str)
            - If matches, returns (True, "")
            - If different, returns (False, reason)
        """
        current = self.get(component)
        if stored_version == str(current):
            return True, ""
        
        reason = f"Version mismatch: {stored_version} → {current}"
        logger.warning("Schema migration needed: %s", reason)
        return False, reason
    
    def list_versions(self) -> dict[str, dict[str, Any]]:
        """Return all registered versions as dictionary.
        
        Returns:
            Dictionary mapping component names to version info
        """
        result = {}
        seen = set()
        for key, version in self._versions.items():
            if version.component not in seen:
                result[version.component] = version.to_dict()
                seen.add(version.component)
        return result


# Global singleton registry
_REGISTRY: SchemaRegistry | None = None


def get_registry() -> SchemaRegistry:
    """Get or create the global schema registry.
    
    Returns:
        SchemaRegistry singleton
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SchemaRegistry()
    return _REGISTRY


def reset_registry() -> None:
    """Reset registry to defaults (for testing).
    
    Note: This is primarily for unit testing. Do not call in production.
    """
    global _REGISTRY
    _REGISTRY = None

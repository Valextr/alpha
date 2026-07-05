from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class FeatureMeta:
    """Metadata for a single feature."""

    name: str
    description: str
    category: str
    lookback: int
    depends_on: list[str] = field(default_factory=list)


class FeatureRegistry:
    """Singleton registry for all features."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._features = {}
        return cls._instance

    def register(self, name, *, description, category, lookback, depends_on=None):
        """Register a feature and return a decorator for auto-registration."""
        meta = FeatureMeta(
            name=name,
            description=description,
            category=category,
            lookback=lookback,
            depends_on=depends_on or [],
        )
        self._features[name] = meta

        def decorator(fn):
            return fn

        return decorator

    def list_features(self):
        """Return all registered features."""
        return list(self._features.values())

    def get_feature(self, name):
        """Get metadata for a feature by name."""
        return self._features.get(name)

    def features_by_category(self):
        """Group features by category."""
        result = {}
        for meta in self._features.values():
            result.setdefault(meta.category, []).append(meta)
        return result

    def validate_dependencies(self):
        """Check that all feature dependencies exist."""
        known = set(self._features.keys())
        missing = []
        for meta in self._features.values():
            for dep in meta.depends_on:
                if dep not in known and dep not in missing:
                    missing.append(dep)
        return missing

    def reset(self):
        """Clear all registered features. Useful for testing."""
        self._features.clear()

    def __len__(self):
        return len(self._features)

    def __contains__(self, name):
        return name in self._features


registry = FeatureRegistry()

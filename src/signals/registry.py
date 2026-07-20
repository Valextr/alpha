"""Signal registry — mirrors the FeatureRegistry pattern."""

from __future__ import annotations

from typing import Any, Callable

from .base import SignalMeta


class SignalRegistry:
    """Singleton registry for all signals."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._signals: dict[str, SignalMeta] = {}
            cls._instance._fns: dict[str, Callable[..., Any]] = {}
        return cls._instance

    def register(
        self,
        name: str,
        *,
        description: str,
        category: str,
        parameters: dict | None = None,
        depends_on: list[str] | None = None,
        requires_features: list[str] | None = None,
    ):
        """Register a signal and return a decorator."""
        meta = SignalMeta(
            name=name,
            description=description,
            category=category,
            parameters=parameters or {},
            depends_on=depends_on or [],
            requires_features=requires_features or [],
        )
        self._signals[name] = meta

        def decorator(fn):
            self._fns[name] = fn
            return fn

        return decorator

    def list_signals(self) -> list[SignalMeta]:
        """Return all registered signal metadata."""
        return list(self._signals.values())

    def get_signal(self, name: str) -> SignalMeta | None:
        """Get metadata for a signal by name."""
        return self._signals.get(name)

    def get_generate_fn(self, name: str) -> Callable | None:
        """Get the generation function for a signal."""
        return self._fns.get(name)

    def signals_by_category(self) -> dict[str, list[SignalMeta]]:
        """Group signals by category."""
        result: dict[str, list[SignalMeta]] = {}
        for meta in self._signals.values():
            result.setdefault(meta.category, []).append(meta)
        return result

    def validate_dependencies(self) -> list[str]:
        """Check that all signal dependencies exist."""
        known = set(self._signals.keys())
        missing = []
        for meta in self._signals.values():
            for dep in meta.depends_on:
                if dep not in known and dep not in missing:
                    missing.append(dep)
        return missing

    def reset(self) -> None:
        """Clear all registered signals. Useful for testing."""
        self._signals.clear()
        self._fns.clear()

    def __len__(self) -> int:
        return len(self._signals)

    def __contains__(self, name: str) -> bool:
        return name in self._signals


registry = SignalRegistry()
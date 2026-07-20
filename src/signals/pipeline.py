"""Signal pipeline — generates all registered signals on a DataFrame."""

from __future__ import annotations

import importlib

import polars as pl

from .registry import registry


def _signal_module(signal_name: str) -> str | None:
    """Determine which module a signal belongs to."""
    if signal_name.startswith("mean_reversion"):
        return "mean_reversion"
    elif signal_name.startswith("momentum"):
        return "momentum"
    # Extend as new signal modules are added
    return None


def generate_all(df: pl.DataFrame, categories: list[str] | None = None) -> pl.DataFrame:
    """Generate all registered signals (or a filtered subset).

    Signals are applied in dependency order. Each signal function
    receives the current DataFrame state and returns an enriched copy.

    Args:
        df: Feature-enriched DataFrame sorted by (ticker, date).
        categories: Optional category filter (e.g. ["mean_reversion"]).

    Returns:
        DataFrame with all signal columns appended.
    """
    if df.is_empty():
        return df

    all_signals = registry.list_signals()
    if categories:
        category_set = set(categories)
        all_signals = [s for s in all_signals if s.category in category_set]

    # Sort by depends_on length (leaf signals first)
    all_signals.sort(key=lambda s: len(s.depends_on))

    current = df
    for meta in all_signals:
        module_name = _signal_module(meta.name)
        if module_name is None:
            continue

        try:
            mod = importlib.import_module(f".{module_name}", package="src.signals")
        except ImportError:
            continue

        func_name = f"generate_{meta.name}"
        func = getattr(mod, func_name, None)
        if func is not None:
            try:
                current = func(current)
            except Exception as e:
                print(f"Warning: Failed to generate {meta.name}: {e}")

    return current


def generate_all_with_forward_returns(
    df: pl.DataFrame,
    categories: list[str] | None = None,
    horizons: list[int] | None = None,
) -> pl.DataFrame:
    """Generate signals and attach forward returns for validation.

    Forward returns are computed AFTER signal generation, so signal
    computation itself remains free of future leakage.

    Args:
        df: Feature-enriched DataFrame sorted by (ticker, date).
        categories: Optional category filter.
        horizons: Forward horizons in trading days (default: [1, 5, 21]).

    Returns:
        DataFrame with signal columns AND forward_return_<h> columns.
    """
    from .base import compute_forward_returns

    if horizons is None:
        horizons = [1, 5, 21]

    current = generate_all(df, categories=categories)
    current = compute_forward_returns(current, horizons=horizons)
    return current
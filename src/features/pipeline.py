from __future__ import annotations

import importlib

import polars as pl

from .registry import registry
from . import price, volatility  # noqa: F401 - triggers auto-registration
from .base import validate_features


def _feature_module(feature_name):
    """Determine which module a feature belongs to by name convention."""
    if feature_name.startswith(("return_", "log_return_", "cum_return_",
                                 "drawdown_", "price_displacement_")):
        return "price"
    elif feature_name.startswith(("vol_", "vol_of_vol", "vol_ratio")):
        return "volatility"
    elif feature_name.startswith(("relative_volume", "accum_", "volume_")):
        return "volume"
    elif feature_name.startswith("cs_"):
        return "cross_sectional"
    elif feature_name.startswith("regime_"):
        return "regime"
    elif feature_name.startswith(("frac_", "kalman_", "hurst_")):
        return "advanced"
    return None


def compute_features(df, categories=None):
    """Compute all registered features on a DataFrame.

    Features are applied in dependency order: features with no
    dependencies first, then those that depend on them.

    Args:
        df: Input DataFrame with columns at minimum:
            ticker, date, open, high, low, close, volume.
            Must be sorted by (ticker, date).
        categories: Optional list of categories to compute.
            If None, computes all registered categories.

    Returns:
        DataFrame with all feature columns appended.
    """
    if df.is_empty():
        return df

    all_features = registry.list_features()

    if categories:
        category_set = set(categories)
        all_features = [f for f in all_features if f.category in category_set]

    # Sort by lookback (shorter lookback first = dependencies resolved first)
    all_features.sort(key=lambda f: (f.lookback, len(f.depends_on)))

    current = df
    for meta in all_features:
        module_name = _feature_module(meta.name)
        if module_name is None:
            continue

        try:
            mod = importlib.import_module(f".{module_name}", package="src.features")
        except ImportError:
            continue

        func_name = f"compute_{meta.name}"
        func = getattr(mod, func_name, None)
        if func is not None:
            try:
                current = func(current)
            except Exception as e:
                print(f"Warning: Failed to compute {meta.name}: {e}")

    return current


def compute_and_validate(df, categories=None):
    """Compute features and return validation results.

    Args:
        df: Input DataFrame sorted by (ticker, date).
        categories: Optional category filter.

    Returns:
        Tuple of (enriched DataFrame, validation dict).
    """
    enriched = compute_features(df, categories=categories)
    validation = validate_features(enriched)
    return enriched, validation

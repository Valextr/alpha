"""Integration test for Batches 1-3: volume, cross_sectional, regime features."""

import sys
from pathlib import Path

import polars as pl

# Add project root to path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.features.pipeline import compute_features, compute_and_validate
from src.features.registry import registry


def load_gold_sample(tickers=None, year="2023"):
    """Load gold layer data for specified tickers."""
    gold_dir = root / "data" / "gold" / "daily" / f"year={year}"
    files = list(gold_dir.glob("ticker=*/part-0.parquet"))

    if tickers:
        files = [f for f in files if any(t in str(f) for t in tickers)]

    frames = []
    for f in sorted(files):
        df = pl.read_parquet(str(f))
        frames.append(df)

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])


def test_registry():
    """Verify all features are registered."""
    features = registry.list_features()
    categories = registry.features_by_category()

    print(f"Total registered features: {len(features)}")
    for cat, feats in sorted(categories.items()):
        print(f"  {cat}: {len(feats)} features")

    # Check new batches
    assert "volume" in categories, "volume category missing"
    assert "cross_sectional" in categories, "cross_sectional category missing"
    assert "regime" in categories, "regime category missing"

    assert len(categories["volume"]) == 4, f"Expected 4 volume features, got {len(categories['volume'])}"
    assert len(categories["cross_sectional"]) == 5, f"Expected 5 cross_sectional features, got {len(categories['cross_sectional'])}"
    assert len(categories["regime"]) == 4, f"Expected 4 regime features, got {len(categories['regime'])}"

    # Check dependencies
    missing = registry.validate_dependencies()
    assert not missing, f"Missing dependencies: {missing}"

    print("\nRegistry: OK")
    return True


def test_volume_features():
    """Test volume features on sample data."""
    print("\n--- Testing Volume Features ---")
    df = load_gold_sample(tickers=["AAPL", "MSFT", "SPY"])
    assert not df.is_empty(), "No sample data loaded"
    print(f"Input: {len(df)} rows, {df['ticker'].n_unique()} tickers, cols: {df.columns}")

    enriched, validation = compute_and_validate(df, categories=["volume"])

    expected_cols = ["relative_volume_21d", "volume_zscore_63d",
                     "accumulation_distribution", "volume_shock"]
    for col in expected_cols:
        assert col in enriched.columns, f"Missing column: {col}"

    # Check values
    rel_vol = enriched["relative_volume_21d"]
    assert (rel_vol.fill_null(0) >= 0).all(), "relative_volume should be non-negative"

    shock = enriched["volume_shock"]
    assert shock.is_in([0, 1]).all(), "volume_shock should be binary"

    print(f"Output cols: {len(enriched.columns)} (+{len(expected_cols)} volume features)")
    print(f"Validation: {validation}")
    print("Volume: OK")
    return enriched


def test_cross_sectional_features(df=None):
    """Test cross-sectional features on sample data."""
    print("\n--- Testing Cross-Sectional Features ---")

    if df is None:
        # Need multiple tickers for cross-sectional to make sense
        df = load_gold_sample(tickers=["AAPL", "MSFT", "GOOGL", "SPY", "QQQ", "JPM", "JNJ"])

    assert not df.is_empty(), "No sample data"
    print(f"Input: {len(df)} rows, {df['ticker'].n_unique()} tickers")

    # First compute price + volume (dependencies)
    df = compute_features(df, categories=["price", "volatility", "volume"])
    print(f"After price+vol+volume: {len(df.columns)} cols")

    # Now compute cross-sectional
    enriched, validation = compute_and_validate(df, categories=["cross_sectional"])

    expected_cols = ["cs_return_zscore_21d", "cs_return_rank_21d",
                     "cs_vol_rank_21d", "cs_volume_rank_21d",
                     "sector_relative_return_21d"]
    for col in expected_cols:
        assert col in enriched.columns, f"Missing column: {col}"

    # Check rank features are 0-1
    for rank_col in ["cs_return_rank_21d", "cs_vol_rank_21d", "cs_volume_rank_21d"]:
        vals = enriched[rank_col].drop_nulls()
        if len(vals) > 0:
            assert (vals >= 0).all() and (vals <= 1).all(), f"{rank_col} should be 0-1"

    print(f"Output cols: {len(enriched.columns)} (+5 cross-sectional)")
    print(f"Validation: {validation}")
    print("Cross-Sectional: OK")
    return enriched


def test_regime_features(df=None):
    """Test regime features on sample data."""
    print("\n--- Testing Regime Features ---")

    if df is None:
        df = load_gold_sample(tickers=["AAPL", "MSFT", "GOOGL", "SPY", "QQQ", "JPM", "JNJ"])

    assert not df.is_empty(), "No sample data"

    # Compute all dependencies first
    df = compute_features(df, categories=["price", "volatility", "volume",
                                           "cross_sectional"])
    print(f"After all deps: {len(df.columns)} cols")

    enriched, validation = compute_and_validate(df, categories=["regime"])

    expected_cols = ["regime_market_trend", "regime_vol_state",
                     "regime_breadth", "regime_vol_regime"]
    for col in expected_cols:
        assert col in enriched.columns, f"Missing column: {col}"

    # Check ranges
    breadth = enriched["regime_breadth"].drop_nulls()
    if len(breadth) > 0:
        assert (breadth >= 0).all() and (breadth <= 1).all(), "breadth should be 0-1"

    vol_state = enriched["regime_vol_state"].drop_nulls()
    if len(vol_state) > 0:
        assert (vol_state >= 0).all(), "vol_state should be non-negative"

    vol_regime = enriched["regime_vol_regime"]
    assert vol_regime.is_in([0, 1]).all(), "vol_regime should be binary"

    print(f"Output cols: {len(enriched.columns)} (+4 regime)")
    print(f"Validation: {validation}")
    print("Regime: OK")
    return enriched


def test_full_pipeline():
    """Test full pipeline: all categories end-to-end."""
    print("\n--- Testing Full Pipeline (Batches 0-3) ---")
    df = load_gold_sample(tickers=["AAPL", "MSFT", "GOOGL", "SPY", "QQQ",
                                    "JPM", "JNJ", "XOM", "PG", "NVDA"])

    assert not df.is_empty(), "No sample data"
    input_cols = len(df.columns)
    print(f"Input: {len(df)} rows, {df['ticker'].n_unique()} tickers, {input_cols} cols")

    enriched, validation = compute_and_validate(df)

    # Should have all categories
    features = registry.list_features()
    # Exclude advanced (not built yet)
    non_advanced = [f for f in features if f.category != "advanced"]
    expected_new = len(non_advanced)

    print(f"\nExpected {expected_new} features (excluding advanced)")
    print(f"New feature columns: {len(enriched.columns) - input_cols}")
    print(f"Total columns: {len(enriched.columns)}")

    # Show sample values
    print("\nSample values (last row per ticker):")
    sample = enriched.group_by("ticker").last().select([
        "ticker", "date",
        "relative_volume_21d", "volume_shock",
        "cs_return_rank_21d", "sector_relative_return_21d",
        "regime_breadth", "regime_vol_regime",
    ]).sort("ticker")
    print(sample)

    print(f"\nValidation:")
    print(f"  NaN rates (>1%): {len(validation['nan_rates'])} columns")
    print(f"  Inf columns: {len(validation['inf_columns'])}")
    print(f"  Constant columns: {len(validation['constant_columns'])}")

    assert not validation["inf_columns"], f"Inf values found: {validation['inf_columns']}"

    print("\nFull Pipeline: OK")
    return enriched


if __name__ == "__main__":
    print("=" * 60)
    print("Feature Pipeline Tests - Batches 1-3")
    print("=" * 60)

    test_registry()
    test_volume_features()
    test_cross_sectional_features()
    test_regime_features()
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

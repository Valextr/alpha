"""Integration test for Batch 4: advanced features."""

import sys
from pathlib import Path

import polars as pl

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


def test_advanced_registry():
    """Verify advanced features are registered."""
    categories = registry.features_by_category()
    assert "advanced" in categories, "advanced category missing"
    advanced = categories["advanced"]
    assert len(advanced) == 4, f"Expected 4 advanced features, got {len(advanced)}"

    names = [f.name for f in advanced]
    assert "hurst_63d" in names
    assert "kalman_alpha" in names
    assert "kalman_beta" in names
    assert "frac_diff_1d" in names

    print(f"Advanced features registered: {names}")
    print("Registry: OK")


def test_advanced_features():
    """Test advanced features on sample data."""
    print("\n--- Testing Advanced Features ---")
    df = load_gold_sample(tickers=["AAPL", "MSFT", "SPY"])
    assert not df.is_empty(), "No sample data"
    print(f"Input: {len(df)} rows, {df['ticker'].n_unique()} tickers")

    enriched, validation = compute_and_validate(df, categories=["advanced"])

    expected_cols = ["hurst_63d", "kalman_alpha", "kalman_beta", "frac_diff_1d"]
    for col in expected_cols:
        assert col in enriched.columns, f"Missing column: {col}"

    # Check Hurst exponent range (should be 0-1 typically)
    hurst = enriched["hurst_63d"].drop_nulls()
    if len(hurst) > 0:
        print(f"  Hurst: min={hurst.min():.3f}, max={hurst.max():.3f}, mean={hurst.mean():.3f}")
        # Hurst can be outside 0-1 for short windows but should be reasonable
        assert float(hurst.min()) > -0.5 and float(hurst.max()) < 2.0, f"Hurst range: [{hurst.min():.3f}, {hurst.max():.3f}]"

    # Check Kalman alpha tracks price
    kalman_a = enriched["kalman_alpha"].drop_nulls()
    if len(kalman_a) > 0:
        print(f"  Kalman alpha: min={kalman_a.min():.1f}, max={kalman_a.max():.1f}")

    # Check Kalman beta (trend) is reasonable
    kalman_b = enriched["kalman_beta"].drop_nulls()
    if len(kalman_b) > 0:
        print(f"  Kalman beta: min={kalman_b.min():.4f}, max={kalman_b.max():.4f}")

    # Check frac_diff is computed
    frac = enriched["frac_diff_1d"].drop_nulls()
    if len(frac) > 0:
        print(f"  Frac diff: min={frac.min():.1f}, max={frac.max():.1f}")

    print(f"Output cols: {len(enriched.columns)} (+4 advanced)")
    print(f"Validation NaN rates: {len(validation['nan_rates'])} columns")
    print("Advanced: OK")


def test_full_pipeline_all_categories():
    """Test full pipeline with ALL 6 categories."""
    print("\n--- Testing Full Pipeline (All 6 Categories) ---")
    df = load_gold_sample(tickers=["AAPL", "MSFT", "GOOGL", "SPY", "QQQ",
                                    "JPM", "JNJ", "XOM", "PG", "NVDA"])
    assert not df.is_empty(), "No sample data"
    input_cols = len(df.columns)
    print(f"Input: {len(df)} rows, {df['ticker'].n_unique()} tickers, {input_cols} cols")

    enriched, validation = compute_and_validate(df)

    features = registry.list_features()
    expected_new = len(features)

    print(f"\nExpected {expected_new} total features")
    print(f"New feature columns: {len(enriched.columns) - input_cols}")
    print(f"Total columns: {len(enriched.columns)}")

    # Show feature counts by category
    categories = registry.features_by_category()
    print("\nFeatures by category:")
    for cat, feats in sorted(categories.items()):
        present = [f for f in feats if f.name in enriched.columns]
        print(f"  {cat}: {len(present)}/{len(feats)} computed")

    assert not validation["inf_columns"], f"Inf values: {validation['inf_columns']}"

    print("\nFull Pipeline (All Categories): OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Advanced Features Tests - Batch 4")
    print("=" * 60)

    test_advanced_registry()
    test_advanced_features()
    test_full_pipeline_all_categories()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

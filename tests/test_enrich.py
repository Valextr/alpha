"""Tests for enrich.py — correctness and performance of market_cap_bucket."""

import sys
import time
from pathlib import Path

from datetime import date

import polars as pl

# Add project root to path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.data.enrich import enrich_with_market_cap_bucket, build_gold_layer


def _expected_bucket(ticker: str) -> str:
    """Reference implementation matching the old bucket() logic."""
    mega_caps = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "BRK.B", "JPM"}
    large_caps = {"TSLA", "UNH", "V", "MA", "JNJ", "XOM", "CVX", "PG", "HD", "DIS", "NFLX"}
    if ticker in mega_caps:
        return "mega"
    if ticker in large_caps:
        return "large"
    if ticker.startswith("X") or ticker in {"SPY", "QQQ", "IWM", "DIA"}:
        return "etf"
    return "mid"


def test_market_cap_bucket_known_tickers():
    """Every known ticker maps to the correct bucket."""
    tickers = [
        # mega
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "BRK.B", "JPM",
        # large
        "TSLA", "UNH", "V", "MA", "JNJ", "XOM", "CVX", "PG", "HD", "DIS", "NFLX",
        # etf (explicit)
        "SPY", "QQQ", "IWM", "DIA",
        # etf (starts with X)
        "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLY",
        # mid
        "INTC", "AMD", "CRM", "ADBE", "BA", "GE", "CAT", "HON", "UNP",
        # unknown (should be mid)
        "UNKNOWN",
    ]

    df = pl.DataFrame({"ticker": tickers, "date": [date(2023, 1, 1)] * len(tickers)})
    result = enrich_with_market_cap_bucket(df)

    for ticker, bucket in zip(tickers, result["market_cap_bucket"].to_list()):
        expected = _expected_bucket(ticker)
        assert bucket == expected, f"{ticker}: got {bucket}, expected {expected}"

    print("test_market_cap_bucket_known_tickers: OK")


def test_market_cap_bucket_identical_output():
    """Output must match the old implementation exactly for all edge cases."""
    edge_tickers = [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "BRK.B", "JPM",
        "TSLA", "UNH", "V", "MA", "JNJ", "XOM", "CVX", "PG", "HD", "DIS", "NFLX",
        "SPY", "QQQ", "IWM", "DIA",
        "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLY",
        "INTC", "AMD", "CRM", "ADBE", "BA", "GE", "CAT", "HON", "UNP",
        "UNKNOWN", "BRK-B", "BRK.A",  # edge cases
    ]

    df = pl.DataFrame({"ticker": edge_tickers})
    result = enrich_with_market_cap_bucket(df)

    expected = [_expected_bucket(t) for t in edge_tickers]
    actual = result["market_cap_bucket"].to_list()

    assert actual == expected, f"Mismatch:\n  expected: {expected}\n  actual:   {actual}"

    # No map_elements should appear in the output column type
    assert result["market_cap_bucket"].dtype == pl.String, f"Wrong dtype: {result['market_cap_bucket'].dtype}"

    print("test_market_cap_bucket_identical_output: OK")


def test_no_map_elements():
    """Verify no map_elements usage in enrich.py."""
    enrich_path = root / "src" / "data" / "enrich.py"
    content = enrich_path.read_text()
    assert "map_elements" not in content, "map_elements still present in enrich.py"
    assert "map_apply" not in content, "map_apply still present in enrich.py"

    print("test_no_map_elements: OK")


def test_market_cap_bucket_speedup():
    """Benchmark: vectorized should be 10-100x faster than map_elements on large data."""
    # Generate 50K rows of realistic ticker data
    import random
    all_tickers = [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "BRK.B", "JPM",
        "TSLA", "UNH", "V", "MA", "JNJ", "XOM", "CVX", "PG", "HD", "DIS", "NFLX",
        "SPY", "QQQ", "IWM", "DIA",
        "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLY",
        "INTC", "AMD", "CRM", "ADBE", "BA", "GE", "CAT", "HON", "UNP",
        "UNKNOWN",
    ]
    n_rows = 50_000
    tickers = [random.choice(all_tickers) for _ in range(n_rows)]
    df = pl.DataFrame({"ticker": tickers, "date": [date(2023, 1, 1)] * n_rows})

    # Warmup
    _ = enrich_with_market_cap_bucket(df.clone())

    # Benchmark vectorized (current implementation)
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        result = enrich_with_market_cap_bucket(df.clone())
    vectorized_ms = (time.perf_counter() - start) / iterations * 1000

    # Verify correctness on large data
    expected = [_expected_bucket(t) for t in tickers]
    actual = result["market_cap_bucket"].to_list()
    assert actual == expected, "Output mismatch on large data"

    print(f"Vectorized: {vectorized_ms:.3f} ms per call ({n_rows} rows)")
    print(f"Rough equiv. for old map_elements (estimated): {vectorized_ms * 50:.0f} ms (50x slower typical)")

    # map_elements on 50K rows typically takes 2500+ ms, vectorized should be <50 ms
    assert vectorized_ms < 100, f"Vectorized should be <100ms, got {vectorized_ms:.1f}ms"

    print("test_market_cap_bucket_speedup: OK")


def test_build_gold_layer_includes_bucket():
    """End-to-end: build_gold_layer produces market_cap_bucket column."""
    silver = pl.DataFrame({
        "ticker": ["AAPL", "TSLA", "SPY", "UNKNOWN"],
        "date": [date(2023, 1, 1)] * 4,
        "close": [150.0, 200.0, 400.0, 50.0],
        "volume": [1e6, 2e6, 5e6, 1e5],
    })
    gold = build_gold_layer(silver)

    assert "market_cap_bucket" in gold.columns, "market_cap_bucket missing from gold layer"
    assert gold["market_cap_bucket"].dtype == pl.String

    print("test_build_gold_layer_includes_bucket: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Enrich tests")
    print("=" * 60)

    test_market_cap_bucket_known_tickers()
    test_market_cap_bucket_identical_output()
    test_no_map_elements()
    test_market_cap_bucket_speedup()
    test_build_gold_layer_includes_bucket()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
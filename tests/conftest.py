"""Fixtures for synthetic OHLCV data and corporate actions.

All fixtures produce small, deterministic DataFrames so data-pipeline
unit tests run fast without network access or on-disk data.

Resource guardrails appended below prevent runaway memory consumption
during test runs.
"""

import sys

# Production resolves business-logic modules from the private repo first
# (the systemd service does sys.path.insert(0, beta)); the
# public repo ships stubs for src/features, src/risk, src/ensemble, etc.
# Pinning the private 'src' package here — at conftest import time, before any
# test module imports src.* — keeps resolution deterministic in every xdist
# worker, where per-worker path juggling would otherwise randomly resolve the
# public stubs and fail ~14 test modules with ImportError.
_ALPHA_PRIVATE = "/home/Vale/working/beta"
if _ALPHA_PRIVATE not in sys.path:
    sys.path.insert(0, _ALPHA_PRIVATE)
import src  # noqa: F401  (pins sys.modules['src'] to the private tree)

from datetime import date, timedelta

import gc
import os

import polars as pl
import pytest


# ── helpers ──────────────────────────────────────────────────────────

def _trading_dates(start: date, n: int) -> list[date]:
    """Return *n* weekday dates starting from *start* (Mon–Fri only).
    """
    dates: list[date] = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:  # Mon=0 … Fri=4
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _make_bars(
    ticker: str,
    num_days: int = 60,
    base_close: float = 100.0,
    start_date: date = date(2023, 1, 3),
    volume: int = 1_000_000,
) -> pl.DataFrame:
    """Create deterministic daily bars for a single ticker.

    Prices walk up by $1 each day starting from *base_close*.
    Open = close, high = close+1, low = close-1 (clamped >= 1).
    """
    dates = _trading_dates(start_date, num_days)
    closes = [base_close + i for i in range(num_days)]
    return pl.DataFrame(
        {
            "ticker": [ticker] * num_days,
            "date": dates,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [max(c - 1, 1.0) for c in closes],
            "close": closes,
            "volume": [volume] * num_days,
        }
    )


def _merge_bars(*frames: pl.DataFrame) -> pl.DataFrame:
    """Concatenate per-ticker bar frames into one."""
    return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])


# ── pytest fixtures ──────────────────────────────────────────────────

@pytest.fixture
def single_ticker_bars() -> pl.DataFrame:
    """60 daily bars for AAPL starting Jan 3 2023."""
    return _make_bars("AAPL", num_days=60, base_close=100.0)


@pytest.fixture
def two_ticker_bars() -> pl.DataFrame:
    """60 bars each for AAPL & MSFT."""
    return _merge_bars(
        _make_bars("AAPL", num_days=60, base_close=100.0),
        _make_bars("MSFT", num_days=60, base_close=250.0),
    )


@pytest.fixture
def bars_with_zero_volume() -> pl.DataFrame:
    """Bars where 20 % of days have zero volume (for zero-volume checks)."""
    df = _make_bars("TEST", num_days=50, base_close=50.0)
    zero_idx = list(range(0, 10, 1))  # 10/50 = 20 %
    df = df.with_columns(
        pl.when(pl.arange(0, len(df)).is_in(zero_idx))
        .then(pl.lit(0))
        .otherwise(pl.col("volume"))
        .alias("volume")
    )
    return df


@pytest.fixture
def bars_with_negative_price() -> pl.DataFrame:
    """Bars containing negative close prices (should never happen)."""
    df = _make_bars("BAD", num_days=20, base_close=30.0)
    # Make a few rows negative
    df = df.with_columns(
        pl.when(pl.arange(0, len(df)).is_in([5, 6, 7]))
        .then(pl.lit(-10.0))
        .otherwise(pl.col("close"))
        .alias("close"),
    )
    return df


@pytest.fixture
def bars_with_large_price_gap() -> pl.DataFrame:
    """Bars where one day jumps 50 % (for price gap detection)."""
    dates = _trading_dates(date(2023, 1, 3), 40)
    closes = [100.0 + i for i in range(40)]
    # Inject a 50 % jump at row 20
    closes[20] = 150.0
    closes = [closes[i] + (0.0 if i <= 20 else (closes[20] + (i - 20))) for i in range(40)]
    # Simpler: just set row 20 to 150 and continue from there
    closes = [100.0 + i for i in range(20)] + [150.0] + [150.0 + i for i in range(1, 20)]
    return pl.DataFrame(
        {
            "ticker": ["GAP"] * 40,
            "date": dates,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [max(c - 1, 1.0) for c in closes],
            "close": closes,
            "volume": [1_000_000] * 40,
        }
    )


@pytest.fixture
def bars_with_date_gaps() -> pl.DataFrame:
    """Bars with >5 day gaps between trading dates."""
    dates = _trading_dates(date(2023, 1, 3), 16) + _trading_dates(
        date(2023, 1, 3) + timedelta(days=80), 16
    )
    closes = [100.0 + i for i in range(32)]
    return pl.DataFrame(
        {
            "ticker": ["GAPD"] * 32,
            "date": dates,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [max(c - 1, 1.0) for c in closes],
            "close": closes,
            "volume": [1_000_000] * 32,
        }
    )


# ── splits & dividends ───────────────────────────────────────────────

@pytest.fixture
def splits_2x() -> pl.DataFrame:
    """A 2:1 split for AAPL on 2023-02-15."""
    return pl.DataFrame(
        {
            "ticker": ["AAPL"],
            "date": [date(2023, 2, 15)],
            "action_type": ["split"],
            "factor": [2.0],
        }
    )


@pytest.fixture
def splits_empty() -> pl.DataFrame:
    """Empty split DataFrame."""
    return pl.DataFrame(
        {
            "ticker": pl.Series([], dtype=pl.Utf8),
            "date": pl.Series([], dtype=pl.Date),
            "action_type": pl.Series([], dtype=pl.Utf8),
            "factor": pl.Series([], dtype=pl.Float64),
        }
    )


@pytest.fixture
def dividends_fixture() -> pl.DataFrame:
    """Two quarterly dividends for AAPL."""
    return pl.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "ex_date": [date(2023, 2, 10), date(2023, 5, 12)],
            "amount": [0.23, 0.24],
        }
    )


@pytest.fixture
def dividends_empty() -> pl.DataFrame:
    """Empty dividend DataFrame."""
    return pl.DataFrame(
        {
            "ticker": pl.Series([], dtype=pl.Utf8),
            "ex_date": pl.Series([], dtype=pl.Date),
            "amount": pl.Series([], dtype=pl.Float64),
        }
    )


@pytest.fixture
def corporate_actions_empty() -> pl.DataFrame:
    """Empty corporate actions DataFrame."""
    return pl.DataFrame(
        {
            "ticker": pl.Series([], dtype=pl.Utf8),
            "date": pl.Series([], dtype=pl.Date),
            "action_type": pl.Series([], dtype=pl.Utf8),
            "factor": pl.Series([], dtype=pl.Float64),
        }
    )


# ── silver-layer fixtures (for enrich tests) ─────────────────────────

@pytest.fixture
def silver_bars() -> pl.DataFrame:
    """Silver-layer bars for two tickers (includes silver columns)."""
    bars = _merge_bars(
        _make_bars("AAPL", num_days=70, base_close=150.0),
        _make_bars("MSFT", num_days=70, base_close=300.0),
    )
    return bars.with_columns(
        pl.col("close").alias("adj_close"),
        pl.lit(1.0, dtype=pl.Float64).alias("split_factor"),
        pl.lit(0.0, dtype=pl.Float64).alias("dividend_yield"),
        pl.lit(True).alias("is_market_date"),
        pl.lit("good").alias("data_quality"),
    )


# ── Resource guardrails ──────────────────────────────────────────────
# Prevents runaway memory consumption during test runs.
#
# Layer 1: Per-test RSS delta watchdog — fails if a single test grows
#   more than MEMORY_DELTA_MB (default 2 GB).
# Layer 2: Session-level RSS cap — aborts the run if total process RSS
#   exceeds SESSION_RSS_MB (default 32 GB on a 64 GB machine).
# Layer 3: gc.collect() after every test to reclaim Python-managed memory.
# Layer 4: Real-data tests marked as 'slow' — skipped by default via
#   pyproject.toml addopts.

DEFAULT_MEMORY_DELTA_MB = 2048
MEMORY_DELTA_MB = int(os.environ.get("ALPHA_TEST_MEMORY_DELTA_MB", DEFAULT_MEMORY_DELTA_MB))
MEMORY_DELTA_BYTES = MEMORY_DELTA_MB * 1024 * 1024

# Session-level RSS cap: abort if the pytest process exceeds this.
# On a 64 GB machine with Hermes + vLLM + OS overhead, 4 GB is a safe cap.
# Override via ALPHA_TEST_SESSION_RSS_MB.
DEFAULT_SESSION_RSS_MB = 4 * 1024
SESSION_RSS_MB = int(os.environ.get("ALPHA_TEST_SESSION_RSS_MB", DEFAULT_SESSION_RSS_MB))
SESSION_RSS_BYTES = SESSION_RSS_MB * 1024 * 1024

# Minimum available system memory — if the system drops below this, abort.
# Default: 4 GB (leave headroom for OS, Hermes, vLLM, etc.).
DEFAULT_MIN_AVAILABLE_MB = 4 * 1024
MIN_AVAILABLE_MB = int(os.environ.get("ALPHA_TEST_MIN_AVAILABLE_MB", DEFAULT_MIN_AVAILABLE_MB))
MIN_AVAILABLE_BYTES = MIN_AVAILABLE_MB * 1024 * 1024


@pytest.fixture(autouse=True)
def _memory_guard():
    """Check memory growth during each test."""
    import psutil

    process = psutil.Process()
    before_mem = process.memory_info().rss

    yield

    gc.collect()

    after_mem = process.memory_info().rss
    delta = after_mem - before_mem

    if delta > MEMORY_DELTA_BYTES:
        pytest.fail(
            f"Test consumed {delta / 1024 / 1024:.0f} MB (limit: {MEMORY_DELTA_MB} MB). "
            f"Set ALPHA_TEST_MEMORY_DELTA_MB to override."
        )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Pre-test check — refuse to start the next test if memory is tight."""
    import psutil

    # Check available system memory
    vm = psutil.virtual_memory()
    if vm.available < MIN_AVAILABLE_BYTES:
        pytest.exit(
            f"Available system memory ({vm.available / 1024 / 1024:.0f} MB) below minimum "
            f"({MIN_AVAILABLE_MB} MB).  Aborting to prevent swap thrashing. "
            f"Set ALPHA_TEST_MIN_AVAILABLE_MB to override, or run "
            f"'pytest -m not slow' to skip heavy tests."
        )

    # Check process RSS
    process = psutil.Process()
    rss = process.memory_info().rss
    threshold = SESSION_RSS_BYTES * 0.8
    if rss > threshold:
        pytest.exit(
            f"Pre-test RSS ({rss / 1024 / 1024:.0f} MB) exceeded 80 % of cap "
            f"({SESSION_RSS_MB} MB).  Aborting to prevent swap thrashing. "
            f"Set ALPHA_TEST_SESSION_RSS_MB to override, or run "
            f"'pytest -m not slow' to skip heavy tests."
        )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    """Post-test check — hard abort if we've blown past the cap."""
    if nextitem is None:
        return  # last test, nothing to protect

    import psutil

    process = psutil.Process()
    rss = process.memory_info().rss

    if rss > SESSION_RSS_BYTES:
        pytest.exit(
            f"Post-test RSS ({rss / 1024 / 1024:.0f} MB) exceeded cap "
            f"({SESSION_RSS_MB} MB).  Aborting to prevent OOM. "
            f"Set ALPHA_TEST_SESSION_RSS_MB to override, or run "
            f"'pytest -m not slow' to skip heavy tests."
        )


def pytest_configure(config):
    """Register the 'slow' marker and print memory limits."""
    config.addinivalue_line("markers", "slow: marks tests as slow (real data, large datasets)")
    print(f"Alpha test memory delta limit: {MEMORY_DELTA_MB} MB per test")
    print(f"Alpha test session RSS cap: {SESSION_RSS_MB} MB total")
    print(f"Alpha test min available memory: {MIN_AVAILABLE_MB} MB system")

    # Check swap usage at session start — if swap is >90% used, warn/abort.
    # With swappiness=10, existing zram usage doesn't cause thrashing.
    # Only abort if swap is nearly full (no room for new allocations).
    import psutil
    swap = psutil.swap_memory()
    swap_pct = swap.percent
    if swap_pct > 90:
        print(
            f"\nWARNING: Swap usage at {swap_pct:.0f}% ({swap.used / 1024 / 1024:.0f} MB / "
            f"{swap.total / 1024 / 1024:.0f} MB). Tests may thrash.\n"
            f"Run 'sudo swapoff -a && sudo swapon -a' to clear.\n"
        )
        ok_pct = float(os.environ.get("ALPHA_TEST_SWAP_PCT_OK", "90"))
        if swap_pct > ok_pct:
            pytest.exit(
                f"Swap usage ({swap_pct:.0f}%) exceeds threshold ({ok_pct:.0f}%). "
                f"Aborting to prevent thrashing. Set ALPHA_TEST_SWAP_PCT_OK to override."
            )
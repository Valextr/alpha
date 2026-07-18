"""Fixtures for synthetic OHLCV data and corporate actions.

All fixtures produce small, deterministic DataFrames so data-pipeline
unit tests run fast without network access or on-disk data.
"""

from datetime import date, timedelta

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
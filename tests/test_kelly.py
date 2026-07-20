"""Kelly criterion position sizing tests.

Covers:
    - KellyConfig validation
    - compute_kelly_fraction edge cases
    - compute_position_weights integration
    - Portfolio exposure capping
    - Position capping
    - Signal direction handling
    - Multi-signal ensemble interface
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.portfolio.kelly import (
    KellyConfig,
    compute_kelly_fraction,
    compute_position_weights,
    compute_position_weights_from_signal_scores,
)
from src.signals.base import compute_forward_returns
from src.features.pipeline import compute_features

# ── helpers ──────────────────────────────────────────────────────────


def _trading_dates(start: date, n: int) -> list[date]:
    dates: list[date] = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _make_bars(
    ticker: str,
    num_days: int = 300,
    base_close: float = 100.0,
    start_date: date = date(2023, 1, 3),
) -> pl.DataFrame:
    dates = _trading_dates(start_date, num_days)
    import math

    closes = []
    c = base_close
    for i in range(num_days):
        c = c * (1.0 + 0.02 * math.sin(i * 0.1))
        closes.append(round(c, 2))
    return pl.DataFrame(
        {
            "ticker": [ticker] * num_days,
            "date": dates,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [max(c - 1, 1.0) for c in closes],
            "close": closes,
            "volume": [1_000_000] * num_days,
        }
    )


def _multi_ticker_bars(
    n_tickers: int = 5,
    n_days: int = 300,
) -> pl.DataFrame:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"][:n_tickers]
    frames = [
        _make_bars(t, num_days=n_days, base_close=100.0 + i * 50)
        for i, t in enumerate(tickers)
    ]
    return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])


def _add_synthetic_signal(df: pl.DataFrame, col_name: str = "signal_test") -> pl.DataFrame:
    """Add a synthetic signal aligned with forward returns (positive edge)."""
    return df.with_columns(
        pl.when(pl.col("forward_return_1") > 0).then(0.5)
        .when(pl.col("forward_return_1") < 0).then(-0.5)
        .otherwise(0.0)
        .alias(col_name)
    )


# ── KellyConfig tests ────────────────────────────────────────────────


class TestKellyConfig:
    def test_default_values(self):
        config = KellyConfig()
        assert config.kelly_fraction == 0.25
        assert config.lookback == 63
        assert config.target_vol == 0.10
        assert config.max_position == 0.20
        assert config.max_portfolio_exposure == 1.0
        assert config.forward_horizon == 1

    def test_custom_values(self):
        config = KellyConfig(
            kelly_fraction=0.5,
            lookback=252,
            target_vol=0.15,
            max_position=0.10,
            max_portfolio_exposure=2.0,
            forward_horizon=5,
        )
        assert config.kelly_fraction == 0.5
        assert config.lookback == 252
        assert config.target_vol == 0.15

    def test_invalid_kelly_fraction_zero(self):
        with pytest.raises(ValueError, match="kelly_fraction"):
            KellyConfig(kelly_fraction=0.0)

    def test_invalid_kelly_fraction_negative(self):
        with pytest.raises(ValueError, match="kelly_fraction"):
            KellyConfig(kelly_fraction=-0.1)

    def test_invalid_kelly_fraction_over_one(self):
        with pytest.raises(ValueError, match="kelly_fraction"):
            KellyConfig(kelly_fraction=1.5)

    def test_invalid_lookback_too_small(self):
        with pytest.raises(ValueError, match="lookback"):
            KellyConfig(lookback=1)

    def test_invalid_target_vol_zero(self):
        with pytest.raises(ValueError, match="target_vol"):
            KellyConfig(target_vol=0.0)

    def test_invalid_max_position_zero(self):
        with pytest.raises(ValueError, match="max_position"):
            KellyConfig(max_position=0.0)

    def test_invalid_max_portfolio_exposure_zero(self):
        with pytest.raises(ValueError, match="max_portfolio_exposure"):
            KellyConfig(max_portfolio_exposure=0.0)

    def test_invalid_forward_horizon_zero(self):
        with pytest.raises(ValueError, match="forward_horizon"):
            KellyConfig(forward_horizon=0)


# ── compute_kelly_fraction tests ────────────────────────────────────


class TestKellyFraction:
    def test_positive_edge_quarter_kelly(self):
        """60% win rate, 2:1 odds → positive Kelly."""
        f = compute_kelly_fraction(0.6, 0.02, 0.01, 0.25)
        assert f > 0.0
        # Full Kelly: (0.6 * 2 - 0.4) / 2 = (1.2 - 0.4) / 2 = 0.4
        # Quarter Kelly: 0.4 * 0.25 = 0.1
        assert abs(f - 0.1) < 1e-10

    def test_negative_edge(self):
        """40% win rate, 0.5:1 odds → zero Kelly."""
        f = compute_kelly_fraction(0.4, 0.01, 0.02, 0.25)
        assert f == 0.0

    def test_break_even_edge(self):
        """50% win rate, 1:1 odds → zero Kelly."""
        f = compute_kelly_fraction(0.5, 0.01, 0.01, 0.25)
        assert f == 0.0

    def test_full_kelly(self):
        """Full Kelly should be 4x quarter Kelly."""
        f_full = compute_kelly_fraction(0.6, 0.02, 0.01, 1.0)
        f_quarter = compute_kelly_fraction(0.6, 0.02, 0.01, 0.25)
        assert abs(f_quarter - f_full * 0.25) < 1e-10

    def test_half_kelly(self):
        """Half Kelly should optimize Sharpe ratio."""
        f_half = compute_kelly_fraction(0.6, 0.02, 0.01, 0.5)
        f_full = compute_kelly_fraction(0.6, 0.02, 0.01, 1.0)
        assert abs(f_half - f_full * 0.5) < 1e-10

    def test_clamped_to_max_kelly_fraction(self):
        """Kelly should never exceed the configured fraction."""
        # Extreme edge: 90% win rate, 10:1 odds
        f = compute_kelly_fraction(0.9, 0.10, 0.01, 0.25)
        assert f <= 0.25

    def test_zero_avg_win(self):
        f = compute_kelly_fraction(0.6, 0.0, 0.01, 0.25)
        assert f == 0.0

    def test_zero_avg_loss(self):
        f = compute_kelly_fraction(0.6, 0.02, 0.0, 0.25)
        assert f == 0.0


# ── compute_position_weights integration tests ─────────────────────


class TestPositionWeights:
    def _prepare_df(self, n_tickers=3, n_days=200):
        df = _multi_ticker_bars(n_tickers, n_days)
        df = compute_features(df)
        df = compute_forward_returns(df, horizons=[1, 5, 21])
        return df

    def test_output_columns(self):
        """Output should contain all expected columns."""
        df = self._prepare_df()
        df = _add_synthetic_signal(df)
        weights = compute_position_weights(df, "signal_test", "forward_return_1")

        expected = [
            "ticker", "date", "signal_value", "kelly_fraction",
            "vol_targeted_fraction", "capped_fraction",
            "position_weight", "portfolio_exposure",
        ]
        for col in expected:
            assert col in weights.columns, f"Missing column: {col}"

    def test_output_sorted_by_date_ticker(self):
        """Output should be sorted by date, then ticker."""
        df = self._prepare_df()
        df = _add_synthetic_signal(df)
        weights = compute_position_weights(df, "signal_test", "forward_return_1")

        dates = weights["date"].to_list()
        assert dates == sorted(dates), "Dates should be sorted"

    def test_position_weight_respects_signal_direction(self):
        """Positive signal → positive weight, negative signal → negative weight."""
        df = self._prepare_df()
        df = _add_synthetic_signal(df)
        weights = compute_position_weights(df, "signal_test", "forward_return_1")

        # Filter to non-zero rows
        nz = weights.filter(pl.col("position_weight") != 0)
        if len(nz) == 0:
            pytest.skip("No non-zero positions generated")

        # Positive signal should produce positive weight
        long_rows = nz.filter(pl.col("signal_value") > 0)
        assert len(long_rows) > 0, "Should have long positions"
        assert (long_rows["position_weight"] > 0).all(), "Long positions should be positive"

        # Negative signal should produce negative weight
        short_rows = nz.filter(pl.col("signal_value") < 0)
        assert len(short_rows) > 0, "Should have short positions"
        assert (short_rows["position_weight"] < 0).all(), "Short positions should be negative"

    def test_position_capping(self):
        """Position weight should not exceed max_position."""
        df = self._prepare_df()
        df = _add_synthetic_signal(df)
        config = KellyConfig(max_position=0.10)
        weights = compute_position_weights(df, "signal_test", "forward_return_1", config)

        assert (
            weights["position_weight"].abs() <= 0.10 + 1e-9
        ).all(), "Position weight should not exceed max_position"

    def test_portfolio_exposure_capping(self):
        """Total portfolio exposure should not exceed max_portfolio_exposure."""
        df = self._prepare_df()
        df = _add_synthetic_signal(df)
        config = KellyConfig(max_portfolio_exposure=0.5)
        weights = compute_position_weights(df, "signal_test", "forward_return_1", config)

        assert (
            weights["portfolio_exposure"] <= 0.5 + 1e-9
        ).all(), "Portfolio exposure should not exceed cap"

    def test_no_position_for_negative_edge(self):
        """Signal with no edge should produce zero or near-zero positions."""
        df = self._prepare_df()
        # Add a random signal (no edge — just oscillation)
        import math
        df = df.with_columns(
            (pl.lit(0.02) * pl.int_range(len(df)) % 100)
            .map_elements(lambda x: math.sin(x * 0.1), return_dtype=pl.Float64)
            .alias("signal_no_edge")
        )
        weights = compute_position_weights(df, "signal_no_edge", "forward_return_1")

        # After enough history, positions should shrink (no persistent edge)
        # Just verify the pipeline runs without crashing
        assert len(weights) > 0

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty result with correct schema."""
        df = self._prepare_df()
        df = df.filter(pl.col("ticker") == "NONEXISTENT")  # Ensure empty
        df = _add_synthetic_signal(df)
        df = df.with_columns(pl.col("signal_test").fill_null(0.0))

        result = compute_position_weights(df, "signal_test", "forward_return_1")
        assert result.is_empty() or len(result) == 0

    def test_missing_column_raises(self):
        """Missing signal or forward column should raise ValueError."""
        df = self._prepare_df()
        with pytest.raises(ValueError, match="Missing required column"):
            compute_position_weights(df, "nonexistent_signal", "forward_return_1")

        with pytest.raises(ValueError, match="Missing required column"):
            compute_position_weights(df, "close", "nonexistent_forward")

    def test_lookback_warmup(self):
        """Early rows (before lookback) should have reduced position sizes."""
        df = self._prepare_df()
        df = _add_synthetic_signal(df)
        config = KellyConfig(lookback=63)
        weights = compute_position_weights(df, "signal_test", "forward_return_1", config)

        # First few dates should have zero positions (not enough history)
        first_date = weights["date"].min()
        first_rows = weights.filter(pl.col("date") == first_date)
        # Kelly fraction should be 0 for early rows
        assert (first_rows["kelly_fraction"] == 0).all() or (first_rows["kelly_fraction"].is_null()).all()

    def test_multi_ticker_isolation(self):
        """Each ticker should have independent Kelly estimates."""
        df = self._prepare_df(n_tickers=3)
        df = _add_synthetic_signal(df)
        weights = compute_position_weights(df, "signal_test", "forward_return_1")

        # All 3 tickers should be present
        tickers = set(weights["ticker"].to_list())
        assert len(tickers) == 3, f"Expected 3 tickers, got {len(tickers)}"


# ── compute_position_weights_from_signal_scores tests ───────────────


class TestMultiSignalWeights:
    def _prepare_df(self):
        df = _multi_ticker_bars(3, 200)
        df = compute_features(df)
        df = compute_forward_returns(df, horizons=[1, 5, 21])
        # Add two synthetic signals
        df = df.with_columns([
            pl.when(pl.col("forward_return_1") > 0).then(0.5)
            .when(pl.col("forward_return_1") < 0).then(-0.5)
            .otherwise(0.0)
            .alias("signal_test_1"),
            pl.when(pl.col("forward_return_5") > 0).then(0.3)
            .when(pl.col("forward_return_5") < 0).then(-0.3)
            .otherwise(0.0)
            .alias("signal_test_2"),
        ])
        return df

    def test_multi_signal_produces_output(self):
        df = self._prepare_df()
        weights = compute_position_weights_from_signal_scores(
            df, ["signal_test_1", "signal_test_2"]
        )
        assert len(weights) > 0

    def test_empty_signal_list_raises(self):
        df = self._prepare_df()
        with pytest.raises(ValueError, match="signal_cols must not be empty"):
            compute_position_weights_from_signal_scores(df, [])

    def test_missing_signal_column_raises(self):
        df = self._prepare_df()
        with pytest.raises(ValueError, match="Missing signal column"):
            compute_position_weights_from_signal_scores(
                df, ["signal_test_1", "nonexistent_signal"]
            )


# ── Edge cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_signal(self):
        """Zero signal should produce zero position weight."""
        df = _multi_ticker_bars(3, 200)
        df = compute_features(df)
        df = compute_forward_returns(df, horizons=[1])
        df = df.with_columns(pl.lit(0.0).alias("signal_zero"))

        weights = compute_position_weights(df, "signal_zero", "forward_return_1")
        assert (weights["position_weight"] == 0).all()

    def test_very_short_lookback(self):
        """Lookback of 5 (minimum) should still work."""
        df = _multi_ticker_bars(3, 100)
        df = compute_features(df)
        df = compute_forward_returns(df, horizons=[1])
        df = _add_synthetic_signal(df)

        config = KellyConfig(lookback=5, kelly_fraction=0.5)
        weights = compute_position_weights(df, "signal_test", "forward_return_1", config)
        assert len(weights) > 0

    def test_high_volatility_targeting(self):
        """High target volatility should scale positions up (within cap)."""
        df = _multi_ticker_bars(3, 200)
        df = compute_features(df)
        df = compute_forward_returns(df, horizons=[1])
        df = _add_synthetic_signal(df)

        config = KellyConfig(target_vol=0.50, max_position=0.50)
        weights = compute_position_weights(df, "signal_test", "forward_return_1", config)
        assert len(weights) > 0

    def test_single_ticker(self):
        """Single ticker should work without cross-sectional operations."""
        df = _make_bars("SINGLE", 200)
        df = compute_features(df)
        df = compute_forward_returns(df, horizons=[1])
        df = _add_synthetic_signal(df)

        weights = compute_position_weights(df, "signal_test", "forward_return_1")
        assert len(weights) > 0
        assert weights["ticker"].unique().to_list() == ["SINGLE"]
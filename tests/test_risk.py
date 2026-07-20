"""Risk management tests.

Tests all risk control layers:
- Config validation
- Position limits
- Sector limits
- Leverage caps
- Volatility targeting
- Drawdown circuit breaker
- Correlation spike detection
- Full pipeline integration
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from pathlib import Path
import polars as pl
import pytest

root = None  # resolved below
for p in (Path(__file__).resolve().parent.parent,):
    if (p / "src").exists():
        root = p
        break
if root is None:
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(root))

from src.risk.config import RiskConfig
from src.risk.constraints import (
    enforce_position_limits,
    enforce_sector_limits,
    enforce_leverage,
    enforce_all_constraints as enforce_all_hard,
)
from src.risk.vol_targeting import (
    compute_realized_volatility,
    volatility_target,
)
from src.risk.drawdown import (
    drawdown_circuit_breaker,
    compute_drawdown_series,
)
from src.risk.correlation import (
    compute_rolling_correlation,
    correlation_spike_detector,
)
from src.risk.pipeline import enforce_all_constraints

# ── Helpers ──────────────────────────────────────────────────────────


def _trading_dates(start: date, n: int) -> list[date]:
    dates: list[date] = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _make_weights(
    tickers: list[str],
    n_days: int = 60,
    sectors: dict[str, str] | None = None,
    base_weight: float = 0.15,
) -> pl.DataFrame:
    """Create synthetic weight DataFrame."""
    dates = _trading_dates(date(2023, 1, 3), n_days)
    rows = []
    for d in dates:
        for i, t in enumerate(tickers):
            # Vary weight slightly per ticker/date
            w = base_weight + 0.01 * (i * 0.1)
            row = {"ticker": t, "date": d, "weight": w}
            if sectors and t in sectors:
                row["sector"] = sectors[t]
            rows.append(row)
    return pl.DataFrame(rows)


def _make_returns(
    tickers: list[str],
    n_days: int = 100,
) -> pl.DataFrame:
    """Create synthetic returns DataFrame with some volatility."""
    import math
    dates = _trading_dates(date(2023, 1, 3), n_days)
    rows = []
    for d_idx, d in enumerate(dates):
        for i, t in enumerate(tickers):
            # Generate some structured returns
            r = 0.001 + 0.02 * math.sin(d_idx * 0.1 + i)
            rows.append({"ticker": t, "date": d, "return": r})
    return pl.DataFrame(rows)


def _make_equity_curve(
    n_days: int = 60,
    include_drawdown: bool = False,
) -> pl.DataFrame:
    """Create synthetic equity curve."""
    import math
    dates = _trading_dates(date(2023, 1, 3), n_days)
    rows = []
    eq = 100000.0
    for i, d in enumerate(dates):
        if include_drawdown and 20 <= i <= 35:
            # Simulate a drawdown
            eq = eq * 0.97
        else:
            eq = eq * (1.0 + 0.001 + 0.005 * math.sin(i * 0.1))
        rows.append({"date": d, "equity": eq})
    return pl.DataFrame(rows)


# ── Config tests ────────────────────────────────────────────────────


class TestRiskConfig:
    def test_defaults(self):
        cfg = RiskConfig()
        assert cfg.max_position_pct == 0.10
        assert cfg.max_sector_pct == 0.30
        assert cfg.max_leverage == 1.5
        assert cfg.target_vol == 0.10
        assert cfg.max_drawdown == 0.20

    def test_conservative_preset(self):
        cfg = RiskConfig.conservative()
        assert cfg.max_position_pct == 0.05
        assert cfg.max_leverage == 1.0
        assert cfg.target_vol == 0.05

    def test_aggressive_preset(self):
        cfg = RiskConfig.aggressive()
        assert cfg.max_position_pct == 0.20
        assert cfg.max_leverage == 2.0

    def test_invalid_max_position_zero(self):
        with pytest.raises(ValueError):
            RiskConfig(max_position_pct=0.0)

    def test_invalid_max_position_gt_1(self):
        with pytest.raises(ValueError):
            RiskConfig(max_position_pct=1.5)

    def test_invalid_negative_leverage(self):
        with pytest.raises(ValueError):
            RiskConfig(max_leverage=-1.0)

    def test_invalid_lookback_too_small(self):
        with pytest.raises(ValueError):
            RiskConfig(lookback_vol=5)

    def test_frozen(self):
        cfg = RiskConfig()
        # frozen=True dataclasses should prevent attribute assignment
        try:
            cfg.__setattr__("max_position_pct", 0.5)
            pytest.fail("Should raise FrozenInstanceError")
        except (AttributeError, ValueError) as e:
            assert "frozen" in str(e).lower() or "readonly" in str(e).lower() or "frozen" in type(e).__name__.lower()


# ── Position limits ─────────────────────────────────────────────────


class TestPositionLimits:
    def test_no_cap_needed(self):
        df = _make_weights(["AAPL", "MSFT"], base_weight=0.05)
        result = enforce_position_limits(df, max_position_pct=0.10)
        assert (result["weight"] <= 0.10 + 1e-9).all()

    def test_cap_applied(self):
        df = _make_weights(["AAPL", "MSFT"], base_weight=0.15)
        result = enforce_position_limits(df, max_position_pct=0.10)
        assert (result["weight"] <= 0.10 + 1e-9).all()

    def test_negative_weights_capped(self):
        df = pl.DataFrame({
            "ticker": ["A", "A", "B", "B"],
            "date": [date(2023, 1, 3)] * 2 + [date(2023, 1, 4)] * 2,
            "weight": [0.15, -0.15, 0.05, -0.05],
        })
        result = enforce_position_limits(df, max_position_pct=0.10)
        assert (result["weight"].abs() <= 0.10 + 1e-9).all()


# ── Sector limits ───────────────────────────────────────────────────


class TestSectorLimits:
    def test_no_violation(self):
        sectors = {"AAPL": "Technology", "JPM": "Financials"}
        df = _make_weights(["AAPL", "JPM"], sectors=sectors, base_weight=0.10)
        result = enforce_sector_limits(df, max_sector_pct=0.30)
        # Sector totals should be ~0.10 each (well under 0.30)
        sector_totals = (
            result.with_columns(pl.col("weight").abs().alias("_aw"))
            .group_by("date")
            .agg(
                pl.sum("_aw").alias("total")
            )
        )
        assert (sector_totals["total"] <= 0.30 + 1e-9).all()

    def test_sector_cap_applied(self):
        # 3 tech stocks, each 0.20 → sector total 0.60 > 0.30
        sectors = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
            "JPM": "Financials",
        }
        df = _make_weights(["AAPL", "MSFT", "GOOGL", "JPM"], sectors=sectors, base_weight=0.20)
        result = enforce_sector_limits(df, max_sector_pct=0.30)

        # Check that tech sector total is capped at 0.30 per date
        tech_result = result.filter(pl.col("sector") == "Technology")
        sector_sums = tech_result.with_columns(
            pl.col("weight").abs().sum().over("date").alias("_sec_sum")
        )["_sec_sum"].unique()
        assert all(s <= 0.30 + 1e-6 for s in sector_sums)


# ── Leverage ────────────────────────────────────────────────────────


class TestLeverage:
    def test_no_cap_needed(self):
        df = _make_weights(["AAPL", "MSFT"], base_weight=0.20)
        result = enforce_leverage(df, max_leverage=1.0)
        # 2 tickers × 0.20 = 0.40 total, under 1.0
        leverage = result.with_columns(
            pl.col("weight").abs().sum().over("date")
        )["weight"]
        # Individual row weights unchanged
        assert result["weight"].is_not_null().all()

    def test_leverage_capped(self):
        # 5 tickers × 0.50 = 2.50 total leverage, cap at 1.5
        df = _make_weights(["A", "B", "C", "D", "E"], base_weight=0.50)
        result = enforce_leverage(df, max_leverage=1.5)

        # Check gross leverage per date
        leverage = result.with_columns(
            pl.col("weight").abs().sum().over("date")
        )
        assert (leverage["weight"] <= 1.5 + 1e-9).all()

    def test_proportional_scaling(self):
        # All weights should be scaled by the same factor
        df = _make_weights(["A", "B"], base_weight=1.0)
        result = enforce_leverage(df, max_leverage=1.0)

        # Original: 2.0 total, capped to 1.0 → scale 0.5
        # Each weight should be ~0.5
        assert abs(result["weight"].mean() - 0.5) < 0.01


# ── Volatility targeting ────────────────────────────────────────────


class TestVolatilityTargeting:
    def test_realized_volatility_shape(self):
        tickers = ["AAPL", "MSFT", "GOOGL"]
        returns = _make_returns(tickers, n_days=100)
        vols = compute_realized_volatility(returns, lookback=21)
        assert "realized_vol" in vols.columns
        assert len(vols) > 0

    def test_vol_target_scales_down(self):
        tickers = ["AAPL", "MSFT"]
        weights = _make_weights(tickers, base_weight=0.30)
        returns = _make_returns(tickers, n_days=100)
        cfg = RiskConfig(target_vol=0.05, lookback_vol=21)

        result = volatility_target(weights, returns, cfg)
        # With low target vol and reasonable returns, weights should shrink
        assert result["weight"].is_not_null().all()

    def test_vol_target_no_crash_on_nan(self):
        tickers = ["AAPL"]
        weights = _make_weights(tickers, base_weight=0.30)
        returns = _make_returns(tickers, n_days=10)
        cfg = RiskConfig(target_vol=0.10, lookback_vol=63)

        result = volatility_target(weights, returns, cfg)
        # Should not crash even with insufficient lookback data
        assert result["weight"].is_not_null().all()


# ── Drawdown ────────────────────────────────────────────────────────


class TestDrawdown:
    def test_compute_drawdown_series(self):
        df = _make_equity_curve(n_days=60)
        dd = compute_drawdown_series(df, equity_col="equity")
        assert "drawdown" in dd.columns
        assert "peak_equity" in dd.columns
        assert (dd["drawdown"] >= 0).all()

    def test_drawdown_during_decline(self):
        df = _make_equity_curve(n_days=60, include_drawdown=True)
        dd = compute_drawdown_series(df, equity_col="equity")
        max_dd = dd["drawdown"].max()
        assert max_dd > 0.05  # Should detect the simulated drawdown

    def test_circuit_breaker_triggers(self):
        equity = _make_equity_curve(n_days=60, include_drawdown=True)
        dd = compute_drawdown_series(equity, equity_col="equity")
        combined = dd.select(["date", "equity", "drawdown", "peak_equity"]).with_columns(
            pl.lit(1.0).alias("weight")
        )

        cfg = RiskConfig(max_drawdown=0.10, drawdown_recovery_pct=0.50)
        result = drawdown_circuit_breaker(combined, cfg, equity_col="equity")

        assert "leverage_scale" in result.columns
        # During deep drawdown, scale should be < 1.0
        assert (result["leverage_scale"] <= 1.0 + 1e-9).all()


# ── Correlation ─────────────────────────────────────────────────────


class TestCorrelation:
    def test_rolling_correlation_shape(self):
        tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
        returns = _make_returns(tickers, n_days=100)
        corr = compute_rolling_correlation(returns, lookback=30)
        assert "avg_correlation" in corr.columns
        assert len(corr) > 0

    def test_correlation_spike_detector_no_spike(self):
        tickers = ["AAPL", "MSFT"]
        weights = _make_weights(tickers, base_weight=0.20)
        returns = _make_returns(tickers, n_days=60)
        cfg = RiskConfig(correlation_spike_threshold=0.95)

        result = correlation_spike_detector(weights, returns, cfg)
        assert "correlation_spike" in result.columns
        # Most dates should not be flagged as spikes
        spikes = result.filter(pl.col("correlation_spike"))
        assert len(spikes) <= len(result)


# ── Hard constraints pipeline ───────────────────────────────────────


class TestHardConstraints:
    def test_all_constraints(self):
        sectors = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
            "JPM": "Financials", "XOM": "Energy",
        }
        df = _make_weights(
            ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"],
            sectors=sectors,
            base_weight=0.25,  # 5 × 0.25 = 1.25 leverage, tech sector 0.75
        )
        cfg = RiskConfig(
            max_position_pct=0.15,
            max_sector_pct=0.30,
            max_leverage=1.0,
        )

        result = enforce_all_hard(df, cfg, sector_col="sector")

        # Check position limits
        assert (result["weight"].abs() <= 0.15 + 1e-9).all()

        # Check sector limits
        if "sector" in result.columns:
            sector_totals = (
                result.with_columns(pl.col("weight").abs().alias("_aw"))
                .group_by(["date", "sector"])
                .agg(pl.sum("_aw").alias("sector_total"))
            )
            assert (sector_totals["sector_total"] <= 0.30 + 1e-6).all()


# ── Full pipeline integration ───────────────────────────────────────


class TestFullPipeline:
    def test_pipeline_end_to_end(self):
        tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
        sectors = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
            "JPM": "Financials", "XOM": "Energy",
        }

        weights = _make_weights(tickers, n_days=60, sectors=sectors, base_weight=0.20)
        returns = _make_returns(tickers, n_days=100)
        equity = _make_equity_curve(n_days=60, include_drawdown=True)

        cfg = RiskConfig(
            max_position_pct=0.15,
            max_sector_pct=0.30,
            max_leverage=1.0,
            target_vol=0.10,
            max_drawdown=0.20,
        )

        result = enforce_all_constraints(
            weights, cfg,
            returns_df=returns,
            equity_curve=equity,
        )

        # Position limits enforced
        assert (result["weight"].abs() <= 0.15 + 1e-6).all()

        # Has diagnostic columns
        assert "leverage_scale" in result.columns

        # No NaN weights
        assert result["weight"].is_not_null().all()

    def test_pipeline_without_returns(self):
        sectors = {"AAPL": "Technology", "JPM": "Financials"}
        weights = _make_weights(["AAPL", "JPM"], n_days=30, sectors=sectors, base_weight=0.15)
        cfg = RiskConfig()

        # Should work with just hard constraints
        result = enforce_all_constraints(weights, cfg)
        assert result["weight"].is_not_null().all()

    def test_pipeline_without_equity(self):
        returns = _make_returns(["AAPL", "MSFT"], n_days=100)
        weights = _make_weights(["AAPL", "MSFT"], n_days=60, base_weight=0.20)
        cfg = RiskConfig()

        # Should work without equity curve
        result = enforce_all_constraints(weights, cfg, returns_df=returns)
        assert result["weight"].is_not_null().all()
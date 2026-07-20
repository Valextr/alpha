"""End-to-end portfolio pipeline.

Wires together:
    1. Kelly criterion position sizing (signal-driven weights)
    2. Risk management pipeline (vol targeting, drawdown, correlation, hard caps)
    3. Portfolio construction (risk parity, HRP, Ledoit-Wolf)

The pipeline supports two modes:

    - ``kelly`` mode: Kelly criterion produces initial weights, then risk
      management constraints are applied. Use this for signal-driven
      alpha strategies.

    - ``construction`` mode: portfolio construction method (risk parity,
      HRP) produces structural weights, then Kelly signal scores modulate
      the direction/magnitude. Risk management is applied last.

Typical usage:

    from src.portfolio.integration import PortfolioPipeline

    # Kelly mode
    pipeline = PortfolioPipeline(mode="kelly")
    positions = pipeline.run(
        signals=signals_df,
        returns=returns_df,
        prices=prices_df,
        equity_curve=equity_df,
    )

    # Construction mode (risk parity)
    pipeline = PortfolioPipeline(mode="construction", construction_method="risk_parity")
    positions = pipeline.run(signals=signals_df, returns=returns_df, equity_curve=equity_df)

Output:
    A DataFrame with columns:
        ticker, date, weight (final), signal_value, kelly_fraction,
        position_weight_raw, vol_scale, drawdown_scale, correlation_scale,
        constraint_adjustments, portfolio_exposure, method
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from .kelly import KellyConfig, compute_position_weights, compute_position_weights_from_signal_scores
from .construction import (
    risk_parity_weights,
    hrp_weights,
    pivot_returns,
)
from ..risk import RiskConfig, enforce_all_constraints

# Core columns the risk pipeline preserves
_RISK_CORE_COLS = {"ticker", "date", "weight", "sector"}


@dataclass
class PipelineConfig:
    """Configuration for the portfolio pipeline.

    Combines KellyConfig, RiskConfig, and construction method into one
    interface. All defaults match the individual module defaults.

    Args:
        mode: Pipeline mode. ``"kelly"`` for signal-driven sizing,
            ``"construction"`` for structural allocation.
        construction_method: Method for construction mode
            (``"risk_parity"``, ``"hrp"``).
        kelly: Kelly criterion configuration.
        risk: Risk management configuration.
        signal_cols: Signal column names to use. If mode is "kelly" and
            this is a list, signals are averaged before sizing.
        returns_lookback: Number of trading days for rolling returns
            in construction mode.
    """

    mode: Literal["kelly", "construction"] = "kelly"
    construction_method: Literal["risk_parity", "hrp"] = "risk_parity"
    kelly: KellyConfig = field(default_factory=KellyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    signal_cols: list[str] | None = None
    returns_lookback: int = 63


class PortfolioPipeline:
    """End-to-end portfolio pipeline: Kelly → Risk → Construction.

    Args:
        config: Pipeline configuration. If None, uses PipelineConfig() defaults.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(
        self,
        *,
        signals: pl.DataFrame,
        returns: pl.DataFrame | None = None,
        prices: pl.DataFrame | None = None,
        equity_curve: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Run the full pipeline.

        Args:
            signals: DataFrame with at minimum ``ticker``, ``date``, and
                signal columns. May include forward return columns for
                Kelly edge estimation.
            returns: DataFrame with ``ticker``, ``date``, ``return`` columns.
                Required for risk management (vol targeting, correlation).
            prices: DataFrame with ``ticker``, ``date``, ``close`` columns.
                Required for construction mode (used to compute returns).
            equity_curve: DataFrame with ``date``, ``equity`` columns.
                Optional; enables drawdown circuit breaker.

        Returns:
            DataFrame with final position weights and diagnostic columns.
        """
        if self.config.mode == "kelly":
            return self._run_kelly(
                signals=signals,
                returns=returns,
                equity_curve=equity_curve,
            )
        else:
            return self._run_construction(
                signals=signals,
                returns=returns,
                prices=prices,
                equity_curve=equity_curve,
            )

    # ------------------------------------------------------------------
    # Kelly mode
    # ------------------------------------------------------------------

    def _run_kelly(
        self,
        *,
        signals: pl.DataFrame,
        returns: pl.DataFrame | None = None,
        equity_curve: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Kelly mode: Kelly sizing → risk management."""

        # Step 1: Kelly sizing
        if self.config.signal_cols and len(self.config.signal_cols) > 1:
            weights = compute_position_weights_from_signal_scores(
                signals, self.config.signal_cols, self.config.kelly
            )
        elif self.config.signal_cols:
            weights = compute_position_weights(
                signals,
                self.config.signal_cols[0],
                f"forward_return_{self.config.kelly.forward_horizon}",
                self.config.kelly,
            )
        else:
            # Auto-detect: find columns matching 'signal_*' or 'ensemble_*'
            sig_cols = [c for c in signals.columns if c.startswith(("signal_", "ensemble_"))]
            if not sig_cols:
                raise ValueError(
                    "No signal columns found. Provide signal_cols in PipelineConfig "
                    f"or use columns prefixed with 'signal_' or 'ensemble_'. "
                    f"Available: {signals.columns}"
                )
            if len(sig_cols) > 1:
                weights = compute_position_weights_from_signal_scores(
                    signals, sig_cols, self.config.kelly
                )
            else:
                weights = compute_position_weights(
                    signals,
                    sig_cols[0],
                    f"forward_return_{self.config.kelly.forward_horizon}",
                    self.config.kelly,
                )

        # Rename position_weight to weight for the risk pipeline
        weights = weights.rename({"position_weight": "weight"})

        # Step 2: Risk management (preserve extra columns around the call)
        result = self._apply_risk_with_preservation(
            weights,
            returns_df=returns,
            equity_curve=equity_curve,
        )

        # Add method tag
        return result.with_columns(pl.lit("kelly").alias("method"))

    # ------------------------------------------------------------------
    # Construction mode
    # ------------------------------------------------------------------

    def _run_construction(
        self,
        *,
        signals: pl.DataFrame,
        returns: pl.DataFrame | None = None,
        prices: pl.DataFrame | None = None,
        equity_curve: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Construction mode: structural allocation → signal modulation → risk."""

        # Compute returns if not provided
        returns_df = self._get_returns_array(signals, returns, prices)

        # Step 1: Portfolio construction
        weight_array, weight_tickers = self._compute_construction_weights(returns_df)

        # Step 2: Build weight DataFrame from construction weights
        tickers = signals["ticker"].unique().to_list()
        dates = signals["date"].sort().to_list()

        # Rebalance every N days
        rebalance_interval = self.config.returns_lookback
        weight_rows = []
        for i in range(0, len(dates), rebalance_interval):
            period_dates = dates[i : i + rebalance_interval]
            for j, w in enumerate(weight_array):
                ticker = weight_tickers[j] if j < len(weight_tickers) else tickers[j] if j < len(tickers) else f"ticker_{j}"
                for d in period_dates:
                    weight_rows.append({"ticker": ticker, "date": d, "weight": float(w)})

        weights_df = pl.DataFrame(weight_rows)

        # Step 3: Modulate with signal direction
        if self.config.signal_cols:
            sig_cols = self.config.signal_cols
        else:
            sig_cols = [c for c in signals.columns if c.startswith(("signal_", "ensemble_"))]

        if sig_cols:
            signal_expr = pl.col(sig_cols[0])
            for c in sig_cols[1:]:
                signal_expr = signal_expr + pl.col(c)
            signal_expr = signal_expr / len(sig_cols)
            signal_agg = signals.select([
                "ticker", "date",
                signal_expr.alias("signal_value"),
            ])
            weights_df = weights_df.join(signal_agg, on=["ticker", "date"], how="left")
            weights_df = weights_df.with_columns(
                (
                    pl.when(pl.col("signal_value") > 0)
                    .then(pl.col("weight"))
                    .when(pl.col("signal_value") < 0)
                    .then(pl.col("weight") * 0.5)
                    .otherwise(pl.col("weight"))
                ).alias("weight"),
                pl.col("signal_value").fill_null(0.0),
            )

        # Step 4: Risk management
        result = self._apply_risk_with_preservation(
            weights_df,
            returns_df=returns,
            equity_curve=equity_curve,
        )

        return result.with_columns(pl.lit(self.config.construction_method).alias("method"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_risk_with_preservation(
        self,
        weights: pl.DataFrame,
        *,
        returns_df: pl.DataFrame | None = None,
        equity_curve: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Run risk pipeline, preserving extra columns the pipeline would drop.

        The risk pipeline (vol_targeting, correlation) hard-selects only
        [ticker, date, weight] + risk diagnostics, so any additional columns
        (e.g. signal_value, kelly_fraction, capped_fraction) get lost.

        Workaround: extract extra columns before the pipeline, rejoin after.
        """
        # Identify extra columns beyond the risk core
        extra_cols = [c for c in weights.columns if c not in _RISK_CORE_COLS]

        if extra_cols:
            # Extract key + extras before risk pipeline
            extras = weights.select(["ticker", "date"] + extra_cols).unique()
            risk_input = weights.select(["ticker", "date", "weight"] +
                                        (["sector"] if "sector" in weights.columns else []))
        else:
            extras = None
            risk_input = weights

        # Run risk pipeline on core columns only
        result = enforce_all_constraints(
            risk_input,
            self.config.risk,
            returns_df=returns_df,
            equity_curve=equity_curve,
            weight_col="weight",
        )

        # Rejoin extras
        if extras is not None:
            result = result.join(extras, on=["ticker", "date"], how="left")

        return result

    def _get_returns_array(
        self,
        signals: pl.DataFrame,
        returns_df: pl.DataFrame | None,
        prices_df: pl.DataFrame | None,
    ) -> pl.DataFrame:
        """Get returns data for construction methods.

        Priority: returns_df > signals with forward returns > prices.
        """
        if returns_df is not None:
            return returns_df

        forward_cols = [c for c in signals.columns if c.startswith("forward_return_")]
        if forward_cols:
            return signals.select(["ticker", "date", pl.col(forward_cols[0]).alias("return")])

        if prices_df is not None:
            return prices_df.with_columns(
                (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("return")
            )

        raise ValueError(
            "No returns data available for construction mode. "
            "Provide returns, prices, or include forward_return columns in signals."
        )

    def _compute_construction_weights(
        self,
        returns_df: pl.DataFrame,
    ) -> tuple:
        """Compute structural weights from returns data.

        Returns:
            (weight_array, ticker_list) tuple.
        """
        returns_matrix, tickers = pivot_returns(returns_df)

        if self.config.construction_method == "hrp":
            weights = hrp_weights(returns_matrix)
        else:
            weights = risk_parity_weights(returns_matrix)

        return weights, tickers

    # ------------------------------------------------------------------
    # Portfolio analytics
    # ------------------------------------------------------------------

    def analyze(
        self,
        positions: pl.DataFrame,
        returns: pl.DataFrame | None = None,
    ) -> dict:
        """Compute portfolio-level analytics on the output positions.

        Args:
            positions: Output from ``run()``.
            returns: Per-ticker daily returns for volatility calculation.

        Returns:
            Dictionary with portfolio_exposure, num_positions, num_dates,
            mean_weight, weight_std, and (if returns provided) portfolio_vol
            and portfolio_sharpe.
        """
        stats = positions.with_columns([
            pl.col("weight").abs().sum().over("date").alias("_exposure"),
            pl.col("weight").is_not_null().sum().over("date").alias("_positions"),
        ]).select([
            pl.col("_exposure").mean().alias("portfolio_exposure_mean"),
            pl.col("_exposure").max().alias("portfolio_exposure_max"),
            pl.col("_positions").mean().alias("num_positions_mean"),
            pl.col("date").n_unique().alias("num_dates"),
            pl.col("weight").mean().alias("mean_weight"),
            pl.col("weight").std().alias("weight_std"),
        ])

        result = stats.to_dict(as_series=False)

        if returns is not None:
            port_return = positions.join(returns, on=["ticker", "date"], how="inner").group_by("date").agg(
                (pl.col("weight") * pl.col("return")).sum().alias("port_return")
            )
            if len(port_return) > 1:
                mean_ret = float(port_return["port_return"].mean())
                std_ret = float(port_return["port_return"].std())
                annual_vol = std_ret * (252**0.5)
                annual_ret = mean_ret * 252
                sharpe = annual_ret / annual_vol if annual_vol > 0 else 0.0
                result["portfolio_vol_annual"] = annual_vol
                result["portfolio_return_annual"] = annual_ret
                result["portfolio_sharpe"] = sharpe

        return result

    # ------------------------------------------------------------------
    # Multi-strategy comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        *,
        signals: pl.DataFrame,
        returns: pl.DataFrame | None = None,
        prices: pl.DataFrame | None = None,
        equity_curve: pl.DataFrame | None = None,
        modes: list[Literal["kelly", "construction"]] | None = None,
        construction_methods: list[Literal["risk_parity", "hrp"]] | None = None,
    ) -> dict[str, pl.DataFrame]:
        """Compare multiple portfolio construction approaches on the same data.

        Args:
            signals: Signal DataFrame.
            returns: Returns DataFrame.
            prices: Prices DataFrame.
            equity_curve: Equity curve for drawdown tracking.
            modes: Modes to compare (default: ["kelly", "construction"]).
            construction_methods: Methods for construction mode
                (default: ["risk_parity", "hrp"]).

        Returns:
            Dictionary mapping strategy name to positions DataFrame.
        """
        modes = modes or ["kelly", "construction"]
        construction_methods = construction_methods or ["risk_parity", "hrp"]

        results = {}

        for mode in modes:
            if mode == "kelly":
                cfg = PipelineConfig(
                    mode="kelly",
                    kelly=self.config.kelly,
                    risk=self.config.risk,
                    signal_cols=self.config.signal_cols,
                )
                results["kelly"] = PortfolioPipeline(cfg).run(
                    signals=signals,
                    returns=returns,
                    equity_curve=equity_curve,
                )
            else:
                for method in construction_methods:
                    cfg = PipelineConfig(
                        mode="construction",
                        construction_method=method,
                        kelly=self.config.kelly,
                        risk=self.config.risk,
                        signal_cols=self.config.signal_cols,
                        returns_lookback=self.config.returns_lookback,
                    )
                    results[method] = PortfolioPipeline(cfg).run(
                        signals=signals,
                        returns=returns,
                        prices=prices,
                        equity_curve=equity_curve,
                    )

        return results
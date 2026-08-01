"""Live trading runner — daily trading loop.

Pipeline:
    yfinance → gold data → features → signals → ensemble → portfolio targets
    → IBBroker → IBKR paper trading → monitoring → daily report

Usage:
    from src.live.runner import LiveRunner

    runner = LiveRunner()
    runner.run_day()  # execute one trading cycle
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, Optional

import polars as pl
import yfinance as yf

from src.ensemble.pipeline import EnsemblePipeline, EnsemblePipelineConfig
from src.execution.engine import ExecutionConfig, ExecutionEngine
from src.execution.ib_broker import IBBroker
from src.execution.monitoring_bridge import MonitoringBridge
from src.execution.models import Position as ExecPosition
from src.features.pipeline import compute_features
from src.live.config import LiveConfig
from src.signals.pipeline import generate_all

logger = logging.getLogger(__name__)


@dataclass
class DayResult:
    """Result of a single trading day."""

    date: date
    tickers_traded: list[str] = field(default_factory=list)
    orders_placed: int = 0
    orders_filled: int = 0
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    drawdown: float = 0.0
    positions: dict = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)
    error: Optional[str] = None


class LiveRunner:
    """Daily live trading runner.

    Connects to IBKR paper trading, fetches market data via yfinance,
    runs the full signal pipeline, and executes trades.

    Args:
        config: Live trading configuration.
    """

    def __init__(self, config: Optional[LiveConfig] = None):
        self.config = config or LiveConfig()
        self.engine: Optional[ExecutionEngine] = None
        self.broker: Optional[IBBroker] = None
        self.monitoring: Optional[MonitoringBridge] = None
        self._connected: bool = False

    def connect(self) -> None:
        """Connect to IBKR and initialize the execution pipeline."""
        logger.info("Connecting to IBKR...")

        # Create IBBroker (reads .env for credentials)
        self.broker = IBBroker()
        self.broker.connect()

        # Create monitoring bridge
        self.monitoring = MonitoringBridge(
            initial_balance=self.broker.get_portfolio_value(),
        )

        # Create execution engine with IB broker
        exec_config = ExecutionConfig(
            max_positions=self.config.max_positions,
            max_position_weight=self.config.max_position_weight,
            daily_loss_limit_pct=self.config.daily_loss_limit_pct,
            max_drawdown_pct=self.config.max_drawdown_pct,
        )
        self.engine = ExecutionEngine(
            exec_config,
            broker=self.broker,
            monitoring_bridge=self.monitoring,
        )
        self.engine.start()

        self._connected = True
        logger.info(
            f"Connected. Portfolio value: ${self.engine.get_portfolio_value():,.2f}"
        )

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self.engine:
            self.engine.stop()
        self._connected = False
        logger.info("Disconnected from IBKR")

    def fetch_data(self, end_date: Optional[date] = None) -> pl.DataFrame:
        """Fetch latest market data via yfinance and merge with gold data.

        Args:
            end_date: Override the end date (defaults to today).

        Returns:
            Gold data DataFrame with latest prices appended.
        """
        if end_date is None:
            end_date = date.today()

        logger.info(f"Fetching data for {self.config.tickers}...")

        # Load existing gold data
        gold_dir = self.config.data_dir / "gold" / "daily"
        files = sorted(gold_dir.rglob("*.parquet"))
        if files:
            existing = pl.concat(
                [pl.read_parquet(str(f)) for f in files], how="vertical_relaxed"
            )
        else:
            existing = pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "date": pl.Date,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Int64,
                    "vwap": pl.Float64,
                    "adj_close": pl.Float64,
                    "split_factor": pl.Float64,
                    "dividend_yield": pl.Float64,
                    "is_market_date": pl.Boolean,
                    "data_quality": pl.Utf8,
                    "sector": pl.Utf8,
                    "avg_volume_20d": pl.Float64,
                    "avg_volume_60d": pl.Float64,
                    "volume_ratio": pl.Float64,
                    "market_cap_bucket": pl.Utf8,
                    "universe_date": pl.Date,
                    "source": pl.Utf8,
                }
            )

        # Fetch latest data from yfinance
        new_rows: list[pl.DataFrame] = []
        for ticker in self.config.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(
                    start=(end_date - timedelta(days=30)).isoformat(),
                    end=end_date.isoformat(),
                    auto_adjust=False,
                )
                if hist.empty:
                    logger.warning(f"No data for {ticker}")
                    continue

                df = pl.from_pandas(hist.reset_index())
                # Rename date column
                for candidate in ["Date", "Dates", "timestamp", "Datetime"]:
                    if candidate in df.columns:
                        df = df.rename({candidate: "date"})
                        break

                # Standardize columns
                for src, dst in [
                    ("Open", "open"), ("High", "high"), ("Low", "low"),
                    ("Close", "close"), ("Volume", "volume"),
                ]:
                    if src in df.columns:
                        df = df.rename({src: dst})

                if "ticker" not in df.columns:
                    df = df.with_columns(pl.lit(ticker).alias("ticker"))

                # Ensure date is Date type
                if "date" in df.columns and df["date"].dtype != pl.Date:
                    df = df.with_columns(
                        pl.col("date").cast(pl.Datetime).dt.date().alias("date")
                    )

                # Select only the core columns we have from yfinance
                core_cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
                existing_cols = [c for c in core_cols if c in df.columns]
                row = df.select(existing_cols)

                # Add default values for enrichment columns
                row = row.with_columns(
                    pl.lit(None, dtype=pl.Float64).alias("vwap"),
                    pl.lit(None, dtype=pl.Float64).alias("adj_close"),
                    pl.lit(1.0, dtype=pl.Float64).alias("split_factor"),
                    pl.lit(0.0, dtype=pl.Float64).alias("dividend_yield"),
                    pl.lit(True, dtype=pl.Boolean).alias("is_market_date"),
                    pl.lit("ok", dtype=pl.Utf8).alias("data_quality"),
                    pl.lit("").alias("sector"),
                    pl.lit(0.0, dtype=pl.Float64).alias("avg_volume_20d"),
                    pl.lit(0.0, dtype=pl.Float64).alias("avg_volume_60d"),
                    pl.lit(0.0, dtype=pl.Float64).alias("volume_ratio"),
                    pl.lit("large", dtype=pl.Utf8).alias("market_cap_bucket"),
                    pl.lit(True, dtype=pl.Boolean).alias("universe_date"),
                    pl.lit("yfinance", dtype=pl.Utf8).alias("source"),
                )

                # Reorder to match gold schema exactly
                gold_cols = ["close", "date", "high", "low", "open", "source", "ticker", "volume",
                             "vwap", "adj_close", "split_factor", "dividend_yield",
                             "is_market_date", "data_quality", "sector",
                             "avg_volume_20d", "avg_volume_60d", "volume_ratio",
                             "market_cap_bucket", "universe_date"]
                row = row.select(gold_cols)

                new_rows.append(row)
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")

        if new_rows:
            new_data = pl.concat(new_rows, how="vertical_relaxed")
            # Merge: keep existing rows, add new ones (avoid duplicates)
            if len(existing) > 0:
                existing_dates = existing.select(["ticker", "date"]).unique()
                new_data = new_data.join(
                    existing_dates,
                    on=["ticker", "date"],
                    how="anti",
                )
                merged = pl.concat([existing, new_data], how="vertical_relaxed")
            else:
                merged = new_data
        else:
            merged = existing

        return merged.sort(["ticker", "date"])

    def compute_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Run the full signal pipeline: features → signals → ensemble.

        Args:
            df: Gold data DataFrame.

        Returns:
            DataFrame with ensemble signal column.
        """
        logger.info("Computing features...")
        enriched = compute_features(df)

        logger.info("Generating signals...")
        signals = generate_all(enriched)

        logger.info(f"Running ensemble ({self.config.ensemble_mode})...")
        ensemble_config = EnsemblePipelineConfig(
            mode=self.config.ensemble_mode,
            ic_lookback=self.config.ic_lookback,
            ic_target_horizon=self.config.ic_target_horizon,
            weight_method="positive_ic",
        )
        pipeline = EnsemblePipeline(ensemble_config)
        result = pipeline.run(signals)

        return result

    def generate_targets(self, signal_df: pl.DataFrame) -> dict[str, float]:
        """Generate portfolio targets from ensemble signals.

        Args:
            signal_df: DataFrame with ensemble signal column.

        Returns:
            Dict mapping ticker -> target_weight.
        """
        if self.engine is None:
            logger.error("Engine not initialized")
            return {}

        # Determine the ensemble output column name
        if self.config.ensemble_mode == "ic_weighted":
            signal_col = "signal_ensemble"
        else:
            signal_col = "ensemble_prediction"

        if signal_col not in signal_df.columns:
            logger.warning(f"Signal column '{signal_col}' not found. Available: {signal_df.columns}")
            # Try to find any ensemble-like column
            for col in signal_df.columns:
                if "ensemble" in col.lower():
                    signal_col = col
                    break
            else:
                logger.error("No ensemble signal column found")
                return {}

        return self.engine.generate_targets_from_signals(
            signal_df,
            signal_column=signal_col,
            top_n=self.config.top_n,
            min_signal_strength=self.config.min_signal_strength,
        )

    def execute_targets(self, targets: dict[str, float]) -> list:
        """Execute portfolio targets via the engine.

        Args:
            targets: Dict mapping ticker -> target_weight.

        Returns:
            List of orders placed.
        """
        if self.engine is None:
            logger.error("Engine not initialized")
            return []

        if not targets:
            logger.info("No targets to execute")
            return []

        logger.info(f"Executing targets for {len(targets)} tickers: {targets}")
        orders = self.engine.execute_targets(targets)
        return orders

    def record_eod(self) -> dict:
        """Record end-of-day metrics.

        Returns:
            Dict with EOD metrics.
        """
        if self.engine is None:
            return {}

        snapshot = self.engine.get_snapshot()
        portfolio_value = self.engine.get_portfolio_value()

        metrics = {
            "portfolio_value": portfolio_value,
            "num_positions": snapshot.num_positions,
            "drawdown": snapshot.drawdown,
            "daily_pnl": snapshot.daily_pnl,
            "num_orders_today": snapshot.num_orders_today,
        }

        if self.monitoring:
            # Update monitoring from engine state
            fills = self.broker.get_fills() if self.broker else []
            self.monitoring.update_from_engine(
                positions=snapshot.positions,
                fills=fills,
                portfolio_value=portfolio_value,
                daily_pnl=snapshot.daily_pnl,
                peak_equity=self.engine._peak_equity,
            )

        logger.info(f"EOD: value=${portfolio_value:,.2f}, drawdown={snapshot.drawdown:.2%}")
        return metrics

    def generate_report(self, result: DayResult) -> str:
        """Generate a daily report.

        Args:
            result: DayResult from the trading cycle.

        Returns:
            Report path.
        """
        report = {
            "date": str(result.date),
            "portfolio_value": result.portfolio_value,
            "daily_pnl": result.daily_pnl,
            "drawdown": result.drawdown,
            "orders_placed": result.orders_placed,
            "orders_filled": result.orders_filled,
            "tickers_traded": result.tickers_traded,
            "positions": {
                k: {
                    "quantity": v.quantity,
                    "avg_cost": v.avg_cost,
                    "current_price": v.current_price,
                    "unrealized_pnl": v.unrealized_pnl,
                }
                for k, v in result.positions.items()
            },
            "alerts": result.alerts,
        }

        if result.error:
            report["error"] = result.error

        report_path = self.config.report_dir / f"day_{result.date}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to {report_path}")
        return str(report_path)

    def run_day(self, end_date: Optional[date] = None) -> DayResult:
        """Run a single trading day.

        Full pipeline: fetch data → compute signals → execute → report.

        Args:
            end_date: Override the trading date (defaults to today).

        Returns:
            DayResult with all metrics.
        """
        if end_date is None:
            end_date = date.today()

        result = DayResult(date=end_date)

        try:
            # Step 1: Fetch data
            df = self.fetch_data(end_date)
            logger.info(f"Loaded {len(df)} rows of data")

            # Step 2: Compute signals
            signal_df = self.compute_signals(df)

            # Step 3: Generate targets
            targets = self.generate_targets(signal_df)
            result.tickers_traded = list(targets.keys())

            # Step 4: Execute
            if targets:
                orders = self.execute_targets(targets)
                result.orders_placed = len(orders)
                result.orders_filled = sum(
                    1 for o in orders if o.status.value == "filled"
                )
            else:
                logger.info("No targets generated — no trades")

            # Step 5: Record EOD
            metrics = self.record_eod()
            result.portfolio_value = metrics.get("portfolio_value", 0.0)
            result.daily_pnl = metrics.get("daily_pnl", 0.0)
            result.drawdown = metrics.get("drawdown", 0.0)

            if self.engine:
                result.positions = self.engine.get_positions()

            # Step 6: Get alerts
            if self.monitoring:
                alerts = self.monitoring.get_alerts()
                result.alerts = [str(a) for a in alerts]

            # Step 7: Generate report
            report_path = self.generate_report(result)
            logger.info(f"Day complete. Report: {report_path}")

        except Exception as e:
            result.error = str(e)
            logger.error(f"Error during trading day: {e}", exc_info=True)

        return result

    def run(self, end_date: Optional[date] = None) -> DayResult:
        """Full trading cycle: connect → run_day → disconnect.

        Args:
            end_date: Override the trading date.

        Returns:
            DayResult with all metrics.
        """
        try:
            self.connect()
            return self.run_day(end_date)
        finally:
            self.disconnect()
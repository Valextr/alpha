#!/usr/bin/env python
"""Backtest runner — simulates trading on historical data.

Usage:
    uv run python backtest.py              # Default: 2025-01-01 to 2025-12-31
    uv run python backtest.py --start 2024-01-01 --end 2024-12-31
    uv run python backtest.py --days 60    # Last 60 trading days
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.ensemble.pipeline import EnsemblePipeline, EnsemblePipelineConfig
from src.execution.broker import PaperBroker
from src.execution.engine import ExecutionConfig, ExecutionEngine
from src.execution.monitoring_bridge import MonitoringBridge
from src.features.pipeline import compute_features
from src.live.config import LiveConfig
from src.signals.pipeline import generate_all

logging.basicConfig(
    level=logging.WARNING,  # Quiet — only show errors
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpha backtest runner")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date")
    parser.add_argument("--days", type=int, default=None, help="Last N trading days")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0, help="Starting cash")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def load_gold_data(data_dir: Path, start: date, end: date) -> pl.DataFrame:
    """Load gold data for the date range."""
    gold_dir = data_dir / "gold" / "daily"
    files = sorted(gold_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No gold data at {gold_dir}")

    df = pl.concat([pl.read_parquet(str(f)) for f in files], how="vertical_relaxed")
    df = df.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    return df.sort(["ticker", "date"])


def run_backtest(config: LiveConfig, start: date, end: date, initial_cash: float = 1_000_000.0) -> pl.DataFrame:
    """Run the full backtest pipeline."""
    data_dir = config.data_dir

    # Load data
    print(f"Loading data [{start} to {end}]...")
    df = load_gold_data(data_dir, start, end)
    tickers = df["ticker"].unique().to_list()
    dates = df["date"].unique().sort().to_list()
    print(f"  {len(tickers)} tickers, {len(dates)} days, {len(df)} rows")

    # Compute features and signals ONCE for the full period
    print("Computing features...")
    enriched = compute_features(df)
    print("Generating signals...")
    signals = generate_all(enriched)
    print("Running ensemble...")
    ensemble_config = EnsemblePipelineConfig(
        mode=config.ensemble_mode,
        ic_lookback=config.ic_lookback,
        ic_target_horizon=config.ic_target_horizon,
    )
    pipeline = EnsemblePipeline(ensemble_config)
    result = pipeline.run(signals)

    # Build price source for PaperBroker from gold data
    price_source = {}
    for ticker in tickers:
        ticker_df = result.filter(pl.col("ticker") == ticker).select(["date", "close"])
        ticker_df = ticker_df.with_columns(pl.col("close").alias("price"))
        price_source[ticker] = ticker_df

    # Set up paper broker with historical prices
    broker = PaperBroker(
        initial_cash=initial_cash,
        commission_per_share=0.005,
        slippage_bps=10.0,
        price_source=price_source,
    )

    # Set up execution engine
    exec_config = ExecutionConfig(
        initial_cash=initial_cash,
        max_positions=config.max_positions,
        max_position_weight=config.max_position_weight,
        daily_loss_limit_pct=config.daily_loss_limit_pct,
        max_drawdown_pct=config.max_drawdown_pct,
    )
    monitoring = MonitoringBridge(initial_balance=initial_cash)
    engine = ExecutionEngine(exec_config, broker=broker, monitoring_bridge=monitoring)
    engine.start()

    # Trade day by day
    print("Trading...")
    equity_curve = []
    signal_col = "signal_ensemble" if config.ensemble_mode == "ic_weighted" else "ensemble_prediction"

    for i, trade_date in enumerate(dates):
        # Get data up to and including this date (point-in-time correct)
        day_signals = result.filter(pl.col("date") == trade_date)

        if day_signals.is_empty():
            continue

        # Generate targets
        targets = engine.generate_targets_from_signals(
            day_signals,
            signal_column=signal_col,
            top_n=config.top_n,
            min_signal_strength=config.min_signal_strength,
        )

        # Execute with price source
        if targets:
            engine.execute_targets(targets, price_source={
                t: price_source.get(t, pl.DataFrame()) for t in targets
            })

        # Record EOD
        portfolio_value = broker.get_portfolio_value()
        positions = broker.get_positions()
        daily_pnl = broker.record_eod()

        equity_curve.append({
            "date": trade_date,
            "portfolio_value": float(portfolio_value),
            "daily_pnl": float(daily_pnl),
            "num_positions": len(positions),
            "orders_today": engine._num_orders_today,
        })

        # Reset daily counters
        engine._num_orders_today = 0
        engine._num_trades_today = 0

        if i % 50 == 0 and i > 0:
            print(f"  Day {i}/{len(dates)}: ${portfolio_value:,.2f}")

    engine.stop()

    return pl.DataFrame(equity_curve)


def print_summary(equity: pl.DataFrame, initial_cash: float) -> None:
    """Print backtest summary."""
    if equity.is_empty():
        print("No trades executed.")
        return

    final_value = float(equity["portfolio_value"].max())
    total_return = (final_value - initial_cash) / initial_cash * 100

    # Calculate metrics
    daily_returns = equity["daily_pnl"] / equity.shift(1)["portfolio_value"].fill_null(initial_cash)
    total_days = len(equity)
    total_orders = int(equity["orders_today"].sum())

    print("\n" + "=" * 60)
    print("Backtest Summary")
    print("=" * 60)
    print(f"Period:      {equity['date'].min()} to {equity['date'].max()} ({total_days} days)")
    print(f"Initial:     ${initial_cash:,.2f}")
    print(f"Final:       ${final_value:,.2f}")
    print(f"Total P&L:   ${final_value - initial_cash:+,.2f} ({total_return:+.2f}%)")
    print(f"Total Orders: {total_orders}")
    print(f"Max Positions: {int(equity['num_positions'].max())}")
    print("=" * 60)

    # Save equity curve
    output = Path("reports/live/backtest_equity.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    equity.write_csv(output)
    print(f"\nEquity curve saved to {output}")


def main() -> None:
    args = parse_args()

    config = LiveConfig.from_env()
    if args.tickers:
        config.tickers = [t.strip() for t in args.tickers.split(",")]

    if args.days:
        end = date.today()
        start = end - timedelta(days=args.days * 7 // 5)  # Approximate trading days
    else:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)

    print("=" * 60)
    print("Alpha Backtest Runner")
    print("=" * 60)
    print(f"Period: {start} to {end}")
    print(f"Tickers: {', '.join(config.tickers)}")
    print(f"Initial Cash: ${config.initial_cash if hasattr(config, 'initial_cash') else args.initial_cash:,.2f}")
    print("=" * 60)

    equity = run_backtest(config, start, end, args.initial_cash)
    print_summary(equity, args.initial_cash)


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Intraday backtest runner — simulates trading on 1-minute bar data.

Architecture:
    1. Load 1-minute bars from Alpaca parquet files
    2. Compute intraday features (returns, vol, VWAP, volume)
    3. Generate intraday signals (VWAP mean reversion)
    4. Trade bar-by-bar using PaperBroker

Usage:
    uv run python backtest_intraday.py --start 2025-01-01 --end 2025-12-31
    uv run python backtest_intraday.py --days 90
    uv run python backtest_intraday.py --tickers AAPL,MSFT
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.data.alpaca import download_intraday
from src.execution.broker import PaperBroker
from src.execution.engine import ExecutionConfig, ExecutionEngine
from src.execution.monitoring_bridge import MonitoringBridge
from src.features.intraday import compute_intraday_features
from src.signals.base import compute_forward_returns
from src.signals.vwap_reversion import generate_vwap_mean_reversion_60b

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpha intraday backtest runner")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date")
    parser.add_argument("--days", type=int, default=None, help="Last N trading days")
    parser.add_argument("--tickers", type=str, default="AAPL,AMZN,GOOGL,META,MSFT", help="Comma-separated tickers")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0, help="Starting cash")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def load_intraday_data(
    tickers: list[str],
    start: date,
    end: date,
    data_dir: Path,
) -> pl.DataFrame:
    """Load intraday 1-minute bars from parquet files.

    Downloads from Alpaca if data doesn't exist.
    """
    intraday_dir = data_dir / "intraday" / "1min"

    # Check if data exists
    existing_files = list(intraday_dir.rglob("*.parquet"))
    if existing_files:
        print(f"Loading {len(existing_files)} existing parquet files...")
        df = pl.concat([pl.read_parquet(str(f)) for f in existing_files], how="vertical_relaxed")
        df = df.filter(
            (pl.col("date") >= start) &
            (pl.col("date") <= end) &
            pl.col("ticker").is_in(tickers)
        )
    else:
        print(f"No existing data. Downloading from Alpaca...")
        download_intraday(tickers, start, end, data_dir=data_dir)
        existing_files = list(intraday_dir.rglob("*.parquet"))
        df = pl.concat([pl.read_parquet(str(f)) for f in existing_files], how="vertical_relaxed")
        df = df.filter(
            (pl.col("date") >= start) &
            (pl.col("date") <= end) &
            pl.col("ticker").is_in(tickers)
        )

    # Sort by ticker, datetime
    df = df.sort(["ticker", "datetime"])
    return df


def run_intraday_backtest(
    df: pl.DataFrame,
    initial_cash: float = 1_000_000.0,
) -> pl.DataFrame:
    """Run the intraday backtest pipeline.

    Trades bar-by-bar using VWAP mean reversion signals.
    """
    tickers = df["ticker"].unique().to_list()
    total_bars = len(df)
    print(f"  {len(tickers)} tickers, {total_bars} bars")

    # Compute features
    print("Computing intraday features...")
    enriched = compute_intraday_features(df)

    # Generate VWAP signal
    print("Generating VWAP mean reversion signals...")
    signals = generate_vwap_mean_reversion_60b(enriched)

    # Compute forward returns for validation
    print("Computing forward returns...")
    signals = compute_forward_returns(signals, horizons=[1, 5, 20])

    # Check signal quality
    signal_col = "signal_vwap_mean_reversion_60b"
    from src.signals.base import rank_ic
    ic_1b = rank_ic(signal_col, "forward_return_1", signals)
    ic_5b = rank_ic(signal_col, "forward_return_5", signals)
    ic_20b = rank_ic(signal_col, "forward_return_20", signals)
    print(f"  Signal IC: 1-bar={ic_1b:+.4f}, 5-bar={ic_5b:+.4f}, 20-bar={ic_20b:+.4f}")

    # Build price source for PaperBroker (per-ticker, per-datetime)
    print("Building price sources...")
    price_source = {}
    for ticker in tickers:
        ticker_df = signals.filter(pl.col("ticker") == ticker).select(["datetime", "close"])
        ticker_df = ticker_df.with_columns(pl.col("close").alias("price"))
        price_source[ticker] = ticker_df

    # Set up paper broker
    broker = PaperBroker(
        initial_cash=initial_cash,
        commission_per_share=0.005,
        slippage_bps=10.0,
        price_source=price_source,
    )

    # Set up execution engine (relaxed limits for intraday)
    exec_config = ExecutionConfig(
        initial_cash=initial_cash,
        max_positions=len(tickers),
        max_position_weight=0.20,  # 20% max per ticker
        daily_loss_limit_pct=1.0,  # Disable daily loss limit (100%)
        max_drawdown_pct=1.0,  # Disable drawdown limit (100%)
        min_cash_reserve=0.0,  # No cash reserve for intraday
    )
    monitoring = MonitoringBridge(initial_balance=initial_cash)
    engine = ExecutionEngine(exec_config, broker=broker, monitoring_bridge=monitoring)
    engine.start()

    # Trade bar-by-bar
    print("Trading bar-by-bar...")
    equity_curve = []
    signal_col = "signal_vwap_mean_reversion_60b"

    # Group by datetime for cross-sectional trading
    datetimes = signals["datetime"].unique().sort().to_list()

    for i, dt in enumerate(datetimes):
        # Get signals for this bar
        bar_signals = signals.filter(pl.col("datetime") == dt)

        if bar_signals.is_empty():
            continue

        # Generate targets from signals
        targets = engine.generate_targets_from_signals(
            bar_signals,
            signal_column=signal_col,
            top_n=len(tickers),
            min_signal_strength=0.1,  # Only trade if signal is meaningful
        )

        if targets:
            engine.execute_targets(targets, price_source={
                t: price_source.get(t, pl.DataFrame()) for t in targets
            })

        # Record equity (skip daily loss check for intraday)
        portfolio_value = broker.get_portfolio_value()
        positions = broker.get_positions()
        # Don't call record_eod() — it's designed for daily trading
        # Just track P&L manually
        prev_value = equity_curve[-1]["portfolio_value"] if equity_curve else initial_cash
        bar_pnl = portfolio_value - prev_value

        equity_curve.append({
            "datetime": dt,
            "portfolio_value": float(portfolio_value),
            "pnl": float(bar_pnl),
            "num_positions": len(positions),
        })

        if i % 10000 == 0 and i > 0:
            print(f"  Bar {i}/{len(datetimes)}: ${portfolio_value:,.2f}")

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
    total_bars = len(equity)
    total_pnl = float(equity["pnl"].sum())
    max_positions = int(equity["num_positions"].max())

    # Win rate
    winning_bars = int((equity["pnl"] > 0).sum())
    win_rate = winning_bars / total_bars * 100 if total_bars > 0 else 0

    print("\n" + "=" * 60)
    print("Intraday Backtest Summary")
    print("=" * 60)
    print(f"Period:      {equity['datetime'].min()} to {equity['datetime'].max()}")
    print(f"Total Bars:  {total_bars}")
    print(f"Initial:     ${initial_cash:,.2f}")
    print(f"Final:       ${final_value:,.2f}")
    print(f"Total P&L:   ${total_pnl:+,.2f} ({total_return:+.2f}%)")
    print(f"Win Rate:    {win_rate:.1f}%")
    print(f"Max Positions: {max_positions}")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]

    if args.days:
        end = date.today()
        start = end - timedelta(days=args.days * 7 // 5)  # Approximate trading days
    else:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)

    print("=" * 60)
    print("Alpha Intraday Backtest Runner")
    print("=" * 60)
    print(f"Period: {start} to {end}")
    print(f"Tickers: {tickers}")
    print(f"Initial Cash: ${args.initial_cash:,.2f}")
    print("=" * 60)

    data_dir = Path(__file__).resolve().parent / "data"

    # Load data
    print("Loading intraday data...")
    df = load_intraday_data(tickers, start, end, data_dir)

    if df.is_empty():
        print("No data available.")
        return

    # Run backtest
    equity = run_intraday_backtest(df, initial_cash=args.initial_cash)

    # Print summary
    print_summary(equity, args.initial_cash)

    # Save equity curve
    output = Path("reports/intraday/backtest_equity.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    equity.write_csv(str(output))
    print(f"\nEquity curve saved to {output}")


if __name__ == "__main__":
    main()
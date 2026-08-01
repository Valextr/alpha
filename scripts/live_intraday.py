#!/usr/bin/env python
"""Live intraday trading session entry point.

Connects to IBKR Gateway, streams 1-minute bars, computes VWAP MR signals
with filters (ADX, time-of-day, volume), and executes trades via IBBroker.

Usage:
    # Dry run (PaperBroker, no live orders)
    uv run python scripts/live_intraday.py --dry-run

    # Live paper trading (IBKR Gateway must be running)
    uv run python scripts/live_intraday.py --live

    # Custom ticker
    uv run python scripts/live_intraday.py --ticker GOOGL

    # Multiple tickers
    uv run python scripts/live_intraday.py --tickers MSFT,GOOGL,AMZN

Note: Market hours are 9:30 AM - 4:00 PM ET (14:30 - 21:00 UTC winter).
The script will wait for market open if started early.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from datetime import date, datetime, timezone

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live.intraday_runner import IntradayRunner, IntradayRunnerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live intraday trading session")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Connect to IBKR Gateway for live paper trading",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with PaperBroker (no live orders)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="MSFT",
        help="Single ticker to trade (default: MSFT)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers (overrides --ticker)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=1_000_000.0,
        help="Starting cash balance (default: 1M)",
    )
    parser.add_argument(
        "--max-shares",
        type=int,
        default=1000,
        help="Maximum shares per position (default: 1000)",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.005,
        help="Stop loss percentage (default: 0.5%%)",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.0075,
        help="Take profit percentage (default: 0.75%%)",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest on historical data instead of live session",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-01-01",
        help="Backtest start date (default: 2025-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-12-31",
        help="Backtest end date (default: 2025-12-31)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve tickers
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        tickers = [args.ticker]

    # Build config
    config = IntradayRunnerConfig(
        tickers=tickers,
        timeframe="1min",
        initial_cash=args.initial_cash,
        max_position_shares=args.max_shares,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        use_ibkr=args.live,
        backtest=args.backtest,
        data_dir=Path("data/intraday/1min"),
        signal_column="signal_vwap_mean_reversion_filtered",
    )

    runner = IntradayRunner(config)

    print("=" * 60)
    print("Alpha Intraday Trading Session")
    print("=" * 60)
    print(f"Date:       {date.today()}")
    print(f"Tickers:    {', '.join(tickers)}")
    print(f"Mode:       {'LIVE (IBKR)' if args.live else 'DRY RUN' if args.dry_run else 'BACKTEST'}")
    print(f"Cash:       ${args.initial_cash:,.0f}")
    print(f"Max shares: {args.max_shares}")
    print(f"Stop loss:  {args.stop_loss:.2%}")
    print(f"Take profit:{args.take_profit:.2%}")
    print("=" * 60)

    # Register signal handler for graceful shutdown
    def handle_signal(signum, frame):
        print("\n\nInterrupt received — shutting down...")
        runner.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        if args.backtest:
            print(f"\nRunning backtest: {args.start_date} to {args.end_date}")
            results = runner.run_backtest(
                start_date=args.start_date,
                end_date=args.end_date,
                tickers=tickers,
            )
            agg = results.get("aggregate", {})
            print("\n" + "=" * 60)
            print("Backtest Results:")
            print(f"  Sessions:   {agg.get('total_sessions', 0)}")
            print(f"  Total P&L:  ${agg.get('total_pnl', 0):,.2f}")
            print(f"  Total trades: {agg.get('total_trades', 0)}")
            print(f"  Win rate:   {agg.get('win_rate_pct', 0):.1f}%")
            print(f"  Avg return: {agg.get('avg_session_return_pct', 0):.2f}%")
            print(f"  Max DD:     {agg.get('max_drawdown_pct', 0):.2f}%")
            print("=" * 60)

        elif args.live:
            print("\nStarting live session...")
            print("  Press Ctrl+C to stop at any time")
            print("  Market hours: 9:30 AM - 4:00 PM ET")
            print()
            metrics = runner.run_session()

            print("\n" + "=" * 60)
            print("Session Complete:")
            for ticker, m in metrics.items():
                print(f"\n  {ticker}:")
                print(f"    P&L:      ${m.get('total_pnl', 0):,.2f}")
                print(f"    Trades:   {m.get('total_trades', 0)}")
                print(f"    Win rate: {m.get('win_rate_pct', 0):.1f}%")
            print("=" * 60)

        else:
            # Default to dry run
            print("\nStarting dry run session...")
            print("  Press Ctrl+C to stop at any time")
            print()
            metrics = runner.run_session()

            print("\n" + "=" * 60)
            print("Dry Run Complete:")
            for ticker, m in metrics.items():
                print(f"\n  {ticker}:")
                print(f"    P&L:      ${m.get('total_pnl', 0):,.2f}")
                print(f"    Trades:   {m.get('total_trades', 0)}")
                print(f"    Win rate: {m.get('win_rate_pct', 0):.1f}%")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\nSession interrupted by user")
    except Exception as e:
        log.error(f"Session failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
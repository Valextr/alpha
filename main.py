#!/usr/bin/env python
"""Alpha live trading entry point.

Usage:
    uv run python main.py              # Run today's trading cycle
    uv run python main.py --date 2026-07-29  # Run for a specific date
    uv run python main.py --dry-run    # Run pipeline without placing orders
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from src.live.config import LiveConfig
from src.live.runner import LiveRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpha live trading runner")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Trading date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline without placing orders.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers to trade.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse date
    end_date = None
    if args.date:
        try:
            end_date = date.fromisoformat(args.date)
        except ValueError:
            print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD.")
            sys.exit(1)

    # Load config
    config = LiveConfig.from_env()

    # Override tickers if specified
    if args.tickers:
        config.tickers = [t.strip() for t in args.tickers.split(",")]

    print("=" * 60)
    print("Alpha Live Trading Runner")
    print("=" * 60)
    print(f"Date:       {end_date or date.today()}")
    print(f"Tickers:    {', '.join(config.tickers)}")
    print(f"Mode:       {config.ensemble_mode}")
    print(f"Max Pos:    {config.max_positions}")
    print(f"Dry Run:    {args.dry_run}")
    print("=" * 60)

    runner = LiveRunner(config)

    if args.dry_run:
        print("\n[Dry run mode — no orders will be placed]\n")
        # Just test the data pipeline
        df = runner.fetch_data(end_date)
        print(f"Fetched {len(df)} rows of data")
        signal_df = runner.compute_signals(df)
        print(f"Signal columns: {[c for c in signal_df.columns if 'ensemble' in c.lower()]}")
        print("\nDry run complete.")
        return

    # Run the full trading cycle
    result = runner.run(end_date)

    # Print summary
    print("\n" + "=" * 60)
    print("Trading Day Summary")
    print("=" * 60)
    print(f"Portfolio Value: ${result.portfolio_value:,.2f}")
    print(f"Daily P&L:       ${result.daily_pnl:+,.2f}")
    print(f"Drawdown:        {result.drawdown:.2%}")
    print(f"Orders Placed:   {result.orders_placed}")
    print(f"Orders Filled:   {result.orders_filled}")
    print(f"Positions:       {len(result.positions)}")
    if result.tickers_traded:
        print(f"Tickers Traded:  {', '.join(result.tickers_traded)}")
    if result.alerts:
        print(f"Alerts:          {len(result.alerts)}")
        for alert in result.alerts:
            print(f"  - {alert}")
    if result.error:
        print(f"ERROR: {result.error}")
    print("=" * 60)


if __name__ == "__main__":
    main()
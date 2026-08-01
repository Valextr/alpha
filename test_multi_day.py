#!/usr/bin/env python
"""Multi-day backtest runner — simulates consecutive trading days.

Usage:
    uv run python test_multi_day.py              # Last 10 trading days
    uv run python test_multi_day.py --days 30    # Last 30 trading days
    uv run python test_multi_day.py --start 2024-01-01 --end 2024-12-31  # Date range
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.live.config import LiveConfig
from src.live.runner import LiveRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-day backtest runner")
    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Number of trading days to simulate (default: 10)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def is_trading_day(d: date) -> bool:
    """Check if a date is a trading day (Mon-Fri, no major holidays)."""
    # Simple check: Mon-Fri
    if d.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    # Major US market holidays (approximate)
    # Note: This is a simplified check — doesn't account for moving holidays
    holidays = {
        (1, 1),  # New Year's
        (7, 4),  # Independence Day
        (12, 25),  # Christmas
    }
    return (d.month, d.day) not in holidays


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config = LiveConfig.from_env()

    # Determine date range
    if args.start and args.end:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
    else:
        end = date.today()
        # Count backwards to find enough trading days
        days_counted = 0
        current = end
        while days_counted < args.days:
            if is_trading_day(current):
                days_counted += 1
            current -= timedelta(days=1)
        start = current + timedelta(days=1)

    # Generate trading days
    trading_days = []
    current = start
    while current <= end and len(trading_days) < args.days:
        if is_trading_day(current):
            trading_days.append(current)
        current += timedelta(days=1)

    print("=" * 60)
    print("Multi-Day Backtest Runner")
    print("=" * 60)
    print(f"Trading days: {len(trading_days)}")
    print(f"Date range: {trading_days[0]} to {trading_days[-1]}")
    print(f"Tickers: {', '.join(config.tickers)}")
    print(f"Mode: {config.ensemble_mode}")
    print("=" * 60)

    runner = LiveRunner(config)
    results = []

    # Connect once for the entire test
    runner.connect()

    try:
        for i, day in enumerate(trading_days, 1):
            print(f"\n[{i}/{len(trading_days)}] Running {day}...")

            result = runner.run_day(day)

            results.append({
                "date": day,
                "portfolio_value": result.portfolio_value,
                "daily_pnl": result.daily_pnl,
                "drawdown": result.drawdown,
                "orders_placed": result.orders_placed,
                "orders_filled": result.orders_filled,
                "positions": len(result.positions),
                "error": result.error,
            })

            # Print daily summary
            print(f"  Portfolio: ${result.portfolio_value:,.2f}")
            print(f"  Orders: {result.orders_placed} placed, {result.orders_filled} filled")
            print(f"  Positions: {len(result.positions)}")
            if result.error:
                print(f"  ERROR: {result.error}")

    finally:
        runner.disconnect()

    # Print final summary
    print("\n" + "=" * 60)
    print("Backtest Summary")
    print("=" * 60)

    if results:
        initial_value = results[0]["portfolio_value"]
        final_value = results[-1]["portfolio_value"]
        total_pnl = final_value - initial_value
        total_return = total_pnl / initial_value * 100 if initial_value else 0

        total_orders = sum(r["orders_placed"] for r in results)
        total_fills = sum(r["orders_filled"] for r in results)
        errors = sum(1 for r in results if r["error"])

        print(f"Initial Value: ${initial_value:,.2f}")
        print(f"Final Value:   ${final_value:,.2f}")
        print(f"Total P&L:     ${total_pnl:+,.2f} ({total_return:+.2f}%)")
        print(f"Total Orders:  {total_orders} placed, {total_fills} filled")
        print(f"Errors:        {errors}/{len(results)} days")
        print(f"Max Drawdown:  {max(r['drawdown'] for r in results):.2%}")

        # Save results to CSV
        results_df = pl.DataFrame(results)
        output_path = Path("reports/live/multi_day_test.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.write_csv(output_path)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
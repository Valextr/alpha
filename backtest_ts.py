#!/usr/bin/env python
"""Backtest with TS signals on 2025 data (5 tickers only).

Memory-efficient: loads only original 5 tickers, uses minimal features.
"""

import polars as pl
from datetime import date
from src.data.config import get_config
from src.features.pipeline import compute_features
from src.signals.pipeline import generate_all
from src.ensemble.pipeline import EnsemblePipeline, EnsemblePipelineConfig
from src.execution.broker import PaperBroker
from src.execution.engine import ExecutionConfig, ExecutionEngine
from src.execution.monitoring_bridge import MonitoringBridge
from src.signals.base import compute_forward_returns

def load_2025_data():
    """Load just 2025 gold data for original 5 tickers."""
    config = get_config()
    gold_dir = config.data_dir / "gold" / "daily"
    files = sorted(gold_dir.rglob("*.parquet"))
    small_files = [f for f in files if any(t in str(f) for t in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"])]
    df = pl.concat([pl.read_parquet(str(f)) for f in small_files], how="vertical_relaxed")
    df = df.filter((pl.col("date") >= date(2025, 1, 1)) & (pl.col("date") <= date(2025, 12, 31)))
    return df.sort(["ticker", "date"])

print("Loading 2025 data...")
df = load_2025_data()
print(f"Data: {len(df)} rows, {df['ticker'].n_unique()} tickers")

print("Computing features...")
enriched = compute_features(df)

print("Generating signals...")
signals = generate_all(enriched)

print("Computing forward returns...")
signals = compute_forward_returns(signals, horizons=[5])

# TS signals
ts_signals = [
    "signal_mean_reversion_ts_21d",
    "signal_mean_reversion_ts_63d",
    "signal_momentum_ts_21d",
    "signal_momentum_ts_63d",
]

# CS signals for comparison
cs_signals = [
    "signal_mean_reversion_21d",
    "signal_mean_reversion_63d",
    "signal_momentum_21d",
    "signal_momentum_63d",
]

def run_backtest(signal_cols, label, initial_cash=1_000_000.0):
    """Run backtest with given signals."""
    print(f"\n{'='*60}")
    print(f"Backtest: {label}")
    print(f"{'='*60}")

    # Run ensemble
    pipeline = EnsemblePipeline(EnsemblePipelineConfig(
        mode="ic_weighted",
        ic_target_horizon=5,
        signal_columns=signal_cols,
        weight_method="positive_ic",
    ))
    result = pipeline.run(signals)

    # Build price source
    price_source = {}
    for ticker in df["ticker"].unique().to_list():
        ticker_df = result.filter(pl.col("ticker") == ticker).select(["date", "close"])
        price_source[ticker] = ticker_df

    # Set up paper broker
    broker = PaperBroker(
        initial_cash=initial_cash,
        commission_per_share=0.005,
        slippage_bps=10.0,
        price_source=price_source,
    )

    exec_config = ExecutionConfig(
        initial_cash=initial_cash,
        max_positions=10,
        max_position_weight=0.25,
        daily_loss_limit_pct=0.05,
        max_drawdown_pct=0.15,
    )
    monitoring = MonitoringBridge(initial_balance=initial_cash)
    engine = ExecutionEngine(exec_config, broker=broker, monitoring_bridge=monitoring)
    engine.start()

    # Trade day by day
    dates = result["date"].unique().sort().to_list()
    equity_curve = []

    for i, trade_date in enumerate(dates):
        day_signals = result.filter(pl.col("date") == trade_date)
        if day_signals.is_empty():
            continue

        targets = engine.generate_targets_from_signals(
            day_signals,
            signal_column="signal_ensemble",
            min_signal_strength=0.1,
        )

        if targets:
            engine.execute_targets(targets, price_source={
                t: price_source.get(t, pl.DataFrame()) for t in targets
            })

        portfolio_value = broker.get_portfolio_value()
        daily_pnl = broker.record_eod()

        equity_curve.append({
            "date": trade_date,
            "portfolio_value": float(portfolio_value),
            "daily_pnl": float(daily_pnl),
        })

        engine._num_orders_today = 0
        engine._num_trades_today = 0

    engine.stop()

    # Calculate results
    equity_df = pl.DataFrame(equity_curve)
    final_value = float(equity_df.tail(1)["portfolio_value"].item())
    total_return = (final_value - initial_cash) / initial_cash * 100
    max_drawdown = (initial_cash - equity_df["portfolio_value"].min()) / initial_cash * 100

    print(f"Final value: ${final_value:,.2f}")
    print(f"Return: {total_return:+.2f}%")
    print(f"Max drawdown: {max_drawdown:.2f}%")
    print(f"Trading days: {len(equity_df)}")

    return {
        "label": label,
        "final": final_value,
        "return": total_return,
        "max_drawdown": max_drawdown,
        "days": len(equity_df),
    }

# Run CS backtest
cs_result = run_backtest(cs_signals, "CS Signals (4 signals)")

# Run TS backtest
ts_result = run_backtest(ts_signals, "TS Signals (4 signals)")

# Print comparison
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"{'Metric':<20} {'CS':>12} {'TS':>12} {'Change':>12}")
print("-" * 60)
print(f"{'Final Value':<20} ${cs_result['final']:>11,.2f} ${ts_result['final']:>11,.2f} ${ts_result['final'] - cs_result['final']:>+11,.2f}")
print(f"{'Return':<20} {cs_result['return']:>11.2f}% {ts_result['return']:>11.2f}% {ts_result['return'] - cs_result['return']:>+11.2f}%")
print(f"{'Max Drawdown':<20} {cs_result['max_drawdown']:>11.2f}% {ts_result['max_drawdown']:>11.2f}% {ts_result['max_drawdown'] - cs_result['max_drawdown']:>+11.2f}%")
print(f"{'Trading Days':<20} {cs_result['days']:>11} {ts_result['days']:>11}")
print(f"{'='*60}")
"""Intraday execution engine for bar-by-bar trading.

This engine is designed for intraday mean reversion strategies that:
- Trade on 1-hour bars
- Compute signals on every bar
- Place orders based on signal strength
- Track intraday P&L
- Close positions at market close
- Support stop loss/take profit

Architecture:
    1. Load 1-minute data from Alpaca parquet files
    2. Resample to bar interval (1-hour)
    3. Compute intraday features (VWAP, returns, volume)
    4. Generate intraday signals (VWAP mean reversion)
    5. Trade bar-by-bar using PaperBroker
    6. Close positions at market close
    7. Track intraday P&L and metrics

Known issues:
- Position sizing needs tuning (max_position_shares=1000 is conservative)
- Risk limits (50% max drawdown) are relaxed to avoid triggering too early
- Engine expects data at data/intraday/1min/ticker={TICKER}/year={YYYY}/month={MM}/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path
from typing import Optional

import polars as pl

from src.data.alpaca import download_intraday
from src.execution.broker import PaperBroker
from src.features.intraday import compute_intraday_features
from src.signals.base import compute_forward_returns
from src.signals.vwap_reversion import generate_vwap_mean_reversion_60b

log = logging.getLogger(__name__)


@dataclass
class IntradayConfig:
    """Configuration for intraday trading."""
    
    # Trading parameters
    initial_cash: float = 1_000_000.0
    max_position_shares: int = 1_000  # Max shares to hold (tuned from 10,000)
    signal_threshold: float = 0.0  # Minimum signal strength to trade
    slippage_bps: float = 10.0  # Slippage in basis points (0.1%)
    commission_per_share: float = 0.005  # Commission per share
    
    # Risk management
    stop_loss_pct: float = 0.5  # Stop loss at 0.5%
    take_profit_pct: float = 0.75  # Take profit at 0.75%
    max_drawdown_pct: float = 50.0  # Max drawdown at 50% (relaxed to avoid early triggers)
    
    # Time parameters
    market_open: time = time(9, 30)  # 9:30 AM ET
    market_close: time = time(16, 0)  # 4:00 PM ET
    bar_interval: str = "1h"  # 1-hour bars
    signal_horizon: int = 20  # 20-bar horizon (5 hours)
    
    # Data parameters
    data_dir: Optional[Path] = None
    tickers: list[str] = field(default_factory=lambda: ["MSFT"])
    start_date: date = field(default_factory=lambda: date(2025, 1, 1))
    end_date: date = field(default_factory=lambda: date(2025, 3, 31))


class IntradayEngine:
    """Intraday execution engine for bar-by-bar trading.
    
    Supports per-ticker position tracking for multi-ticker portfolios.
    Closes all positions at end of each trading session.
    Enforces per-position stop loss and take profit on every bar.
    """
    
    def __init__(self, config: Optional[IntradayConfig] = None):
        self.config = config or IntradayConfig()
        self.equity_curve: list[dict] = []
        self.trades: list[dict] = []
        # Per-ticker state
        self.positions: dict[str, int] = {t: 0 for t in self.config.tickers}
        self.entry_prices: dict[str, Optional[float]] = {t: None for t in self.config.tickers}
        self._last_prices: dict[str, float] = {t: 0.0 for t in self.config.tickers}
        self.cash: float = self.config.initial_cash
        self.peak_equity: float = self.config.initial_cash
        # Session tracking
        self._current_session_date: Optional[date] = None
    
    def _data_dir(self) -> Path:
        """Get the data directory, defaulting to project data/intraday/1min."""
        if self.config.data_dir is not None:
            return self.config.data_dir
        return Path(__file__).resolve().parents[3] / "data" / "intraday" / "1min"
    
    def load_data(self) -> pl.DataFrame:
        """Load 1-minute data and resample to bar interval."""
        data_dir = self._data_dir()
        
        # Download data if not exists
        for ticker in self.config.tickers:
            files = [f for f in data_dir.rglob("*.parquet") if f.parts and f"ticker={ticker}" in f.parts]
            if not files:
                log.info(f"Downloading {ticker} data...")
                download_intraday(
                    [ticker],
                    self.config.start_date,
                    self.config.end_date,
                    data_dir=data_dir.parent.parent,
                )
        
        # Load and filter to regular session
        all_bars = []
        for ticker in self.config.tickers:
            files = [f for f in data_dir.rglob("*.parquet") if f.parts and f"ticker={ticker}" in f.parts]
            if not files:
                log.warning(f"No data files for {ticker}")
                continue
            
            df = pl.concat([pl.read_parquet(str(f)) for f in files], how="vertical_relaxed")
            df = df.sort(["ticker", "datetime"])
            
            # Filter to date range
            df = df.filter(
                (pl.col("datetime").dt.date() >= self.config.start_date) &
                (pl.col("datetime").dt.date() <= self.config.end_date)
            )
            
            # Filter to regular session (9:30 AM - 4:00 PM ET = 14:30 - 21:00 UTC)
            df = df.with_columns(
                pl.col("datetime").dt.hour().alias("hour"),
                pl.col("datetime").dt.minute().alias("minute"),
            ).filter(
                ((pl.col("hour") >= 14) & (pl.col("hour") < 21)) |
                ((pl.col("hour") == 14) & (pl.col("minute") >= 30))
            )
            
            all_bars.append(df)
        
        if not all_bars:
            raise ValueError("No data loaded")

        df = pl.concat(all_bars, how="vertical_relaxed")
        df = df.sort("datetime")

        # Resample to bar interval (per ticker)
        bars_list = []
        for ticker in self.config.tickers:
            ticker_df = df.filter(pl.col("ticker") == ticker)
            if ticker_df.is_empty():
                continue
            ticker_bars = ticker_df.group_by_dynamic(
                "datetime",
                every=self.config.bar_interval,
                include_boundaries=True,
            ).agg(
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
                pl.col("ticker").first(),
                pl.col("symbol").first(),
                ((pl.col("close") * pl.col("volume")).sum() / pl.col("volume").sum()).alias("vwap_raw"),
            )
            bars_list.append(ticker_bars)

        bars = pl.concat(bars_list, how="vertical_relaxed")
        bars = bars.sort("datetime")

        return bars
    
    def compute_signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        """Compute intraday features and generate signals."""
        enriched = compute_intraday_features(bars)
        signals = generate_vwap_mean_reversion_60b(enriched)
        signals = compute_forward_returns(signals, horizons=[self.config.signal_horizon])
        
        # Add date column for compatibility
        signals = signals.with_columns(
            pl.col("datetime").dt.date().alias("date")
        )
        
        return signals
    
    def _portfolio_value(self, bar: dict) -> float:
        """Calculate portfolio value using current prices."""
        ticker = bar.get("ticker", self.config.tickers[0])
        close = bar.get("close", 0.0)
        # Track last known price per ticker
        if ticker in self._last_prices:
            self._last_prices[ticker] = close
        
        value = self.cash
        for t, pos in self.positions.items():
            if pos != 0 and t in self._last_prices:
                value += pos * self._last_prices[t]
        return value
    
    def _close_all_positions(self, bar: dict) -> list[dict]:
        """Close all open positions at current bar prices. Returns list of trades."""
        trades = []
        close = float(bar.get("close", 0.0))
        dt = bar["datetime"]
        
        for t in list(self.positions.keys()):
            if self.positions[t] == 0:
                continue
            # Use last known price for this ticker
            price = self._last_prices.get(t, close)
            shares_to_trade = -self.positions[t]
            trade_price = price * (1 - self.config.slippage_bps / 10000 if shares_to_trade > 0 else 1 + self.config.slippage_bps / 10000)
            commission = abs(shares_to_trade) * self.config.commission_per_share
            self.positions[t] = 0
            self.cash -= shares_to_trade * trade_price + commission
            self.entry_prices[t] = None
            trade = {
                "datetime": dt,
                "ticker": t,
                "shares": shares_to_trade,
                "price": trade_price,
                "commission": commission,
                "signal": 0.0,
                "position": 0,
            }
            trades.append(trade)
            self.trades.append(trade)
        
        return trades
    
    def _check_sl_tp(self, bar: dict) -> Optional[dict]:
        """Check stop loss and take profit for the current bar's ticker."""
        ticker = bar.get("ticker", self.config.tickers[0])
        close = bar["close"]
        dt = bar["datetime"]
        
        if self.positions[ticker] == 0 or self.entry_prices[ticker] is None:
            return None
        
        entry = self.entry_prices[ticker]
        pnl_pct = (close - entry) / entry if self.positions[ticker] > 0 else (entry - close) / entry
        
        # Stop loss: close if loss exceeds threshold
        if pnl_pct <= -self.config.stop_loss_pct / 100:
            shares_to_trade = -self.positions[ticker]
            trade_price = close * (1 - self.config.slippage_bps / 10000 if shares_to_trade > 0 else 1 + self.config.slippage_bps / 10000)
            commission = abs(shares_to_trade) * self.config.commission_per_share
            self.positions[ticker] = 0
            self.cash -= shares_to_trade * trade_price + commission
            self.entry_prices[ticker] = None
            log.info(f"SL triggered for {ticker}: {pnl_pct*100:.2f}% loss")
            trade = {
                "datetime": dt,
                "ticker": ticker,
                "shares": shares_to_trade,
                "price": trade_price,
                "commission": commission,
                "signal": 0.0,
                "position": 0,
            }
            self.trades.append(trade)
            return trade
        
        # Take profit: close if gain exceeds threshold
        if pnl_pct >= self.config.take_profit_pct / 100:
            shares_to_trade = -self.positions[ticker]
            trade_price = close * (1 - self.config.slippage_bps / 10000 if shares_to_trade > 0 else 1 + self.config.slippage_bps / 10000)
            commission = abs(shares_to_trade) * self.config.commission_per_share
            self.positions[ticker] = 0
            self.cash -= shares_to_trade * trade_price + commission
            self.entry_prices[ticker] = None
            log.info(f"TP triggered for {ticker}: {pnl_pct*100:.2f}% gain")
            trade = {
                "datetime": dt,
                "ticker": ticker,
                "shares": shares_to_trade,
                "price": trade_price,
                "commission": commission,
                "signal": 0.0,
                "position": 0,
            }
            self.trades.append(trade)
            return trade
        
        return None
    
    def _check_session_change(self, bar: dict) -> bool:
        """Check if we've entered a new trading session. Closes positions if so."""
        bar_date = bar.get("date")
        if bar_date is None:
            # Fallback: extract date from datetime
            dt_val = bar.get("datetime")
            if dt_val is not None:
                if callable(getattr(dt_val, "date", None)):
                    bar_date = dt_val.date()
                elif hasattr(dt_val, "date"):
                    bar_date = dt_val.date
            if bar_date is None:
                return False
        
        if self._current_session_date is None:
            self._current_session_date = bar_date
            return False
        
        if bar_date != self._current_session_date:
            # New session — close all positions at last known prices
            log.info(f"Session change: {self._current_session_date} -> {bar_date}, closing all positions")
            self._close_all_positions(bar)
            self._current_session_date = bar_date
            return True
        
        return False
    
    def _check_risk_limits(self, portfolio_value: float) -> bool:
        """Check if risk limits are breached."""
        # Check drawdown
        drawdown = (self.peak_equity - portfolio_value) / self.peak_equity * 100
        if drawdown > self.config.max_drawdown_pct:
            return False
        
        return True
    
    def trade_bar(self, bar: dict) -> Optional[dict]:
        """Trade a single bar based on signal."""
        ticker = bar.get("ticker", self.config.tickers[0])
        signal = bar.get("signal_vwap_mean_reversion_60b")
        close = bar["close"]
        datetime = bar["datetime"]
        
        # Check SL/TP before anything else
        sl_tp_trade = self._check_sl_tp(bar)
        if sl_tp_trade is not None:
            return sl_tp_trade
        
        if signal is None:
            return None
        
        # Check risk limits
        portfolio_value = self._portfolio_value(bar)
        if not self._check_risk_limits(portfolio_value):
            log.warning("Risk limits breached, closing all positions")
            self._close_all_positions(bar)
            return None
        
        # Check if signal meets threshold
        if abs(signal) < self.config.signal_threshold:
            return None
        
        # Calculate target position based on signal strength
        # Signal > 0 means price is below VWAP (oversold) → buy
        # Signal < 0 means price is above VWAP (overbought) → sell
        target_position = signal * self.config.max_position_shares
        
        # Adjust position
        shares_to_trade = int(target_position - self.positions[ticker])
        
        if abs(shares_to_trade) < 1:
            return None
        
        # Execute trade
        trade_price = close * (1 + self.config.slippage_bps / 10000 if shares_to_trade > 0 else 1 - self.config.slippage_bps / 10000)
        commission = abs(shares_to_trade) * self.config.commission_per_share
        
        # Update position and cash
        self.positions[ticker] += shares_to_trade
        self.cash -= shares_to_trade * trade_price + commission
        self.entry_prices[ticker] = trade_price if self.positions[ticker] != 0 else None
        
        # Update peak equity
        portfolio_value = self._portfolio_value(bar)
        if portfolio_value > self.peak_equity:
            self.peak_equity = portfolio_value
        
        # Record trade
        trade = {
            "datetime": datetime,
            "ticker": ticker,
            "shares": shares_to_trade,
            "price": trade_price,
            "commission": commission,
            "signal": signal,
            "position": self.positions[ticker],
        }
        self.trades.append(trade)
        
        return trade
    
    def run(self) -> pl.DataFrame:
        """Run the intraday trading engine."""
        # Load data and compute signals
        bars = self.load_data()
        signals = self.compute_signals(bars)
        
        log.info(f"Loaded {len(bars):,} bars")
        log.info(f"Trading {len(signals):,} bars with signals")
        
        # Trade bar-by-bar
        for bar in signals.iter_rows(named=True):
            # Check for session change (EOD position closure)
            self._check_session_change(bar)
            
            trade = self.trade_bar(bar)
            
            # Record equity
            portfolio_value = self._portfolio_value(bar)
            total_position = sum(self.positions.values())
            self.equity_curve.append({
                "datetime": bar["datetime"],
                "portfolio_value": portfolio_value,
                "position": total_position,
                "cash": self.cash,
                "trade": trade is not None,
            })
        
        # Convert to DataFrame
        equity = pl.DataFrame(self.equity_curve)
        
        return equity
    
    def get_metrics(self, equity: pl.DataFrame) -> dict:
        """Calculate trading metrics."""
        if equity.is_empty():
            return {}
        
        final_value = float(equity["portfolio_value"].max())
        initial_value = self.config.initial_cash
        total_return = (final_value - initial_value) / initial_value * 100
        total_pnl = final_value - initial_value
        
        # Win rate
        returns = equity["portfolio_value"].pct_change().drop_nulls()
        winning_bars = int((returns > 0).sum())
        total_bars = len(returns)
        win_rate = winning_bars / total_bars * 100 if total_bars > 0 else 0
        
        # Sharpe ratio (annualized)
        sharpe = float(returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0
        
        # Max drawdown
        peak = equity["portfolio_value"].cum_max()
        drawdown = (equity["portfolio_value"] - peak) / peak
        max_drawdown = float(drawdown.min() * 100)
        
        return {
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "num_trades": len(self.trades),
            "num_bars": len(equity),
        }
    
    def print_summary(self, equity: pl.DataFrame) -> None:
        """Print trading summary."""
        metrics = self.get_metrics(equity)
        
        print("\n" + "=" * 60)
        print("Intraday Trading Summary")
        print("=" * 60)
        print(f"Period:      {equity['datetime'].min()} to {equity['datetime'].max()}")
        print(f"Total Bars:  {metrics['num_bars']}")
        print(f"Trades:      {metrics['num_trades']}")
        print(f"Initial:     ${metrics['initial_value']:,.2f}")
        print(f"Final:       ${metrics['final_value']:,.2f}")
        print(f"Total P&L:   ${metrics['total_pnl']:+,.2f} ({metrics['total_return']:+.2f}%)")
        print(f"Win Rate:    {metrics['win_rate']:.1f}%")
        print(f"Sharpe:      {metrics['sharpe']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print("=" * 60)


def run_intraday_backtest(config: Optional[IntradayConfig] = None) -> pl.DataFrame:
    """Run an intraday backtest with the given configuration."""
    engine = IntradayEngine(config)
    equity = engine.run()
    engine.print_summary(equity)
    return equity


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    
    # Run backtest with default config
    equity = run_intraday_backtest()
    
    # Save results
    output = Path("reports/intraday/intraday_backtest.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    equity.write_csv(str(output))
    print(f"\nEquity curve saved to {output}")
"""Build and test all 7 orthogonal signal candidates.

Generates signal modules, runs IC tests across all timeframes/horizons/tickers,
saves results to docs/ORTHOGONAL-SIGNAL-RESULTS.md, and commits.

Run: cd ~/working/alpha && uv run python scripts/build_orthogonal_signals.py
"""

from __future__ import annotations

import glob
import logging
import subprocess
from datetime import date
from pathlib import Path

import polars as pl
from scipy.stats import spearmanr

from src.features.intraday import compute_intraday_features
from src.signals.base import compute_forward_returns

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT = Path("/home/Vale/working/alpha")
SIGNALS_DIR = PROJECT / "src" / "signals"

TICKERS = ["MSFT", "GOOGL", "AAPL", "AMZN", "META", "QQQ", "INTC"]
TIMEFRAMES = ["15m", "30m", "1h"]
HORIZONS = [5, 20]
START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 12, 31)


# ── Signal definitions ──────────────────────────────────────────────

def generate_macd(df: pl.DataFrame) -> pl.DataFrame:
    """MACD momentum signal."""
    ema12 = df.with_columns(
        pl.col("close").ewm_mean(com=1 / 12 - 1).over("ticker").alias("_ema12")
    )
    ema26 = ema12.with_columns(
        pl.col("close").ewm_mean(com=1 / 26 - 1).over("ticker").alias("_ema26")
    )
    macd = ema26.with_columns(
        (pl.col("_ema12") - pl.col("_ema26")).alias("_macd")
    )
    signal = macd.with_columns(
        pl.col("_macd").ewm_mean(com=1 / 9 - 1).over("ticker").alias("_signal")
    )
    raw = signal.with_columns(
        (pl.col("_macd") - pl.col("_signal")).alias("_raw")
    )
    lb, ms = 60, 30
    mean = raw["_raw"].rolling_mean(lb, min_samples=ms).over("ticker")
    std = raw["_raw"].rolling_std(lb, min_samples=ms).over("ticker")
    return raw.with_columns(
        ((raw["_raw"] - mean) / std.replace(0, None)).tanh().alias("signal_macd")
    ).drop(["_ema12", "_ema26", "_macd", "_signal", "_raw", "mean", "std"])


def generate_roc(df: pl.DataFrame) -> pl.DataFrame:
    """Rate of Change (momentum) signal."""
    period = 10
    raw = df.with_columns(
        ((pl.col("close") / pl.col("close").shift(period).over("ticker")) - 1).alias("_raw")
    )
    lb, ms = 60, 30
    mean = raw["_raw"].rolling_mean(lb, min_samples=ms).over("ticker")
    std = raw["_raw"].rolling_std(lb, min_samples=ms).over("ticker")
    return raw.with_columns(
        ((raw["_raw"] - mean) / std.replace(0, None)).tanh().alias("signal_roc")
    ).drop(["_raw"])


def generate_volume_momentum(df: pl.DataFrame) -> pl.DataFrame:
    """Volume-weighted momentum signal."""
    period = 10
    vol_ret = df.with_columns(
        (pl.col("close") / pl.col("close").shift(period).over("ticker") - 1).alias("_vret")
    ).with_columns(
        (pl.col("_vret") * pl.col("volume")).alias("_vmom")
    ).with_columns(
        pl.col("_vmom").rolling_mean(period, min_samples=5).over("ticker").alias("_raw")
    )
    lb, ms = 60, 30
    mean = vol_ret["_raw"].rolling_mean(lb, min_samples=ms).over("ticker")
    std = vol_ret["_raw"].rolling_std(lb, min_samples=ms).over("ticker")
    return vol_ret.with_columns(
        ((vol_ret["_raw"] - mean) / std.replace(0, None)).tanh().alias("signal_volume_momentum")
    ).drop(["_vret", "_vmom", "_raw"])


def generate_hurst(df: pl.DataFrame) -> pl.DataFrame:
    """Hurst exponent regime signal (directional: positive when mean-reverting)."""
    # Simplified Hurst via rolling R/S approximation
    # Use log returns and rolling variance ratio as proxy
    period = 20
    lr = df.with_columns(
        pl.col("close").log().diff().over("ticker").alias("_lr")
    )
    var1 = lr["_lr"].rolling_var(window_size=5, min_samples=3).over("ticker")
    var2 = lr["_lr"].rolling_var(window_size=10, min_samples=5).over("ticker")
    raw = lr.with_columns(
        ((var1 / var2.replace(0, None)) - 1).alias("_raw")
    )
    # Positive when variance ratio > 1 (persistent/trending), invert for mean-reversion
    lb, ms = 60, 30
    mean = raw["_raw"].rolling_mean(lb, min_samples=ms).over("ticker")
    std = raw["_raw"].rolling_std(lb, min_samples=ms).over("ticker")
    return raw.with_columns(
        (-((raw["_raw"] - mean) / std.replace(0, None))).tanh().alias("signal_hurst")
    ).drop(["_lr", "_raw"])


def generate_relative_volume(df: pl.DataFrame) -> pl.DataFrame:
    """Relative volume filter as directional signal."""
    period = 60
    avg_vol = pl.col("volume").rolling_mean(period, min_samples=30).over("ticker")
    raw = df.with_columns(
        ((pl.col("volume") / avg_vol.replace(0, None)) - 1).alias("_raw")
    )
    # High relative volume = more reliable signals (directional)
    lb, ms = 60, 30
    mean = raw["_raw"].rolling_mean(lb, min_samples=ms).over("ticker")
    std = raw["_raw"].rolling_std(lb, min_samples=ms).over("ticker")
    return raw.with_columns(
        ((raw["_raw"] - mean) / std.replace(0, None)).tanh().alias("signal_relative_volume")
    ).drop(["_raw"])


def generate_cross_asset_momentum(df: pl.DataFrame) -> pl.DataFrame:
    """Cross-asset momentum: stock return vs QQQ return."""
    # This requires QQQ data joined per bar — approximate with rolling return
    # vs rolling volatility as a proxy for relative strength
    period = 10
    ret = df.with_columns(
        pl.col("close").pct_change(period).over("ticker").alias("_ret")
    ).with_columns(
        pl.col("close").pct_change(period).over("ticker").rolling_std(
            20, min_samples=10
        ).over("ticker").alias("_vol")
    )
    raw = ret.with_columns(
        (pl.col("_ret") / pl.col("_vol").replace(0, None)).alias("_raw")
    )
    lb, ms = 60, 30
    mean = raw["_raw"].rolling_mean(lb, min_samples=ms).over("ticker")
    std = raw["_raw"].rolling_std(lb, min_samples=ms).over("ticker")
    return raw.with_columns(
        ((raw["_raw"] - mean) / std.replace(0, None)).tanh().alias("signal_cross_asset_momentum")
    ).drop(["_ret", "_vol", "_raw"])


def generate_order_flow(df: pl.DataFrame) -> pl.DataFrame:
    """Order flow imbalance: buy volume vs sell volume."""
    raw = df.with_columns(
        pl.when(pl.col("close") > pl.col("open")).then(pl.col("volume")).otherwise(0.0).alias("_buy_vol"),
        pl.when(pl.col("close") <= pl.col("open")).then(pl.col("volume")).otherwise(0.0).alias("_sell_vol"),
    ).with_columns(
        ((pl.col("_buy_vol") - pl.col("_sell_vol")) / (pl.col("_buy_vol") + pl.col("_sell_vol")).replace(0, None)).alias("_raw")
    )
    lb, ms = 60, 30
    mean = raw["_raw"].rolling_mean(lb, min_samples=ms).over("ticker")
    std = raw["_raw"].rolling_std(lb, min_samples=ms).over("ticker")
    return raw.with_columns(
        ((raw["_raw"] - mean) / std.replace(0, None)).tanh().alias("signal_order_flow")
    ).drop(["_buy_vol", "_sell_vol", "_raw"])


SIGNAL_GENERATORS = {
    "macd": generate_macd,
    "roc": generate_roc,
    "volume_momentum": generate_volume_momentum,
    "hurst": generate_hurst,
    "relative_volume": generate_relative_volume,
    "cross_asset_momentum": generate_cross_asset_momentum,
    "order_flow": generate_order_flow,
}


# ── Data loading ────────────────────────────────────────────────────

def load_resample(ticker: str, tf: str) -> pl.DataFrame | None:
    data_dir = PROJECT / f"data/intraday/1min/ticker={ticker}"
    dfs = [pl.read_parquet(str(f)) for f in data_dir.rglob("*.parquet")]
    if not dfs:
        return None
    all_data = pl.concat(dfs, how="vertical_relaxed").sort("datetime")
    all_data = all_data.filter(
        (pl.col("datetime").dt.date() >= START_DATE)
        & (pl.col("datetime").dt.date() <= END_DATE)
    ).with_columns(
        pl.col("datetime").dt.hour().alias("hour"),
        pl.col("datetime").dt.minute().alias("minute"),
    ).filter(
        ((pl.col("hour") >= 14) & (pl.col("hour") < 21))
        | ((pl.col("hour") == 14) & (pl.col("minute") >= 30))
    )
    bars = all_data.group_by_dynamic("datetime", every=tf, include_boundaries=True).agg(
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
        pl.col("ticker").first(),
    )
    return bars if len(bars) > 0 else None


# ── IC computation ──────────────────────────────────────────────────

def calc_ic(df: pl.DataFrame, sig_col: str, horizon: int) -> float | None:
    fwd = f"forward_return_{horizon}"
    valid = df.drop_nulls(subset=[sig_col, fwd])
    if len(valid) < 100:
        return None
    c, _ = spearmanr(valid[sig_col].to_list(), valid[fwd].to_list())
    return float(c)


# ── Main ────────────────────────────────────────────────────────────

def main():
    all_ics: dict[tuple[str, str, str, int], float] = {}

    for sig_name, gen in SIGNAL_GENERATORS.items():
        log.info(f"Testing signal: {sig_name}")
        sig_col = f"signal_{sig_name}"

        for ticker in TICKERS:
            for tf in TIMEFRAMES:
                bars = load_resample(ticker, tf)
                if bars is None:
                    continue

                enriched = compute_intraday_features(bars)
                try:
                    sig_df = gen(enriched)
                except Exception as e:
                    log.warning(f"  {ticker}/{tf}: {sig_name} failed: {e}")
                    continue

                if sig_col not in sig_df.columns:
                    log.warning(f"  {ticker}/{tf}: {sig_name} missing column {sig_col}")
                    continue

                with_fwd = compute_forward_returns(sig_df, horizons=HORIZONS)

                for h in HORIZONS:
                    c = calc_ic(with_fwd, sig_col, h)
                    if c is not None:
                        all_ics[(ticker, tf, sig_name, h)] = c

    # ── Write results ──────────────────────────────────────────────
    out = PROJECT / "docs" / "ORTHOGONAL-SIGNAL-RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        f.write("# Orthogonal Signal Test Results\n\n")
        f.write(f"Tested {len(SIGNAL_GENERATORS)} signals across {len(TICKERS)} tickers, "
                f"{len(TIMEFRAMES)} timeframes, {len(HORIZONS)} horizons.\n")
        f.write(f"Total IC calculations: {len(all_ics)}\n\n")

        # Per-signal summary
        f.write("## Per-Signal Summary (avg IC across all tickers)\n\n")
        for sig_name in SIGNAL_GENERATORS:
            f.write(f"### {sig_name}\n\n")
            for tf in TIMEFRAMES:
                for h in HORIZONS:
                    ics = [all_ics[(t, tf, sig_name, h)] for t in TICKERS
                           if (t, tf, sig_name, h) in all_ics]
                    if ics:
                        avg = sum(ics) / len(ics)
                        f.write(f"- {tf}/{h}-bar: avg IC = {avg:+.4f} "
                                f"(range: [{min(ics):+.4f}, {max(ics):+.4f}])\n")
                    else:
                        f.write(f"- {tf}/{h}-bar: N/A\n")
            f.write("\n")

        # Per-ticker detail
        for tf in TIMEFRAMES:
            f.write(f"## {tf} bars\n\n")
            for h in HORIZONS:
                f.write(f"### {h}-bar horizon\n\n")
                f.write("| Ticker | Signal | IC |\n|--------|--------|-----|\n")
                for ticker in TICKERS:
                    items = [(k[2], v) for k, v in all_ics.items()
                             if k[0] == ticker and k[1] == tf and k[3] == h]
                    items.sort(key=lambda x: x[1], reverse=True)
                    for sig_name, ic_val in items:
                        f.write(f"| {ticker} | {sig_name} | {ic_val:+.4f} |\n")
                f.write("\n")

        # Best signals
        f.write("## Best Signals (by avg IC)\n\n")
        avgs = []
        for sig_name in SIGNAL_GENERATORS:
            all_vals = [v for k, v in all_ics.items() if k[2] == sig_name]
            if all_vals:
                avgs.append((sig_name, sum(all_vals) / len(all_vals)))
        avgs.sort(key=lambda x: x[1], reverse=True)
        f.write("| Rank | Signal | Avg IC |\n|------|--------|--------|\n")
        for i, (sig_name, avg) in enumerate(avgs, 1):
            f.write(f"| {i} | {sig_name} | {avg:+.4f} |\n")

    log.info(f"Results saved to {out}")

    # ── Commit ─────────────────────────────────────────────────────
    subprocess.run(["git", "add", str(out)], cwd=PROJECT, check=True)
    subprocess.run(
        ["git", "commit", "-m",
         f"docs: orthogonal signal results ({len(all_ics)} IC calculations)"],
        cwd=PROJECT, check=True, capture_output=True, text=True,
    )
    log.info("Committed")


if __name__ == "__main__":
    main()
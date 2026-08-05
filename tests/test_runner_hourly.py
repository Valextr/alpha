"""Regression tests (Aug 4 2026): multi-minute timeframe support in the
live runner (`IntradayRunnerConfig.timeframe='1h'`).

The runner must aggregate delivered 1-min bars into :30-anchored UTC windows —
identical alignment to `resample_1h`, which every backtest uses — and only
ever route COMPLETED windows to the signal/engine path. Partial or stale
historical windows are never traded (the warm-up contract).

Alignment is verified against polars' own `group_by_dynamic(..., offset='30m')`
(the exact call in resample_1h). All other tests derive window membership at
RUNTIME from `IntradayRunner._window_key` — no hardcoded boundary timestamps.
"""
from datetime import datetime, timedelta, timezone

import polars as pl


def _minute(dt: datetime, close: float = 100.0):
    from src.live.intraday_runner import BarRecord
    return BarRecord(
        ticker="MSFT",
        datetime=dt,
        open=close - 2.0,
        high=close + 3.0,
        low=close - 4.0,
        close=close,
        volume=100,
    )


def _interior_minutes(start: datetime, count: int = 60):
    """Return consecutive minutes starting at `start` that all belong to the
    SAME window as `start`, per IntradayRunner._window_key (runtime-checked)."""
    from src.live.intraday_runner import IntradayRunner

    k0 = IntradayRunner._window_key(start)
    out = []
    for i in range(count):
        dt = start + timedelta(minutes=i)
        if IntradayRunner._window_key(dt) != k0:
            break
        out.append(dt)
    assert len(out) >= 2, "test construction: need at least a partial window"
    return out


class TestRunnerHourlyAggregation:
    def test_window_key_matches_polars_resample_alignment(self):
        from src.live.intraday_runner import IntradayRunner

        k = IntradayRunner._window_key
        times = _interior_minutes(datetime(2025, 6, 4, 12, 31, tzinfo=timezone.utc), 200)
        # include the boundary minute AFTER that window too
        after = times[-1]
        while IntradayRunner._window_key(after) == IntradayRunner._window_key(times[0]):
            after += timedelta(minutes=1)

        df = pl.DataFrame({"dt": times + [after]})
        resampled = df.group_by_dynamic("dt", every="1h", offset="30m").agg(pl.len())
        window_starts = {ts for ts in resampled["dt"]}

        # every minute's key + 30 min must be a polars lower bound
        for t in times + [after]:
            assert k(t) + timedelta(minutes=30) in window_starts

    def test_warm_only_flushes_exactly_completed_windows(self):
        from src.live.intraday_runner import IntradayRunner, IntradayRunnerConfig

        r = IntradayRunner(config=IntradayRunnerConfig(tickers=["MSFT"], timeframe="1h"))
        feed = _interior_minutes(datetime(2025, 6, 4, 13, 31, tzinfo=timezone.utc), 60)
        for dt in feed:
            assert r._ingest_minute("MSFT", _minute(dt), warm_only=True) is None
        # window incomplete (last fed minute still inside it) → nothing flushed
        assert len(r._bar_buffer["MSFT"]) == 0

        # any minute of the NEXT window completes [..:30, next :30)
        key0 = IntradayRunner._window_key(feed[0])
        trigger = feed[-1] + timedelta(hours=1)   # provably in the next window
        assert r._ingest_minute("MSFT", _minute(trigger), warm_only=True) is None

        buf = r._bar_buffer["MSFT"]
        assert len(buf) == 1
        assert buf[0].datetime == key0 + timedelta(minutes=30)

    def test_ohlcv_and_volume_match_polars_resample(self):
        from src.live.intraday_runner import BarRecord, \
            IntradayRunner, IntradayRunnerConfig

        r = IntradayRunner(config=IntradayRunnerConfig(tickers=["MSFT"], timeframe="1h"))
        times = _interior_minutes(datetime(2026, 3, 5, 13, 40, tzinfo=timezone.utc), 60)
        bars = [
            BarRecord(ticker="MSFT", datetime=t,
                      open=10.0 + i, high=12.0 + i, low=9.0 + i,
                      close=11.0 + i, volume=100 * (i + 1))
            for i, t in enumerate(times)
        ]
        for b in bars:
            r._ingest_minute("MSFT", b, warm_only=True)

        # oracle: polars group_by_dynamic over the same minutes
        df = pl.DataFrame({
            "dt": [b.datetime for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        })
        oracle = df.group_by_dynamic("dt", every="1h", offset="30m").agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        )
        assert len(oracle) == 1

        # complete the window with a minute of the next one
        trigger = times[-1] + timedelta(hours=1)
        r._ingest_minute("MSFT", _minute(trigger), warm_only=True)

        expected_label = IntradayRunner._window_key(times[0]) + timedelta(minutes=30)
        recs = [b for b in r._bar_buffer["MSFT"] if b.datetime == expected_label]
        assert len(recs) == 1
        rec = recs[0]
        o = oracle.row(0, named=True)
        assert rec.open == pytest_approx(o["open"])
        assert rec.high == pytest_approx(o["high"])
        assert rec.low == pytest_approx(o["low"])
        assert rec.close == pytest_approx(o["close"])
        assert rec.volume == o["volume"]

    def test_live_flush_routes_only_completed_windows(self, monkeypatch):
        from src.live.intraday_runner import IntradayRunner, IntradayRunnerConfig

        r = IntradayRunner(config=IntradayRunnerConfig(tickers=["MSFT"], timeframe="1h"))
        processed = []
        monkeypatch.setattr(r, "process_bar",
                            lambda ticker, bar: (processed.append(bar.datetime), None)[1])

        feed = _interior_minutes(datetime(2026, 4, 8, 9, 31, tzinfo=timezone.utc), 60)
        for dt in feed:
            assert r._ingest_minute("MSFT", _minute(dt)) is None
        assert processed == []                     # nothing routed mid-window

        key0 = IntradayRunner._window_key(feed[0])
        trigger = feed[-1] + timedelta(hours=1)    # first minute of next window
        r._ingest_minute("MSFT", _minute(trigger))
        assert len(processed) == 1                 # exactly one completed hour
        assert processed[0] == key0 + timedelta(minutes=30)


def pytest_approx(v: float):
    import pytest
    return pytest.approx(v)

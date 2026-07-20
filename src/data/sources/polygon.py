"""Polygon.io data source — production-grade, survivorship-bias-free.

Implements the DataSource ABC using the polygon-api-client library.
Includes delisted tickers via the Tickers API (active=None by default).

Rate limiting:
- Free tier: 5 requests/sec, 12,000/day
- Paid plans: up to 1000/min (varies by tier)
- list_aggs handles pagination internally via Iterator

References:
- https://polygon.io/docs/stocks/get_reference_tickers
- https://polygon.io/docs/stocks/get_aggregates
- https://polygon.io/docs/stocks/get_stock-events-dividends
- https://polygon.io/docs/stocks/get_stock-events-splits
"""

import asyncio
import logging
import time
from collections.abc import Iterator
from datetime import date, datetime
from typing import Optional

import polars as pl

try:
    from polygon import RESTClient
except ImportError:
    raise ImportError(
        "polygon-api-client is required for PolygonDataSource. "
        "Install it with: uv pip install polygon-api-client"
    )

from .base import DataSource

logger = logging.getLogger(__name__)

# Default rate limit params (free tier)
DEFAULT_RPS = 5  # requests per second
DEFAULT_DELAY = 0.2  # seconds between requests (5 req/s = 200ms)


class PolygonDataSource(DataSource):
    """Polygon.io data source for production use.

    Unlike yfinance, Polygon does not suffer from survivorship bias —
    delisted tickers remain queryable via the aggregates endpoint.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.polygon.io",
        rate_limit_rps: int = DEFAULT_RPS,
        request_delay: float = DEFAULT_DELAY,
    ) -> None:
        """Initialize Polygon client.

        Args:
            api_key: Polygon.io API key (from POLYGON_API_KEY env var)
            base_url: Polygon API base URL
            rate_limit_rps: Maximum requests per second
            request_delay: Minimum seconds between API calls
        """
        self._api_key = api_key
        self._client = RESTClient(api_key)
        self._rate_limit_rps = rate_limit_rps
        self._request_delay = request_delay
        self._request_times: list[float] = []

    @property
    def name(self) -> str:
        return "polygon"

    def _rate_limit(self) -> None:
        """Simple sliding-window rate limiter."""
        now = time.monotonic()
        # Remove timestamps older than 1 second
        self._request_times = [
            t for t in self._request_times if now - t < 1.0
        ]
        if len(self._request_times) >= self._rate_limit_rps:
            sleep_time = 1.0 / self._rate_limit_rps - (now - self._request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._request_times.append(time.monotonic())
        time.sleep(self._request_delay)

    async def _run_in_executor(self, coro_func, *args, **kwargs):
        """Run blocking polygon API call in a thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: coro_func(*args, **kwargs))

    def _collect_iterator(self, iterator: Iterator) -> list:
        """Collect all items from a polygon Iterator, applying rate limiting."""
        results = []
        for item in iterator:
            results.append(item)
            self._rate_limit()
        return results

    # ── fetch_daily_bars ─────────────────────────────────────────

    async def fetch_daily_bars(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Fetch daily OHLCV bars from Polygon aggregates endpoint.

        Uses adjusted=False to get raw (unadjusted) prices.
        The silver layer applies split/dividend adjustments.

        Args:
            ticker: Ticker symbol (e.g., 'AAPL')
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            DataFrame with columns: ticker, date, open, high, low, close, volume, vwap
        """
        logger.debug("Fetching %s daily bars [%s, %s]", ticker, start, end)

        empty = pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "date": pl.Date,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
                "vwap": pl.Float64,
            }
        )

        try:
            aggs_iter = self._client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start.isoformat(),
                to=end.isoformat(),
                adjusted=False,
            )
            aggs = await self._run_in_executor(self._collect_iterator, aggs_iter)
        except Exception as e:
            logger.warning("Failed to fetch %s bars: %s", ticker, e)
            return empty

        if not aggs:
            logger.debug("No bars for %s [%s, %s]", ticker, start, end)
            return empty

        # Build DataFrame from raw dicts
        records = []
        for agg in aggs:
            ts_ms = agg.timestamp
            if ts_ms is None:
                continue
            # Polygon timestamp is epoch milliseconds
            bar_date = datetime.fromtimestamp(ts_ms / 1000).date()
            records.append({
                "ticker": ticker,
                "date": bar_date,
                "open": agg.open,
                "high": agg.high,
                "low": agg.low,
                "close": agg.close,
                "volume": int(agg.volume) if agg.volume is not None else 0,
                "vwap": agg.vwap,
            })

        if not records:
            return empty

        df = pl.DataFrame(records)

        # Ensure volume is Int64
        if "volume" in df.columns:
            df = df.with_columns(
                pl.col("volume").cast(pl.Int64)
            )

        return self.validate_bars(df)

    # ── fetch_dividends ──────────────────────────────────────────

    async def fetch_dividends(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Fetch dividend history from Polygon.

        Args:
            ticker: Ticker symbol
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            DataFrame with columns: ticker, ex_date, amount
        """
        logger.debug("Fetching %s dividends [%s, %s]", ticker, start, end)

        empty = pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "ex_date": pl.Date,
                "amount": pl.Float64,
            }
        )

        try:
            div_iter = self._client.list_dividends(
                ticker=ticker,
                ex_dividend_date_gt=start.isoformat(),
                ex_dividend_date_lte=end.isoformat(),
            )
            dividends = await self._run_in_executor(
                self._collect_iterator, div_iter
            )
        except Exception as e:
            logger.warning("Failed to fetch %s dividends: %s", ticker, e)
            return empty

        if not dividends:
            return empty

        records = []
        for div in dividends:
            if div.ex_dividend_date and div.cash_amount is not None:
                records.append({
                    "ticker": ticker,
                    "ex_date": date.fromisoformat(div.ex_dividend_date),
                    "amount": div.cash_amount,
                })

        if not records:
            return empty

        df = pl.DataFrame(records)
        return df

    # ── fetch_corporate_actions ──────────────────────────────────

    async def fetch_corporate_actions(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> Optional[pl.DataFrame]:
        """Fetch splits from Polygon.

        Note: Polygon's splits endpoint only covers splits, not mergers/mergers.

        Args:
            ticker: Ticker symbol
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            DataFrame with columns: ticker, date, action_type, factor
            or None if no splits found.
        """
        logger.debug("Fetching %s splits [%s, %s]", ticker, start, end)

        try:
            splits_iter = self._client.list_splits(
                ticker=ticker,
                execution_date_gt=start.isoformat(),
                execution_date_lte=end.isoformat(),
            )
            splits = await self._run_in_executor(
                self._collect_iterator, splits_iter
            )
        except Exception as e:
            logger.warning("Failed to fetch %s splits: %s", ticker, e)
            return None

        if not splits:
            return None

        records = []
        for split in splits:
            if split.execution_date and split.split_from and split.split_to:
                factor = split.split_to / split.split_from
                records.append({
                    "ticker": ticker,
                    "date": date.fromisoformat(split.execution_date),
                    "action_type": "split",
                    "factor": factor,
                })

        if not records:
            return None

        return pl.DataFrame(records)

    # ── fetch_universe ───────────────────────────────────────────

    async def fetch_universe(self) -> list[str]:
        """Fetch the investable universe from Polygon.

        Returns ALL active US stock tickers (type='cs' = common stock).
        Unlike yfinance, this is a proper universe list.

        NOTE: This endpoint returns thousands of tickers. It makes multiple
        paginated requests. On the free tier, this may take several minutes.

        To include delisted tickers (survivorship-bias-free), set active=None
        in the API call. However, the universe is typically used for current
        universe selection, so we default to active=True.

        Returns:
            List of ticker symbols
        """
        logger.info("Fetching Polygon ticker universe (this may take a while on free tier)...")

        try:
            tickers_iter = self._client.list_tickers(
                type="cs",       # common stock
                market="stocks",
                active=True,
                limit=1000,      # max per page
            )
            tickers = await self._run_in_executor(
                self._collect_iterator, tickers_iter
            )
        except Exception as e:
            logger.error("Failed to fetch universe: %s", e)
            return []

        result = [t.ticker for t in tickers if t.ticker]
        logger.info("Fetched %d active US equity tickers", len(result))
        return result

    async def fetch_delisted_universe(self) -> list[str]:
        """Fetch delisted/inactive tickers.

        Use this to supplement the active universe for survivorship-bias-free
        backtesting. Returns tickers that were once active but are no longer
        trading.

        Returns:
            List of delisted ticker symbols
        """
        logger.info("Fetching delisted tickers from Polygon...")

        try:
            tickers_iter = self._client.list_tickers(
                type="cs",
                market="stocks",
                active=False,
                limit=1000,
            )
            tickers = await self._run_in_executor(
                self._collect_iterator, tickers_iter
            )
        except Exception as e:
            logger.error("Failed to fetch delisted universe: %s", e)
            return []

        result = [t.ticker for t in tickers if t.ticker]
        logger.info("Fetched %d delisted tickers", len(result))
        return result
"""
yfinance wrapper for fetching OHLCV price data.

Handles ticker resolution (symbol -> company name), date alignment
to trading days, and graceful error handling.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def resolve_company_name(ticker: str) -> str:
    """
    Resolve a ticker symbol to its long company name via yfinance.

    Falls back to the ticker itself if the lookup fails.
    """
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ticker
        return name
    except Exception as exc:
        logger.warning("Could not resolve company name for %s: %s", ticker, exc)
        return ticker


def fetch_price_data(
    ticker: str,
    start: datetime,
    end: datetime,
    extra_buffer_days: int = 30,
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data from yfinance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start : datetime
        Start date for the data.
    end : datetime
        End date for the data.
    extra_buffer_days : int
        Extra calendar days prepended so technical indicators have
        enough warmup data.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns [Open, High, Low, Close, Volume] indexed
        by date (timezone-naive), or None on failure.
    """
    buffered_start = start - timedelta(days=extra_buffer_days)
    # yfinance end is exclusive, so add 1 day
    adjusted_end = end + timedelta(days=1)

    try:
        df = yf.download(
            ticker,
            start=buffered_start.strftime("%Y-%m-%d"),
            end=adjusted_end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            logger.warning("No price data returned for %s", ticker)
            return None

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure timezone-naive index
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        # Keep only standard OHLCV columns
        expected = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            logger.warning("Missing columns %s for %s", missing, ticker)
            return None

        df = df[expected].dropna()
        logger.info("Fetched %d rows of price data for %s", len(df), ticker)
        return df

    except Exception as exc:
        logger.error("Price data fetch failed for %s: %s", ticker, exc)
        return None


def get_trading_dates(price_df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the sorted DatetimeIndex of trading days."""
    return price_df.index.sort_values()


def align_to_trading_day(dt: datetime, trading_dates: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    """
    Map *dt* to the correct trading day.

    Articles published after 16:00 ET (approximate market close) are
    attributed to the **next** trading session; otherwise the current day.
    If the date falls on a non-trading day the next available day is used.
    """
    market_close_hour = 16
    date_only = pd.Timestamp(dt.date())

    if dt.hour >= market_close_hour:
        # Attribute to the next trading day
        candidates = trading_dates[trading_dates > date_only]
    else:
        candidates = trading_dates[trading_dates >= date_only]

    if len(candidates) == 0:
        return None
    return candidates[0]

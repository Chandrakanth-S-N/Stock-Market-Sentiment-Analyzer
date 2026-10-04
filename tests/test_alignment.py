"""
Unit tests for time alignment and trading day mapping.
"""

from datetime import datetime

import pandas as pd
import pytest

from data.price_client import align_to_trading_day


@pytest.fixture
def trading_dates() -> pd.DatetimeIndex:
    """Sample trading dates (Mon-Fri, no weekends/holidays)."""
    return pd.DatetimeIndex([
        "2024-01-02",  # Tue
        "2024-01-03",  # Wed
        "2024-01-04",  # Thu
        "2024-01-05",  # Fri
        "2024-01-08",  # Mon (next week)
        "2024-01-09",  # Tue
    ])


class TestAlignToTradingDay:

    def test_during_market_hours(self, trading_dates: pd.DatetimeIndex) -> None:
        """Article published at 10 AM on a trading day -> same day."""
        dt = datetime(2024, 1, 3, 10, 0)
        result = align_to_trading_day(dt, trading_dates)
        assert result == pd.Timestamp("2024-01-03")

    def test_after_market_close(self, trading_dates: pd.DatetimeIndex) -> None:
        """Article published at 5 PM -> next trading day."""
        dt = datetime(2024, 1, 3, 17, 0)
        result = align_to_trading_day(dt, trading_dates)
        assert result == pd.Timestamp("2024-01-04")

    def test_friday_after_close(self, trading_dates: pd.DatetimeIndex) -> None:
        """Friday after close -> Monday (next trading day)."""
        dt = datetime(2024, 1, 5, 18, 0)
        result = align_to_trading_day(dt, trading_dates)
        assert result == pd.Timestamp("2024-01-08")

    def test_weekend_maps_to_monday(self, trading_dates: pd.DatetimeIndex) -> None:
        """Saturday article -> Monday."""
        dt = datetime(2024, 1, 6, 12, 0)  # Saturday
        result = align_to_trading_day(dt, trading_dates)
        assert result == pd.Timestamp("2024-01-08")

    def test_no_future_trading_day(self, trading_dates: pd.DatetimeIndex) -> None:
        """Date beyond the last trading day returns None."""
        dt = datetime(2024, 1, 10, 10, 0)
        result = align_to_trading_day(dt, trading_dates)
        assert result is None

    def test_at_exactly_close(self, trading_dates: pd.DatetimeIndex) -> None:
        """At exactly 16:00 -> next trading day (conservative)."""
        dt = datetime(2024, 1, 4, 16, 0)
        result = align_to_trading_day(dt, trading_dates)
        assert result == pd.Timestamp("2024-01-05")

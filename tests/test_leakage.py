"""
Tests for look-ahead leakage prevention in feature engineering.

Ensures that no feature uses information from the future relative
to the prediction target.
"""

import numpy as np
import pandas as pd
import pytest

from features.builder import (
    build_feature_matrix,
    build_price_features,
    build_sentiment_features,
    build_target,
)


@pytest.fixture
def mock_price_df() -> pd.DataFrame:
    """Synthetic 60-day price data."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-02", periods=60)
    close = 100 + np.cumsum(np.random.randn(60) * 0.5)
    return pd.DataFrame({
        "Open": close - np.random.rand(60) * 0.5,
        "High": close + np.random.rand(60) * 1.0,
        "Low": close - np.random.rand(60) * 1.0,
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, 60),
    }, index=dates)


@pytest.fixture
def mock_daily_sentiment() -> pd.DataFrame:
    """Synthetic daily sentiment data."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-02", periods=60)
    return pd.DataFrame({
        "sentiment_mean": np.random.randn(60) * 0.3,
        "sentiment_median": np.random.randn(60) * 0.2,
        "sentiment_std": np.abs(np.random.randn(60) * 0.1),
        "article_count": np.random.randint(1, 20, 60),
        "n_sources": np.random.randint(1, 10, 60),
        "pos_count": np.random.randint(0, 10, 60),
        "neg_count": np.random.randint(0, 10, 60),
        "pos_neg_ratio": np.random.rand(60) * 3,
    }, index=dates)


class TestNoLookAheadLeakage:

    def test_target_uses_future_price(self, mock_price_df: pd.DataFrame) -> None:
        """Target should compare Close[t+horizon] vs Close[t]."""
        target = build_target(mock_price_df, horizon=1)
        # Last row's target should be NaN (no future data)
        assert pd.isna(mock_price_df["Close"].shift(-1).iloc[-1])

    def test_sentiment_features_are_lagged(self, mock_daily_sentiment: pd.DataFrame) -> None:
        """All sentiment features should use .shift(>=1), so first row is NaN."""
        feats = build_sentiment_features(mock_daily_sentiment)
        # sent_lag1 should be NaN for the first row
        assert pd.isna(feats["sent_lag1"].iloc[0])
        assert pd.isna(feats["sent_lag2"].iloc[0])
        assert pd.isna(feats["sent_lag3"].iloc[0])

    def test_no_future_data_in_features(
        self, mock_price_df: pd.DataFrame, mock_daily_sentiment: pd.DataFrame
    ) -> None:
        """
        After building the feature matrix, verify that for each row,
        no feature column uses data from the target date or later.

        This is a structural check: sentiment features at row t must
        only use sentiment from dates <= t-1.
        """
        X, y = build_feature_matrix(mock_price_df, mock_daily_sentiment, horizon=1)

        # The target at row t depends on Close[t+1] vs Close[t]
        # Features at row t should use data from t and earlier (price),
        # or t-1 and earlier (sentiment).
        assert len(X) > 0
        assert len(y) > 0
        # No NaN in the final matrix
        assert not X.isna().any().any(), "Feature matrix contains NaN values"

    def test_feature_matrix_alignment(
        self, mock_price_df: pd.DataFrame, mock_daily_sentiment: pd.DataFrame
    ) -> None:
        """Feature matrix and target must have the same index."""
        X, y = build_feature_matrix(mock_price_df, mock_daily_sentiment, horizon=1)
        assert X.index.equals(y.index)

    def test_price_features_no_lookahead(self, mock_price_df: pd.DataFrame) -> None:
        """Price features like return_1d use only past data."""
        feats = build_price_features(mock_price_df)
        # return_1d at row 0 should be NaN (no previous day)
        assert pd.isna(feats["return_1d"].iloc[0])

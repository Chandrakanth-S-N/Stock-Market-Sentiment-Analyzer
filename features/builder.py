"""
Feature engineering and time alignment.

Builds the combined feature matrix from daily sentiment scores and
price/volume data, ensuring no look-ahead leakage.

All features are lagged by at least one day relative to the target.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    RSI_PERIOD,
    SMA_LONG,
    SMA_SHORT,
    SENTIMENT_MOMENTUM_WINDOWS,
    VOLATILITY_WINDOW,
    VOLUME_CHANGE_WINDOW,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------


def _compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _compute_sma_crossover(close: pd.Series) -> pd.Series:
    """SMA short/long crossover signal: 1 if short > long, else -1."""
    sma_s = close.rolling(SMA_SHORT, min_periods=SMA_SHORT).mean()
    sma_l = close.rolling(SMA_LONG, min_periods=SMA_LONG).mean()
    return (sma_s > sma_l).astype(int) * 2 - 1


def build_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive technical features from OHLCV data.

    All features use data available *before* the trading day's close,
    so they are safe to use for next-day prediction.

    Returns a DataFrame indexed by Date with feature columns.
    """
    df = price_df.copy()
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["rsi"] = _compute_rsi(df["Close"])
    df["sma_cross"] = _compute_sma_crossover(df["Close"])
    df["volatility"] = df["return_1d"].rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).std()
    df["volume_change"] = df["Volume"].pct_change(VOLUME_CHANGE_WINDOW)
    df["volume_sma_ratio"] = df["Volume"] / df["Volume"].rolling(20, min_periods=10).mean()

    features = df[
        ["return_1d", "return_5d", "rsi", "sma_cross", "volatility", "volume_change", "volume_sma_ratio"]
    ].copy()
    return features


def build_sentiment_features(daily_sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Derive sentiment features from the daily aggregated sentiment.

    Includes lagged values and momentum (rate of change over windows).
    """
    df = daily_sentiment.copy()
    # Lagged sentiment
    df["sent_lag1"] = df["sentiment_mean"].shift(1)
    df["sent_lag2"] = df["sentiment_mean"].shift(2)
    df["sent_lag3"] = df["sentiment_mean"].shift(3)
    df["sent_std_lag1"] = df["sentiment_std"].shift(1)
    df["articles_lag1"] = df["article_count"].shift(1)
    df["pos_neg_ratio_lag1"] = df["pos_neg_ratio"].shift(1)

    # Sentiment momentum
    for w in SENTIMENT_MOMENTUM_WINDOWS:
        df[f"sent_momentum_{w}d"] = df["sentiment_mean"].diff(w)

    features = df[[
        "sent_lag1", "sent_lag2", "sent_lag3",
        "sent_std_lag1", "articles_lag1", "pos_neg_ratio_lag1",
    ] + [f"sent_momentum_{w}d" for w in SENTIMENT_MOMENTUM_WINDOWS]].copy()

    return features


def build_target(price_df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Binary target: 1 if Close[t+horizon] > Close[t], else 0.

    Parameters
    ----------
    price_df : pd.DataFrame
        Must have a 'Close' column.
    horizon : int
        Number of trading days ahead.

    Returns
    -------
    pd.Series
        Named 'target_{horizon}d'.
    """
    future_return = price_df["Close"].shift(-horizon) / price_df["Close"] - 1
    target = (future_return > 0).astype(int)
    target.name = f"target_{horizon}d"
    return target


def build_feature_matrix(
    price_df: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
    horizon: int = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Combine price and sentiment features into a single aligned matrix.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (no NaNs, no future data).
    y : pd.Series
        Binary target.
    """
    price_feats = build_price_features(price_df)
    sent_feats = build_sentiment_features(daily_sentiment)

    # Merge on date index
    combined = price_feats.join(sent_feats, how="left")

    # Fill missing sentiment days with 0 (no news = neutral)
    sent_cols = sent_feats.columns.tolist()
    combined[sent_cols] = combined[sent_cols].fillna(0)

    # Target
    target = build_target(price_df, horizon=horizon)
    combined = combined.join(target, how="left")

    # Drop rows where target is NaN (last `horizon` rows) or features have NaN
    combined = combined.dropna()

    target_col = f"target_{horizon}d"
    X = combined.drop(columns=[target_col])
    y = combined[target_col].astype(int)

    logger.info(
        "Feature matrix: %d samples, %d features, target horizon=%dd",
        len(X), X.shape[1], horizon,
    )
    return X, y


def build_price_only_features(price_df: pd.DataFrame, horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a feature matrix using only price/volume features (no sentiment).

    Used as a baseline to evaluate whether sentiment adds value.
    """
    price_feats = build_price_features(price_df)
    target = build_target(price_df, horizon=horizon)
    combined = price_feats.join(target, how="left").dropna()

    target_col = f"target_{horizon}d"
    X = combined.drop(columns=[target_col])
    y = combined[target_col].astype(int)
    return X, y

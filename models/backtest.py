"""
Backtesting logic for the directional sentiment signal.

Simulates a simple long/flat strategy: go long when the model predicts
"up" (probability > threshold), otherwise stay in cash.  Accounts for
transaction costs and reports performance metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import TRANSACTION_COST_BPS

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest metrics and equity curves."""
    cumulative_return: float
    buy_hold_return: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    n_trades: int
    equity_curve: pd.Series
    buy_hold_curve: pd.Series
    daily_returns: pd.Series


def run_backtest(
    predictions: pd.Series,
    probabilities: pd.Series,
    price_df: pd.DataFrame,
    threshold: float = 0.5,
    cost_bps: float = TRANSACTION_COST_BPS,
) -> Optional[BacktestResult]:
    """
    Backtest the directional signal against buy-and-hold.

    Parameters
    ----------
    predictions : pd.Series
        Binary predictions (1 = long, 0 = flat), indexed by date.
    probabilities : pd.Series
        Predicted probability of up move.
    price_df : pd.DataFrame
        Must contain 'Close' column.
    threshold : float
        Probability threshold above which the strategy goes long.
    cost_bps : float
        Round-trip transaction cost in basis points.

    Returns
    -------
    BacktestResult or None if insufficient data.
    """
    if predictions.empty or len(predictions) < 5:
        return None

    cost_frac = cost_bps / 10_000

    # Align with price data
    close = price_df["Close"].reindex(predictions.index).dropna()
    preds_aligned = predictions.reindex(close.index).dropna()
    probs_aligned = probabilities.reindex(close.index).dropna()

    common_idx = close.index.intersection(preds_aligned.index).intersection(probs_aligned.index)
    if len(common_idx) < 5:
        return None

    close = close.loc[common_idx]
    preds_aligned = preds_aligned.loc[common_idx]
    probs_aligned = probs_aligned.loc[common_idx]

    # Daily returns
    daily_ret = close.pct_change().iloc[1:]
    signal = (probs_aligned > threshold).astype(int)
    position = signal.shift(1).iloc[1:]  # Trade on next day

    # Transaction costs
    trades = position.diff().abs().fillna(0)
    costs = trades * cost_frac

    strategy_ret = position * daily_ret - costs
    buy_hold_ret = daily_ret.copy()

    # Equity curves
    strategy_equity = (1 + strategy_ret).cumprod()
    buy_hold_equity = (1 + buy_hold_ret).cumprod()

    # Metrics
    total_strategy = strategy_equity.iloc[-1] - 1 if len(strategy_equity) > 0 else 0.0
    total_buyhold = buy_hold_equity.iloc[-1] - 1 if len(buy_hold_equity) > 0 else 0.0

    # Sharpe (annualized)
    if strategy_ret.std() > 0:
        sharpe = (strategy_ret.mean() / strategy_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = strategy_equity.cummax()
    drawdown = (strategy_equity - peak) / peak
    max_dd = drawdown.min()

    # Hit rate
    active_days = strategy_ret[position == 1]
    hit_rate = (active_days > 0).mean() if len(active_days) > 0 else 0.0

    n_trades = int(trades.sum())

    return BacktestResult(
        cumulative_return=round(total_strategy, 4),
        buy_hold_return=round(total_buyhold, 4),
        sharpe_ratio=round(sharpe, 4),
        max_drawdown=round(max_dd, 4),
        hit_rate=round(hit_rate, 4),
        n_trades=n_trades,
        equity_curve=strategy_equity,
        buy_hold_curve=buy_hold_equity,
        daily_returns=strategy_ret,
    )

"""
Reusable UI components for the Equity Sentiment Analyzer.

All chart functions use Plotly and adhere to the blue/white palette.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from config import PALETTE

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, sans-serif", size=12, color=PALETTE.text),
    paper_bgcolor=PALETTE.background,
    plot_bgcolor=PALETTE.background,
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=PALETTE.chart_grid, showline=True, linecolor=PALETTE.border),
    yaxis=dict(gridcolor=PALETTE.chart_grid, showline=True, linecolor=PALETTE.border),
    hoverlabel=dict(bgcolor=PALETTE.primary, font_size=12, font_family="Inter"),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=PALETTE.border,
        borderwidth=1,
        font=dict(size=11),
    ),
)


def _apply_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply consistent layout defaults."""
    fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, title=dict(text=title, font_size=14), height=height)
    return fig


def metric_card(label: str, value: str, subtitle: str = "") -> str:
    """Return an HTML metric card."""
    sub_html = f'<div class="metric-sub">{subtitle}</div>' if subtitle else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )


def render_metric_row(metrics: List[Dict[str, str]], cols: int = 4) -> None:
    """Render a row of metric cards."""
    columns = st.columns(cols)
    for i, m in enumerate(metrics):
        with columns[i % cols]:
            st.markdown(
                metric_card(m["label"], m["value"], m.get("subtitle", "")),
                unsafe_allow_html=True,
            )


def disclaimer_block(text: str) -> None:
    """Render the disclaimer."""
    st.markdown(f'<div class="disclaimer">{text}</div>', unsafe_allow_html=True)


def info_box(text: str) -> None:
    """Render an info box."""
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def warning_box(text: str) -> None:
    """Render a warning box."""
    st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def price_chart_with_sentiment(
    price_df: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
    start_date: pd.Timestamp,
) -> go.Figure:
    """Candlestick price chart with sentiment overlay on secondary y-axis."""
    df = price_df[price_df.index >= start_date].copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=["Price", "Sentiment"],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color=PALETTE.up_candle,
            decreasing_line_color=PALETTE.down_candle,
            increasing_fillcolor=PALETTE.up_candle,
            decreasing_fillcolor=PALETTE.down_candle,
            name="OHLC",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Sentiment overlay
    if not daily_sentiment.empty:
        sent = daily_sentiment[daily_sentiment.index >= start_date]
        fig.add_trace(
            go.Bar(
                x=sent.index, y=sent["sentiment_mean"],
                marker_color=[
                    PALETTE.positive if v >= 0 else PALETTE.negative
                    for v in sent["sentiment_mean"]
                ],
                name="Daily Sentiment",
                opacity=0.7,
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color=PALETTE.border, row=2, col=1)

    fig = _apply_layout(fig, height=520)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_annotations(font_size=12, font_color=PALETTE.text_secondary)
    return fig


def sentiment_timeseries(daily_sentiment: pd.DataFrame) -> go.Figure:
    """Line chart of daily sentiment mean with confidence band."""
    fig = go.Figure()
    ds = daily_sentiment.copy()

    upper = ds["sentiment_mean"] + ds["sentiment_std"]
    lower = ds["sentiment_mean"] - ds["sentiment_std"]

    fig.add_trace(go.Scatter(
        x=ds.index, y=upper, mode="lines",
        line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=ds.index, y=lower, mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(46,117,182,0.12)",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=ds.index, y=ds["sentiment_mean"],
        mode="lines+markers",
        line=dict(color=PALETTE.primary, width=2),
        marker=dict(size=4, color=PALETTE.primary),
        name="Sentiment (mean)",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=PALETTE.border)
    return _apply_layout(fig, title="Daily Sentiment Score", height=350)


def sentiment_histogram(articles_df: pd.DataFrame) -> go.Figure:
    """Histogram of per-article compound sentiment scores."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=articles_df["compound"],
        nbinsx=30,
        marker_color=PALETTE.accent,
        marker_line_color=PALETTE.primary,
        marker_line_width=0.5,
        opacity=0.8,
        name="Articles",
    ))
    fig.add_vline(x=0, line_dash="dot", line_color=PALETTE.border)
    return _apply_layout(fig, title="Sentiment Score Distribution", height=320)


def sentiment_vs_return_scatter(
    daily_sentiment: pd.DataFrame,
    price_df: pd.DataFrame,
) -> go.Figure:
    """Scatter plot of daily sentiment vs next-day return."""
    ret = price_df["Close"].pct_change().shift(-1)
    ret.name = "next_day_return"
    merged = daily_sentiment[["sentiment_mean"]].join(ret, how="inner").dropna()

    if merged.empty:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _apply_layout(fig, title="Sentiment vs Next-Day Return", height=350)

    corr, pval = stats.pearsonr(merged["sentiment_mean"], merged["next_day_return"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged["sentiment_mean"],
        y=merged["next_day_return"],
        mode="markers",
        marker=dict(color=PALETTE.accent, size=7, opacity=0.7,
                    line=dict(width=0.5, color=PALETTE.primary)),
        name="Days",
    ))

    # Trend line
    if len(merged) > 2:
        z = np.polyfit(merged["sentiment_mean"], merged["next_day_return"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(merged["sentiment_mean"].min(), merged["sentiment_mean"].max(), 50)
        fig.add_trace(go.Scatter(
            x=x_range, y=p(x_range),
            mode="lines",
            line=dict(color=PALETTE.negative, width=1.5, dash="dash"),
            name=f"Trend (r={corr:.3f}, p={pval:.3f})",
        ))

    fig.update_layout(
        xaxis_title="Daily Sentiment Score",
        yaxis_title="Next-Day Return",
    )
    return _apply_layout(fig, title="Sentiment vs Next-Day Return", height=370)


def article_volume_chart(articles_df: pd.DataFrame) -> go.Figure:
    """Bar chart of article count per day."""
    daily = articles_df.groupby("trading_date").size().reset_index(name="count")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily["trading_date"], y=daily["count"],
        marker_color=PALETTE.accent,
        marker_line_color=PALETTE.primary,
        marker_line_width=0.5,
        name="Articles",
    ))
    return _apply_layout(fig, title="Article Volume Over Time", height=300)


def source_breakdown_chart(articles_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of articles by source."""
    counts = articles_df["source"].value_counts().head(15)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=counts.index[::-1], x=counts.values[::-1],
        orientation="h",
        marker_color=PALETTE.accent,
        marker_line_color=PALETTE.primary,
        marker_line_width=0.5,
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _apply_layout(fig, title="Articles by Source", height=max(250, len(counts) * 25))


def confusion_matrix_chart(cm: np.ndarray) -> go.Figure:
    """Heatmap of the confusion matrix."""
    labels = ["Down", "Up"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        text=cm, texttemplate="%{text}",
        colorscale=[[0, PALETTE.light_tint], [1, PALETTE.primary]],
        showscale=False,
    ))
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
    )
    return _apply_layout(fig, title="Confusion Matrix", height=300)


def feature_importance_chart(importance: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_imp]
    vals = [x[1] for x in sorted_imp]

    fig = go.Figure()
    colors = [PALETTE.primary if "sent" in n else PALETTE.accent for n in names]
    fig.add_trace(go.Bar(
        y=names[::-1], x=vals[::-1],
        orientation="h",
        marker_color=colors[::-1],
        marker_line_width=0,
    ))
    return _apply_layout(fig, title="Feature Importance", height=max(280, len(names) * 22))


def equity_curve_chart(
    strategy: pd.Series,
    buyhold: pd.Series,
) -> go.Figure:
    """Line chart comparing strategy equity curve to buy-and-hold."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strategy.index, y=strategy.values,
        mode="lines",
        line=dict(color=PALETTE.primary, width=2),
        name="Strategy",
    ))
    fig.add_trace(go.Scatter(
        x=buyhold.index, y=buyhold.values,
        mode="lines",
        line=dict(color=PALETTE.negative, width=1.5, dash="dash"),
        name="Buy and Hold",
    ))
    fig.update_layout(yaxis_title="Equity (1 = start)")
    return _apply_layout(fig, title="Backtest: Strategy vs Buy-and-Hold", height=360)

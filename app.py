"""
Equity Sentiment Analyzer — Streamlit entry point.

Orchestrates the data pipeline: fetch news, score sentiment, build
features, train/evaluate the model, and render the analytics UI.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    APP_SUBTITLE,
    APP_TITLE,
    DEFAULT_LOOKBACK,
    DISCLAIMER,
    LOOKBACK_OPTIONS,
    MIN_ARTICLES_FOR_PREDICTION,
    MIN_DAYS_FOR_TRAINING,
    PALETTE,
    TABS,
)
from data.news_client import Article, fetch_news
from data.price_client import (
    align_to_trading_day,
    fetch_price_data,
    get_trading_dates,
    resolve_company_name,
)
from features.builder import (
    build_feature_matrix,
    build_price_only_features,
)
from models.backtest import run_backtest
from models.train import ModelResult, predict_next, train_and_evaluate
from nlp.sentiment import FinanceSentimentScorer, aggregate_daily_sentiment
from ui.components import (
    article_volume_chart,
    confusion_matrix_chart,
    disclaimer_block,
    equity_curve_chart,
    feature_importance_chart,
    info_box,
    metric_card,
    price_chart_with_sentiment,
    render_metric_row,
    sentiment_histogram,
    sentiment_timeseries,
    sentiment_vs_return_scatter,
    source_breakdown_chart,
    warning_box,
)
from ui.styles import inject_custom_css

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"## {APP_TITLE}")
    st.markdown(
        f'<p style="font-size:0.82rem;color:{PALETTE.text_secondary};margin-top:-0.5rem;">'
        f"{APP_SUBTITLE}</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    ticker = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        placeholder="e.g. AAPL, TSLA, INFY.NS",
        help="Enter a valid Yahoo Finance ticker symbol.",
    ).strip().upper()

    lookback = st.selectbox(
        "Lookback Window (days)",
        options=LOOKBACK_OPTIONS,
        index=LOOKBACK_OPTIONS.index(DEFAULT_LOOKBACK),
    )

    st.divider()
    st.markdown(
        f'<p style="font-size:0.78rem;color:{PALETTE.text_secondary};">'
        f"Model: Gradient Boosting / Logistic Regression with walk-forward validation. "
        f"Sentiment: VADER + finance lexicon."
        f"</p>",
        unsafe_allow_html=True,
    )

    run_analysis = st.button("Run Analysis", use_container_width=True)

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_news_cached(ticker: str, company: str, start_str: str, end_str: str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    return fetch_news(ticker, company, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_price_cached(ticker: str, start_str: str, end_str: str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    return fetch_price_data(ticker, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def _resolve_name_cached(ticker: str) -> str:
    return resolve_company_name(ticker)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _build_articles_df(
    articles: list, trading_dates: pd.DatetimeIndex, scorer: FinanceSentimentScorer
) -> pd.DataFrame:
    """Score articles and align them to trading days."""
    rows = []
    for art in articles:
        result = scorer.score_headline_and_description(art.title, art.description)
        td = align_to_trading_day(art.published, trading_dates)
        rows.append({
            "date": art.published,
            "trading_date": td,
            "source": art.source,
            "title": art.title,
            "description": art.description,
            "url": art.url,
            "compound": result.compound,
            "positive": result.positive,
            "negative": result.negative,
            "neutral": result.neutral,
            "label": result.label,
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["trading_date"])
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    return df


def main() -> None:
    """Run the analysis pipeline and render all tabs."""

    if not run_analysis:
        # Landing state
        st.markdown(f"# {APP_TITLE}")
        st.markdown(
            f'<p style="color:{PALETTE.text_secondary};font-size:0.92rem;">'
            f"{APP_SUBTITLE}. Enter a ticker symbol in the sidebar and select "
            f"a lookback window, then press Run Analysis.</p>",
            unsafe_allow_html=True,
        )
        disclaimer_block(DISCLAIMER)
        return

    if not ticker:
        st.error("Please enter a valid ticker symbol.")
        return

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback)
    # Extra buffer for model training
    price_start = end_date - timedelta(days=max(lookback + 120, 250))

    # ── Step 1: Resolve company name ──
    with st.spinner("Resolving ticker..."):
        company_name = _resolve_name_cached(ticker)

    # ── Step 2: Fetch price data ──
    with st.spinner("Fetching price data..."):
        price_df = _fetch_price_cached(ticker, price_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if price_df is None or price_df.empty:
        st.error(f"No price data found for ticker '{ticker}'. Please verify the symbol is valid.")
        return

    trading_dates = get_trading_dates(price_df)

    # ── Step 3: Fetch news ──
    with st.spinner("Fetching news articles..."):
        articles = _fetch_news_cached(
            ticker, company_name,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

    if not articles:
        st.warning(
            f"No news articles found for '{ticker}' ({company_name}) "
            f"in the last {lookback} days. Try a longer lookback window "
            f"or a more widely covered ticker."
        )

    # ── Step 4: Score sentiment ──
    scorer = FinanceSentimentScorer()
    articles_df = _build_articles_df(articles, trading_dates, scorer) if articles else pd.DataFrame()
    daily_sentiment = aggregate_daily_sentiment(articles_df) if not articles_df.empty else pd.DataFrame()

    # ── Step 5: Build features and train model ──
    model_result: Optional[ModelResult] = None
    direction_1d: Optional[str] = None
    prob_1d: Optional[float] = None
    conf_1d: Optional[str] = None
    direction_5d: Optional[str] = None
    prob_5d: Optional[float] = None
    conf_5d: Optional[str] = None
    backtest_result = None
    has_enough_data = (
        not daily_sentiment.empty
        and len(articles_df) >= MIN_ARTICLES_FOR_PREDICTION
        and len(price_df) >= MIN_DAYS_FOR_TRAINING
    )

    if has_enough_data:
        with st.spinner("Building features and training model..."):
            try:
                X, y = build_feature_matrix(price_df, daily_sentiment, horizon=1)
                X_po, y_po = build_price_only_features(price_df, horizon=1)

                if len(X) >= MIN_DAYS_FOR_TRAINING:
                    model_result = train_and_evaluate(X, y, X_po, y_po)

                    # Prediction
                    latest = X.iloc[[-1]]
                    direction_1d, prob_1d, conf_1d = predict_next(model_result, latest)

                    # 5-day prediction (reuse features, different target)
                    try:
                        X5, y5 = build_feature_matrix(price_df, daily_sentiment, horizon=5)
                        if len(X5) >= MIN_DAYS_FOR_TRAINING:
                            mr5 = train_and_evaluate(X5, y5)
                            latest5 = X5.iloc[[-1]]
                            direction_5d, prob_5d, conf_5d = predict_next(mr5, latest5)
                    except Exception:
                        pass

                    # Backtest
                    backtest_result = run_backtest(
                        model_result.oos_predictions,
                        model_result.oos_probabilities,
                        price_df,
                    )

            except Exception as exc:
                logger.error("Model training failed: %s", exc)
                model_result = None

    # ── Render UI ──
    st.markdown(f"# {APP_TITLE}")
    st.markdown(
        f'<p style="color:{PALETTE.text_secondary};font-size:0.88rem;margin-top:-0.5rem;">'
        f"Analysis for <strong>{ticker}</strong> ({company_name}) "
        f"&mdash; {lookback}-day lookback"
        f"</p>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(TABS)

    # ──────────────────── TAB: Overview ────────────────────
    with tabs[0]:
        _render_overview(
            ticker, company_name, articles_df, daily_sentiment,
            direction_1d, prob_1d, conf_1d,
            direction_5d, prob_5d, conf_5d,
            model_result, has_enough_data,
        )

    # ──────────────────── TAB: Sentiment ────────────────────
    with tabs[1]:
        _render_sentiment(articles_df, daily_sentiment, price_df)

    # ──────────────────── TAB: Price and Signals ────────────────────
    with tabs[2]:
        _render_price(price_df, daily_sentiment, start_date)

    # ──────────────────── TAB: Model Performance ────────────────────
    with tabs[3]:
        _render_model(model_result, backtest_result, has_enough_data)

    # ──────────────────── TAB: Articles ────────────────────
    with tabs[4]:
        _render_articles(articles_df)

    # ──────────────────── TAB: Methodology ────────────────────
    with tabs[5]:
        _render_methodology()

    disclaimer_block(DISCLAIMER)


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def _render_overview(
    ticker, company_name, articles_df, daily_sentiment,
    dir_1d, prob_1d, conf_1d,
    dir_5d, prob_5d, conf_5d,
    model_result, has_enough_data,
) -> None:
    overall_sent = 0.0
    if not daily_sentiment.empty:
        overall_sent = daily_sentiment["sentiment_mean"].mean()

    sent_label = "Positive" if overall_sent > 0.1 else ("Negative" if overall_sent < -0.1 else "Neutral")
    n_articles = len(articles_df) if not articles_df.empty else 0

    # Summary metrics
    metrics_row = [
        {"label": "Overall Sentiment", "value": f"{overall_sent:+.3f}", "subtitle": sent_label},
        {"label": "Articles Analyzed", "value": str(n_articles)},
    ]

    if dir_1d and prob_1d:
        metrics_row.append({"label": "Next-Day Outlook", "value": dir_1d, "subtitle": f"{prob_1d:.1%} ({conf_1d})"})
    else:
        metrics_row.append({"label": "Next-Day Outlook", "value": "N/A", "subtitle": "Insufficient data"})

    if dir_5d and prob_5d:
        metrics_row.append({"label": "5-Day Outlook", "value": dir_5d, "subtitle": f"{prob_5d:.1%} ({conf_5d})"})
    else:
        metrics_row.append({"label": "5-Day Outlook", "value": "N/A", "subtitle": "Insufficient data"})

    render_metric_row(metrics_row, cols=4)

    if not has_enough_data:
        warning_box(
            "There is insufficient data to generate a reliable prediction. "
            f"Only {n_articles} article(s) were found. A minimum of "
            f"{MIN_ARTICLES_FOR_PREDICTION} articles and {MIN_DAYS_FOR_TRAINING} "
            f"days of trading data is required. Try a longer lookback window "
            f"or a more widely covered ticker."
        )
    elif model_result and not model_result.is_useful:
        warning_box(
            "The model does not outperform the majority-class baseline on "
            "out-of-sample data. The prediction above should be treated with "
            "skepticism. Sentiment signal may not be informative for this "
            "ticker and time period."
        )

    # Quick chart
    if model_result and not daily_sentiment.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(sentiment_timeseries(daily_sentiment), use_container_width=True, key="overview_ts")
        with col2:
            if not articles_df.empty:
                st.plotly_chart(sentiment_histogram(articles_df), use_container_width=True, key="overview_hist")


def _render_sentiment(articles_df, daily_sentiment, price_df) -> None:
    if articles_df.empty:
        info_box("No articles available for sentiment analysis.")
        return

    st.markdown("## Daily Sentiment")
    if not daily_sentiment.empty:
        st.plotly_chart(sentiment_timeseries(daily_sentiment), use_container_width=True, key="sentiment_ts")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("## Score Distribution")
        st.plotly_chart(sentiment_histogram(articles_df), use_container_width=True, key="sentiment_hist")
    with col2:
        st.markdown("## Sentiment vs Return")
        if not daily_sentiment.empty:
            st.plotly_chart(
                sentiment_vs_return_scatter(daily_sentiment, price_df),
                use_container_width=True,
            )

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("## Article Volume")
        st.plotly_chart(article_volume_chart(articles_df), use_container_width=True)
    with col4:
        st.markdown("## Source Breakdown")
        st.plotly_chart(source_breakdown_chart(articles_df), use_container_width=True)

    # Top headlines
    st.markdown("## Top Positive Headlines")
    top_pos = articles_df.nlargest(5, "compound")[["title", "compound", "source", "date"]]
    if not top_pos.empty:
        st.dataframe(top_pos.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown("## Top Negative Headlines")
    top_neg = articles_df.nsmallest(5, "compound")[["title", "compound", "source", "date"]]
    if not top_neg.empty:
        st.dataframe(top_neg.reset_index(drop=True), use_container_width=True, hide_index=True)


def _render_price(price_df, daily_sentiment, start_date) -> None:
    st.markdown("## Price Chart with Sentiment Overlay")
    fig = price_chart_with_sentiment(
        price_df, daily_sentiment, pd.Timestamp(start_date)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quick stats
    lookback_df = price_df[price_df.index >= pd.Timestamp(start_date)]
    if not lookback_df.empty:
        latest_close = lookback_df["Close"].iloc[-1]
        period_return = (lookback_df["Close"].iloc[-1] / lookback_df["Close"].iloc[0] - 1) * 100
        period_high = lookback_df["High"].max()
        period_low = lookback_df["Low"].min()

        render_metric_row([
            {"label": "Latest Close", "value": f"${latest_close:,.2f}"},
            {"label": "Period Return", "value": f"{period_return:+.2f}%"},
            {"label": "Period High", "value": f"${period_high:,.2f}"},
            {"label": "Period Low", "value": f"${period_low:,.2f}"},
        ], cols=4)


def _render_model(model_result, backtest_result, has_enough_data) -> None:
    if not has_enough_data or model_result is None:
        info_box(
            "Model performance metrics are unavailable. "
            "Insufficient data to train and validate a model. "
            "Increase the lookback window or choose a ticker with "
            "broader news coverage."
        )
        return

    st.markdown(f"## Model: {model_result.model_name}")

    # Metrics
    render_metric_row([
        {"label": "Directional Accuracy", "value": f"{model_result.accuracy:.1%}"},
        {"label": "ROC-AUC", "value": f"{model_result.roc_auc:.3f}"},
        {"label": "F1 Score", "value": f"{model_result.f1:.3f}"},
        {"label": "Precision / Recall", "value": f"{model_result.precision:.2f} / {model_result.recall:.2f}"},
    ], cols=4)

    # Baseline comparison
    st.markdown("## Baseline Comparison")
    baseline_data = {
        "Baseline": ["Always-Up (majority class)", "Previous-Day Direction", "Price-Only Model", "Full Model (sentiment + price)"],
        "Accuracy": [
            f"{model_result.baseline_accuracy:.1%}",
            f"{model_result.prev_day_accuracy:.1%}",
            f"{model_result.price_only_accuracy:.1%}" if model_result.price_only_accuracy else "N/A",
            f"{model_result.accuracy:.1%}",
        ],
        "ROC-AUC": [
            "0.500",
            "N/A",
            f"{model_result.price_only_auc:.3f}" if model_result.price_only_auc else "N/A",
            f"{model_result.roc_auc:.3f}",
        ],
    }
    st.dataframe(pd.DataFrame(baseline_data), use_container_width=True, hide_index=True)

    if not model_result.is_useful:
        warning_box(
            "The full model does not meaningfully outperform the majority-class "
            "baseline. Sentiment features may not contain a reliable signal for "
            "this ticker and time period."
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("## Confusion Matrix")
        st.plotly_chart(confusion_matrix_chart(model_result.confusion), use_container_width=True)
    with col2:
        st.markdown("## Feature Importance")
        st.plotly_chart(feature_importance_chart(model_result.feature_importance), use_container_width=True)

    # Backtest
    if backtest_result:
        st.markdown("## Backtest Results")
        st.markdown(
            f'<p style="font-size:0.82rem;color:{PALETTE.text_secondary};">'
            f"Strategy: go long when model predicts Up (probability > 0.50), otherwise cash. "
            f"Transaction cost: {10} bps per round-trip.</p>",
            unsafe_allow_html=True,
        )
        render_metric_row([
            {"label": "Strategy Return", "value": f"{backtest_result.cumulative_return:+.2%}"},
            {"label": "Buy-and-Hold Return", "value": f"{backtest_result.buy_hold_return:+.2%}"},
            {"label": "Sharpe Ratio", "value": f"{backtest_result.sharpe_ratio:.2f}"},
            {"label": "Max Drawdown", "value": f"{backtest_result.max_drawdown:.2%}"},
        ], cols=4)
        render_metric_row([
            {"label": "Hit Rate", "value": f"{backtest_result.hit_rate:.1%}"},
            {"label": "Number of Trades", "value": str(backtest_result.n_trades)},
        ], cols=4)

        st.plotly_chart(
            equity_curve_chart(backtest_result.equity_curve, backtest_result.buy_hold_curve),
            use_container_width=True,
        )


def _render_articles(articles_df) -> None:
    if articles_df.empty:
        info_box("No articles to display.")
        return

    st.markdown(f"## Analyzed Articles ({len(articles_df)})")

    # Filter
    label_filter = st.multiselect(
        "Filter by sentiment",
        options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"],
    )

    filtered = articles_df[articles_df["label"].isin(label_filter)].copy()
    display_cols = ["date", "source", "title", "compound", "label"]
    filtered_display = filtered[display_cols].sort_values("date", ascending=False).reset_index(drop=True)
    filtered_display.columns = ["Date", "Source", "Headline", "Score", "Label"]
    filtered_display["Date"] = pd.to_datetime(filtered_display["Date"]).dt.strftime("%Y-%m-%d %H:%M")
    filtered_display["Score"] = filtered_display["Score"].round(3)

    st.dataframe(filtered_display, use_container_width=True, hide_index=True, height=500)

    # Export
    csv = articles_df.to_csv(index=False)
    st.download_button(
        label="Download articles as CSV",
        data=csv,
        file_name=f"sentiment_{articles_df.get('ticker', 'data')}.csv",
        mime="text/csv",
    )


def _render_methodology() -> None:
    st.markdown("## Methodology")

    st.markdown("""
### Data Collection

News articles are fetched via Google News RSS (default) or, if an API key
is configured, GNews or NewsAPI.org.  Both the ticker symbol and the
resolved company name are queried to improve coverage.  Articles are
deduplicated by title+URL hash and filtered for relevance.

### Sentiment Analysis

Each headline and description is scored using VADER (Valence Aware
Dictionary and sEntiment Reasoner), extended with a finance-specific
lexicon inspired by the Loughran-McDonald word lists.  The lexicon adds
terms common in financial press (e.g., "beat", "downgrade", "guidance cut",
"record revenue") with calibrated valence scores.

Headlines receive 65% weight and descriptions 35%, reflecting the
editorial emphasis of headline language.  Per-article compound scores
are aggregated to daily scores using exponential recency weighting
(half-life: 3 days) and source-count weighting.

### Feature Engineering

**Sentiment features** (all lagged by at least one day):
- Daily mean, median, and standard deviation of sentiment
- Article count and positive/negative ratio
- Sentiment momentum (3-day and 7-day change)

**Technical features**:
- 1-day and 5-day returns
- RSI (14-period)
- SMA crossover (5/20)
- Realised volatility (10-day rolling std of returns)
- Volume change (5-day) and volume/SMA-20 ratio

### Prediction Model

Two models are trained: L2-regularized logistic regression and gradient
boosting (scikit-learn).  The best is selected by out-of-sample ROC-AUC.

**Validation**: strictly time-ordered walk-forward (TimeSeriesSplit).  No
random shuffling, no look-ahead leakage.  News published after approximate
market close (16:00) is attributed to the next trading session.

**Calibration**: Platt scaling (sigmoid) via CalibratedClassifierCV.

### Backtest

A simple long/flat strategy: go long when the model predicts "up"
(probability > 0.50), otherwise hold cash.  Transaction costs are
assumed at 10 bps per round-trip.  Metrics reported: cumulative return,
buy-and-hold return, annualised Sharpe ratio, maximum drawdown, and
hit rate.

### Limitations

- **Lexicon-based sentiment** captures surface-level polarity but cannot
  understand sarcasm, context, or nuance the way a large language model
  could.
- **Google News RSS** may return fewer articles than paid APIs, especially
  for less-covered tickers or longer lookback windows.
- **Short lookback windows** (7-14 days) often yield too few data points
  for reliable model training.
- **The model is not a predictor of future returns.**  It identifies
  statistical patterns in recent data that may or may not persist.
  All reported metrics are out-of-sample, but the sample is small and
  the signal is noisy.
- Transaction costs, slippage, and market impact are approximated, not
  modelled precisely.
- **This tool is for educational purposes only and does not constitute
  financial advice.**
""")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
else:
    main()

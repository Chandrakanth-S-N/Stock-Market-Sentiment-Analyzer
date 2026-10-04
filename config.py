"""
Application-wide configuration constants.

Centralises colour palette, model hyper-parameters, UI copy,
and feature-engineering settings so nothing is hard-coded elsewhere.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# Colour palette  (restrained blue scheme)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Palette:
    """Immutable colour tokens used across UI and charts."""
    primary: str = "#1F4E79"
    accent: str = "#2E75B6"
    light_tint: str = "#EAF1F8"
    text: str = "#1A1A1A"
    text_secondary: str = "#555555"
    border: str = "#D0D7DE"
    background: str = "#FFFFFF"
    surface: str = "#F8F9FA"
    positive: str = "#2E75B6"       # muted blue for bullish
    negative: str = "#6B7B8D"       # muted slate for bearish
    neutral: str = "#A0AEC0"
    chart_line: str = "#1F4E79"
    chart_fill: str = "#EAF1F8"
    chart_grid: str = "#E8ECF0"
    up_candle: str = "#2E75B6"
    down_candle: str = "#8B9DAF"


PALETTE = Palette()

# ---------------------------------------------------------------------------
# Lookback options
# ---------------------------------------------------------------------------

LOOKBACK_OPTIONS: List[int] = [7, 14, 30, 90]
DEFAULT_LOOKBACK: int = 30

# ---------------------------------------------------------------------------
# Sentiment thresholds
# ---------------------------------------------------------------------------

SENTIMENT_BULLISH_THRESHOLD: float = 0.10
SENTIMENT_BEARISH_THRESHOLD: float = -0.10
CONFIDENCE_BINS: Dict[str, float] = {
    "High": 0.65,
    "Moderate": 0.55,
    "Low": 0.0,
}

# ---------------------------------------------------------------------------
# Model / feature settings
# ---------------------------------------------------------------------------

MIN_ARTICLES_FOR_PREDICTION: int = 5
MIN_DAYS_FOR_TRAINING: int = 30
RSI_PERIOD: int = 14
SMA_SHORT: int = 5
SMA_LONG: int = 20
VOLATILITY_WINDOW: int = 10
VOLUME_CHANGE_WINDOW: int = 5
SENTIMENT_MOMENTUM_WINDOWS: List[int] = [3, 7]
TRANSACTION_COST_BPS: float = 10.0  # basis points per round-trip

# ---------------------------------------------------------------------------
# News settings
# ---------------------------------------------------------------------------

MAX_ARTICLES_PER_QUERY: int = 100
NEWS_REQUEST_TIMEOUT: int = 15  # seconds
RATE_LIMIT_DELAY: float = 1.0  # seconds between requests

# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

MIN_TRAIN_SIZE: int = 60
N_SPLITS: int = 5

# ---------------------------------------------------------------------------
# UI copy
# ---------------------------------------------------------------------------

APP_TITLE: str = "Equity Sentiment Analyzer"
APP_SUBTITLE: str = "Lexicon-based sentiment analysis and directional outlook for equities"
DISCLAIMER: str = (
    "This tool is for informational and educational purposes only. "
    "It does not constitute financial advice, a recommendation, or a "
    "solicitation to buy or sell any security. Past performance and model "
    "outputs do not guarantee future results. Always conduct your own "
    "research and consult a qualified financial adviser before making "
    "investment decisions."
)

# ---------------------------------------------------------------------------
# Tab labels
# ---------------------------------------------------------------------------

TABS: List[str] = [
    "Overview",
    "Sentiment",
    "Price and Signals",
    "Model Performance",
    "Articles",
    "Methodology",
]

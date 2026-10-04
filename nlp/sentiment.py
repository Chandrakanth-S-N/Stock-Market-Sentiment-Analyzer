"""
Sentiment analysis engine.

Combines VADER with a finance-specific lexicon (Loughran-McDonald
inspired terms) to score financial news headlines and descriptions.

The finance lexicon adjusts VADER's valence dictionary before scoring
so that domain-specific terms receive appropriate polarity.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import SENTIMENT_BEARISH_THRESHOLD, SENTIMENT_BULLISH_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Finance lexicon loader
# ---------------------------------------------------------------------------

_LEXICON_DIR = Path(__file__).parent / "lexicon"
_LEXICON_FILE = _LEXICON_DIR / "finance_lexicon.tsv"


def _load_finance_lexicon(path: Path = _LEXICON_FILE) -> Dict[str, float]:
    """Parse the TSV lexicon file into {term: valence}."""
    lexicon: Dict[str, float] = {}
    if not path.exists():
        logger.warning("Finance lexicon file not found at %s", path)
        return lexicon
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit("\t", 1)
            if len(parts) == 2:
                term, valence = parts[0].strip().lower(), float(parts[1])
                lexicon[term] = valence
    logger.info("Loaded %d finance lexicon entries", len(lexicon))
    return lexicon


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    """Sentiment scores for a single piece of text."""
    compound: float
    positive: float
    negative: float
    neutral: float
    label: str  # "Positive", "Negative", "Neutral"


class FinanceSentimentScorer:
    """
    VADER-based sentiment scorer augmented with finance-specific lexicon.

    Usage::

        scorer = FinanceSentimentScorer()
        result = scorer.score("Company beats earnings estimates")
    """

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()
        finance_lex = _load_finance_lexicon()
        # Inject finance terms into VADER's lexicon
        self._analyzer.lexicon.update(finance_lex)
        self._multi_word_terms = {
            k: v for k, v in finance_lex.items() if " " in k
        }

    def score(self, text: str) -> SentimentResult:
        """Score a single text string."""
        # Pre-process: boost multi-word finance terms
        processed = self._inject_multiword(text)
        scores = self._analyzer.polarity_scores(processed)
        compound = scores["compound"]
        label = self._classify(compound)
        return SentimentResult(
            compound=compound,
            positive=scores["pos"],
            negative=scores["neg"],
            neutral=scores["neu"],
            label=label,
        )

    def score_headline_and_description(
        self, headline: str, description: str, headline_weight: float = 0.65
    ) -> SentimentResult:
        """
        Score an article by weighted combination of headline and description.

        Headlines carry more weight because they are editorially crafted
        to convey the primary sentiment signal.
        """
        h = self.score(headline)
        d = self.score(description) if description else h
        desc_weight = 1.0 - headline_weight
        compound = h.compound * headline_weight + d.compound * desc_weight
        pos = h.positive * headline_weight + d.positive * desc_weight
        neg = h.negative * headline_weight + d.negative * desc_weight
        neu = h.neutral * headline_weight + d.neutral * desc_weight
        label = self._classify(compound)
        return SentimentResult(
            compound=round(compound, 4),
            positive=round(pos, 4),
            negative=round(neg, 4),
            neutral=round(neu, 4),
            label=label,
        )

    def _inject_multiword(self, text: str) -> str:
        """Replace multi-word terms with a single scored token."""
        lower = text.lower()
        for term in self._multi_word_terms:
            if term in lower:
                token = term.replace(" ", "_")
                # Add token to lexicon if not present
                if token not in self._analyzer.lexicon:
                    self._analyzer.lexicon[token] = self._multi_word_terms[term]
                text = text.replace(term, token)
                lower = text.lower()
        return text

    @staticmethod
    def _classify(compound: float) -> str:
        if compound >= SENTIMENT_BULLISH_THRESHOLD:
            return "Positive"
        elif compound <= SENTIMENT_BEARISH_THRESHOLD:
            return "Negative"
        return "Neutral"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_daily_sentiment(
    articles_df: pd.DataFrame,
    decay_halflife: int = 3,
) -> pd.DataFrame:
    """
    Aggregate per-article sentiment to daily scores.

    Uses recency weighting (exponential decay) and source-count
    weighting so that days with many independent sources have
    more influence than days with a single article.

    Parameters
    ----------
    articles_df : pd.DataFrame
        Must have columns: trading_date, compound, source.
    decay_halflife : int
        Half-life in days for the exponential recency weight.

    Returns
    -------
    pd.DataFrame
        Indexed by trading_date with columns: sentiment_mean,
        sentiment_median, sentiment_std, article_count, pos_neg_ratio.
    """
    if articles_df.empty:
        return pd.DataFrame()

    df = articles_df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    max_date = df["trading_date"].max()
    days_ago = (max_date - df["trading_date"]).dt.days
    df["recency_weight"] = 0.5 ** (days_ago / decay_halflife)

    grouped = df.groupby("trading_date")

    daily = pd.DataFrame({
        "sentiment_mean": grouped.apply(
            lambda g: (g["compound"] * g["recency_weight"]).sum() / g["recency_weight"].sum()
            if g["recency_weight"].sum() > 0 else 0.0
        ),
        "sentiment_median": grouped["compound"].median(),
        "sentiment_std": grouped["compound"].std().fillna(0),
        "article_count": grouped.size(),
        "n_sources": grouped["source"].nunique(),
    })

    # Positive/negative ratio
    pos_counts = df[df["compound"] > 0].groupby("trading_date").size()
    neg_counts = df[df["compound"] < 0].groupby("trading_date").size()
    daily["pos_count"] = pos_counts.reindex(daily.index, fill_value=0)
    daily["neg_count"] = neg_counts.reindex(daily.index, fill_value=0)
    daily["pos_neg_ratio"] = daily["pos_count"] / (daily["neg_count"] + 1)

    daily.index.name = "trading_date"
    return daily.sort_index()

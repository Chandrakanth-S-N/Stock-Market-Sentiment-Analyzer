"""
Unit tests for the sentiment scoring module.
"""

import pytest

from nlp.sentiment import FinanceSentimentScorer, SentimentResult


@pytest.fixture
def scorer() -> FinanceSentimentScorer:
    return FinanceSentimentScorer()


class TestFinanceSentimentScorer:
    """Tests for FinanceSentimentScorer."""

    def test_positive_headline(self, scorer: FinanceSentimentScorer) -> None:
        result = scorer.score("Company beats earnings estimates with record revenue")
        assert result.compound > 0, f"Expected positive compound, got {result.compound}"
        assert result.label == "Positive"

    def test_negative_headline(self, scorer: FinanceSentimentScorer) -> None:
        result = scorer.score("Company faces lawsuit and major recall of products")
        assert result.compound < 0, f"Expected negative compound, got {result.compound}"
        assert result.label == "Negative"

    def test_neutral_headline(self, scorer: FinanceSentimentScorer) -> None:
        result = scorer.score("Company announces quarterly results today")
        assert -0.2 <= result.compound <= 0.2

    def test_finance_lexicon_boost(self, scorer: FinanceSentimentScorer) -> None:
        """Finance terms should shift scores more than generic VADER."""
        generic = scorer.score("The company did well")
        finance = scorer.score("The company beat estimates with strong earnings")
        assert finance.compound > generic.compound

    def test_downgrade_is_negative(self, scorer: FinanceSentimentScorer) -> None:
        result = scorer.score("Analyst downgrade sends shares lower")
        assert result.compound < 0

    def test_upgrade_is_positive(self, scorer: FinanceSentimentScorer) -> None:
        result = scorer.score("Analyst upgrade on strong guidance")
        assert result.compound > 0

    def test_headline_and_description(self, scorer: FinanceSentimentScorer) -> None:
        result = scorer.score_headline_and_description(
            headline="Company beats earnings",
            description="Revenue also exceeded expectations in all segments",
        )
        assert isinstance(result, SentimentResult)
        assert result.compound > 0

    def test_empty_description_fallback(self, scorer: FinanceSentimentScorer) -> None:
        result = scorer.score_headline_and_description(
            headline="Shares surge on record profits",
            description="",
        )
        assert result.compound > 0

    def test_compound_score_range(self, scorer: FinanceSentimentScorer) -> None:
        """VADER compound should be in [-1, 1]."""
        for text in [
            "Absolutely terrible bankruptcy fraud scandal",
            "Incredible record revenue strong bullish rally",
            "The stock traded flat",
        ]:
            result = scorer.score(text)
            assert -1.0 <= result.compound <= 1.0

    def test_multiword_term_handling(self, scorer: FinanceSentimentScorer) -> None:
        """Multi-word finance terms should be recognized."""
        result = scorer.score("The company issued a guidance cut")
        assert result.compound < 0

    def test_result_fields(self, scorer: FinanceSentimentScorer) -> None:
        result = scorer.score("Test headline")
        assert hasattr(result, "compound")
        assert hasattr(result, "positive")
        assert hasattr(result, "negative")
        assert hasattr(result, "neutral")
        assert hasattr(result, "label")
        assert result.label in ("Positive", "Negative", "Neutral")

"""
News provider interface and concrete implementations.

Supports:
  - Google News RSS (default, no API key required)
  - GNews API (optional, key via st.secrets)
  - NewsAPI.org (optional, key via st.secrets)

The active provider is selected automatically based on available keys.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import feedparser
import requests

from config import MAX_ARTICLES_PER_QUERY, NEWS_REQUEST_TIMEOUT, RATE_LIMIT_DELAY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Article:
    """A single news article."""
    title: str
    description: str
    url: str
    source: str
    published: datetime
    _dedup_key: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        raw = (self.title + self.url).lower().strip()
        self._dedup_key = hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class NewsProvider(ABC):
    """Interface every news provider must implement."""

    @abstractmethod
    def fetch(self, query: str, from_date: datetime, to_date: datetime) -> List[Article]:
        """Return articles matching *query* within the date range."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Google News RSS
# ---------------------------------------------------------------------------

class GoogleNewsRSS(NewsProvider):
    """Fetches news via Google News RSS (no API key needed)."""

    _BASE = "https://news.google.com/rss/search?q={query}+when:{days}d&hl=en&gl=US&ceid=US:en"

    @property
    def name(self) -> str:
        return "Google News RSS"

    def fetch(self, query: str, from_date: datetime, to_date: datetime) -> List[Article]:
        days = max(1, (to_date - from_date).days)
        url = self._BASE.format(query=quote_plus(query), days=days)
        articles: List[Article] = []
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:MAX_ARTICLES_PER_QUERY]:
                pub = self._parse_date(entry.get("published", ""))
                if pub and from_date <= pub <= to_date:
                    title = self._clean(entry.get("title", ""))
                    desc = self._clean(entry.get("summary", entry.get("description", "")))
                    source = entry.get("source", {}).get("title", "Unknown")
                    articles.append(Article(
                        title=title,
                        description=desc,
                        url=entry.get("link", ""),
                        source=source,
                        published=pub,
                    ))
            time.sleep(RATE_LIMIT_DELAY)
        except Exception as exc:
            logger.warning("Google News RSS fetch failed: %s", exc)
        return articles

    @staticmethod
    def _parse_date(raw: str) -> Optional[datetime]:
        for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
            try:
                return datetime.strptime(raw.strip(), fmt).replace(tzinfo=None)
            except ValueError:
                continue
        return None

    @staticmethod
    def _clean(text: str) -> str:
        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", "", text)
        return text.strip()


# ---------------------------------------------------------------------------
# GNews API
# ---------------------------------------------------------------------------

class GNewsAPI(NewsProvider):
    """Fetches news via GNews.io API (requires API key)."""

    _BASE = "https://gnews.io/api/v4/search"

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    @property
    def name(self) -> str:
        return "GNews API"

    def fetch(self, query: str, from_date: datetime, to_date: datetime) -> List[Article]:
        articles: List[Article] = []
        params = {
            "q": query,
            "lang": "en",
            "from": from_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": to_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "max": min(MAX_ARTICLES_PER_QUERY, 100),
            "apikey": self._key,
        }
        try:
            resp = requests.get(self._BASE, params=params, timeout=NEWS_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("articles", []):
                pub = self._parse_date(item.get("publishedAt", ""))
                if pub:
                    articles.append(Article(
                        title=item.get("title", ""),
                        description=item.get("description", ""),
                        url=item.get("url", ""),
                        source=item.get("source", {}).get("name", "Unknown"),
                        published=pub,
                    ))
            time.sleep(RATE_LIMIT_DELAY)
        except Exception as exc:
            logger.warning("GNews API fetch failed: %s", exc)
        return articles

    @staticmethod
    def _parse_date(raw: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------
# NewsAPI.org
# ---------------------------------------------------------------------------

class NewsAPIOrg(NewsProvider):
    """Fetches news via NewsAPI.org (requires API key)."""

    _BASE = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    @property
    def name(self) -> str:
        return "NewsAPI.org"

    def fetch(self, query: str, from_date: datetime, to_date: datetime) -> List[Article]:
        articles: List[Article] = []
        params = {
            "q": query,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": min(MAX_ARTICLES_PER_QUERY, 100),
            "apiKey": self._key,
        }
        try:
            resp = requests.get(self._BASE, params=params, timeout=NEWS_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("articles", []):
                pub = self._parse_date(item.get("publishedAt", ""))
                if pub:
                    articles.append(Article(
                        title=item.get("title", ""),
                        description=item.get("description") or "",
                        url=item.get("url", ""),
                        source=item.get("source", {}).get("name", "Unknown"),
                        published=pub,
                    ))
            time.sleep(RATE_LIMIT_DELAY)
        except Exception as exc:
            logger.warning("NewsAPI.org fetch failed: %s", exc)
        return articles

    @staticmethod
    def _parse_date(raw: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Factory and helpers
# ---------------------------------------------------------------------------

def _get_provider() -> NewsProvider:
    """Select the best available provider based on st.secrets."""
    try:
        import streamlit as st
        secrets: Dict = dict(st.secrets) if hasattr(st, "secrets") else {}
    except Exception:
        secrets = {}

    if secrets.get("NEWSAPI_KEY"):
        logger.info("Using NewsAPI.org provider")
        return NewsAPIOrg(secrets["NEWSAPI_KEY"])
    if secrets.get("GNEWS_KEY"):
        logger.info("Using GNews API provider")
        return GNewsAPI(secrets["GNEWS_KEY"])
    logger.info("Using Google News RSS provider (default)")
    return GoogleNewsRSS()


def _deduplicate(articles: List[Article]) -> List[Article]:
    """Remove duplicates by title+url hash."""
    seen: set = set()
    unique: List[Article] = []
    for a in articles:
        if a._dedup_key not in seen:
            seen.add(a._dedup_key)
            unique.append(a)
    return unique


_NOISE_PATTERNS = re.compile(
    r"(horoscope|lottery|weather forecast|cookie recipe|obituar)",
    re.IGNORECASE,
)


def _filter_irrelevant(articles: List[Article], company: str, ticker: str) -> List[Article]:
    """Keep only articles likely related to the company."""
    company_lower = company.lower()
    ticker_lower = ticker.upper()
    filtered: List[Article] = []
    for a in articles:
        text = (a.title + " " + a.description).lower()
        if _NOISE_PATTERNS.search(text):
            continue
        # Keep if ticker or company name appears
        if ticker_lower.lower() in text or company_lower in text:
            filtered.append(a)
            continue
        # Also keep if clearly financial
        if any(kw in text for kw in ("stock", "share", "market", "earnings", "revenue", "dividend")):
            filtered.append(a)
    return filtered


def fetch_news(ticker: str, company_name: str, from_date: datetime, to_date: datetime) -> List[Article]:
    """
    High-level entry point: fetch, deduplicate, and filter news articles.

    Queries both the ticker symbol and the company name to maximise coverage.
    """
    provider = _get_provider()
    logger.info("Fetching news via %s for %s / %s", provider.name, ticker, company_name)

    articles: List[Article] = []
    for query in [ticker, company_name]:
        batch = provider.fetch(query, from_date, to_date)
        articles.extend(batch)

    articles = _deduplicate(articles)
    articles = _filter_irrelevant(articles, company_name, ticker)
    articles.sort(key=lambda a: a.published, reverse=True)

    logger.info("Fetched %d articles after dedup and filtering", len(articles))
    return articles

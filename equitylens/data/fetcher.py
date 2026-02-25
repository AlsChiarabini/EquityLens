"""
equitylens.data.fetcher
~~~~~~~~~~~~~~~~~~~~~~~

Unified data acquisition layer for the EquityLens research agent.

Data sources (all **100 % free**, no paid API keys required):
  • yfinance         — price history, financial statements, company profile
  • NewsAPI          — recent news headlines  (free tier: 100 req/day)
  • Motley Fool      — earnings-call transcripts (HTML scraping, no key)

Design choices:
  - Disk cache (via *diskcache*) avoids re-fetching identical data within a
    configurable TTL — this matters both for latency and for staying inside
    free-tier rate limits.
  - Automatic retries with exponential back-off via *tenacity*.
  - All independent I/O is parallelised with ThreadPoolExecutor.
  - Both annual AND quarterly financials are fetched so downstream modules
    can compute trailing-twelve-month (TTM) metrics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from diskcache import Cache
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache setup  (default: .cache/ in project root, 24 h TTL)
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(os.getenv("EQUITYLENS_CACHE_DIR", ".cache/fetcher"))
_CACHE_TTL = int(os.getenv("EQUITYLENS_CACHE_TTL", 60 * 60 * 24))  # seconds
_cache = Cache(str(_CACHE_DIR))


def _cache_key(*parts: str) -> str:
    """Deterministic cache key from arbitrary string parts."""
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Retry decorator for transient HTTP / network errors
# ---------------------------------------------------------------------------
_http_retry = retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CompanyProfile:
    """Lightweight company identity card."""
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float
    currency: str
    description: str


@dataclass
class EquityData:
    """All raw data needed downstream by the factor engine and the LLM agent."""
    ticker: str
    profile: CompanyProfile
    prices: pd.DataFrame                # OHLCV daily, 3 yr look-back
    financials: dict[str, pd.DataFrame] # annual + quarterly statements
    info: dict                          # yfinance .info blob
    news: list[dict]                    # [{title, description, url, …}]
    transcripts: list[dict]             # earnings-call transcripts (Motley Fool)
    peers: list[str]                    # comparable tickers
    fetched_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------

class DataFetcher:
    """
    Production-grade data fetcher for single-stock equity research.

    All external I/O is:
      1. Cached to disk (configurable TTL, default 24 h)
      2. Retried with exponential back-off on transient errors
      3. Parallelised where independent

    Usage::

        fetcher = DataFetcher()
        data = fetcher.fetch("AAPL")
    """

    # -- API endpoints -------------------------------------------------------
    NEWS_API_URL     = "https://newsapi.org/v2/everything"
    FOOL_QUOTE_URL   = "https://www.fool.com/quote/{exchange}/{ticker}/"

    # Map yfinance exchange codes → Motley Fool URL slugs
    _EXCHANGE_MAP: dict[str, str] = {
        "NMS": "nasdaq",   # NASDAQ Global Select Market
        "NGM": "nasdaq",   # NASDAQ Global Market
        "NCM": "nasdaq",   # NASDAQ Capital Market
        "NYQ": "nyse",     # NYSE
        "NYS": "nyse",
        "PCX": "nyse",     # NYSE Arca
        "ASE": "amex",     # AMEX
    }

    # -- Tunables ------------------------------------------------------------
    MAX_PEERS: int            = 8
    PRICE_LOOKBACK_YEARS: int = 3
    NEWS_LOOKBACK_DAYS: int   = 7
    NEWS_MAX_ARTICLES: int    = 20
    TRANSCRIPT_LIMIT: int     = 4   # last N quarterly calls

    # Standard browser UA — Motley Fool blocks bare requests
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    def __init__(self) -> None:
        self.news_api_key: str | None = os.getenv("NEWS_API_KEY")

        if not self.news_api_key:
            logger.warning("NEWS_API_KEY not set — news fetching will be skipped.")

    # -- public API ----------------------------------------------------------

    def fetch(self, ticker: str) -> EquityData:
        """Main entry point.  Fetches *everything* for *ticker* in parallel."""
        ticker = ticker.upper().strip()
        logger.info("Fetching data for %s …", ticker)

        # Step 1 — yfinance metadata (needed by several sub-fetchers)
        yf_ticker = yf.Ticker(ticker)
        info = self._cached_info(yf_ticker, ticker)

        if not info or info.get("regularMarketPrice") is None:
            raise ValueError(f"Ticker '{ticker}' not found or delisted.")

        profile = self._build_profile(ticker, info)

        # Step 2 — fire all independent fetches in parallel
        results: dict[str, Any] = {}
        tasks = {
            "prices":      lambda: self._fetch_prices(yf_ticker, ticker),
            "financials":  lambda: self._fetch_financials(yf_ticker, ticker),
            "news":        lambda: self._fetch_news(ticker, profile.name),
            "transcripts": lambda: self._fetch_transcripts(ticker, profile.name, info),
            "peers":       lambda: self._fetch_peers(ticker, info),
        }

        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception:
                    logger.exception("Failed to fetch %s for %s", name, ticker)
                    # Graceful degradation: populate sensible defaults
                    results[name] = (
                        pd.DataFrame() if name in ("prices",) else
                        {} if name == "financials" else []
                    )

        # Validate critical data
        if results["prices"].empty:
            raise ValueError(f"No price data for {ticker} — cannot proceed.")

        logger.info(
            "Done %s  |  prices=%d rows  news=%d  transcripts=%d  peers=%s",
            ticker, len(results["prices"]), len(results["news"]),
            len(results["transcripts"]), results["peers"],
        )

        return EquityData(
            ticker=ticker,
            profile=profile,
            prices=results["prices"],
            financials=results["financials"],
            info=info,
            news=results["news"],
            transcripts=results["transcripts"],
            peers=results["peers"],
        )

    # -- yfinance helpers ----------------------------------------------------

    def _cached_info(self, yf_ticker: yf.Ticker, ticker: str) -> dict:
        """Return yfinance .info with disk-cache layer."""
        key = _cache_key("info", ticker)
        cached = _cache.get(key)
        if cached is not None:
            logger.debug("Cache hit for info/%s", ticker)
            return cached  # type: ignore[return-value]
        info = yf_ticker.info
        _cache.set(key, info, expire=_CACHE_TTL)
        return info

    def _build_profile(self, ticker: str, info: dict) -> CompanyProfile:
        return CompanyProfile(
            ticker=ticker,
            name=info.get("longName", ticker),
            sector=info.get("sector", "Unknown"),
            industry=info.get("industry", "Unknown"),
            market_cap=info.get("marketCap", 0.0),
            currency=info.get("currency", "USD"),
            description=info.get("longBusinessSummary", ""),
        )

    # -- prices --------------------------------------------------------------

    def _fetch_prices(self, yf_ticker: yf.Ticker, ticker: str) -> pd.DataFrame:
        """3-year adjusted OHLCV with basic quality checks."""
        key = _cache_key("prices", ticker)
        cached = _cache.get(key)
        if cached is not None:
            logger.debug("Cache hit for prices/%s", ticker)
            return cached  # type: ignore[return-value]

        end = datetime.today()
        start = end - timedelta(days=365 * self.PRICE_LOOKBACK_YEARS)
        prices = yf_ticker.history(start=start, end=end, auto_adjust=True)

        if prices.empty:
            raise ValueError("No price data returned.")

        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = prices[["Open", "High", "Low", "Close", "Volume"]].copy()

        # --- data quality checks ---
        pct_missing = prices.isna().mean().max()
        if pct_missing > 0.05:
            logger.warning(
                "%s price data has %.1f%% NaN — forward-filling gaps.",
                ticker, pct_missing * 100,
            )
        prices.ffill(inplace=True)
        prices.dropna(inplace=True)

        _cache.set(key, prices, expire=_CACHE_TTL)
        return prices

    # -- financials (annual + quarterly) -------------------------------------

    def _fetch_financials(self, yf_ticker: yf.Ticker, ticker: str) -> dict[str, pd.DataFrame]:
        """Annual AND quarterly statements — enables TTM computation downstream."""
        key = _cache_key("financials", ticker)
        cached = _cache.get(key)
        if cached is not None:
            logger.debug("Cache hit for financials/%s", ticker)
            return cached  # type: ignore[return-value]

        def _safe(attr: str) -> pd.DataFrame:
            try:
                df = getattr(yf_ticker, attr)
                return df if df is not None and not df.empty else pd.DataFrame()
            except Exception as exc:
                logger.warning("Could not fetch %s: %s", attr, exc)
                return pd.DataFrame()

        result = {
            # Annual
            "income_statement": _safe("income_stmt"),
            "balance_sheet":    _safe("balance_sheet"),
            "cash_flow":        _safe("cashflow"),
            # Quarterly
            "income_statement_q": _safe("quarterly_income_stmt"),
            "balance_sheet_q":    _safe("quarterly_balance_sheet"),
            "cash_flow_q":        _safe("quarterly_cashflow"),
        }

        _cache.set(key, result, expire=_CACHE_TTL)
        return result

    # -- news (NewsAPI) ------------------------------------------------------

    @_http_retry
    def _fetch_news(self, ticker: str, company_name: str) -> list[dict]:
        """Recent news via NewsAPI (free tier: 100 req/day)."""
        if not self.news_api_key:
            return []

        key = _cache_key("news", ticker)
        cached = _cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        query = f'"{company_name}" OR "{ticker}"'
        from_date = (
            datetime.today() - timedelta(days=self.NEWS_LOOKBACK_DAYS)
        ).strftime("%Y-%m-%d")

        resp = requests.get(
            self.NEWS_API_URL,
            params={
                "q": query,
                "from": from_date,
                "sortBy": "relevancy",
                "language": "en",
                "pageSize": self.NEWS_MAX_ARTICLES,
                "apiKey": self.news_api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])

        cleaned = [
            {
                "title":        a.get("title", ""),
                "description":  a.get("description", ""),
                "url":          a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
                "source":       a.get("source", {}).get("name", ""),
            }
            for a in articles
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]

        _cache.set(key, cleaned, expire=_CACHE_TTL)
        return cleaned

    # -- earnings transcripts (Motley Fool scraping, free) -------------------

    def _fetch_transcripts(self, ticker: str, company_name: str, info: dict) -> list[dict]:
        """
        Scrape the last N quarterly earnings-call transcripts from The Motley
        Fool.  **No API key required** — pure HTML scraping.

        Strategy:
          1. Visit the ticker's quote page on fool.com (e.g.
             fool.com/quote/nasdaq/aapl/) which lists recent transcript links.
          2. Filter links matching ``/earnings/call-transcripts/`` and the ticker.
          3. Fetch & parse each transcript page with BeautifulSoup.

        Each returned item contains:
            {quarter, year, date, title, content, source_url}
        """
        key = _cache_key("transcripts", ticker)
        cached = _cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        transcripts: list[dict] = []

        try:
            urls = self._search_fool_transcripts(ticker, info)
            for url in urls[: self.TRANSCRIPT_LIMIT]:
                try:
                    transcript = self._scrape_fool_transcript(url)
                    if transcript:
                        transcripts.append(transcript)
                except Exception as exc:
                    logger.warning("Failed to scrape transcript %s: %s", url, exc)
        except Exception as exc:
            logger.warning("Transcript search failed for %s: %s", ticker, exc)

        _cache.set(key, transcripts, expire=_CACHE_TTL)
        return transcripts

    # -- transcript helpers --------------------------------------------------

    @_http_retry
    def _search_fool_transcripts(self, ticker: str, info: dict) -> list[str]:
        """Return a list of Motley Fool transcript URLs for *ticker*.

        Scrapes the ticker's Motley Fool quote page which conveniently lists
        links to the most recent earnings-call transcripts.
        """
        # Resolve the exchange slug (nasdaq, nyse, …)
        yf_exchange = info.get("exchange", "")
        exchange_slug = self._EXCHANGE_MAP.get(yf_exchange, "").lower()

        # If we can't map it, try the two most common exchanges
        slugs_to_try = [exchange_slug] if exchange_slug else ["nasdaq", "nyse"]

        soup: BeautifulSoup | None = None
        for slug in slugs_to_try:
            url = self.FOOL_QUOTE_URL.format(exchange=slug, ticker=ticker.lower())
            try:
                resp = requests.get(url, headers=self._HEADERS, timeout=15)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    break
            except requests.RequestException:
                continue

        if soup is None:
            logger.warning("Could not find Motley Fool quote page for %s", ticker)
            return []

        urls: list[str] = []
        seen: set[str] = set()
        for a_tag in soup.find_all("a", href=True):
            href: str = a_tag["href"]
            if "earnings/call-transcripts" not in href:
                continue
            # We're on the ticker-specific page — every transcript link
            # belongs to this company, so no additional ticker filter needed.
            # Normalise to full URL and deduplicate.
            # Some links appear with a /4056/ prefix — strip it.
            clean = re.sub(r"^/\d+/", "/", href)
            full = clean if clean.startswith("http") else f"https://www.fool.com{clean}"
            if full not in seen:
                seen.add(full)
                urls.append(full)
        return urls

    @_http_retry
    def _scrape_fool_transcript(self, url: str) -> dict | None:
        """Fetch a single Motley Fool transcript page and extract its content."""
        resp = requests.get(url, headers=self._HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # --- title (e.g. "Apple (AAPL) Q1 2025 Earnings Call Transcript") ---
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""

        # Parse quarter & fiscal year from the title
        match = re.search(r"Q(\d)\s+(\d{4})", title)
        quarter = int(match.group(1)) if match else 0
        year = int(match.group(2)) if match else 0

        # --- date ---
        time_tag = soup.find("time")
        date_str = time_tag.get("datetime", "") if time_tag else ""

        # --- transcript body ---
        # Motley Fool wraps the transcript inside <article> or a div with
        # class "article-body".  We grab whichever exists.
        article = soup.find("article") or soup.find("div", class_="article-body")
        if not article:
            return None

        content = article.get_text(separator="\n", strip=True)

        # Drop very short pages (likely paywalled or empty stubs)
        if len(content) < 500:
            return None

        return {
            "quarter":    quarter,
            "year":       year,
            "date":       date_str,
            "title":      title,
            "content":    content,
            "source_url": url,
        }

    # -- peer discovery (sector map) -----------------------------------------

    def _fetch_peers(self, ticker: str, info: dict) -> list[str]:
        """
        Return comparable tickers in the same sector.

        Uses a curated sector map covering the 11 GICS sectors.  The
        analysed ticker is always excluded from the result.
        """
        sector = info.get("sector", "")
        all_peers = _SECTOR_PEERS.get(sector, [])
        return [p for p in all_peers if p != ticker][: self.MAX_PEERS]


# ---------------------------------------------------------------------------
# Curated sector peer map (covers all 11 GICS sectors)
# ---------------------------------------------------------------------------

_SECTOR_PEERS: dict[str, list[str]] = {
    "Technology":              ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AVGO", "ORCL", "CRM"],
    "Financial Services":      ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP"],
    "Healthcare":              ["JNJ", "UNH", "PFE", "ABT", "MRK", "TMO", "DHR", "BMY"],
    "Consumer Cyclical":       ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TGT"],
    "Consumer Defensive":      ["PG", "KO", "PEP", "WMT", "COST", "CL", "GIS", "K"],
    "Industrials":             ["HON", "UPS", "CAT", "DE", "LMT", "RTX", "GE", "MMM"],
    "Energy":                  ["XOM", "CVX", "COP", "SLB", "MPC", "PSX", "VLO", "EOG"],
    "Communication Services":  ["GOOGL", "META", "DIS", "NFLX", "CMCSA", "T", "VZ"],
    "Real Estate":             ["AMT", "PLD", "CCI", "EQIX", "PSA", "O", "SPG", "WELL"],
    "Utilities":               ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL"],
    "Basic Materials":         ["LIN", "APD", "ECL", "SHW", "FCX", "NEM", "NUE", "VMC"],
}
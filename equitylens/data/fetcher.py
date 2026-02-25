"""
equitylens.data.fetcher
~~~~~~~~~~~~~~~~~~~~~~~

Unified data acquisition layer for the EquityLens research agent.

Data sources (all free-tier):
  • yfinance      — price history, financial statements, company profile
  • NewsAPI       — recent news headlines  (FREE_API_KEY: 100 req/day)
  • FMP (FinancialModelingPrep) — earnings-call transcripts, dynamic peer
    discovery via stock screener  (FREE_API_KEY: 250 req/day)

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import yfinance as yf
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
    transcripts: list[dict]             # earnings-call transcripts (FMP)
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
    NEWS_API_URL = "https://newsapi.org/v2/everything"
    FMP_BASE     = "https://financialmodelingprep.com/api/v3"

    # -- Tunables ------------------------------------------------------------
    MAX_PEERS: int            = 8
    PRICE_LOOKBACK_YEARS: int = 3
    NEWS_LOOKBACK_DAYS: int   = 7
    NEWS_MAX_ARTICLES: int    = 20
    TRANSCRIPT_LIMIT: int     = 4   # last N quarterly calls

    def __init__(self) -> None:
        self.news_api_key: str | None = os.getenv("NEWS_API_KEY")
        self.fmp_api_key: str | None  = os.getenv("FMP_API_KEY")

        if not self.news_api_key:
            logger.warning("NEWS_API_KEY not set — news fetching will be skipped.")
        if not self.fmp_api_key:
            logger.warning("FMP_API_KEY not set — transcripts & dynamic peers disabled.")

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
            "transcripts": lambda: self._fetch_transcripts(ticker),
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

    # -- earnings transcripts (FMP, free tier: 250 req/day) ------------------

    @_http_retry
    def _fetch_transcripts(self, ticker: str) -> list[dict]:
        """
        Last N quarterly earnings-call transcripts via FinancialModelingPrep.

        Each item contains:
            {quarter, year, date, content}
        """
        if not self.fmp_api_key:
            return []

        key = _cache_key("transcripts", ticker)
        cached = _cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        transcripts: list[dict] = []
        for i in range(self.TRANSCRIPT_LIMIT):
            url = f"{self.FMP_BASE}/earning_call_transcript/{ticker}"
            try:
                resp = requests.get(
                    url,
                    params={"quarter": 4 - i, "year": datetime.today().year, "apikey": self.fmp_api_key},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                if data:
                    # FMP returns a list; take the first item per quarter
                    entry = data[0] if isinstance(data, list) else data
                    transcripts.append({
                        "quarter": entry.get("quarter", ""),
                        "year":    entry.get("year", ""),
                        "date":    entry.get("date", ""),
                        "content": entry.get("content", ""),
                    })
            except Exception as exc:
                logger.warning("Transcript fetch failed (Q%d): %s", 4 - i, exc)

        _cache.set(key, transcripts, expire=_CACHE_TTL)
        return transcripts

    # -- peer discovery (FMP screener → hardcoded fallback) ------------------

    @_http_retry
    def _fetch_peers(self, ticker: str, info: dict) -> list[str]:
        """
        Two-stage peer discovery:
          1. FMP stock screener — same sector, similar market cap (±50%)
          2. Fallback → hardcoded sector map (no API needed)

        The analysed ticker is always excluded from the peer list.
        """
        sector = info.get("sector", "")
        market_cap = info.get("marketCap", 0)

        # --- Stage 1: dynamic peers via FMP screener ------------------------
        if self.fmp_api_key and sector and market_cap:
            key = _cache_key("peers", ticker, sector, str(int(market_cap)))
            cached = _cache.get(key)
            if cached is not None:
                return cached  # type: ignore[return-value]

            try:
                resp = requests.get(
                    f"{self.FMP_BASE}/stock-screener",
                    params={
                        "sector": sector,
                        "marketCapMoreThan": int(market_cap * 0.5),
                        "marketCapLowerThan": int(market_cap * 1.5),
                        "limit": self.MAX_PEERS + 5,  # over-fetch to allow filtering
                        "apikey": self.fmp_api_key,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                peers = [
                    item["symbol"]
                    for item in resp.json()
                    if item.get("symbol") and item["symbol"] != ticker
                ][:self.MAX_PEERS]

                if peers:
                    _cache.set(key, peers, expire=_CACHE_TTL)
                    return peers
            except Exception as exc:
                logger.warning("FMP peer screener failed: %s — using fallback.", exc)

        # --- Stage 2: hardcoded fallback ------------------------------------
        all_peers = _SECTOR_PEERS.get(sector, [])
        return [p for p in all_peers if p != ticker][:self.MAX_PEERS]


# ---------------------------------------------------------------------------
# Hardcoded sector peer map (fallback when FMP key is absent or rate-limited)
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
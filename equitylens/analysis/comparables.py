"""
equitylens.analysis.comparables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Peer group comparable analysis.

Fetches key valuation multiples for sector peers via yfinance and compares
them against the analyzed stock, giving a relative ranking context.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ComparablesResult:
    """
    Peer group valuation comparison for a single stock.

    ticker          : analyzed stock
    sector          : GICS sector
    peers_analyzed  : list of peer tickers actually fetched
    metrics_df      : DataFrame (tickers × metrics) with valuation multiples
    relative_rank   : metric → rank vs peers (1 = best / cheapest)
    summary_text    : pre-formatted comparison table
    """
    ticker: str
    sector: str
    peers_analyzed: list[str]
    metrics_df: pd.DataFrame
    relative_rank: dict[str, int] = field(default_factory=dict)
    summary_text: str = ""

    def summary(self) -> str:
        return self.summary_text


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class PeerAnalyser:
    """
    Compare a stock against sector peers on key valuation multiples.

    Metrics fetched (all from yfinance .info):
      P/E      : trailing price-to-earnings
      EV/EBITDA: enterprise value to EBITDA
      P/S      : price-to-sales (TTM)
      P/B      : price-to-book
      ROE      : return on equity (higher = better)
      Margin   : net profit margin (higher = better)

    Usage::

        analyser = PeerAnalyser()
        result = analyser.analyse(equity_data, max_peers=5)
        print(result.summary())
    """

    _METRICS: dict[str, str] = {
        "P/E":       "trailingPE",
        "EV/EBITDA": "enterpriseToEbitda",
        "P/S":       "priceToSalesTrailing12Months",
        "P/B":       "priceToBook",
        "ROE %":     "returnOnEquity",
        "Margin %":  "profitMargins",
    }

    # For these metrics, higher = better (don't rank ascending)
    _HIGHER_IS_BETTER = {"ROE %", "Margin %"}

    def analyse(self, data: Any, max_peers: int = 5) -> ComparablesResult:  # data: EquityData
        """
        Fetch peer metrics and return ranked comparison.
        Runs peer fetches in parallel to minimise latency.
        """
        ticker = data.ticker
        peers = (data.peers or [])[:max_peers]
        all_tickers = [ticker] + peers

        rows: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=min(len(all_tickers), 6)) as pool:
            futures = {pool.submit(self._get_metrics, t): t for t in all_tickers}
            for future in as_completed(futures):
                t = futures[future]
                try:
                    rows[t] = future.result()
                except Exception as exc:
                    logger.warning("Metrics fetch failed for %s: %s", t, exc)
                    rows[t] = {m: None for m in self._METRICS}

        df = pd.DataFrame(rows).T
        df.index.name = "Ticker"

        # Compute relative ranks (only for columns with enough data)
        ranks: dict[str, int] = {}
        for col in df.columns:
            valid = df[col].dropna()
            if ticker not in valid.index or len(valid) < 2:
                continue
            ascending = col not in self._HIGHER_IS_BETTER
            ranks[col] = int(valid.rank(ascending=ascending).loc[ticker])

        summary_text = self._format_table(df, ticker, ranks, peers)

        return ComparablesResult(
            ticker=ticker,
            sector=data.profile.sector,
            peers_analyzed=peers,
            metrics_df=df,
            relative_rank=ranks,
            summary_text=summary_text,
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_metrics(self, ticker: str) -> dict:
        info = yf.Ticker(ticker).info
        result: dict[str, Optional[float]] = {}
        for name, key in self._METRICS.items():
            raw = info.get(key)
            result[name] = float(raw) if raw is not None else None
        return result

    def _format_table(
        self,
        df: pd.DataFrame,
        ticker: str,
        ranks: dict[str, int],
        peers: list[str],
    ) -> str:
        cols = df.columns.tolist()
        col_w = 11

        lines = [
            f"Peer Comparison — {ticker}  (sector: {df.shape[0]-1} peers)",
            "-" * (10 + col_w * len(cols)),
            "  " + f"{'Ticker':<8}" + "".join(f"{c:>{col_w}}" for c in cols),
            "  " + "-" * (8 + col_w * len(cols)),
        ]

        # All tickers: analyzed stock first, then peers
        order = [ticker] + [t for t in peers if t in df.index]
        for t in order:
            if t not in df.index:
                continue
            marker = " ★" if t == ticker else "  "
            vals = ""
            for col in cols:
                v = df.loc[t, col]
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    vals += f"{'N/A':>{col_w}}"
                elif col in self._HIGHER_IS_BETTER:
                    vals += f"{v * 100:>{col_w-1}.1f}%"
                else:
                    vals += f"{v:>{col_w}.1f}x"
            lines.append(f"{marker}{t:<8}{vals}")

        # Rank summary
        if ranks:
            n_peers = len(peers) + 1
            rank_parts = [f"{m}=#{r}/{n_peers}" for m, r in ranks.items()]
            lines.append("  " + "─" * (8 + col_w * len(cols)))
            lines.append(f"  ★ Rank vs peers: {', '.join(rank_parts)}")

        return "\n".join(lines)

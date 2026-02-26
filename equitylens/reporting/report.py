"""
equitylens.reporting.report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structured report generation for EquityLens.

Assembles factor scores, aggregation results, LLM sentiment, and peer
comparables into a human-readable equity research report — matching the
output format shown in the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Report data container
# ---------------------------------------------------------------------------

@dataclass
class EquityReport:
    """
    Complete structured equity research report for a single stock.

    Attributes
    ----------
    ticker           : stock symbol
    generated_at     : UTC timestamp
    copeland_score   : primary composite score in [0, 1]
    factor_rank_pct  : percentile rank vs peer universe (0=worst, 1=best)
    signal           : "LONG" | "NEUTRAL" | "SHORT"
    factor_scores    : dict of factor_name → score [0, 1]
    method_scores    : dict of method_name → score [0, 1]
    bull_thesis      : LLM-generated bullish argument
    bear_thesis      : LLM-generated bearish argument
    news_sentiment   : human-readable sentiment label + article count
    n_news           : number of news articles analyzed
    peer_comparison  : pre-formatted peer table (string)
    raw_text         : full formatted report string (ASCII, README style)
    """
    ticker: str
    generated_at: datetime
    copeland_score: float
    factor_rank_pct: Optional[float]
    signal: str
    factor_scores: dict[str, float] = field(default_factory=dict)
    method_scores: dict[str, float] = field(default_factory=dict)
    bull_thesis: str = ""
    bear_thesis: str = ""
    news_sentiment: str = "N/A"
    n_news: int = 0
    peer_comparison: str = ""
    raw_text: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker":         self.ticker,
            "generated_at":   self.generated_at.isoformat(),
            "copeland_score": round(self.copeland_score, 4),
            "signal":         self.signal,
            "factor_rank_pct": round(self.factor_rank_pct, 4) if self.factor_rank_pct else None,
            "factor_scores":  {k: round(v, 4) for k, v in self.factor_scores.items()},
            "method_scores":  {k: round(v, 4) for k, v in self.method_scores.items()},
            "bull_thesis":    self.bull_thesis,
            "bear_thesis":    self.bear_thesis,
            "news_sentiment": self.news_sentiment,
            "n_news":         self.n_news,
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Assembles all analysis outputs into a structured EquityReport.

    Usage::

        gen = ReportGenerator()
        report = gen.generate(factor_scores, agg_result, sentiment_result, comparables_result)
        print(report.raw_text)
    """

    # Factor taxonomy for grouped display (mirrors README)
    _FACTOR_GROUPS: dict[str, list[str]] = {
        "VALUE":         ["book_to_market", "earnings_to_price", "ebitda_to_ev", "sales_to_market"],
        "MOMENTUM":      ["momentum_12_1"],
        "PROFITABILITY": ["gross_profitability", "op_profits_to_book"],
        "INVESTMENT":    ["asset_growth", "accruals", "net_stock_issue"],
        "RISK":          ["beta", "volatility"],
        "LIQUIDITY":     ["dollar_volume", "debt_to_market"],
        "SIZE":          ["size"],
    }

    def generate(
        self,
        factor_scores: Any,           # FactorScores
        agg_result: Any,              # MultiMethodResult
        sentiment_result: Any = None, # SentimentResult | None
        comparables_result: Any = None,  # ComparablesResult | None
        data: Any = None,             # EquityData | None
    ) -> EquityReport:
        ticker = factor_scores.ticker
        copeland = agg_result.copeland_score

        signal = "LONG" if copeland >= 0.65 else ("SHORT" if copeland <= 0.35 else "NEUTRAL")

        # Approximate percentile rank from peer comparison
        factor_rank_pct: Optional[float] = None
        if comparables_result and comparables_result.relative_rank:
            ranks = list(comparables_result.relative_rank.values())
            n_peers = len(comparables_result.peers_analyzed) + 1
            avg_rank = sum(ranks) / len(ranks)
            factor_rank_pct = 1.0 - (avg_rank - 1) / max(n_peers - 1, 1)

        bull_thesis = bear_thesis = ""
        news_sentiment = "N/A"
        n_news = 0
        if sentiment_result:
            bull_thesis = sentiment_result.bull_thesis
            bear_thesis = sentiment_result.bear_thesis
            news_sentiment = (
                f"{sentiment_result.label} ({sentiment_result.n_articles} articles, 7d window)"
            )
            n_news = sentiment_result.n_articles

        raw_text = self._format_report(
            ticker, copeland, signal,
            factor_scores, agg_result,
            sentiment_result, comparables_result,
            factor_rank_pct,
        )

        return EquityReport(
            ticker=ticker,
            generated_at=datetime.utcnow(),
            copeland_score=copeland,
            factor_rank_pct=factor_rank_pct,
            signal=signal,
            factor_scores=dict(factor_scores.scores),
            method_scores=agg_result.to_dict(),
            bull_thesis=bull_thesis,
            bear_thesis=bear_thesis,
            news_sentiment=news_sentiment,
            n_news=n_news,
            peer_comparison=comparables_result.summary_text if comparables_result else "",
            raw_text=raw_text,
        )

    # -----------------------------------------------------------------------
    # ASCII report formatter
    # -----------------------------------------------------------------------

    def _format_report(
        self,
        ticker: str,
        copeland: float,
        signal: str,
        factor_scores: Any,
        agg_result: Any,
        sentiment_result: Any,
        comparables_result: Any,
        factor_rank_pct: Optional[float],
    ) -> str:
        W = 54
        sep  = "═" * W
        thin = "─" * W
        now  = datetime.utcnow().strftime("%Y-%m")

        lines = [
            sep,
            f"  EQUITYLENS REPORT — {ticker}  |  {now}",
            sep,
        ]

        # Composite score row
        bar_len = int(copeland * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"  Copeland Score   : {bar}  {copeland:.2f}")

        if factor_rank_pct is not None:
            top_pct = (1.0 - factor_rank_pct) * 100
            rank_label = f"Top {top_pct:.0f}%" if top_pct <= 50 else f"Bottom {100-top_pct:.0f}%"
            lines.append(f"  Factor Rank      : {rank_label} (vs sector peers)")

        lines.append(f"  Signal           : {signal}")
        lines.append(thin)

        # Factor groups
        scores = factor_scores.scores
        for group, factors in self._FACTOR_GROUPS.items():
            vals = [scores[f] for f in factors if f in scores]
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            bl  = int(avg * 12)
            bar = "█" * bl + "░" * (12 - bl)
            lines.append(f"  {group:<14} {bar}  {avg:.2f}")

        lines.append(thin)

        # Sentiment & theses
        if sentiment_result:
            lines.append(
                f"  News sentiment   : {sentiment_result.label} "
                f"({sentiment_result.n_articles} articles, 7d window)"
            )
            if sentiment_result.bull_thesis:
                bt = sentiment_result.bull_thesis
                if len(bt) > 75:
                    bt = bt[:75] + "..."
                lines.append(f"  Bull thesis: {bt}")
            if sentiment_result.bear_thesis:
                bt = sentiment_result.bear_thesis
                if len(bt) > 75:
                    bt = bt[:75] + "..."
                lines.append(f"  Bear thesis: {bt}")
            lines.append(thin)

        # Method comparison
        lines.append("  Aggregation methods:")
        method_scores = agg_result.to_dict()
        for method, score in sorted(method_scores.items(), key=lambda x: x[1], reverse=True):
            star  = " ★" if method == "copeland" else "  "
            bl    = int(score * 14)
            bar   = "█" * bl + "░" * (14 - bl)
            lines.append(f"  {star}{method:<20} {bar}  {score:.3f}")

        lines.append(sep)
        lines.append("  ★ = Recommended method  (Guidetti et al. EJOR 2026)")
        lines.append(sep)

        return "\n".join(lines)

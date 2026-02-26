# 15 equity factors (Guidetti et al. 2026)
"""
equitylens.factors.anomalies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computes the 15 equity anomaly factors used in:

    Guidetti, M., Insana, A., Chiarabini, L., & Mandreoli, F. (2026).
    "When Pairwise Majority Outperforms Score Aggregation:
     Multi-Criteria Ranking with Equity Anomalies."
    European Journal of Operational Research.

Each factor is a float score where HIGHER = more attractive.
All factors are normalised to [0, 1] via percentile rank at the end,
so they are directly comparable and ready for aggregation.

Factor taxonomy (following the paper):
  VALUE        : book_to_market, earnings_to_price, ebitda_to_ev, sales_to_market
  MOMENTUM     : momentum_12_1
  PROFITABILITY: gross_profitability, op_profits_to_book
  INVESTMENT   : asset_growth (inverted), accruals (inverted), net_stock_issue (inverted)
  RISK         : beta (inverted), volatility (inverted)
  LIQUIDITY    : dollar_volume, debt_to_market (inverted)
  SIZE         : size (inverted — small firms earn premium)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from equitylens.data.fetcher import EquityData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class FactorScores:
    """
    Raw and normalised factor scores for a single stock.

    raw    : dict of factor_name -> float (un-normalised, in natural units)
    scores : dict of factor_name -> float in [0,1]  (higher = more attractive)
    ticker : str
    missing: list of factors that could not be computed (insufficient data)
    """
    ticker: str
    raw: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def available(self) -> list[str]:
        return list(self.scores.keys())

    def summary(self) -> str:
        lines = [f"Factor scores — {self.ticker}", "-" * 40]
        for name, val in sorted(self.scores.items()):
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            lines.append(f"  {name:<28} {bar}  {val:.3f}")
        if self.missing:
            lines.append(f"\n  Missing ({len(self.missing)}): {', '.join(self.missing)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factor engine
# ---------------------------------------------------------------------------

class AnomalyFactors:
    """
    Computes all 15 equity anomaly factors from Guidetti et al. (2026).

    Usage::

        engine = AnomalyFactors()
        scores = engine.compute(equity_data)
        print(scores.summary())
    """

    # Factors where the raw value must be *inverted* before ranking
    # (lower raw value = more attractive stock)
    _INVERT = {
        "asset_growth",
        "accruals",
        "net_stock_issue",
        "beta",
        "volatility",
        "debt_to_market",
        "size",
    }

    def compute(self, data: EquityData) -> FactorScores:
        """Compute all 15 factors and return normalised [0,1] scores."""
        result = FactorScores(ticker=data.ticker)

        # Each _compute_* method returns a float or None
        computations = {
            # VALUE
            "book_to_market":     lambda: self._book_to_market(data),
            "earnings_to_price":  lambda: self._earnings_to_price(data),
            "ebitda_to_ev":       lambda: self._ebitda_to_ev(data),
            "sales_to_market":    lambda: self._sales_to_market(data),
            # MOMENTUM
            "momentum_12_1":      lambda: self._momentum_12_1(data),
            # PROFITABILITY
            "gross_profitability":    lambda: self._gross_profitability(data),
            "op_profits_to_book":     lambda: self._op_profits_to_book(data),
            # INVESTMENT
            "asset_growth":   lambda: self._asset_growth(data),
            "accruals":       lambda: self._accruals(data),
            "net_stock_issue":lambda: self._net_stock_issue(data),
            # RISK
            "beta":       lambda: self._beta(data),
            "volatility": lambda: self._volatility(data),
            # LIQUIDITY
            "dollar_volume":  lambda: self._dollar_volume(data),
            "debt_to_market": lambda: self._debt_to_market(data),
            # SIZE
            "size": lambda: self._size(data),
        }

        for name, fn in computations.items():
            try:
                val = fn()
                if val is not None and np.isfinite(val):
                    result.raw[name] = float(val)
                else:
                    result.missing.append(name)
            except Exception as exc:
                logger.debug("Factor '%s' failed for %s: %s", name, data.ticker, exc)
                result.missing.append(name)

        # Normalise raw scores to [0, 1] using percentile-rank approach.
        # For a single stock we use a sigmoid squeeze around the cross-sectional
        # mean of each factor — this gives stable [0,1] scores even without a
        # full universe. For cross-sectional ranking across multiple stocks,
        # call normalise_universe() instead.
        result.scores = _single_stock_normalise(result.raw, self._INVERT)

        if result.missing:
            logger.warning(
                "%s: %d factor(s) missing: %s",
                data.ticker, len(result.missing), result.missing
            )

        return result

    # -----------------------------------------------------------------------
    # VALUE factors
    # -----------------------------------------------------------------------

    def _book_to_market(self, data: EquityData) -> Optional[float]:
        """
        Book-to-Market ratio.
        Higher B/M → cheaper stock relative to book value → attractive.
        Source: Fama & French (1992); factor 4 in Guidetti et al. (2026).
        """
        bs = data.financials.get("balance_sheet", pd.DataFrame())
        if bs.empty:
            return None

        book_value = _get_item(bs, [
            "Stockholders Equity",
            "Total Stockholders Equity",
            "Common Stock Equity",
        ])
        market_cap = data.info.get("marketCap")

        if book_value is None or market_cap is None or market_cap <= 0:
            return None

        return book_value / market_cap

    def _earnings_to_price(self, data: EquityData) -> Optional[float]:
        """
        Earnings-to-Price (inverse P/E).
        Higher E/P → cheaper stock → attractive.
        Source: Basu (1983); factor 7 in Guidetti et al. (2026).
        """
        eps = data.info.get("trailingEps")
        price = data.info.get("regularMarketPrice") or data.info.get("currentPrice")

        if eps is None or price is None or price <= 0:
            return None

        return eps / price

    def _ebitda_to_ev(self, data: EquityData) -> Optional[float]:
        """
        EBITDA / Enterprise Value.
        Higher → cheaper on cash-flow basis → attractive.
        Source: Loughran & Wellman (2011); factor 8 in Guidetti et al. (2026).
        """
        ebitda = data.info.get("ebitda")
        ev = data.info.get("enterpriseValue")

        if ebitda is None or ev is None or ev <= 0:
            return None

        return ebitda / ev

    def _sales_to_market(self, data: EquityData) -> Optional[float]:
        """
        Sales / Market Cap (inverse Price-to-Sales).
        Higher → cheaper → attractive.
        Source: Barbee et al. (1996); factor 13 in Guidetti et al. (2026).
        """
        revenue = data.info.get("totalRevenue")
        market_cap = data.info.get("marketCap")

        if revenue is None or market_cap is None or market_cap <= 0:
            return None

        return revenue / market_cap

    # -----------------------------------------------------------------------
    # MOMENTUM factor
    # -----------------------------------------------------------------------

    def _momentum_12_1(self, data: EquityData) -> Optional[float]:
        """
        12-1 month price momentum (skip-one-month).
        Return from t-12 to t-1, skipping the most recent month.
        Higher → stronger trend → attractive (in classic momentum).
        Source: Jegadeesh & Titman (1993); factor 10 in Guidetti et al. (2026).
        """
        prices = data.prices["Close"]
        if len(prices) < 260:
            return None

        p_now   = prices.iloc[-22]   # price ~1 month ago  (skip last month)
        p_start = prices.iloc[-252]  # price ~12 months ago

        if p_start <= 0:
            return None

        return (p_now - p_start) / p_start

    # -----------------------------------------------------------------------
    # PROFITABILITY factors
    # -----------------------------------------------------------------------

    def _gross_profitability(self, data: EquityData) -> Optional[float]:
        """
        Gross Profit / Total Assets.
        Higher → more efficient value creation → attractive.
        Source: Novy-Marx (2013); factor 9 in Guidetti et al. (2026).
        """
        is_ = data.financials.get("income_statement", pd.DataFrame())
        bs  = data.financials.get("balance_sheet",    pd.DataFrame())
        if is_.empty or bs.empty:
            return None

        gross_profit = _get_item(is_, ["Gross Profit"])
        total_assets = _get_item(bs,  ["Total Assets"])

        if gross_profit is None or total_assets is None or total_assets <= 0:
            return None

        return gross_profit / total_assets

    def _op_profits_to_book(self, data: EquityData) -> Optional[float]:
        """
        Operating Income / Book Equity.
        Higher → more profitable per unit of equity → attractive.
        Source: Fama & French (2015); factor 12 in Guidetti et al. (2026).
        """
        is_ = data.financials.get("income_statement", pd.DataFrame())
        bs  = data.financials.get("balance_sheet",    pd.DataFrame())
        if is_.empty or bs.empty:
            return None

        op_income = _get_item(is_, ["Operating Income", "EBIT"])
        book_eq   = _get_item(bs,  [
            "Stockholders Equity",
            "Total Stockholders Equity",
            "Common Stock Equity",
        ])

        if op_income is None or book_eq is None or book_eq <= 0:
            return None

        return op_income / book_eq

    # -----------------------------------------------------------------------
    # INVESTMENT factors  (all inverted: lower = more attractive)
    # -----------------------------------------------------------------------

    def _asset_growth(self, data: EquityData) -> Optional[float]:
        """
        Year-over-year growth in total assets.
        Lower (conservative investment) → attractive.
        Source: Cooper et al. (2008); factor 2 in Guidetti et al. (2026).
        """
        bs = data.financials.get("balance_sheet", pd.DataFrame())
        if bs.empty or bs.shape[1] < 2:
            return None

        row = _find_row(bs, ["Total Assets"])
        if row is None:
            return None

        # bs columns are ordered most-recent first
        curr = bs.loc[row, bs.columns[0]]
        prev = bs.loc[row, bs.columns[1]]

        if pd.isna(curr) or pd.isna(prev) or prev == 0:
            return None

        return float((curr - prev) / abs(prev))

    def _accruals(self, data: EquityData) -> Optional[float]:
        """
        Accruals = (Net Income - Operating Cash Flow) / Total Assets.
        Lower accruals → higher earnings quality → attractive.
        Source: Sloan (1996); factor 1 in Guidetti et al. (2026).
        """
        is_ = data.financials.get("income_statement", pd.DataFrame())
        cf  = data.financials.get("cash_flow",         pd.DataFrame())
        bs  = data.financials.get("balance_sheet",     pd.DataFrame())
        if is_.empty or cf.empty or bs.empty:
            return None

        net_income = _get_item(is_, ["Net Income", "Net Income Common Stockholders"])
        cfo        = _get_item(cf,  ["Operating Cash Flow", "Cash Flow From Operations"])
        assets     = _get_item(bs,  ["Total Assets"])

        if any(v is None for v in (net_income, cfo, assets)) or assets <= 0:
            return None

        return float((net_income - cfo) / assets)

    def _net_stock_issue(self, data: EquityData) -> Optional[float]:
        """
        Net stock issuance = log(shares_t / shares_{t-1}).
        Lower (buybacks > issuances) → attractive.
        Source: Pontiff & Woodgate (2008); factor 11 in Guidetti et al. (2026).

        We compare "Ordinary Shares Number" (or "Share Issued") across the
        two most recent annual balance sheets.  Falls back to yfinance's
        ``sharesOutstanding`` vs the older period if the row is missing.
        """
        bs = data.financials.get("balance_sheet", pd.DataFrame())
        if bs.empty or bs.shape[1] < 2:
            return None

        row = _find_row(bs, [
            "Ordinary Shares Number",
            "Share Issued",
            "Common Stock",
        ])

        if row is not None:
            curr = bs.loc[row, bs.columns[0]]
            prev = bs.loc[row, bs.columns[1]]
            if not pd.isna(curr) and not pd.isna(prev) and prev > 0:
                return float(np.log(curr / prev))

        # Fallback: if balance-sheet rows are unavailable, approximation
        # from diluted shares in income statement.
        is_ = data.financials.get("income_statement", pd.DataFrame())
        if not is_.empty and is_.shape[1] >= 2:
            row2 = _find_row(is_, ["Diluted Average Shares", "Basic Average Shares"])
            if row2 is not None:
                curr = is_.loc[row2, is_.columns[0]]
                prev = is_.loc[row2, is_.columns[1]]
                if not pd.isna(curr) and not pd.isna(prev) and prev > 0:
                    return float(np.log(curr / prev))

        return None

    # -----------------------------------------------------------------------
    # RISK factors  (both inverted: lower risk = more attractive)
    # -----------------------------------------------------------------------

    def _beta(self, data: EquityData) -> Optional[float]:
        """
        Market beta (CAPM).  Lower beta → defensive → attractive in this ranking.
        Source: Frazzini & Pedersen (2014) BAB; factor 3 in Guidetti et al. (2026).
        """
        beta = data.info.get("beta")
        return float(beta) if beta is not None else None

    def _volatility(self, data: EquityData) -> Optional[float]:
        """
        Annualised realised volatility (252-day).
        Lower vol → attractive (low-vol anomaly).
        Source: Ang et al. (2006); factor 15 in Guidetti et al. (2026).
        """
        prices = data.prices["Close"]
        if len(prices) < 30:
            return None

        returns = prices.pct_change().dropna()
        return float(returns.std() * np.sqrt(252))

    # -----------------------------------------------------------------------
    # LIQUIDITY factors
    # -----------------------------------------------------------------------

    def _dollar_volume(self, data: EquityData) -> Optional[float]:
        """
        Average daily dollar trading volume (log-scaled).
        Higher → more liquid → attractive.
        Source: Brennan et al. (1998); factor 6 in Guidetti et al. (2026).
        """
        prices = data.prices
        if prices.empty:
            return None

        dollar_vol = (prices["Close"] * prices["Volume"]).tail(63).mean()  # ~1 quarter
        if dollar_vol <= 0:
            return None

        return float(np.log1p(dollar_vol))

    def _debt_to_market(self, data: EquityData) -> Optional[float]:
        """
        Total Debt / Market Cap.
        Lower leverage → more conservative → attractive (inverted factor).
        Source: Bhandari (1988); factor 5 in Guidetti et al. (2026).
        """
        total_debt = data.info.get("totalDebt")
        market_cap = data.info.get("marketCap")

        if total_debt is None or market_cap is None or market_cap <= 0:
            return None

        return float(total_debt / market_cap)

    # -----------------------------------------------------------------------
    # SIZE factor  (inverted: smaller firms earn size premium)
    # -----------------------------------------------------------------------

    def _size(self, data: EquityData) -> Optional[float]:
        """
        Log(Market Capitalisation).
        Lower size → small-cap premium → attractive (inverted).
        Source: Banz (1981); factor 14 in Guidetti et al. (2026).
        """
        market_cap = data.info.get("marketCap")
        if market_cap is None or market_cap <= 0:
            return None

        return float(np.log(market_cap))


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

# Empirical priors for z-standardisation across the U.S. equity universe.
# Derived from typical large-/mid-cap distributions so that a sigmoid
# produces meaningful [0,1] scores even for a *single* stock.
# (mean, std) — factors in _INVERT are negated *before* standardisation.
_FACTOR_PRIORS: dict[str, tuple[float, float]] = {
    "book_to_market":      (0.40,  0.35),
    "earnings_to_price":   (0.04,  0.04),
    "ebitda_to_ev":        (0.08,  0.06),
    "sales_to_market":     (0.80,  0.70),
    "momentum_12_1":       (0.10,  0.25),
    "gross_profitability": (0.25,  0.20),
    "op_profits_to_book":  (0.25,  0.25),
    "asset_growth":        (0.08,  0.12),   # inverted: negated then z-scored
    "accruals":            (-0.03, 0.06),   # inverted
    "net_stock_issue":     (0.00,  0.05),   # inverted
    "beta":                (1.00,  0.40),   # inverted
    "volatility":          (0.30,  0.15),   # inverted
    "dollar_volume":       (18.0,  2.50),
    "debt_to_market":      (0.50,  0.40),   # inverted
    "size":                (23.5,  2.00),   # inverted
}


def _single_stock_normalise(
    raw: dict[str, float],
    invert_set: set[str],
) -> dict[str, float]:
    """
    Maps raw factor values to [0, 1] using a z-scored sigmoid transform.

    For each factor:
      1. Invert if in *invert_set* (multiply by -1)
      2. Z-standardise against empirical market priors  (mean, std)
      3. Apply sigmoid:  1 / (1 + exp(-z))

    The priors in ``_FACTOR_PRIORS`` are calibrated on typical U.S. large-/
    mid-cap equities so that 0.5 ≈ 'market-average'.  Cross-sectional
    percentile ranking across a full universe is in ``normalise_universe()``.
    """
    scores: dict[str, float] = {}

    for name, val in raw.items():
        v = -val if name in invert_set else val

        # Z-standardise against market priors
        mu, sigma = _FACTOR_PRIORS.get(name, (0.0, 1.0))
        z = (v - mu) / sigma if sigma > 0 else 0.0

        # Clip to avoid numerical extremes, then sigmoid
        z = float(np.clip(z, -5, 5))
        scores[name] = float(1.0 / (1.0 + np.exp(-z)))

    return scores


def normalise_universe(
    factor_df: pd.DataFrame,
    invert_set: set[str] | None = None,
) -> pd.DataFrame:
    """
    Cross-sectional percentile-rank normalisation for a universe of stocks.

    Args:
        factor_df : DataFrame with shape (n_stocks, n_factors), raw values.
        invert_set: factor names where lower raw = better.

    Returns:
        DataFrame of same shape with values in [0, 1].

    Used in notebooks/02_copeland_vs_baselines.ipynb to replicate
    the cross-sectional ranking methodology of Guidetti et al. (2026).
    """
    invert_set = invert_set or set()
    result = factor_df.copy()

    for col in result.columns:
        series = result[col]
        if col in invert_set:
            series = -series
        # Percentile rank: each stock's position in [0,1] within the universe
        result[col] = series.rank(pct=True)

    return result


# ---------------------------------------------------------------------------
# DataFrame lookup helpers
# ---------------------------------------------------------------------------

def _find_row(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first index label in *candidates* that exists in df.index."""
    for c in candidates:
        if c in df.index:
            return c
    return None


def _get_item(df: pd.DataFrame, candidates: list[str]) -> Optional[float]:
    """
    Extract the most recent (first column) value for the first matching row.
    yfinance returns financial statements with most-recent period first.
    """
    row = _find_row(df, candidates)
    if row is None:
        return None

    val = df.loc[row].iloc[0]

    if pd.isna(val):
        return None

    return float(val)
"""
equitylens.data.preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data-quality pipeline applied to raw EquityData before factor computation.

Steps:
  1. Validate completeness — flag tickers with insufficient history
  2. Forward-fill & interpolate small price gaps
  3. Winsorise extreme financial-statement values (1st / 99th percentile)
  4. Currency normalisation (stub — all USD for now)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from equitylens.data.fetcher import EquityData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess(data: EquityData, *, min_price_rows: int = 200) -> EquityData:
    """
    Run the full cleaning pipeline on an EquityData object **in-place**
    and return the same reference.

    Raises ValueError if data is too thin to be usable.
    """
    _validate_prices(data, min_rows=min_price_rows)
    data.prices = _clean_prices(data.prices, data.ticker)
    data.financials = _clean_financials(data.financials, data.ticker)
    return data


# ---------------------------------------------------------------------------
# Price cleaning
# ---------------------------------------------------------------------------


def _validate_prices(data: EquityData, *, min_rows: int = 200) -> None:
    if data.prices is None or len(data.prices) < min_rows:
        raise ValueError(
            f"{data.ticker}: only {len(data.prices) if data.prices is not None else 0} "
            f"price rows — need at least {min_rows}."
        )


def _clean_prices(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Forward-fill small gaps, drop remaining NaNs, sort chronologically."""
    prices = prices.sort_index()

    gap_pct = prices.isna().mean().max()
    if gap_pct > 0:
        logger.info("%s: filling %.1f%% price gaps via ffill.", ticker, gap_pct * 100)
        prices = prices.ffill().bfill()

    prices = prices.dropna()
    return prices


# ---------------------------------------------------------------------------
# Financial-statement cleaning
# ---------------------------------------------------------------------------


def _clean_financials(
    financials: dict[str, pd.DataFrame], ticker: str
) -> dict[str, pd.DataFrame]:
    """Winsorise each statement's numeric columns at 1st / 99th percentile."""
    cleaned: dict[str, pd.DataFrame] = {}
    for name, df in financials.items():
        if df.empty:
            cleaned[name] = df
            continue
        cleaned[name] = _winsorise(df, ticker=ticker, label=name)
    return cleaned


def _winsorise(
    df: pd.DataFrame,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
    ticker: str = "",
    label: str = "",
) -> pd.DataFrame:
    """Clip numeric columns to [lower, upper] percentiles row-wise."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    if num_cols.empty:
        return df

    df = df.copy()
    for col in num_cols:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        clipped = df[col].clip(lo, hi)
        n_clipped = (clipped != df[col]).sum()
        if n_clipped:
            logger.debug(
                "%s/%s: winsorised %d values in '%s'.", ticker, label, n_clipped, col
            )
        df[col] = clipped
    return df

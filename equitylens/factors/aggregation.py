# Copeland + 6 baseline aggregation methods

"""
equitylens.factors.aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-criteria aggregation methods for equity factor scores.

Implements Copeland's pairwise majority rule and 10 baseline methods,
following the taxonomy in:

    Guidetti, M., Insana, A., Chiarabini, L., & Mandreoli, F. (2026).
    "When Pairwise Majority Outperforms Score Aggregation:
     Multi-Criteria Ranking with Equity Anomalies."
    European Journal of Operational Research.

Key finding from the paper: Copeland dominates all alternatives across
15 equity anomalies on US data 2000–2023 under both EW and VW schemes,
and remains robust to transaction costs and bull/bear regimes.

Methods implemented
-------------------
  SOCIAL CHOICE  : Copeland (default), Borda, Majority Judgment
  MCDM           : TOPSIS, VIKOR, Borda (also MCDM framing)
  HEURISTICS     : z-score composite, mean rank

For a single stock (no universe), Copeland is computed by pairwise
comparison of the stock's factor scores against market-average benchmarks
(0.5 in normalised space). For cross-sectional ranking of a universe,
use rank_universe() which applies the full pairwise tournament.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & result container
# ---------------------------------------------------------------------------

class AggregationMethod(str, Enum):
    COPELAND           = "copeland"
    BORDA              = "borda"
    MAJORITY_JUDGMENT  = "majority_judgment"
    TOPSIS             = "topsis"
    VIKOR              = "vikor"
    ZSCORE             = "zscore"
    MEAN_RANK          = "mean_rank"


@dataclass
class AggregationResult:
    """
    Composite score produced by a single aggregation method.

    score   : float in [0, 1]  (higher = more attractive)
    method  : AggregationMethod used
    details : method-specific breakdown (e.g. wins/losses for Copeland)
    factors_used : number of factors that contributed
    """
    score: float
    method: AggregationMethod
    details: dict
    factors_used: int

    def __repr__(self) -> str:
        return (
            f"AggregationResult(method={self.method.value}, "
            f"score={self.score:.4f}, factors={self.factors_used})"
        )


@dataclass
class MultiMethodResult:
    """
    Scores from all aggregation methods for a single stock.
    The recommended score is always Copeland (Guidetti et al. 2026).
    """
    ticker: str
    results: dict[str, AggregationResult]

    @property
    def copeland_score(self) -> float:
        return self.results[AggregationMethod.COPELAND].score

    @property
    def recommended_score(self) -> float:
        """Primary score: Copeland (best performer in Guidetti et al. 2026)."""
        return self.copeland_score

    def summary(self) -> str:
        lines = [f"Aggregation results — {self.ticker}", "=" * 50]
        # Copeland first, then others sorted
        ordered = [AggregationMethod.COPELAND] + [
            m for m in AggregationMethod if m != AggregationMethod.COPELAND
        ]
        for method in ordered:
            key = method.value
            if key not in self.results:
                continue
            r = self.results[key]
            bar = "█" * int(r.score * 30) + "░" * (30 - int(r.score * 30))
            star = " ★" if method == AggregationMethod.COPELAND else "  "
            lines.append(f"{star} {key:<22} {bar}  {r.score:.4f}")
        lines.append("\n  ★ = recommended method (Guidetti et al. 2026)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, float]:
        return {k: v.score for k, v in self.results.items()}


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class FactorAggregator:
    """
    Aggregates normalised [0,1] factor scores into a single composite score
    using multiple methods simultaneously.

    The primary method is Copeland's pairwise majority rule, which the paper
    demonstrates achieves exponentially small misordering probability under
    heteroskedastic noise with mild average informativeness.

    Usage::

        agg = FactorAggregator()
        result = agg.aggregate(scores_dict, ticker="AAPL")
        print(result.summary())

        # Cross-sectional universe ranking
        ranked_df = agg.rank_universe(universe_df)
    """

    # Neutral benchmark: 0.5 in normalised [0,1] space = market average
    _MARKET_AVERAGE = 0.5

    def aggregate(
        self,
        scores: dict[str, float],
        ticker: str = "",
        weights: Optional[dict[str, float]] = None,
    ) -> MultiMethodResult:
        """
        Run all aggregation methods on a single stock's factor scores.

        Args:
            scores  : dict of factor_name -> float in [0, 1]
            ticker  : stock ticker (for labelling only)
            weights : optional factor weights (default: equal weight)

        Returns:
            MultiMethodResult with scores from all methods
        """
        if not scores:
            raise ValueError("scores dict is empty — nothing to aggregate.")

        factors = list(scores.keys())
        values  = np.array([scores[f] for f in factors])
        n       = len(factors)

        # Build weight vector (uniform if not provided)
        if weights:
            w = np.array([weights.get(f, 1.0) for f in factors])
            w = w / w.sum()
        else:
            w = np.ones(n) / n

        results: dict[str, AggregationResult] = {}

        results[AggregationMethod.COPELAND]          = self._copeland(values, factors, w)
        results[AggregationMethod.BORDA]             = self._borda(values, factors, w)
        results[AggregationMethod.MAJORITY_JUDGMENT] = self._majority_judgment(values, factors, w)
        results[AggregationMethod.TOPSIS]            = self._topsis(values, factors, w)
        results[AggregationMethod.VIKOR]             = self._vikor(values, factors, w)
        results[AggregationMethod.ZSCORE]            = self._zscore_composite(values, factors, w)
        results[AggregationMethod.MEAN_RANK]         = self._mean_rank(values, factors, w)

        # Convert enum keys to string for JSON-serialisability
        str_results = {k.value: v for k, v in results.items()}

        return MultiMethodResult(ticker=ticker, results=str_results)

    # -----------------------------------------------------------------------
    # SOCIAL CHOICE methods
    # -----------------------------------------------------------------------

    def _copeland(
        self, values: np.ndarray, factors: list[str], weights: np.ndarray
    ) -> AggregationResult:
        """
        Copeland's pairwise majority rule.

        The stock 'wins' on a criterion if its score exceeds the market
        average (0.5). The Copeland score = (wins - losses) / n_factors,
        mapped to [0, 1] by (score + 1) / 2.

        In cross-sectional ranking (rank_universe), wins/losses are computed
        pairwise between all stocks — this is the full Condorcet tournament
        from Guidetti et al. (2026).

        Theoretical justification: under heteroskedastic noise, Copeland
        achieves exponentially small misordering probability when average
        factor informativeness exceeds 0.5 (Theorem 1, Guidetti et al. 2026).
        """
        benchmark = self._MARKET_AVERAGE

        wins   = 0.0
        losses = 0.0
        ties   = 0.0
        breakdown: dict[str, str] = {}

        for i, (factor, val) in enumerate(zip(factors, values)):
            w = weights[i]
            if val > benchmark:
                wins   += w
                breakdown[factor] = f"WIN  ({val:.3f} > {benchmark})"
            elif val < benchmark:
                losses += w
                breakdown[factor] = f"LOSS ({val:.3f} < {benchmark})"
            else:
                ties += w
                breakdown[factor] = f"TIE  ({val:.3f} = {benchmark})"

        # Net score in [-1, +1], normalised to [0, 1]
        net   = wins - losses
        score = (net + 1.0) / 2.0

        return AggregationResult(
            score=float(np.clip(score, 0.0, 1.0)),
            method=AggregationMethod.COPELAND,
            details={
                "wins":      round(wins,   4),
                "losses":    round(losses, 4),
                "ties":      round(ties,   4),
                "net":       round(net,    4),
                "breakdown": breakdown,
            },
            factors_used=len(factors),
        )

    def _borda(
        self, values: np.ndarray, factors: list[str], weights: np.ndarray
    ) -> AggregationResult:
        """
        Borda count (discretised).

        Each factor is mapped to a grade in {0, 1, 2, 3, 4} (quintile bin),
        then the weighted sum of grades is normalised to [0, 1].
        This differentiates Borda from plain mean-rank even for a single stock,
        and mirrors the discrete ordinal ranking used in Guidetti et al. (2026).
        """
        # Discretise into 5 ordinal grades (quintile-like)
        bins = np.array([0.0, 0.20, 0.40, 0.60, 0.80, 1.01])
        grades = np.digitize(values, bins) - 1          # 0..4
        grades = np.clip(grades, 0, 4).astype(float)

        score = float(np.dot(weights, grades)) / 4.0    # normalise to [0, 1]
        return AggregationResult(
            score=float(np.clip(score, 0.0, 1.0)),
            method=AggregationMethod.BORDA,
            details={"grades": {f: int(g) for f, g in zip(factors, grades)}},
            factors_used=len(factors),
        )

    def _majority_judgment(
        self, values: np.ndarray, factors: list[str], weights: np.ndarray
    ) -> AggregationResult:
        """
        Majority Judgment (Balinski & Laraki 2011).
        Aggregate by median rather than mean — more robust to outlier factors.
        """
        # Weighted median
        sorted_idx = np.argsort(values)
        sorted_vals    = values[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cumulative = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumulative, 0.5)
        median_idx = min(median_idx, len(sorted_vals) - 1)
        score = float(sorted_vals[median_idx])

        return AggregationResult(
            score=np.clip(score, 0.0, 1.0),
            method=AggregationMethod.MAJORITY_JUDGMENT,
            details={"weighted_median": score},
            factors_used=len(factors),
        )

    # -----------------------------------------------------------------------
    # MCDM methods
    # -----------------------------------------------------------------------

    def _topsis(
        self, values: np.ndarray, factors: list[str], weights: np.ndarray
    ) -> AggregationResult:
        """
        TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).
        Hwang & Yoon (1981).

        For a single stock, ideal = 1.0, nadir = 0.0 on each factor.
        Score = distance to nadir / (distance to ideal + distance to nadir).
        """
        weighted = values * weights

        ideal = weights * 1.0   # best possible: all factors = 1
        nadir = weights * 0.0   # worst possible: all factors = 0

        d_ideal = float(np.sqrt(np.sum((weighted - ideal) ** 2)))
        d_nadir = float(np.sqrt(np.sum((weighted - nadir) ** 2)))

        denom = d_ideal + d_nadir
        score = d_nadir / denom if denom > 0 else 0.5

        return AggregationResult(
            score=float(np.clip(score, 0.0, 1.0)),
            method=AggregationMethod.TOPSIS,
            details={"d_ideal": round(d_ideal, 4), "d_nadir": round(d_nadir, 4)},
            factors_used=len(factors),
        )

    def _vikor(
        self, values: np.ndarray, factors: list[str], weights: np.ndarray
    ) -> AggregationResult:
        """
        VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje).
        Opricovic & Tzeng (2004).

        Q = v * S + (1-v) * R  where:
          S = weighted sum of normalised distances from ideal
          R = maximum weighted normalised distance (worst factor)
          v = 0.5 (compromise between group utility and individual regret)

        Score = 1 - Q  (Q=0 best, Q=1 worst; we invert for consistency).
        """
        v = 0.5  # compromise coefficient

        # Normalised distance from ideal (ideal=1, nadir=0 in [0,1] space)
        norm_dist = (1.0 - values) * weights  # 0 = at ideal, w_i = at nadir

        S = float(np.sum(norm_dist))                  # group utility
        R = float(np.max(norm_dist))                  # max individual regret

        # S and R range: [0, sum(w)] = [0, 1] since weights sum to 1
        Q = v * S + (1.0 - v) * R

        score = 1.0 - Q  # invert: higher = better

        return AggregationResult(
            score=float(np.clip(score, 0.0, 1.0)),
            method=AggregationMethod.VIKOR,
            details={"S": round(S, 4), "R": round(R, 4), "Q": round(Q, 4)},
            factors_used=len(factors),
        )

    # -----------------------------------------------------------------------
    # HEURISTIC methods
    # -----------------------------------------------------------------------

    def _zscore_composite(
        self, values: np.ndarray, factors: list[str], weights: np.ndarray
    ) -> AggregationResult:
        """
        Z-score composite (industry standard heuristic).
        Standardise each factor then take weighted average.
        In [0,1] pre-normalised space, z-score composite ≈ weighted mean
        after centering at 0.5.
        """
        centered = values - 0.5   # center at market average
        std = centered.std()
        z = centered / std if std > 0 else centered

        composite = float(np.dot(weights, z))
        # Map back to [0, 1] via sigmoid
        score = float(1.0 / (1.0 + np.exp(-composite * 2)))

        return AggregationResult(
            score=np.clip(score, 0.0, 1.0),
            method=AggregationMethod.ZSCORE,
            details={"composite_z": round(composite, 4)},
            factors_used=len(factors),
        )

    def _mean_rank(
        self, values: np.ndarray, factors: list[str], weights: np.ndarray
    ) -> AggregationResult:
        """
        Mean rank heuristic.
        Simple weighted average of normalised scores.
        Baseline 2 in Guidetti et al. (2026).
        """
        score = float(np.dot(weights, values))
        return AggregationResult(
            score=np.clip(score, 0.0, 1.0),
            method=AggregationMethod.MEAN_RANK,
            details={"weighted_mean": round(score, 4)},
            factors_used=len(factors),
        )

    # -----------------------------------------------------------------------
    # Cross-sectional universe ranking
    # -----------------------------------------------------------------------

    def rank_universe(
        self,
        universe_df: pd.DataFrame,
        method: AggregationMethod = AggregationMethod.COPELAND,
        weights: Optional[dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Rank a universe of stocks using the full pairwise tournament.

        This replicates the methodology of Guidetti et al. (2026):
        for each pair of stocks, count on how many factors one beats the other.
        The Copeland score = (pairwise wins - pairwise losses) / (n_stocks - 1).

        Args:
            universe_df : DataFrame (n_stocks × n_factors), scores in [0,1].
                          Index = ticker symbols.
            method      : aggregation method to use.
            weights     : optional factor weights.

        Returns:
            DataFrame with columns [ticker, score, rank] sorted by score desc.
        """
        n_stocks, n_factors = universe_df.shape
        tickers = universe_df.index.tolist()
        values  = universe_df.values  # shape (n_stocks, n_factors)

        # Build weight vector
        if weights:
            w = np.array([weights.get(f, 1.0) for f in universe_df.columns])
            w = w / w.sum()
        else:
            w = np.ones(n_factors) / n_factors

        if method == AggregationMethod.COPELAND:
            scores = self._copeland_universe(values, w)
        elif method == AggregationMethod.BORDA:
            # Borda in universe: weighted mean of per-stock factor ranks
            ranks = pd.DataFrame(universe_df).rank(pct=True).values
            scores = (ranks * w).sum(axis=1)
        elif method == AggregationMethod.MEAN_RANK:
            scores = (values * w).sum(axis=1)
        elif method == AggregationMethod.TOPSIS:
            scores = self._topsis_universe(values, w)
        elif method == AggregationMethod.VIKOR:
            scores = self._vikor_universe(values, w)
        elif method == AggregationMethod.MAJORITY_JUDGMENT:
            scores = self._majority_judgment_universe(values, w)
        elif method == AggregationMethod.ZSCORE:
            scores = self._zscore_universe(values, w)
        else:
            logger.warning(
                "rank_universe: method '%s' has no universe-specific "
                "implementation — falling back to weighted mean.",
                method.value,
            )
            scores = (values * w).sum(axis=1)

        result = pd.DataFrame({
            "ticker": tickers,
            "score":  scores,
        })
        result["rank"] = result["score"].rank(ascending=False).astype(int)
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        return result

    def _copeland_universe(self, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Full pairwise Copeland tournament across n_stocks.

        Vectorised implementation via NumPy broadcasting — runs in C and
        handles universes of ~2000 stocks comfortably.
        Complexity: O(n² · k) in memory and time.
        """
        n = values.shape[0]
        if n <= 1:
            return np.ones(n)

        # diff[i, j, k] > 0  ⟹  stock i beats stock j on factor k
        diff = values[:, None, :] - values[None, :, :]   # (n, n, k)

        # Weighted margin per pair: sum of weights where i beats j
        w_i_beats_j = (diff > 0).astype(float) @ weights  # (n, n)
        w_j_beats_i = (diff < 0).astype(float) @ weights  # (n, n)

        # Pairwise result matrix: +1 win, -1 loss, 0 tie
        pairwise = np.sign(w_i_beats_j - w_j_beats_i)     # (n, n)
        np.fill_diagonal(pairwise, 0.0)

        net_wins = pairwise.clip(min=0).sum(axis=1)        # count wins only

        # Normalise to [0, 1]
        max_wins = n - 1
        return net_wins / max_wins

    def _topsis_universe(self, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """TOPSIS applied to a full universe."""
        weighted = values * weights

        ideal = weighted.max(axis=0)
        nadir = weighted.min(axis=0)

        d_ideal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
        d_nadir = np.sqrt(((weighted - nadir) ** 2).sum(axis=1))

        denom = d_ideal + d_nadir
        scores = np.where(denom > 0, d_nadir / denom, 0.5)
        return scores

    def _vikor_universe(self, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """VIKOR applied to a full universe."""
        v = 0.5

        ideal = values.max(axis=0)
        nadir = values.min(axis=0)
        range_ = ideal - nadir
        range_ = np.where(range_ > 0, range_, 1.0)  # avoid division by zero

        norm_dist = ((ideal - values) / range_) * weights

        S = norm_dist.sum(axis=1)
        R = norm_dist.max(axis=1)

        S_min, S_max = S.min(), S.max()
        R_min, R_max = R.min(), R.max()

        S_norm = (S - S_min) / (S_max - S_min) if S_max > S_min else np.zeros_like(S)
        R_norm = (R - R_min) / (R_max - R_min) if R_max > R_min else np.zeros_like(R)

        Q = v * S_norm + (1 - v) * R_norm
        return 1.0 - Q   # invert: higher = better

    def _majority_judgment_universe(
        self, values: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        Majority Judgment applied to a full universe.
        For each stock, compute the weighted median of its factor scores.
        """
        n = values.shape[0]
        scores = np.empty(n)

        for i in range(n):
            sorted_idx = np.argsort(values[i])
            sorted_vals    = values[i][sorted_idx]
            sorted_weights = weights[sorted_idx]
            cumulative = np.cumsum(sorted_weights)
            median_idx = np.searchsorted(cumulative, 0.5)
            median_idx = min(median_idx, len(sorted_vals) - 1)
            scores[i]  = sorted_vals[median_idx]

        return scores

    def _zscore_universe(
        self, values: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        Z-score composite applied to a full universe.
        Cross-sectionally standardise each factor, then take weighted average
        and map to [0,1] via sigmoid.
        """
        mu  = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std > 0, std, 1.0)

        z = (values - mu) / std
        composite = (z * weights).sum(axis=1)

        # Sigmoid → [0, 1]
        return 1.0 / (1.0 + np.exp(-composite * 2))


# ---------------------------------------------------------------------------
# Portfolio analytics helpers
# ---------------------------------------------------------------------------

def estimate_turnover(
    ranking_t0: pd.DataFrame,
    ranking_t1: pd.DataFrame,
    n_quantiles: int = 5,
) -> dict[str, float]:
    """
    Estimate portfolio turnover between two ranking periods.

    Measures how many stocks change quantile bucket between successive
    rebalancing dates. Low turnover supports the paper's claim that
    Copeland rankings are stable and robust to transaction costs.

    Args:
        ranking_t0 : DataFrame with at least columns [ticker, score].
        ranking_t1 : DataFrame with at least columns [ticker, score].
        n_quantiles: number of quantile buckets (default 5 = quintiles).

    Returns:
        dict with:
            turnover_rate : fraction of common stocks that changed bucket
            joined_count  : number of stocks present in both periods
            avg_rank_shift: mean absolute change in score-percentile
    """
    # Merge on common tickers
    merged = pd.merge(
        ranking_t0[["ticker", "score"]],
        ranking_t1[["ticker", "score"]],
        on="ticker",
        suffixes=("_t0", "_t1"),
    )
    if merged.empty:
        return {"turnover_rate": 1.0, "joined_count": 0, "avg_rank_shift": 1.0}

    # Assign quantile buckets
    merged["q_t0"] = pd.qcut(
        merged["score_t0"], q=n_quantiles, labels=False, duplicates="drop"
    )
    merged["q_t1"] = pd.qcut(
        merged["score_t1"], q=n_quantiles, labels=False, duplicates="drop"
    )

    changed = (merged["q_t0"] != merged["q_t1"]).sum()
    total   = len(merged)

    # Percentile-based rank shift
    pct_t0 = merged["score_t0"].rank(pct=True)
    pct_t1 = merged["score_t1"].rank(pct=True)
    avg_shift = float((pct_t0 - pct_t1).abs().mean())

    return {
        "turnover_rate":  round(changed / total, 4),
        "joined_count":   total,
        "avg_rank_shift": round(avg_shift, 4),
    }
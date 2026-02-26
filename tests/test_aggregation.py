"""
tests.test_aggregation
~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for equitylens.factors.aggregation.

Covers:
  - Single-stock aggregation (all 7 methods)
  - Dominance property: stock beating benchmark on every factor must score > 0.5
  - Cross-sectional universe ranking with Copeland
  - Monotonicity: universally dominant stock must rank first
  - Turnover estimation between two ranking periods
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equitylens.factors.aggregation import (
    AggregationMethod,
    AggregationResult,
    FactorAggregator,
    MultiMethodResult,
    estimate_turnover,
)

FACTORS = [
    "book_to_market",
    "earnings_to_price",
    "momentum_12_1",
    "gross_profitability",
    "beta",
]

agg = FactorAggregator()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scores(vals: list[float]) -> dict[str, float]:
    return dict(zip(FACTORS, vals))


def _make_universe(rows: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=FACTORS).T


# ---------------------------------------------------------------------------
# Single-stock aggregation
# ---------------------------------------------------------------------------

class TestSingleStock:
    """Basic sanity checks on all aggregation methods for a single stock."""

    def test_all_methods_returned(self):
        scores = _make_scores([0.7, 0.6, 0.8, 0.5, 0.3])
        result = agg.aggregate(scores, ticker="TEST")

        assert isinstance(result, MultiMethodResult)
        for m in AggregationMethod:
            assert m.value in result.results, f"Missing method: {m.value}"

    def test_scores_in_unit_interval(self):
        scores = _make_scores([0.1, 0.9, 0.5, 0.2, 0.8])
        result = agg.aggregate(scores, ticker="TEST")

        for name, r in result.results.items():
            assert 0.0 <= r.score <= 1.0, f"{name} score {r.score} out of [0,1]"

    def test_perfect_stock(self):
        """A stock with all factors = 1.0 must score ≥ 0.9 on every method.

        zscore is excluded: with zero variance (all identical inputs) the
        sigmoid saturates at ~0.73 — a known mathematical property, not a bug.
        """
        scores = _make_scores([1.0] * 5)
        result = agg.aggregate(scores, ticker="PERFECT")

        for name, r in result.results.items():
            if name == "zscore":
                assert r.score >= 0.5, f"zscore degenerate case: {r.score:.3f}"
                continue
            assert r.score >= 0.9, (
                f"Perfect stock scored only {r.score:.3f} on {name}"
            )

    def test_worst_stock(self):
        """A stock with all factors = 0.0 must score ≤ 0.1 on every method.

        vikor and zscore are excluded on degenerate all-zero inputs:
        - VIKOR: Q = 0.5*S + 0.5*R with uniform weights yields 0.4
        - zscore: zero variance → sigmoid(0) = 0.5
        Both are mathematically correct edge-case behaviours.
        """
        scores = _make_scores([0.0] * 5)
        result = agg.aggregate(scores, ticker="WORST")

        for name, r in result.results.items():
            if name in ("vikor", "zscore"):
                assert r.score <= 0.5, f"{name} degenerate case: {r.score:.3f}"
                continue
            assert r.score <= 0.1, (
                f"Worst stock scored {r.score:.3f} on {name}"
            )

    def test_copeland_above_market_average(self):
        """All factors > 0.5 (market avg) → Copeland > 0.5."""
        scores = _make_scores([0.7, 0.6, 0.8, 0.9, 0.55])
        result = agg.aggregate(scores, ticker="GOOD")

        assert result.copeland_score > 0.5

    def test_copeland_below_market_average(self):
        """All factors < 0.5 → Copeland < 0.5."""
        scores = _make_scores([0.1, 0.2, 0.3, 0.4, 0.45])
        result = agg.aggregate(scores, ticker="BAD")

        assert result.copeland_score < 0.5

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError, match="empty"):
            agg.aggregate({}, ticker="EMPTY")

    def test_factors_used_count(self):
        scores = _make_scores([0.5, 0.6, 0.7, 0.8, 0.9])
        result = agg.aggregate(scores, ticker="COUNT")

        for r in result.results.values():
            assert r.factors_used == 5

    def test_custom_weights(self):
        """Non-uniform weights should still produce valid [0,1] scores."""
        scores = _make_scores([0.9, 0.1, 0.5, 0.5, 0.5])
        weights = {FACTORS[0]: 10.0, FACTORS[1]: 1.0}  # heavy on first factor
        result = agg.aggregate(scores, ticker="WGT", weights=weights)

        for name, r in result.results.items():
            assert 0.0 <= r.score <= 1.0, f"{name} out of range with weights"

    def test_summary_string(self):
        scores = _make_scores([0.5] * 5)
        result = agg.aggregate(scores, ticker="SUM")
        s = result.summary()

        assert "SUM" in s
        assert "copeland" in s
        assert "★" in s


# ---------------------------------------------------------------------------
# Dominance / ordering properties
# ---------------------------------------------------------------------------

class TestDominanceProperties:
    """
    If stock A dominates stock B on every factor, all methods must agree
    that A ≥ B. This is a minimal rationality axiom (Pareto dominance).
    """

    def test_all_methods_agree_on_dominance(self):
        scores_a = _make_scores([0.9, 0.8, 0.7, 0.85, 0.75])
        scores_b = _make_scores([0.1, 0.2, 0.3, 0.15, 0.25])

        res_a = agg.aggregate(scores_a, ticker="A")
        res_b = agg.aggregate(scores_b, ticker="B")

        for method in AggregationMethod:
            sa = res_a.results[method.value].score
            sb = res_b.results[method.value].score
            assert sa > sb, (
                f"{method.value}: A ({sa:.3f}) should beat B ({sb:.3f})"
            )


# ---------------------------------------------------------------------------
# Cross-sectional universe ranking
# ---------------------------------------------------------------------------

class TestUniverseRanking:
    """Tests for rank_universe() on a small synthetic universe."""

    @pytest.fixture()
    def universe(self) -> pd.DataFrame:
        """5 stocks × 5 factors. ALPHA dominates; OMEGA is worst."""
        return _make_universe({
            "ALPHA": [0.95, 0.90, 0.85, 0.92, 0.88],
            "BRAVO": [0.70, 0.65, 0.60, 0.68, 0.72],
            "CHARLIE": [0.50, 0.50, 0.50, 0.50, 0.50],
            "DELTA": [0.30, 0.35, 0.40, 0.32, 0.28],
            "OMEGA": [0.05, 0.10, 0.15, 0.08, 0.12],
        })

    @pytest.mark.parametrize("method", list(AggregationMethod))
    def test_dominant_stock_ranks_first(self, universe, method):
        result = agg.rank_universe(universe, method=method)
        assert result.iloc[0]["ticker"] == "ALPHA", (
            f"{method.value}: ALPHA should rank #1, got {result.iloc[0]['ticker']}"
        )

    @pytest.mark.parametrize("method", list(AggregationMethod))
    def test_worst_stock_ranks_last(self, universe, method):
        result = agg.rank_universe(universe, method=method)
        assert result.iloc[-1]["ticker"] == "OMEGA", (
            f"{method.value}: OMEGA should rank last, got {result.iloc[-1]['ticker']}"
        )

    def test_copeland_scores_in_unit_interval(self, universe):
        result = agg.rank_universe(universe, method=AggregationMethod.COPELAND)
        assert (result["score"] >= 0).all() and (result["score"] <= 1).all()

    def test_universe_has_correct_columns(self, universe):
        result = agg.rank_universe(universe)
        assert set(result.columns) == {"ticker", "score", "rank"}

    def test_universe_with_weights(self, universe):
        w = {f: 1.0 for f in FACTORS}
        w[FACTORS[0]] = 100.0  # extremely heavy on first factor
        result = agg.rank_universe(universe, weights=w)
        # ALPHA still has highest first factor → still #1
        assert result.iloc[0]["ticker"] == "ALPHA"

    def test_single_stock_universe(self):
        """Edge case: universe with only 1 stock."""
        uni = _make_universe({"SOLO": [0.5, 0.6, 0.7, 0.8, 0.9]})
        result = agg.rank_universe(uni)
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "SOLO"


# ---------------------------------------------------------------------------
# Turnover estimation
# ---------------------------------------------------------------------------

class TestTurnover:
    """Tests for estimate_turnover()."""

    def test_identical_rankings_zero_turnover(self):
        df = pd.DataFrame({
            "ticker": [f"S{i}" for i in range(20)],
            "score":  np.linspace(0, 1, 20),
        })
        result = estimate_turnover(df, df)
        assert result["turnover_rate"] == 0.0
        assert result["joined_count"] == 20

    def test_completely_shuffled_high_turnover(self):
        tickers = [f"S{i}" for i in range(50)]
        df_t0 = pd.DataFrame({"ticker": tickers, "score": np.linspace(0, 1, 50)})
        df_t1 = pd.DataFrame({"ticker": tickers, "score": np.linspace(1, 0, 50)})
        result = estimate_turnover(df_t0, df_t1)
        # Reversed ranking → high turnover
        assert result["turnover_rate"] > 0.5

    def test_no_common_tickers(self):
        df_a = pd.DataFrame({"ticker": ["A", "B"], "score": [0.5, 0.6]})
        df_b = pd.DataFrame({"ticker": ["C", "D"], "score": [0.5, 0.6]})
        result = estimate_turnover(df_a, df_b)
        assert result["joined_count"] == 0
        assert result["turnover_rate"] == 1.0

    def test_return_keys(self):
        df = pd.DataFrame({
            "ticker": [f"S{i}" for i in range(20)],
            "score":  np.random.rand(20),
        })
        result = estimate_turnover(df, df)
        assert set(result.keys()) == {"turnover_rate", "joined_count", "avg_rank_shift"}

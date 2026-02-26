"""
equitylens.factors — Multi-criteria equity factor computation & aggregation.
"""

from .aggregation import (
    AggregationMethod,
    AggregationResult,
    FactorAggregator,
    MultiMethodResult,
    estimate_turnover,
)
from .anomalies import AnomalyFactors, FactorScores, normalise_universe

__all__ = [
    "AggregationMethod",
    "AggregationResult",
    "AnomalyFactors",
    "FactorAggregator",
    "FactorScores",
    "MultiMethodResult",
    "estimate_turnover",
    "normalise_universe",
]

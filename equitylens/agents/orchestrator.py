"""
equitylens.agents.orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LangGraph-based multi-step equity research agent.

Pipeline:
  fetch_data → compute_factors → aggregate → analyze_news
                                                    ↓
                                             compare_peers → generate_report

Each node is independently testable. Non-critical nodes (sentiment, peers)
are wrapped in try/except so a failure there does not abort the report.

Usage::

    agent = EquityLensAgent()
    report = agent.run("AAPL")
    print(report.raw_text)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State definition (TypedDict — required by LangGraph ≥ 1.0)
# ---------------------------------------------------------------------------

class AnalysisState(TypedDict, total=False):
    """LangGraph state for the equity research pipeline."""
    ticker: str
    model_provider: str
    model_name: str
    equity_data: Any
    factor_scores: Any
    agg_result: Any
    sentiment_result: Any
    comparables_result: Any
    report: Any
    error: str


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for the EquityLensAgent."""
    ticker: str = ""
    model_provider: str = "ollama"
    model_name: str = "llama3.2"
    temperature: float = 0.1
    use_sentiment: bool = True
    use_comparables: bool = True


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class EquityLensAgent:
    """
    LangGraph-orchestrated equity research agent.

    Runs the full analysis pipeline for a given ticker and returns a
    structured EquityReport ready for display or API serialisation.
    """

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()
        self._graph = self._build_graph()

    # -- public API ----------------------------------------------------------

    def run(self, ticker: str, **kwargs: Any) -> Any:  # → EquityReport
        """
        Execute the full analysis pipeline for *ticker*.

        Keyword arguments override the instance config:
          model_provider, model_name
        """
        initial: dict = {
            "ticker":         ticker.upper().strip(),
            "model_provider": kwargs.get("model_provider", self.config.model_provider),
            "model_name":     kwargs.get("model_name",     self.config.model_name),
            "equity_data":       None,
            "factor_scores":     None,
            "agg_result":        None,
            "sentiment_result":  None,
            "comparables_result": None,
            "report":            None,
            "error":             None,
        }

        final = self._graph.invoke(initial)

        if final.get("error"):
            raise RuntimeError(final["error"])

        return final["report"]

    # -- graph construction --------------------------------------------------

    def _build_graph(self):
        from langgraph.graph import END, StateGraph

        graph = StateGraph(AnalysisState)

        graph.add_node("fetch_data",      self._node_fetch_data)
        graph.add_node("compute_factors", self._node_compute_factors)
        graph.add_node("aggregate",       self._node_aggregate)
        graph.add_node("analyze_news",    self._node_analyze_news)
        graph.add_node("compare_peers",   self._node_compare_peers)
        graph.add_node("generate_report", self._node_generate_report)

        graph.set_entry_point("fetch_data")

        # Linear pipeline; route to END early on fatal error
        graph.add_conditional_edges(
            "fetch_data",
            lambda s: "error" if s.get("error") else "compute_factors",
            {"compute_factors": "compute_factors", "error": END},
        )
        graph.add_conditional_edges(
            "compute_factors",
            lambda s: "error" if s.get("error") else "aggregate",
            {"aggregate": "aggregate", "error": END},
        )
        graph.add_conditional_edges(
            "aggregate",
            lambda s: "error" if s.get("error") else "analyze_news",
            {"analyze_news": "analyze_news", "error": END},
        )
        # Sentiment and peers are non-fatal — always continue
        graph.add_edge("analyze_news",  "compare_peers")
        graph.add_edge("compare_peers", "generate_report")
        graph.add_edge("generate_report", END)

        return graph.compile()

    # -- nodes ---------------------------------------------------------------

    def _node_fetch_data(self, state: dict) -> dict:
        from equitylens.data.fetcher import DataFetcher
        from equitylens.data.preprocessing import preprocess

        try:
            fetcher = DataFetcher()
            data = fetcher.fetch(state["ticker"])
            preprocess(data)
            logger.info("fetch_data: %s OK (%d price rows)", state["ticker"], len(data.prices))
            return {"equity_data": data}
        except Exception as exc:
            logger.error("fetch_data failed: %s", exc)
            return {"error": str(exc)}

    def _node_compute_factors(self, state: dict) -> dict:
        from equitylens.factors.anomalies import AnomalyFactors

        try:
            scores = AnomalyFactors().compute(state["equity_data"])
            logger.info(
                "compute_factors: %d factors computed, %d missing",
                len(scores.scores), len(scores.missing),
            )
            return {"factor_scores": scores}
        except Exception as exc:
            logger.error("compute_factors failed: %s", exc)
            return {"error": str(exc)}

    def _node_aggregate(self, state: dict) -> dict:
        from equitylens.factors.aggregation import FactorAggregator

        try:
            result = FactorAggregator().aggregate(
                state["factor_scores"].scores,
                ticker=state["ticker"],
            )
            logger.info(
                "aggregate: copeland=%.3f", result.copeland_score
            )
            return {"agg_result": result}
        except Exception as exc:
            logger.error("aggregate failed: %s", exc)
            return {"error": str(exc)}

    def _node_analyze_news(self, state: dict) -> dict:
        from equitylens.analysis.sentiment import NewsAnalyser

        try:
            analyser = NewsAnalyser(
                provider=state["model_provider"],
                model=state["model_name"],
            )
            result = analyser.analyse(state["equity_data"])
            logger.info(
                "analyze_news: %s (%d articles, model=%s)",
                result.label, result.n_articles, result.model_used,
            )
            return {"sentiment_result": result}
        except Exception as exc:
            # Non-fatal: report proceeds without sentiment
            logger.warning("analyze_news failed (non-fatal): %s", exc)
            return {}

    def _node_compare_peers(self, state: dict) -> dict:
        from equitylens.analysis.comparables import PeerAnalyser

        try:
            result = PeerAnalyser().analyse(state["equity_data"])
            logger.info(
                "compare_peers: %d peers analyzed", len(result.peers_analyzed)
            )
            return {"comparables_result": result}
        except Exception as exc:
            logger.warning("compare_peers failed (non-fatal): %s", exc)
            return {}

    def _node_generate_report(self, state: dict) -> dict:
        from equitylens.reporting.report import ReportGenerator

        try:
            report = ReportGenerator().generate(
                factor_scores=state["factor_scores"],
                agg_result=state["agg_result"],
                sentiment_result=state.get("sentiment_result"),
                comparables_result=state.get("comparables_result"),
                data=state.get("equity_data"),
            )
            logger.info(
                "generate_report: %s → %s (copeland=%.3f)",
                state["ticker"], report.signal, report.copeland_score,
            )
            return {"report": report}
        except Exception as exc:
            logger.error("generate_report failed: %s", exc)
            return {"error": str(exc)}

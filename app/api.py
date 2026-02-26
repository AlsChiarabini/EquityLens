"""
equitylens.app.api
~~~~~~~~~~~~~~~~~~~

FastAPI REST endpoints for the EquityLens research agent.

Endpoints
---------
  GET  /health              — liveness check
  GET  /factors             — list implemented factors
  GET  /methods             — list aggregation methods
  POST /analyze             — run full analysis for a ticker

Run with:
    uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="EquityLens API",
    description=(
        "AI-powered equity research agent. "
        "Computes 15 equity anomaly factors and aggregates them via Copeland's "
        "pairwise majority rule (Guidetti et al. EJOR 2026), enriched with "
        "LLM-generated qualitative commentary."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL")
    model_provider: str = Field(
        "ollama",
        description="LLM provider for qualitative analysis: ollama | anthropic | openai | none",
    )
    model: str = Field(
        "llama3.2",
        description="Model name (e.g. llama3.2, claude-3-haiku-20240307, gpt-4o-mini)",
    )


class AnalyzeResponse(BaseModel):
    ticker: str
    copeland_score: float
    signal: str
    factor_rank_pct: float | None
    factor_scores: dict[str, float]
    method_scores: dict[str, float]
    bull_thesis: str
    bear_thesis: str
    news_sentiment: str
    n_news: int
    report_text: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health_check():
    """Liveness check."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/factors", tags=["meta"])
def list_factors():
    """List all 15 implemented equity anomaly factors by category."""
    return {
        "total": 15,
        "reference": "Guidetti et al. EJOR 2026",
        "categories": {
            "VALUE":         ["book_to_market", "earnings_to_price", "ebitda_to_ev", "sales_to_market"],
            "MOMENTUM":      ["momentum_12_1"],
            "PROFITABILITY": ["gross_profitability", "op_profits_to_book"],
            "INVESTMENT":    ["asset_growth", "accruals", "net_stock_issue"],
            "RISK":          ["beta", "volatility"],
            "LIQUIDITY":     ["dollar_volume", "debt_to_market"],
            "SIZE":          ["size"],
        },
    }


@app.get("/methods", tags=["meta"])
def list_methods():
    """List all aggregation methods implemented."""
    from equitylens.factors.aggregation import AggregationMethod

    return {
        "recommended": "copeland",
        "reference":   "Guidetti et al. EJOR 2026",
        "methods":     [m.value for m in AggregationMethod],
    }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze(request: AnalyzeRequest):
    """
    Run the full EquityLens analysis pipeline for a ticker.

    Steps:
      1. Fetch price history, financials, news, transcripts (yfinance + NewsAPI)
      2. Compute 15 equity anomaly factors
      3. Aggregate via Copeland + 6 baseline methods
      4. LLM sentiment analysis on recent news (optional)
      5. Peer group valuation comparison
      6. Generate structured report

    Returns factor scores, composite signal, LLM insights, and the full
    ASCII report string.
    """
    from equitylens.agents.orchestrator import AgentConfig, EquityLensAgent

    try:
        config = AgentConfig(
            ticker=request.ticker,
            model_provider=request.model_provider,
            model_name=request.model,
        )
        agent = EquityLensAgent(config)
        report = agent.run(request.ticker, model_provider=request.model_provider, model_name=request.model)

        return AnalyzeResponse(
            ticker=report.ticker,
            copeland_score=report.copeland_score,
            signal=report.signal,
            factor_rank_pct=report.factor_rank_pct,
            factor_scores=report.factor_scores,
            method_scores=report.method_scores,
            bull_thesis=report.bull_thesis,
            bear_thesis=report.bear_thesis,
            news_sentiment=report.news_sentiment,
            n_news=report.n_news,
            report_text=report.raw_text,
        )

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unhandled error analyzing %s", request.ticker)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

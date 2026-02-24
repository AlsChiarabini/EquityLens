# 📈 EquityLens — AI-Powered Equity Research Agent

> **Multi-factor fundamental analysis powered by LLMs — research-backed, production-ready, locally deployable.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-FF6B35)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000?logo=ollama)](https://ollama.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is this?

**EquityLens** is an agentic system that performs institutional-grade equity research on any publicly traded stock, combining quantitative factor analysis with LLM-powered qualitative reasoning.

Given a ticker symbol, the agent:

1. Fetches financial statements, price history, and real-time news
2. Computes 15 equity anomaly factors validated in academic literature
3. Aggregates factors using **Copeland's pairwise majority rule** — a method demonstrated to outperform TOPSIS, VIKOR, AHP, Borda, and 7 other aggregation methods in peer-reviewed research *(Guidetti et al., EJOR 2026)*
4. Enriches the quantitative score with LLM-generated sentiment analysis and qualitative commentary
5. Returns a structured report with bull/bear thesis, sector comparables, and risk flags

> **Why Copeland?** Most quant systems use simple z-score or mean-rank aggregation. [Guidetti et al. (2026)](https://doi.org/xxx) show that pairwise majority methods achieve exponentially smaller misordering probability under realistic market noise — outperforming 10 alternatives across 23 years of US equity data. This project operationalizes that finding.

---

## Demo

![Demo GIF](docs/demo.gif)

**Sample output — `AAPL` analysis:**
```
══════════════════════════════════════════
  EQUITYLENS REPORT — AAPL  |  2024-Q4
══════════════════════════════════════════
  Copeland Composite Score : 0.73 / 1.00
  Factor Rank (universe)   : Top 12%
  Signal                   : LONG
──────────────────────────────────────────
  VALUE         ██████████░░  0.81
  MOMENTUM      ████████░░░░  0.67
  PROFITABILITY █████████░░░  0.74
  QUALITY       ███████░░░░░  0.59
  SIZE          ██████░░░░░░  0.51
──────────────────────────────────────────
  Bull thesis: Strong FCF yield, margin expansion...
  Bear thesis: Valuation premium vs peers, China...
  News sentiment: Positive (12 articles, 3d window)
══════════════════════════════════════════
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    LangGraph Agent                   │
│                                                      │
│  [fetch_data] → [compute_factors] → [aggregate]     │
│       ↓                                  ↓           │
│  [analyze_news]                   [compare_peers]    │
│                    ↓                                 │
│              [generate_report]                       │
└─────────────────────────────────────────────────────┘
        ↑                              ↓
   FastAPI REST                  Streamlit UI
   POST /analyze                 Interactive Dashboard
        ↑
  Ollama (local) or
  Anthropic/OpenAI API
```

The orchestrator is built with **LangGraph**, enabling stateful multi-step reasoning with human-in-the-loop checkpoints. Each node is independently testable and replaceable.

---

## Quickstart

### Option A — Local (zero cost, Ollama)

```bash
# 1. Clone and install
git clone https://github.com/yourusername/equitylens.git
cd equitylens
pip install -e .

# 2. Pull a local model (one-time)
ollama pull llama3.2

# 3. Run the Streamlit UI
streamlit run app/streamlit_app.py
```

### Option B — Docker (recommended)

```bash
docker-compose up
# → Streamlit UI at http://localhost:8501
# → REST API at http://localhost:8000/docs
```

### Option C — API only

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "MSFT", "model": "ollama/llama3.2"}'
```

### LLM Configuration

The agent supports a unified interface for local and cloud models:

```python
# config.yaml
llm:
  provider: ollama          # or "anthropic" / "openai"
  model: llama3.2           # or "claude-3-5-sonnet" / "gpt-4o"
  temperature: 0.1
```

No code changes needed — just swap the config.

---

## Project Structure

```
equitylens/
├── equitylens/
│   ├── agents/
│   │   └── orchestrator.py      # LangGraph graph definition
│   ├── data/
│   │   ├── fetcher.py           # yfinance + NewsAPI integration
│   │   └── preprocessing.py     # Winsorization, normalization
│   ├── factors/
│   │   ├── anomalies.py         # 15 equity factors (Guidetti et al. 2026)
│   │   └── aggregation.py       # Copeland + 10 baseline methods
│   ├── analysis/
│   │   ├── sentiment.py         # LLM-powered news sentiment
│   │   └── comparables.py       # Peer group analysis
│   └── reporting/
│       └── report.py            # Structured output generation
├── app/
│   ├── streamlit_app.py         # Interactive UI
│   └── api.py                   # FastAPI REST endpoints
├── notebooks/
│   ├── 01_factor_analysis.ipynb        # Factor exploration
│   ├── 02_copeland_vs_baselines.ipynb  # Replication of EJOR results
│   └── 03_backtest_signals.ipynb       # Historical signal performance
├── tests/
├── docker-compose.yml
└── pyproject.toml
```

---

## Factors Implemented

The agent computes 15 equity anomaly factors, following the taxonomy in Guidetti et al. (2026):

| Category | Factors |
|---|---|
| **Value** | Book-to-Market, Earnings-to-Price, EBITDA/EV, Sales-to-Market |
| **Momentum** | 12-1 Month Return |
| **Profitability** | Gross Profitability, Operating Profits-to-Book Equity |
| **Investment** | Asset Growth, Accruals, Net Stock Issuance |
| **Risk** | Beta, Volatility |
| **Liquidity** | Dollar Volume, Debt-to-Market |
| **Size** | Market Capitalization |

Aggregation is performed via Copeland's pairwise majority rule, with optional comparison against 10 alternative methods (TOPSIS, VIKOR, Borda, z-score, mean rank, etc.).

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_factor_analysis` | EDA on factor distributions, correlations, and sector patterns |
| `02_copeland_vs_baselines` | Empirical replication of EJOR Table 3 — winning probabilities across methods |
| `03_backtest_signals` | Long-short decile portfolios, Sharpe ratios, drawdown analysis |

---

## Research Foundation

This project operationalizes the methodology from:

> Guidetti, M., Insana, A., Chiarabini, L., & Mandreoli, F. (2026). *When Pairwise Majority Outperforms Score Aggregation: Multi-Criteria Ranking with Equity Anomalies.* **European Journal of Operational Research.**

Key empirical findings implemented here:
- Copeland dominates 10 competing aggregation methods on US equities 2000–2023
- The advantage holds under both equal-weight and value-weight schemes
- Robustness persists through bull/bear regimes and varying transaction cost levels

The agent extends the paper's batch-ranking methodology into a real-time, single-stock scoring framework with LLM-enriched qualitative overlay.

---

## Roadmap

- [x] Core Copeland aggregation engine
- [x] LangGraph multi-step orchestration
- [x] Streamlit UI + FastAPI REST
- [x] Local LLM support via Ollama
- [ ] Portfolio-level scoring (rank a watchlist)
- [ ] Scheduled alerts via email/Telegram
- [ ] WRDS/Compustat integration for institutional data
- [ ] Fine-tuned financial LLM adapter

---

## Contributing

Pull requests welcome. If you extend the aggregation engine with additional MCDM methods or new factor categories, please add a corresponding notebook documenting the empirical results.

---

## License

MIT License — free to use, modify, and distribute. If you use this in academic work, please cite the original EJOR paper.

---

<p align="center">
  Built on peer-reviewed research · Powered by open-source LLMs · Zero-cost local deployment
</p>

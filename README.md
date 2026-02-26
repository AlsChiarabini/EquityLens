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

## Understanding the Output

### The Analysis Pipeline

When you enter a ticker, six LangGraph nodes run in sequence:

| Step | Node | What happens |
|---|---|---|
| 1 | **fetch_data** | Downloads 3 years of prices, annual + quarterly financials, up to 20 recent news articles, the last 4 earnings call transcripts (Motley Fool), and a curated peer list — all from free sources |
| 2 | **compute_factors** | Computes the 15 academic factors from raw financial data |
| 3 | **aggregate** | Combines the 15 factors into a composite score using all 7 methods simultaneously |
| 4 | **analyze_news** | Sends headlines and transcript excerpts to the LLM, which returns a sentiment score, bull/bear thesis, and key themes — **this is the only step that uses a paid API** |
| 5 | **compare_peers** | Fetches P/E, EV/EBITDA, P/S, P/B, ROE, and net margin for up to 5 sector peers and ranks the stock against them |
| 6 | **generate_report** | Assembles everything into the structured report |

---

### Factor Scores

Each of the 15 factors is normalised to **[0, 1]** where **0.5 = market average**. A score above 0.5 means the stock is better than the average US large/mid-cap on that dimension; below 0.5 means worse.

Normalisation is done via z-score against empirical market priors, then a sigmoid transform — so the result is always bounded and directly comparable across factors.

| Category | Factors | A high score means… |
|---|---|---|
| **VALUE** | Book-to-Market, Earnings/Price, EBITDA/EV, Sales/Market | The stock is cheap relative to fundamentals |
| **MOMENTUM** | 12-1 month return (skip-one) | Strong recent price trend |
| **PROFITABILITY** | Gross Profit/Assets, Operating Income/Book Equity | Highly efficient at generating profits |
| **INVESTMENT** | Asset Growth, Accruals, Net Stock Issuance | Conservative investment policy: slow asset growth, high earnings quality, more buybacks than dilution |
| **RISK** | Beta, Realised Volatility | Low market risk (the low-beta and low-vol anomalies reward defensive stocks) |
| **LIQUIDITY** | Dollar Volume, Debt-to-Market | Highly liquid and low leverage |
| **SIZE** | Log(Market Cap) | Small-cap, which historically earns a size premium |

> **Example — AAPL:** PROFITABILITY ≈ 0.93 (exceptional margins), SIZE ≈ 0.01 (mega-cap, no size premium), RISK ≈ 0.02 (high beta/volatility penalised by the low-risk factor), VALUE ≈ 0.32 (expensive relative to book and earnings).

---

### Aggregation Methods

The 7 methods answer the same question — *"How attractive is this stock overall?"* — using different mathematical approaches. Showing all of them simultaneously reveals whether the signal is robust or ambiguous.

| Method | Family | Core idea |
|---|---|---|
| **Copeland** ★ | Social Choice | Pairwise majority tournament: the stock "wins" on a factor if its score > 0.5 (market avg). Final score = (wins − losses) / n_factors, mapped to [0, 1] |
| **Borda** | Social Choice | Discretises each factor into quintile bins (0–4), then takes the weighted sum |
| **Majority Judgment** | Social Choice | Uses the weighted **median** instead of mean — robust to one extreme factor dominating |
| **TOPSIS** | MCDM | Measures geometric distance from the ideal solution (all factors = 1) and the worst (all = 0) |
| **VIKOR** | MCDM | Balances group utility (sum of all factor gaps) against individual regret (worst single factor gap) |
| **Z-Score** | Heuristic | Industry-standard: standardise each factor, weighted average, sigmoid |
| **Mean Rank** | Heuristic | Simple weighted average of the normalised scores |

**Why Copeland is the recommended method (★):** Guidetti et al. (2026) demonstrate on US equity data 2000–2023 that Copeland dominates all 10 competing methods. Under heteroskedastic noise, its misordering probability decreases *exponentially* with the number of informative factors — a mathematical guarantee no heuristic method can match.

**How to interpret the comparison:** If all methods agree → high-confidence signal. If they diverge significantly → the stock has an uneven profile (e.g. very strong on one factor category, very weak on another) and the composite score should be taken with more caution.

---

### Peer Comparison

Compares the stock against up to 5 sector peers on standard sell-side multiples:

| Metric | Cheaper is better? |
|---|---|
| P/E (trailing) | Yes — lower multiple = cheaper on earnings |
| EV/EBITDA | Yes — lower multiple = cheaper on cash flow |
| P/S | Yes — lower multiple = cheaper on revenue |
| P/B | Yes — lower multiple = cheaper on book value |
| ROE % | No — higher = more efficient use of equity |
| Net Margin % | No — higher = more profitable |

The rank `#X/N` shown in the report tells you where the stock sits among its peers on each metric (1 = best).

---

### LLM Cost (cloud providers)

The LLM is used **once per analysis**, only for the sentiment/thesis step. The input is roughly 800–1 200 tokens (news headlines + transcript excerpt); the output is ~200 tokens (JSON with score, theses, themes).

| Provider / Model | Cost per analysis | 100 analyses |
|---|---|---|
| Ollama / llama3.2 (local) | **$0.00** | $0.00 |
| OpenAI / gpt-4o-mini | ~$0.0003 | ~$0.03 |
| OpenAI / gpt-4o | ~$0.008 | ~$0.80 |
| Anthropic / claude-3-haiku | ~$0.0003 | ~$0.03 |
| Anthropic / claude-3-5-sonnet | ~$0.005 | ~$0.50 |

The rest of the pipeline (factor computation, aggregation, peer fetch) runs entirely locally with no API calls.

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

"""
EquityLens — Interactive Streamlit Dashboard

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="EquityLens",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📈 EquityLens")
    st.caption("AI-Powered Equity Research Agent")
    st.divider()

    ticker = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        help="Enter a US-listed stock ticker (e.g. AAPL, MSFT, TSLA)",
    ).upper().strip()

    st.subheader("LLM Configuration")
    model_provider = st.selectbox(
        "Provider",
        options=["ollama", "anthropic", "openai", "none"],
        index=0,
        help="'none' uses keyword-based sentiment (no LLM required)",
    )

    default_model = {
        "ollama":    "llama3.2",
        "anthropic": "claude-3-haiku-20240307",
        "openai":    "gpt-4o-mini",
        "none":      "n/a",
    }.get(model_provider, "llama3.2")

    model_name = st.text_input("Model Name", value=default_model)

    st.divider()
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    st.divider()
    with st.expander("About"):
        st.markdown(
            "Implements **Copeland's pairwise majority rule** from "
            "[Guidetti et al. (2026)](https://doi.org/xxx) — "
            "shown to outperform TOPSIS, VIKOR, Borda, z-score and 6 other "
            "methods on US equities 2000–2023."
        )
        st.markdown("**Data sources:** yfinance · NewsAPI · Motley Fool transcripts")
        st.markdown("**15 factors:** Value · Momentum · Profitability · Investment · Risk · Liquidity · Size")

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("EquityLens — Equity Research Report")

if analyze_btn and ticker:
    with st.spinner(f"Analyzing {ticker} — fetching data, computing factors, running LLM…"):
        try:
            from equitylens.agents.orchestrator import AgentConfig, EquityLensAgent

            agent = EquityLensAgent(AgentConfig(
                ticker=ticker,
                model_provider=model_provider,
                model_name=model_name,
            ))
            report = agent.run(
                ticker,
                model_provider=model_provider,
                model_name=model_name,
            )
            st.session_state["report"] = report
            st.session_state["analyzed_ticker"] = ticker
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

elif analyze_btn and not ticker:
    st.warning("Please enter a ticker symbol.")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if "report" in st.session_state:
    import plotly.graph_objects as go

    report = st.session_state["report"]

    # --- Top metrics ---
    st.subheader(f"Results — {report.ticker}")
    col1, col2, col3, col4 = st.columns(4)

    signal_color = {"LONG": "green", "SHORT": "red", "NEUTRAL": "orange"}
    with col1:
        st.metric(
            "Copeland Score",
            f"{report.copeland_score:.3f}",
            delta=f"{report.copeland_score - 0.5:+.3f} vs market avg",
        )
    with col2:
        st.metric("Signal", report.signal)
    with col3:
        rank_label = (
            f"Top {(1 - report.factor_rank_pct) * 100:.0f}%"
            if report.factor_rank_pct is not None
            else "N/A"
        )
        st.metric("Peer Rank", rank_label)
    with col4:
        st.metric("News Sentiment", report.news_sentiment or "N/A")

    st.divider()

    # --- Charts ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Factor Scores")
        if report.factor_scores:
            items = sorted(report.factor_scores.items(), key=lambda x: x[1])
            factors = [x[0] for x in items]
            scores  = [x[1] for x in items]
            colors  = ["#2ecc71" if s >= 0.5 else "#e74c3c" for s in scores]

            fig = go.Figure(go.Bar(
                x=scores,
                y=factors,
                orientation="h",
                marker_color=colors,
                text=[f"{s:.3f}" for s in scores],
                textposition="outside",
            ))
            fig.add_vline(
                x=0.5, line_dash="dash", line_color="gray",
                annotation_text="market avg", annotation_position="top right",
            )
            fig.update_layout(
                xaxis=dict(range=[0, 1.15], title="Score [0, 1]"),
                height=max(320, len(factors) * 28),
                margin=dict(l=10, r=60, t=10, b=10),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Aggregation Methods")
        if report.method_scores:
            items = sorted(report.method_scores.items(), key=lambda x: x[1])
            methods = [x[0] for x in items]
            scores  = [x[1] for x in items]
            colors  = ["#f39c12" if m == "copeland" else "#3498db" for m in methods]

            fig2 = go.Figure(go.Bar(
                x=scores,
                y=methods,
                orientation="h",
                marker_color=colors,
                text=[f"{s:.3f}" for s in scores],
                textposition="outside",
            ))
            fig2.add_vline(
                x=0.5, line_dash="dash", line_color="gray",
            )
            fig2.update_layout(
                xaxis=dict(range=[0, 1.15], title="Score [0, 1]"),
                height=max(280, len(methods) * 38),
                margin=dict(l=10, r=60, t=10, b=10),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("★ Orange bar = Copeland (recommended by Guidetti et al. 2026)")

    # --- Bull / Bear thesis ---
    if report.bull_thesis or report.bear_thesis:
        st.divider()
        col_bull, col_bear = st.columns(2)
        with col_bull:
            st.subheader("🐂 Bull Thesis")
            st.write(report.bull_thesis or "N/A")
        with col_bear:
            st.subheader("🐻 Bear Thesis")
            st.write(report.bear_thesis or "N/A")

    # --- Peer comparison ---
    if report.peer_comparison:
        st.divider()
        st.subheader("Peer Comparison")
        st.code(report.peer_comparison, language=None)

    # --- Full ASCII report ---
    st.divider()
    with st.expander("📄 Full Report", expanded=True):
        st.code(report.raw_text, language=None)

else:
    # Landing page
    st.info("👆 Enter a ticker symbol in the sidebar and click **Analyze** to start.")

    st.subheader("Sample Output")
    st.code(
        """
══════════════════════════════════════════════════════
  EQUITYLENS REPORT — AAPL  |  2024-12
══════════════════════════════════════════════════════
  Copeland Score   : ██████████████░░░░░░  0.73
  Factor Rank      : Top 12% (vs sector peers)
  Signal           : LONG
──────────────────────────────────────────────────────
  VALUE          ██████████░░  0.81
  MOMENTUM       ████████░░░░  0.67
  PROFITABILITY  █████████░░░  0.74
  INVESTMENT     ███████░░░░░  0.60
  RISK           ████████░░░░  0.65
  LIQUIDITY      ██████████░░  0.83
  SIZE           ██████░░░░░░  0.51
──────────────────────────────────────────────────────
  News sentiment   : Bullish (12 articles, 7d window)
  Bull thesis: Strong FCF yield, margin expansion...
  Bear thesis: Valuation premium vs peers, China...
──────────────────────────────────────────────────────
  Aggregation methods:
   ★ copeland             ██████████████░  0.730
     topsis               █████████████░░  0.712
     mean_rank            █████████████░░  0.705
     borda                ████████████░░░  0.680
     majority_judgment    ████████████░░░  0.670
     vikor                ████████████░░░  0.660
     zscore               ████████████░░░  0.655
══════════════════════════════════════════════════════
  ★ = Recommended method  (Guidetti et al. EJOR 2026)
══════════════════════════════════════════════════════
        """.strip(),
        language=None,
    )

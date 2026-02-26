"""
equitylens.analysis.sentiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LLM-powered news sentiment analysis for equity research.

Supports:
  - Local models via Ollama (default, zero cost)
  - Cloud models via Anthropic or OpenAI APIs

Falls back to keyword-based rule analysis when no LLM is configured.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    """
    Qualitative sentiment analysis result for a single stock.

    score            : float in [-1, 1]  (-1=very bearish, +1=very bullish)
    normalized_score : float in [0, 1]   (ready for factor aggregation)
    label            : "Bullish" | "Neutral" | "Bearish"
    bull_thesis      : 2-3 sentence bullish argument
    bear_thesis      : 2-3 sentence bearish argument
    key_themes       : list of 3-5 key themes extracted from news
    n_articles       : number of news articles analyzed
    model_used       : LLM provider/model that produced the result
    """
    ticker: str
    score: float
    normalized_score: float
    label: str
    bull_thesis: str
    bear_thesis: str
    key_themes: list[str] = field(default_factory=list)
    n_articles: int = 0
    model_used: str = "none"

    def summary(self) -> str:
        bar_len = int(self.normalized_score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines = [
            f"News Sentiment — {self.ticker}",
            "-" * 40,
            f"  Label  : {self.label}",
            f"  Score  : {bar}  {self.normalized_score:.3f}",
            f"  Sources: {self.n_articles} articles  |  Model: {self.model_used}",
            "",
            f"  Bull thesis: {self.bull_thesis}",
            f"  Bear thesis: {self.bear_thesis}",
        ]
        if self.key_themes:
            lines.append(f"  Themes : {', '.join(self.key_themes)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class NewsAnalyser:
    """
    LLM-powered qualitative analysis of news and earnings call transcripts.

    Usage::

        analyser = NewsAnalyser(provider="ollama", model="llama3.2")
        result = analyser.analyse(equity_data)
        print(result.summary())
    """

    _SYSTEM_PROMPT = (
        "You are an expert equity research analyst. "
        "Analyze the provided news articles and earnings call transcripts for a stock. "
        "Return ONLY a valid JSON object with this exact structure:\n"
        '{"score": <float -1 to 1>, "label": <"Bullish"|"Neutral"|"Bearish">, '
        '"bull_thesis": <string>, "bear_thesis": <string>, "key_themes": [<strings>]}\n'
        "score: -1=very bearish, 0=neutral, +1=very bullish. "
        "bull_thesis and bear_thesis: 2-3 sentences each. "
        "key_themes: 3-5 short themes. Be concise and base analysis on provided content."
    )

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.2",
        temperature: float = 0.1,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self._client: Optional[tuple[str, Any]] = self._init_client()

    def _init_client(self) -> Optional[tuple[str, Any]]:
        if self.provider == "ollama":
            try:
                import ollama
                return ("ollama", ollama)
            except ImportError:
                logger.warning("ollama package not installed — using rule-based analysis.")
                return None

        if self.provider == "anthropic":
            try:
                import anthropic
                return ("anthropic", anthropic.Anthropic())
            except (ImportError, Exception) as exc:
                logger.warning("Anthropic init failed: %s", exc)
                return None

        if self.provider == "openai":
            try:
                import openai
                return ("openai", openai.OpenAI())
            except (ImportError, Exception) as exc:
                logger.warning("OpenAI init failed: %s", exc)
                return None

        return None

    def analyse(self, data: Any) -> SentimentResult:  # data: EquityData
        """
        Analyze news and transcripts for the given equity data.
        Returns SentimentResult with qualitative insights.
        """
        ticker = data.ticker
        n_articles = len(data.news)

        if not data.news and not data.transcripts:
            return self._empty_result(ticker, "No news or transcripts available.")

        context = self._build_context(data)

        if self._client is None:
            return self._rule_based(data, context)

        try:
            raw = self._call_llm(context)
            return self._parse_response(raw, ticker, n_articles)
        except Exception as exc:
            logger.warning(
                "LLM sentiment failed for %s (%s) — falling back to rule-based.", ticker, exc
            )
            return self._rule_based(data, context)

    # -----------------------------------------------------------------------
    # Context builder
    # -----------------------------------------------------------------------

    def _build_context(self, data: Any) -> str:
        parts = [
            f"TICKER: {data.ticker}",
            f"COMPANY: {data.profile.name} ({data.profile.sector})",
        ]

        if data.news:
            parts.append(f"\n--- RECENT NEWS ({len(data.news)} articles) ---")
            for i, article in enumerate(data.news[:10]):
                title = article.get("title", "")
                desc = (article.get("description") or "")[:200]
                source = article.get("source", "")
                date = (article.get("published_at") or "")[:10]
                parts.append(f"{i + 1}. [{source}, {date}] {title}. {desc}")

        if data.transcripts:
            t = data.transcripts[0]
            parts.append(f"\n--- LATEST EARNINGS CALL: {t.get('title', '')} ---")
            parts.append(t.get("content", "")[:1500])

        return "\n".join(parts)

    # -----------------------------------------------------------------------
    # LLM calls
    # -----------------------------------------------------------------------

    def _call_llm(self, context: str) -> str:
        prompt = f"{context}\n\nProvide your analysis as JSON:"
        provider, client = self._client  # type: ignore[misc]

        if provider == "ollama":
            resp = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": self.temperature},
            )
            return resp["message"]["content"]

        if provider == "anthropic":
            resp = client.messages.create(
                model=self.model,
                max_tokens=512,
                system=self._SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        if provider == "openai":
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=512,
            )
            return resp.choices[0].message.content

        raise ValueError(f"Unknown provider: {provider}")

    # -----------------------------------------------------------------------
    # Response parsing
    # -----------------------------------------------------------------------

    def _parse_response(self, raw: str, ticker: str, n_articles: int) -> SentimentResult:
        # Strip markdown code fences if present
        json_str = raw
        match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try to find raw JSON object
            obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if obj_match:
                json_str = obj_match.group(0)

        payload = json.loads(json_str.strip())

        score = float(payload.get("score", 0.0))
        score = max(-1.0, min(1.0, score))
        normalized = (score + 1.0) / 2.0

        label = payload.get("label", "Neutral")
        if label not in ("Bullish", "Bearish", "Neutral"):
            label = "Bullish" if score > 0.2 else ("Bearish" if score < -0.2 else "Neutral")

        return SentimentResult(
            ticker=ticker,
            score=score,
            normalized_score=normalized,
            label=label,
            bull_thesis=payload.get("bull_thesis", ""),
            bear_thesis=payload.get("bear_thesis", ""),
            key_themes=payload.get("key_themes", []),
            n_articles=n_articles,
            model_used=f"{self.provider}/{self.model}",
        )

    # -----------------------------------------------------------------------
    # Rule-based fallback
    # -----------------------------------------------------------------------

    def _rule_based(self, data: Any, context: str) -> SentimentResult:
        """Keyword-counting sentiment when no LLM is available."""
        _POS = {
            "beat", "exceed", "growth", "profit", "gain", "surge", "record",
            "strong", "positive", "upgrade", "bullish", "outperform",
            "raise", "increase", "expand", "accelerate", "momentum", "buy",
        }
        _NEG = {
            "miss", "decline", "loss", "drop", "fall", "weak", "cut",
            "downgrade", "bearish", "underperform", "decrease",
            "concern", "risk", "warning", "slow", "layoff", "sell",
        }

        pos = neg = 0
        for article in data.news:
            text = (
                (article.get("title") or "") + " " + (article.get("description") or "")
            ).lower()
            pos += sum(1 for w in _POS if w in text)
            neg += sum(1 for w in _NEG if w in text)

        total = pos + neg
        score = (pos - neg) / total if total else 0.0
        normalized = (score + 1.0) / 2.0
        label = "Bullish" if score > 0.2 else ("Bearish" if score < -0.2 else "Neutral")

        return SentimentResult(
            ticker=data.ticker,
            score=score,
            normalized_score=normalized,
            label=label,
            bull_thesis=f"{pos} positive signals detected in recent news headlines.",
            bear_thesis=f"{neg} negative signals detected in recent news headlines.",
            key_themes=[],
            n_articles=len(data.news),
            model_used="rule-based",
        )

    def _empty_result(self, ticker: str, reason: str) -> SentimentResult:
        return SentimentResult(
            ticker=ticker,
            score=0.0,
            normalized_score=0.5,
            label="Neutral",
            bull_thesis=reason,
            bear_thesis=reason,
            key_themes=[],
            n_articles=0,
            model_used="none",
        )

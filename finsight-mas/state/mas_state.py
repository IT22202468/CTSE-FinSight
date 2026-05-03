# state/mas_state.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class Article(BaseModel):
    """A single news article fetched from an RSS source."""
    id: str                           # SHA256(url)[:12] — dedup key
    title: str
    summary: str
    url: str
    source: str                       # "Yahoo Finance", "Reuters", etc.
    tickers_mentioned: list[str] = [] # ["AAPL", "NVDA"]
    published_at: str = ""


class SentimentResult(BaseModel):
    """LLM sentiment classification for one article."""
    article_id: str
    ticker: str
    label: str                        # "BULLISH" | "BEARISH" | "NEUTRAL"
    confidence: float                 # 0.0–1.0
    reason: str                       # One-sentence justification from LLM


class RiskSignal(BaseModel):
    """Composite risk score for one ticker."""
    ticker: str
    sentiment_score: float            # Avg bearish weight across articles (0–1)
    price_momentum_7d: float          # % change: positive = up, negative = down
    volatility_14d: float             # Normalised std dev of returns (0–1)
    composite_risk: float             # 0.55*sentiment + 0.45*volatility
    risk_tier: str                    # "HIGH" | "MEDIUM" | "LOW"
    articles_analysed: int
    current_price: float = 0.0


class MASState(BaseModel):
    """Global state passed through all four agents."""
    run_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_timestamp: datetime = Field(default_factory=datetime.now)

    # Configuration (set at startup, read-only for agents)
    watchlist: list[str] = []
    alert_threshold: float = 0.70

    # Agent 1 output
    raw_articles: list[Article] = []

    # Agent 2 output
    sentiment_results: list[SentimentResult] = []

    # Agent 3 output
    risk_signals: list[RiskSignal] = []

    # Agent 4 output
    final_report_path: Optional[str] = None
    high_risk_tickers: list[str] = []

    # System-wide error accumulator
    errors: list[str] = []

# tools/market_tools.py
import ast
import json
import math
from typing import Type
import yfinance as yf
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, model_validator
from tenacity import RetryError
from config.logger import log_tool_call, log_tool_result, log_error
from tools.retry_utils import network_retry
from tools.input_guards import unwrap_json_envelope

AGENT = "MarketCorrelatorAgent"


class FetchStockDataInput(BaseModel):
    """Input schema for FetchStockDataTool."""
    ticker: str = Field(..., description="The stock ticker symbol, e.g. 'AAPL'.")
    period: str = Field(default="14d", description="Time period: '7d', '14d', '1mo', etc.")

    @model_validator(mode="before")
    @classmethod
    def unwrap_json_envelope(cls, values):
        return unwrap_json_envelope(values)

    @field_validator("ticker", mode="before")
    @classmethod
    def coerce_ticker_string(cls, v):
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("ticker list is empty")
            return str(v[0])
        return v


class FetchStockDataTool(BaseTool):
    name: str = "fetch_stock_data"
    description: str = (
        "Fetches OHLCV stock price history and current price for a ticker. "
        "Returns JSON with: ticker, current_price, prices (list of floats), dates (list), momentum_7d (float %)."
    )
    args_schema: Type[BaseModel] = FetchStockDataInput

    @staticmethod
    @network_retry
    def _fetch_history(ticker: str, period: str):
        t = yf.Ticker(ticker.upper())
        hist = t.history(period=period)
        if hist.empty:
            raise ValueError(f"Empty history returned for {ticker}")
        return hist

    def _run(self, ticker: str, period: str = "14d") -> str:
        log_tool_call(AGENT, self.name, {"ticker": ticker, "period": period})
        try:
            try:
                hist = self._fetch_history(ticker, period)
            except (RetryError, Exception) as e:
                err = f"Failed to fetch {ticker} after retries: {e}"
                log_error(AGENT, err)
                return json.dumps({"error": err, "ticker": ticker.upper(),
                                   "prices": [], "momentum_7d": 0.0, "current_price": 0.0})

            closes = hist["Close"].tolist()
            dates  = hist.index.strftime("%Y-%m-%d").tolist()

            momentum_7d = ((closes[-1] - closes[-7]) / closes[-7]) * 100 if len(closes) >= 7 else 0.0
            current_price = closes[-1]

            result = {
                "ticker": ticker.upper(),
                "current_price": round(current_price, 2),
                "prices": [round(p, 4) for p in closes],
                "dates": dates,
                "momentum_7d": round(momentum_7d, 4),
            }
            log_tool_result(AGENT, self.name, f"{ticker}: ${current_price:.2f}, momentum={momentum_7d:.2f}%")
            return json.dumps(result)

        except Exception as e:
            err = f"Unexpected error fetching {ticker}: {e}"
            log_error(AGENT, err)
            return json.dumps({"error": err, "ticker": ticker.upper(),
                               "prices": [], "momentum_7d": 0.0, "current_price": 0.0})


# ─── Tool 2: Compute Risk Signal ──────────────────────────────────────────

def _coerce_float_list(v):
    """Accept a JSON/Python-repr string in addition to an actual list of floats."""
    if isinstance(v, list):
        return [float(x) for x in v]
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
    return v


class ComputeRiskSignalInput(BaseModel):
    """Input schema for ComputeRiskSignalTool."""
    ticker: str = Field(..., description="Ticker symbol.")
    sentiment_scores: list[float] = Field(..., description="List of bearish sentiment weights (0.0–1.0) from articles.")
    articles_analysed: int = Field(default=0, description="Number of articles used.")
    # Price fields — required for full accuracy but default to neutral if omitted
    prices: list[float] = Field(default_factory=list, description="List of closing prices from fetch_stock_data.")
    momentum_7d: float = Field(default=0.0, description="7-day price momentum % from fetch_stock_data.")
    current_price: float = Field(default=0.0, description="Current stock price from fetch_stock_data.")
    # Confidence weights — default to equal weighting if omitted
    sentiment_confidences: list[float] = Field(default_factory=list, description="Confidence weights for each sentiment score.")

    @model_validator(mode="before")
    @classmethod
    def unwrap_json_envelope(cls, values):
        return unwrap_json_envelope(values)

    @field_validator("sentiment_scores", "sentiment_confidences", "prices", mode="before")
    @classmethod
    def coerce_float_list(cls, v):
        return _coerce_float_list(v)


class ComputeRiskSignalTool(BaseTool):
    name: str = "compute_risk_signal"
    description: str = (
        "Computes a composite risk score (0.0–1.0) and risk tier (HIGH/MEDIUM/LOW) for a ticker "
        "by combining weighted sentiment with 14-day price volatility. "
        "Formula: risk = 0.55 × sentiment + 0.45 × volatility."
    )
    args_schema: Type[BaseModel] = ComputeRiskSignalInput

    def _run(
        self,
        ticker: str,
        sentiment_scores: list[float],
        sentiment_confidences: list[float],
        prices: list[float],
        momentum_7d: float,
        articles_analysed: int,
        current_price: float = 0.0,
    ) -> str:
        """
        Compute composite risk signal for a ticker.

        Args:
            ticker: Stock ticker symbol.
            sentiment_scores: List of bearish weights (0=bullish, 1=bearish).
            sentiment_confidences: Matching confidence values for weighted avg.
            prices: Closing price list for std dev computation.
            momentum_7d: 7-day % price change.
            articles_analysed: Number of source articles.
            current_price: Latest stock price.

        Returns:
            JSON string: RiskSignal dict or {"error": "..."}.
        """
        log_tool_call(AGENT, self.name, {"ticker": ticker, "articles": articles_analysed})
        try:
            # Weighted sentiment average
            if sentiment_scores and sentiment_confidences:
                total_weight = sum(sentiment_confidences)
                if total_weight > 0:
                    sentiment_score = sum(s * c for s, c in zip(sentiment_scores, sentiment_confidences)) / total_weight
                else:
                    sentiment_score = 0.5
            else:
                sentiment_score = 0.5

            # Volatility: std dev of daily log returns, normalised to 0–1
            if len(prices) >= 2:
                log_returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices)) if prices[i-1] > 0]
                raw_vol = float(np.std(log_returns)) if log_returns else 0.0
                # Normalise: typical daily vol ~0.01–0.04 → map to 0–1
                volatility_14d = min(1.0, raw_vol / 0.04)
            else:
                volatility_14d = 0.0

            # Composite formula
            composite_risk = round(0.55 * sentiment_score + 0.45 * volatility_14d, 4)
            composite_risk = max(0.0, min(1.0, composite_risk))

            # Risk tier
            if composite_risk >= 0.70:
                risk_tier = "HIGH"
            elif composite_risk >= 0.40:
                risk_tier = "MEDIUM"
            else:
                risk_tier = "LOW"

            result = {
                "ticker": ticker.upper(),
                "sentiment_score": round(sentiment_score, 4),
                "price_momentum_7d": round(momentum_7d, 4),
                "volatility_14d": round(volatility_14d, 4),
                "composite_risk": composite_risk,
                "risk_tier": risk_tier,
                "articles_analysed": articles_analysed,
                "current_price": round(current_price, 2),
            }
            log_tool_result(AGENT, self.name, f"{ticker}: risk={composite_risk:.2f} [{risk_tier}]")
            return json.dumps(result)

        except Exception as e:
            err = f"Risk computation failed for {ticker}: {str(e)}"
            log_error(AGENT, err)
            return json.dumps({"error": err})

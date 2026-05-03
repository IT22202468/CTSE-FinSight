# tools/market_tools.py
import json
import math
from typing import Type
import yfinance as yf
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from config.logger import log_tool_call, log_tool_result, log_error

AGENT = "MarketCorrelatorAgent"


# ─── Tool 1: Fetch Stock Data ─────────────────────────────────────────────

class FetchStockDataInput(BaseModel):
    """Input schema for FetchStockDataTool."""
    ticker: str = Field(..., description="The stock ticker symbol, e.g. 'AAPL'.")
    period: str = Field(default="14d", description="Time period: '7d', '14d', '1mo', etc.")


class FetchStockDataTool(BaseTool):
    name: str = "fetch_stock_data"
    description: str = (
        "Fetches OHLCV stock price history and current price for a ticker via yfinance (free). "
        "Returns JSON with: ticker, current_price, prices (list), dates (list), momentum_7d (%)."
    )
    args_schema: Type[BaseModel] = FetchStockDataInput

    def _run(self, ticker: str, period: str = "14d") -> str:
        """
        Fetch historical price data for a stock ticker.

        Args:
            ticker: Stock ticker symbol (uppercase).
            period: yfinance period string — '7d', '14d', '1mo'.

        Returns:
            JSON string with price data dict, or {"error": "..."} on failure.
        """
        log_tool_call(AGENT, self.name, {"ticker": ticker, "period": period})
        try:
            t = yf.Ticker(ticker.upper())
            hist = t.history(period=period)

            if hist.empty:
                err = f"No price data for {ticker}"
                log_error(AGENT, err)
                return json.dumps({"error": err})

            closes = hist["Close"].tolist()
            dates  = hist.index.strftime("%Y-%m-%d").tolist()

            # 7-day momentum
            if len(closes) >= 7:
                momentum_7d = ((closes[-1] - closes[-7]) / closes[-7]) * 100
            else:
                momentum_7d = 0.0

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
            err = f"Failed to fetch {ticker}: {str(e)}"
            log_error(AGENT, err)
            return json.dumps({"error": err})


# ─── Tool 2: Compute Risk Signal ──────────────────────────────────────────

class ComputeRiskSignalInput(BaseModel):
    """Input schema for ComputeRiskSignalTool."""
    ticker: str = Field(..., description="Ticker symbol.")
    sentiment_scores: list[float] = Field(..., description="List of bearish sentiment weights (0.0–1.0) from articles.")
    sentiment_confidences: list[float] = Field(..., description="Confidence weights for each sentiment score.")
    prices: list[float] = Field(..., description="List of closing prices for volatility calculation.")
    momentum_7d: float = Field(..., description="7-day price momentum as percentage (e.g. -5.3).")
    articles_analysed: int = Field(..., description="Number of articles used.")
    current_price: float = Field(default=0.0, description="Current stock price.")


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

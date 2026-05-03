# evaluation/test_correlator.py
import json
import pytest
from hypothesis import given, strategies as st, settings
from tools.market_tools import FetchStockDataTool, ComputeRiskSignalTool

stock_tool = FetchStockDataTool()
risk_tool  = ComputeRiskSignalTool()

# ── Happy Path ──────────────────────────────────────────────────────────────

def test_fetch_stock_returns_prices():
    result = json.loads(stock_tool._run(ticker="AAPL"))
    assert "prices" in result
    assert isinstance(result["prices"], list)
    assert len(result["prices"]) > 0

def test_fetch_stock_returns_current_price():
    result = json.loads(stock_tool._run(ticker="MSFT"))
    assert "current_price" in result
    assert result["current_price"] > 0

def test_compute_risk_returns_valid_tier():
    result = json.loads(risk_tool._run(
        ticker="NVDA",
        sentiment_scores=[0.8],
        sentiment_confidences=[0.9],
        prices=[100, 98, 95, 93, 90, 88, 85, 83, 80, 78, 75, 73, 70, 68],
        momentum_7d=-12.5,
        articles_analysed=3,
        current_price=68.0,
    ))
    assert result["risk_tier"] in ("HIGH", "MEDIUM", "LOW")

# ── Edge Cases ──────────────────────────────────────────────────────────────

def test_fetch_invalid_ticker_returns_error():
    result = json.loads(stock_tool._run(ticker="ZZZZZZNOTREAL"))
    assert "error" in result

def test_risk_with_empty_prices_returns_zero_volatility():
    result = json.loads(risk_tool._run(
        ticker="AAPL",
        sentiment_scores=[0.5],
        sentiment_confidences=[0.8],
        prices=[],
        momentum_7d=0.0,
        articles_analysed=1,
    ))
    assert result["volatility_14d"] == 0.0

# ── Property-Based ──────────────────────────────────────────────────────────

@given(
    sentiment_scores=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=5),
    sentiment_confidences=st.lists(st.floats(min_value=0.1, max_value=1.0), min_size=1, max_size=5),
    prices=st.lists(st.floats(min_value=10.0, max_value=1000.0), min_size=2, max_size=14),
)
@settings(max_examples=20)
def test_composite_risk_always_in_0_to_1(sentiment_scores, sentiment_confidences, prices):
    """Composite risk must always be within [0.0, 1.0] for any valid inputs."""
    # Pad lists to same length
    n = min(len(sentiment_scores), len(sentiment_confidences))
    result = json.loads(risk_tool._run(
        ticker="TEST",
        sentiment_scores=sentiment_scores[:n],
        sentiment_confidences=sentiment_confidences[:n],
        prices=prices,
        momentum_7d=0.0,
        articles_analysed=n,
    ))
    if "error" not in result:
        assert 0.0 <= result["composite_risk"] <= 1.0

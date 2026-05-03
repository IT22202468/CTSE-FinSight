# evaluation/test_sentiment.py
import json
import pytest
from hypothesis import given, strategies as st, settings
from tools.sentiment_tools import ClassifySentimentTool, ExtractFinancialEntitiesTool

sentiment_tool = ClassifySentimentTool()
entity_tool    = ExtractFinancialEntitiesTool()

# ── Happy Path ──────────────────────────────────────────────────────────────

def test_classify_returns_required_fields():
    result = json.loads(sentiment_tool._run(text="Apple reports record revenue.", ticker="AAPL"))
    for field in ["label", "confidence"]:
        assert field in result, f"Missing: {field}"

def test_classify_label_is_valid_enum():
    result = json.loads(sentiment_tool._run(text="Apple reports record revenue.", ticker="AAPL"))
    assert result["label"] in ("BULLISH", "BEARISH", "NEUTRAL")

def test_classify_confidence_in_range():
    result = json.loads(sentiment_tool._run(text="Tesla falls 15% after recall.", ticker="TSLA"))
    assert 0.0 <= result["confidence"] <= 1.0

def test_extract_entities_returns_list():
    result = json.loads(entity_tool._run(text="Apple CEO Tim Cook announced record Q3 earnings."))
    assert "entities" in result
    assert isinstance(result["entities"], list)

# ── Edge Cases ──────────────────────────────────────────────────────────────

def test_classify_empty_text_returns_neutral():
    result = json.loads(sentiment_tool._run(text="", ticker="AAPL"))
    assert result["label"] == "NEUTRAL"

def test_classify_nonsense_text_does_not_crash():
    result = json.loads(sentiment_tool._run(text="asdfgh jkl qwerty 12345", ticker="MSFT"))
    assert "label" in result   # must return something valid

def test_confidence_never_exceeds_1():
    result = json.loads(sentiment_tool._run(text="NVIDIA achieves world-record AI chip sales.", ticker="NVDA"))
    assert result["confidence"] <= 1.0

# ── LLM-as-a-Judge ─────────────────────────────────────────────────────────

def test_obvious_bearish_classified_bearish():
    """
    LLM-as-a-Judge: Unambiguously negative article should return BEARISH.
    llama3.2:3b must score confidence >= 0.65 on obvious cases.
    """
    text = "Tesla stock crashes 25% after massive recall and CEO resignation announcement."
    result = json.loads(sentiment_tool._run(text=text, ticker="TSLA"))
    assert result["label"] == "BEARISH", f"Expected BEARISH, got {result['label']}"
    assert result["confidence"] >= 0.65, f"Confidence too low: {result['confidence']}"

def test_obvious_bullish_classified_bullish():
    """LLM-as-a-Judge: Clear positive news must be BULLISH."""
    text = "Microsoft beats all estimates with record cloud revenue, stock up 12% premarket."
    result = json.loads(sentiment_tool._run(text=text, ticker="MSFT"))
    assert result["label"] == "BULLISH"

# ── Property-Based ──────────────────────────────────────────────────────────

@given(text=st.text(min_size=10, max_size=200))
@settings(max_examples=10)
def test_classify_never_crashes_on_arbitrary_text(text):
    result = json.loads(sentiment_tool._run(text=text, ticker="AAPL"))
    assert "label" in result
    assert result["label"] in ("BULLISH", "BEARISH", "NEUTRAL")

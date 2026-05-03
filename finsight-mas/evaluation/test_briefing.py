# evaluation/test_briefing.py
import json
import pytest
from pathlib import Path
from hypothesis import given, strategies as st, settings
from tools.report_tools import InsertDBRecordTool, GenerateHTMLReportTool

db_tool   = InsertDBRecordTool()
html_tool = GenerateHTMLReportTool()

SAMPLE_SIGNAL = {
    "ticker": "AAPL",
    "sentiment_score": 0.7,
    "price_momentum_7d": -5.2,
    "volatility_14d": 0.65,
    "composite_risk": 0.68,
    "risk_tier": "MEDIUM",
    "articles_analysed": 4,
    "current_price": 182.50,
}

SAMPLE_SIGNALS = [
    {**SAMPLE_SIGNAL, "ticker": "AAPL", "risk_tier": "MEDIUM", "composite_risk": 0.68},
    {**SAMPLE_SIGNAL, "ticker": "TSLA", "risk_tier": "HIGH", "composite_risk": 0.82},
    {**SAMPLE_SIGNAL, "ticker": "MSFT", "risk_tier": "LOW", "composite_risk": 0.32},
]

# ── Happy Path ──────────────────────────────────────────────────────────────

def test_db_insert_returns_success():
    result = json.loads(db_tool._run(signal_json=json.dumps(SAMPLE_SIGNAL), run_id="test_001"))
    assert result.get("success") is True

def test_html_report_is_created():
    result = json.loads(html_tool._run(signals_json=json.dumps(SAMPLE_SIGNALS), run_id="test_001"))
    assert "report_path" in result
    assert Path(result["report_path"]).exists()

def test_html_report_contains_all_tickers():
    result = json.loads(html_tool._run(signals_json=json.dumps(SAMPLE_SIGNALS), run_id="test_002"))
    html_content = Path(result["report_path"]).read_text()
    for sig in SAMPLE_SIGNALS:
        assert sig["ticker"] in html_content

# ── Edge Cases ──────────────────────────────────────────────────────────────

def test_html_with_empty_signals_does_not_crash():
    result = json.loads(html_tool._run(signals_json=json.dumps([]), run_id="test_empty"))
    assert "report_path" in result

def test_db_with_invalid_json_returns_error():
    result = json.loads(db_tool._run(signal_json="not json", run_id="test_bad"))
    assert "error" in result

# ── Property-Based ──────────────────────────────────────────────────────────

@given(risk_score=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=15)
def test_risk_tier_matches_score(risk_score):
    """High score must always map to correct tier in report."""
    signal = {**SAMPLE_SIGNAL, "composite_risk": risk_score,
              "risk_tier": "HIGH" if risk_score >= 0.70 else "MEDIUM" if risk_score >= 0.40 else "LOW"}
    result = json.loads(html_tool._run(signals_json=json.dumps([signal]), run_id=f"test_{risk_score:.2f}"))
    assert "report_path" in result

# evaluation/test_fetcher.py
import json
import pytest
from hypothesis import given, strategies as st, settings
from tools.news_tools import FetchRSSFeedTool, FilterByTickersTool

fetch_tool  = FetchRSSFeedTool()
filter_tool = FilterByTickersTool()

# ── Happy Path ──────────────────────────────────────────────────────────────

def test_fetch_rss_returns_list():
    result = json.loads(fetch_tool._run(url="https://finance.yahoo.com/rss/topstories"))
    assert isinstance(result, list), "Expected list of articles"
    assert len(result) > 0

def test_fetch_rss_article_has_required_fields():
    result = json.loads(fetch_tool._run(url="https://finance.yahoo.com/rss/topstories"))
    article = result[0]
    for field in ["id", "title", "summary", "url", "source"]:
        assert field in article, f"Missing field: {field}"

def test_fetch_rss_respects_max_articles():
    result = json.loads(fetch_tool._run(url="https://finance.yahoo.com/rss/topstories", max_articles=5))
    assert len(result) <= 5

# ── Edge Cases ──────────────────────────────────────────────────────────────

def test_fetch_rss_invalid_url_returns_error():
    result = json.loads(fetch_tool._run(url="https://this-does-not-exist-xyz.com/rss"))
    assert "error" in result, "Expected error key on invalid URL"

def test_filter_with_no_matching_tickers_returns_empty():
    articles_json = json.dumps([{
        "id": "abc123",
        "title": "General economic outlook for 2025",
        "summary": "Economists predict moderate growth.",
        "url": "https://example.com",
        "source": "Test",
        "tickers_mentioned": [],
        "published_at": ""
    }])
    result = json.loads(filter_tool._run(articles_json=articles_json, tickers=["AAPL", "TSLA"]))
    assert result == [], f"Expected empty list, got {result}"

def test_filter_invalid_json_returns_error():
    result = json.loads(filter_tool._run(articles_json="not json", tickers=["AAPL"]))
    assert "error" in result

# ── Property-Based ──────────────────────────────────────────────────────────

@given(tickers=st.lists(st.sampled_from(["AAPL", "MSFT", "NVDA", "TSLA"]), min_size=1, max_size=4))
@settings(max_examples=10)
def test_filter_output_always_subset_of_input(tickers):
    raw = json.loads(fetch_tool._run(url="https://finance.yahoo.com/rss/topstories", max_articles=10))
    if isinstance(raw, list) and raw:
        filtered = json.loads(filter_tool._run(articles_json=json.dumps(raw), tickers=tickers))
        assert len(filtered) <= len(raw), "Filtered list cannot exceed input size"

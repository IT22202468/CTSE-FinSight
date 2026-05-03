# evaluation/test_fetcher.py
import json

import pytest
from hypothesis import given, settings, strategies as st

from tools.news_tools import FetchTickerNewsTool, FilterByTickersTool

fetch_tool = FetchTickerNewsTool()
filter_tool = FilterByTickersTool()

# ── Happy Path ──────────────────────────────────────────────────────────────


def test_fetch_ticker_news_returns_list():
    result = json.loads(fetch_tool._run(tickers=["AAPL"], max_per_ticker=8))
    assert isinstance(result, list), "Expected list of articles"


def test_fetch_ticker_news_article_has_required_fields():
    result = json.loads(fetch_tool._run(tickers=["MSFT"], max_per_ticker=5))
    if not result:
        pytest.skip("No Yahoo Finance headlines returned (network or API)")
    article = result[0]
    for field in ["id", "title", "summary", "url", "source", "tickers_mentioned"]:
        assert field in article, f"Missing field: {field}"


def test_fetch_ticker_news_respects_max_per_ticker():
    result = json.loads(fetch_tool._run(tickers=["AAPL"], max_per_ticker=3))
    assert len(result) <= 3


# ── Edge Cases ──────────────────────────────────────────────────────────────


def test_fetch_ticker_news_unknown_symbol_returns_empty_or_small():
    result = json.loads(fetch_tool._run(tickers=["ZZZZINVALID99"], max_per_ticker=5))
    assert isinstance(result, list)


def test_filter_with_no_matching_tickers_returns_empty():
    articles_json = json.dumps(
        [
            {
                "id": "abc123",
                "title": "General economic outlook for 2025",
                "summary": "Economists predict moderate growth.",
                "url": "https://example.com",
                "source": "Test",
                "tickers_mentioned": [],
                "published_at": "",
            }
        ]
    )
    result = json.loads(filter_tool._run(articles_json=articles_json, tickers=["AAPL", "TSLA"]))
    assert result == [], f"Expected empty list, got {result}"


def test_filter_keeps_article_when_ticker_pre_tagged():
    articles_json = json.dumps(
        [
            {
                "id": "x1",
                "title": "Quarterly update",
                "summary": "Revenue discussion without symbols.",
                "url": "https://example.com/a",
                "source": "Test",
                "tickers_mentioned": ["NVDA"],
                "published_at": "",
            }
        ]
    )
    result = json.loads(filter_tool._run(articles_json=articles_json, tickers=["NVDA", "AAPL"]))
    assert len(result) == 1
    assert "NVDA" in result[0]["tickers_mentioned"]


def test_filter_invalid_json_returns_error():
    result = json.loads(filter_tool._run(articles_json="not json", tickers=["AAPL"]))
    assert "error" in result


# ── Property-Based ──────────────────────────────────────────────────────────


@given(tickers=st.lists(st.sampled_from(["AAPL", "MSFT", "NVDA", "TSLA"]), min_size=1, max_size=2))
@settings(max_examples=8)
def test_filter_output_always_subset_of_input(tickers):
    raw = json.loads(fetch_tool._run(tickers=["AAPL"], max_per_ticker=8))
    if isinstance(raw, list) and raw:
        filtered = json.loads(filter_tool._run(articles_json=json.dumps(raw), tickers=tickers))
        assert len(filtered) <= len(raw), "Filtered list cannot exceed input size"

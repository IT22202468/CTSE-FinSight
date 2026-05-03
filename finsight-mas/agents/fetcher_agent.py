# agents/fetcher_agent.py
import json
from crewai import Agent, Task
from tools.news_tools import FetchTickerNewsTool, FilterByTickersTool
from config import OLLAMA_MODEL, LLM_MAX_ITER_FETCHER, LLM_MAX_RETRY, MAX_NEWS_PER_TICKER

fetcher_agent = Agent(
    role="Financial News Aggregator",
    goal=(
        "Fetch financial news from Yahoo Finance (per ticker), filter to watchlist tickers, "
        "deduplicate, and return structured article data."
    ),
    backstory=(
        "You are a financial data engineer who ingests market headlines via yfinance. "
        "You are precise about deduplication and always tag which tickers each article mentions."
    ),
    tools=[FetchTickerNewsTool(), FilterByTickersTool()],
    llm=f"ollama/{OLLAMA_MODEL}",
    verbose=True,
    allow_delegation=False,
    max_iter=LLM_MAX_ITER_FETCHER,
    max_retry_limit=LLM_MAX_RETRY,
)


def make_fetcher_task(watchlist: list[str]) -> Task:
    tickers_json = json.dumps(watchlist)
    return Task(
        description=(
            "Fetch financial news and filter by watchlist tickers.\n\n"
            "STEP 1 — Call fetch_ticker_news ONCE with:\n"
            f"  tickers: {tickers_json}   (must be this JSON array of strings, not a prose description)\n"
            f"  max_per_ticker: {MAX_NEWS_PER_TICKER}\n\n"
            "STEP 2 — Call filter_by_tickers with:\n"
            "  articles_json: the exact JSON array string returned from fetch_ticker_news in step 1\n"
            f"  tickers: {tickers_json}\n\n"
            "IMPORTANT RULES:\n"
            "  - Do not use RSS or fetch_rss_feed; use only fetch_ticker_news for headlines.\n"
            f"  - filter_by_tickers 'tickers' must be a JSON array like {tickers_json}, never a description string.\n"
            "  - articles_json must be real JSON from step 1, never placeholder empty objects.\n\n"
            "Return: 'Fetched N articles covering tickers: X, Y, Z.'"
        ),
        expected_output="A plain-English summary of how many articles were fetched and which tickers appeared.",
        agent=fetcher_agent,
    )

# agents/fetcher_agent.py
import json
from crewai import Agent, Task
from tools.news_tools import FetchRSSFeedTool, FilterByTickersTool
from config import OLLAMA_MODEL, LLM_MAX_ITER, LLM_MAX_RETRY, RSS_FEEDS

fetcher_agent = Agent(
    role="Financial News Aggregator",
    goal=(
        "Fetch financial news from RSS feeds, filter to watchlist tickers, "
        "deduplicate, and return structured article data."
    ),
    backstory=(
        "You are a financial data engineer who ingests news feeds. "
        "You are precise about deduplication and always tag which tickers each article mentions."
    ),
    tools=[FetchRSSFeedTool(), FilterByTickersTool()],
    llm=f"ollama/{OLLAMA_MODEL}",
    verbose=True,
    allow_delegation=False,
    max_iter=LLM_MAX_ITER,
    max_retry_limit=LLM_MAX_RETRY,
)


def make_fetcher_task(watchlist: list[str]) -> Task:
    feed_calls = "\n".join(
        f'  fetch_rss_feed(url="{f}")' for f in RSS_FEEDS
    )
    tickers_json = json.dumps(watchlist)
    return Task(
        description=(
            "Fetch financial news and filter by watchlist tickers.\n\n"
            "STEP 1 — Call fetch_rss_feed ONCE PER URL (each call takes a single string URL, NOT a list):\n"
            f"{feed_calls}\n\n"
            "STEP 2 — Merge all article JSON arrays into one combined JSON string.\n\n"
            "STEP 3 — Call filter_by_tickers with:\n"
            "  articles_json: the combined JSON array string from step 2\n"
            f"  tickers: {tickers_json}\n\n"
            "IMPORTANT RULES:\n"
            "  - fetch_rss_feed 'url' must be a plain string, never a list.\n"
            f"  - filter_by_tickers 'tickers' must be a JSON array like {tickers_json}, never a description string.\n"
            "  - articles_json must be real JSON from step 2, never placeholder empty objects.\n\n"
            "Return: 'Fetched N articles covering tickers: X, Y, Z.'"
        ),
        expected_output="A plain-English summary of how many articles were fetched and which tickers appeared.",
        agent=fetcher_agent,
    )

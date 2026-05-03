# agents/fetcher_agent.py
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
    feeds_str = "\n".join(f"- {f}" for f in RSS_FEEDS)
    return Task(
        description=(
            f"Fetch financial news from these RSS feeds:\n{feeds_str}\n\n"
            f"Filter articles to only those mentioning these tickers: {watchlist}.\n"
            "Use fetch_rss_feed for each feed URL, then filter_by_tickers on the combined results.\n"
            "Return a summary: 'Fetched N articles covering tickers: X, Y, Z.'"
        ),
        expected_output="A plain-English summary of how many articles were fetched and which tickers appeared.",
        agent=fetcher_agent,
    )

# tools/news_tools.py
import ast
import feedparser
import hashlib
import json
import re
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, model_validator
from config.logger import log_tool_call, log_tool_result, log_error

AGENT = "NewsFetcherAgent"


def _unwrap_json_envelope(values: dict) -> dict:
    """
    Detect and unwrap the LLM anti-pattern of packing all args into a single
    JSON-string value under an arbitrary key, e.g. {'rss': '{"url":"https://..."}'}.
    """
    if len(values) == 1:
        only_value = next(iter(values.values()))
        if isinstance(only_value, str):
            try:
                inner = json.loads(only_value)
                if isinstance(inner, dict):
                    return inner
            except (json.JSONDecodeError, ValueError):
                pass
    return values


# ─── Tool 1: Fetch RSS Feed ────────────────────────────────────────────────

class FetchRSSFeedInput(BaseModel):
    """Input schema for FetchRSSFeedTool."""
    url: str = Field(..., description="The RSS feed URL to fetch articles from.")
    max_articles: int = Field(default=15, description="Maximum articles to return.")

    @model_validator(mode="before")
    @classmethod
    def unwrap_json_envelope(cls, values):
        return _unwrap_json_envelope(values)

    @field_validator("url", mode="before")
    @classmethod
    def coerce_url_string(cls, v):
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("url list is empty")
            return v[0]
        return v


class FetchRSSFeedTool(BaseTool):
    name: str = "fetch_rss_feed"
    description: str = (
        "Fetches and parses a financial news RSS feed from a URL. "
        "Returns a JSON list of article dicts with id, title, summary, url, source, published_at."
    )
    args_schema: Type[BaseModel] = FetchRSSFeedInput

    def _run(self, url: str, max_articles: int = 15) -> str:
        """
        Fetch and parse an RSS feed.

        Args:
            url: The RSS feed URL to parse.
            max_articles: Maximum number of articles to return.

        Returns:
            JSON string: list of article dicts on success, or {"error": "..."} on failure.
        """
        log_tool_call(AGENT, self.name, {"url": url, "max_articles": max_articles})
        try:
            feed = feedparser.parse(url)
            if not feed.entries:
                err = f"No entries found at {url}"
                log_error(AGENT, err)
                return json.dumps({"error": err})

            articles = []
            for entry in feed.entries[:max_articles]:
                raw_url = entry.get("link", "")
                article = {
                    "id": hashlib.sha256(raw_url.encode()).hexdigest()[:12],
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", "")[:500],   # cap to 500 chars
                    "url": raw_url,
                    "source": feed.feed.get("title", url),
                    "tickers_mentioned": [],                      # filled by next tool
                    "published_at": entry.get("published", ""),
                }
                articles.append(article)

            log_tool_result(AGENT, self.name, f"Fetched {len(articles)} articles from {url}")
            return json.dumps(articles)

        except Exception as e:
            err = f"Failed to fetch RSS from {url}: {str(e)}"
            log_error(AGENT, err)
            return json.dumps({"error": err})


# ─── Tool 2: Filter by Tickers ────────────────────────────────────────────

class FilterByTickersInput(BaseModel):
    """Input schema for FilterByTickersTool."""
    articles_json: str = Field(..., description="JSON string of article list from fetch_rss_feed.")
    tickers: list[str] = Field(..., description="List of ticker symbols to match, e.g. ['AAPL', 'TSLA'].")

    @model_validator(mode="before")
    @classmethod
    def unwrap_json_envelope(cls, values):
        return _unwrap_json_envelope(values)

    @field_validator("tickers", mode="before")
    @classmethod
    def coerce_tickers_list(cls, v):
        """Accept a JSON array string or Python list-repr in addition to an actual list."""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        return v


class FilterByTickersTool(BaseTool):
    name: str = "filter_by_tickers"
    description: str = (
        "Scans article titles and summaries for mentions of ticker symbols. "
        "Annotates each article with tickers_mentioned and returns only relevant articles. "
        "Input: JSON string of articles + list of tickers."
    )
    args_schema: Type[BaseModel] = FilterByTickersInput

    def _run(self, articles_json: str, tickers: list[str]) -> str:
        """
        Filter articles to only those mentioning watchlist tickers.

        Args:
            articles_json: JSON string of article list from fetch_rss_feed.
            tickers: List of ticker symbols to search for.

        Returns:
            JSON string: filtered + annotated article list, or {"error": "..."}.
        """
        log_tool_call(AGENT, self.name, {"tickers": tickers, "article_count": "see input"})
        try:
            articles = json.loads(articles_json)
            if isinstance(articles, dict) and "error" in articles:
                return articles_json  # propagate upstream error

            # Build per-ticker company name map for fuzzy matching
            TICKER_NAMES = {
                "AAPL": ["Apple", "AAPL"],
                "MSFT": ["Microsoft", "MSFT"],
                "NVDA": ["Nvidia", "NVDA", "NVidia"],
                "TSLA": ["Tesla", "TSLA"],
                "GOOGL": ["Google", "Alphabet", "GOOGL"],
                "AMZN": ["Amazon", "AMZN"],
                "META": ["Meta", "Facebook", "META"],
                "NFLX": ["Netflix", "NFLX"],
            }

            relevant = []
            for article in articles:
                text = f"{article['title']} {article['summary']}"
                found = []
                for ticker in tickers:
                    aliases = TICKER_NAMES.get(ticker, [ticker])
                    if any(re.search(rf'\b{re.escape(a)}\b', text, re.IGNORECASE) for a in aliases):
                        found.append(ticker)
                if found:
                    article["tickers_mentioned"] = found
                    relevant.append(article)

            log_tool_result(AGENT, self.name, f"{len(relevant)} relevant articles from {len(articles)} total")
            return json.dumps(relevant)

        except json.JSONDecodeError as e:
            err = f"Invalid JSON in articles_json: {e}"
            log_error(AGENT, err)
            return json.dumps({"error": err})
        except Exception as e:
            err = f"Filtering failed: {str(e)}"
            log_error(AGENT, err)
            return json.dumps({"error": err})

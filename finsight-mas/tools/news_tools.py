# tools/news_tools.py
import ast
import hashlib
import json
import re
from typing import Any, Type

import yfinance as yf
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, model_validator
from tenacity import RetryError
from config.logger import log_tool_call, log_tool_result, log_error
from tools.retry_utils import network_retry
from tools.input_guards import unwrap_json_envelope, guard_error, validate_json_list_field

AGENT = "NewsFetcherAgent"


def _normalize_yf_news_item(raw: dict[str, Any], ticker: str) -> dict[str, Any] | None:
    """Map a yfinance news entry to our article schema (supports nested `content` and legacy shapes)."""
    content = raw.get("content")
    if isinstance(content, dict):
        title = (content.get("title") or "").strip()
        summary = (content.get("summary") or content.get("description") or "")[:500]
        published_at = content.get("pubDate") or content.get("displayTime") or ""
        source = (content.get("provider") or {}).get("displayName") or "Yahoo Finance"
        canon = content.get("canonicalUrl")
        click = content.get("clickThroughUrl")
        url = ""
        if isinstance(canon, dict):
            url = (canon.get("url") or "").strip()
        if not url and isinstance(click, dict):
            url = (click.get("url") or "").strip()
        raw_id = (content.get("id") or raw.get("id") or url or title) or ""
    else:
        title = (raw.get("title") or "").strip()
        summary = (raw.get("summary") or "")[:500]
        published_at = str(raw.get("providerPublishTime", raw.get("pubDate", "")))
        source = str(raw.get("publisher", "Yahoo Finance"))
        url = (raw.get("link") or "").strip()
        raw_id = (raw.get("uuid") or raw.get("id") or url or title) or ""

    if not title and not url:
        return None

    key = url if url else raw_id
    article_id = hashlib.sha256(key.encode()).hexdigest()[:12]
    t = ticker.upper()
    return {
        "id": article_id,
        "title": title,
        "summary": summary,
        "url": url,
        "source": source,
        "tickers_mentioned": [t],
        "published_at": published_at,
    }


# ─── Tool 1: Yahoo Finance ticker news (JSON, not RSS) ─────────────────────

class FetchTickerNewsInput(BaseModel):
    """Input schema for FetchTickerNewsTool."""
    tickers: list[str] = Field(..., description="Stock symbols to pull Yahoo Finance news for, e.g. ['AAPL', 'MSFT'].")
    max_per_ticker: int = Field(default=10, ge=1, le=50, description="Max headlines per symbol.")

    @model_validator(mode="before")
    @classmethod
    def unwrap_json_envelope(cls, values):
        return unwrap_json_envelope(values)

    @field_validator("tickers", mode="before")
    @classmethod
    def coerce_tickers_list(cls, v):
        if isinstance(v, list):
            return [str(x).strip().upper() for x in v if str(x).strip()]
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(x).strip().upper() for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                pass
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, list):
                    return [str(x).strip().upper() for x in parsed if str(x).strip()]
            except (ValueError, SyntaxError):
                pass
        raise ValueError("'tickers' must be a non-empty list of ticker symbols (or a JSON array string).")


class FetchTickerNewsTool(BaseTool):
    name: str = "fetch_ticker_news"
    description: str = (
        "Fetches recent headlines for given stock symbols via Yahoo Finance (JSON API through yfinance). "
        "Returns a JSON list of article dicts with id, title, summary, url, source, published_at, tickers_mentioned. "
        "Prefer this over RSS feeds; call once with the full watchlist."
    )
    args_schema: Type[BaseModel] = FetchTickerNewsInput

    @staticmethod
    @network_retry
    def _fetch_raw(ticker: str, count: int) -> list[dict[str, Any]]:
        t = yf.Ticker(ticker)
        if hasattr(t, "get_news"):
            out = t.get_news(count=count)
            return list(out) if out else []
        news = t.news
        return list(news) if isinstance(news, list) and news else []

    def _run(self, tickers: list[str], max_per_ticker: int = 10) -> str:
        log_tool_call(AGENT, self.name, {"tickers": tickers, "max_per_ticker": max_per_ticker})
        if not tickers:
            log_error(AGENT, "fetch_ticker_news called with empty tickers")
            return json.dumps([])

        by_url: dict[str, dict[str, Any]] = {}
        try:
            for ticker in tickers:
                try:
                    raw_list = self._fetch_raw(ticker, max_per_ticker)
                except (RetryError, Exception) as e:
                    log_error(AGENT, f"yfinance news failed for {ticker}: {e}")
                    continue

                for raw in raw_list:
                    if not isinstance(raw, dict):
                        continue
                    art = _normalize_yf_news_item(raw, ticker)
                    if not art:
                        continue
                    u = art["url"] or art["id"]
                    if u in by_url:
                        prev = set(by_url[u].get("tickers_mentioned") or [])
                        prev.update(art.get("tickers_mentioned") or [])
                        by_url[u]["tickers_mentioned"] = sorted(prev)
                    else:
                        by_url[u] = art

            merged = list(by_url.values())
            log_tool_result(AGENT, self.name, f"Fetched {len(merged)} unique articles for {len(tickers)} tickers")
            return json.dumps(merged)

        except Exception as e:
            err = f"Unexpected error in fetch_ticker_news: {e}"
            log_error(AGENT, err)
            return json.dumps([])


# ─── Tool 2: Filter by Tickers ────────────────────────────────────────────

class FilterByTickersInput(BaseModel):
    """Input schema for FilterByTickersTool."""
    articles_json: str = Field(..., description="JSON string of article list from fetch_ticker_news.")
    tickers: list[str] = Field(..., description="List of ticker symbols to match, e.g. ['AAPL', 'TSLA'].")

    @model_validator(mode="before")
    @classmethod
    def unwrap_json_envelope(cls, values):
        return unwrap_json_envelope(values)

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
        # ── Input guard ───────────────────────────────────────────────────
        err = validate_json_list_field(articles_json, "articles_json")
        if err:
            log_error(AGENT, f"filter_by_tickers bad articles_json: {err}")
            return guard_error([err])
        if not tickers:
            log_error(AGENT, "filter_by_tickers called with empty tickers list")
            return guard_error(["'tickers' must be a non-empty list of ticker symbols."])

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
                pre = article.get("tickers_mentioned") or []
                found = []
                for ticker in tickers:
                    if ticker in pre:
                        found.append(ticker)
                        continue
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

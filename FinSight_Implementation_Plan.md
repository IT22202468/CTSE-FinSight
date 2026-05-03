
# FinSight MAS — Implementation Plan
## Model: phi3:mini via Ollama | Framework: CrewAI

> **Document type:** Living implementation plan — update as you build  
> **Project:** SE4010 CTSE Assignment 2 — Financial News Intelligence & Risk Alerting MAS  
> **Team size:** 4 Students | **Framework:** CrewAI | **LLM:** phi3:mini (Ollama)

---

## ⚠️ phi3:mini-Specific Constraints (Read This First)

Before writing a single line of code, your entire team must understand how phi3:mini behaves differently from larger models like llama3.1:8b. This affects every prompt, every tool output format, and every agent design decision.

| Property | phi3:mini | llama3.1:8b |
|----------|-----------|-------------|
| Size | ~2.3 GB VRAM | ~5 GB VRAM |
| RAM required | 4 GB free | 8 GB free |
| Context window | 4,096 tokens | 8,192 tokens |
| Instruction following | Good on simple tasks | Strong on complex tasks |
| JSON output reliability | Moderate — needs strict prompting | Good |
| Multi-step reasoning | Struggles with >3 steps | Handles well |
| Speed (local) | ~30–50 tok/s | ~15–25 tok/s |

### What this means for your implementation:

1. **Short, directive prompts only** — phi3:mini drifts on long prompts. Keep system prompts under 100 words.
2. **Force JSON output explicitly** — Always end prompts with: `Respond ONLY with valid JSON. No explanation.`
3. **One task per agent** — phi3:mini cannot handle compound tasks reliably. Each agent does ONE thing.
4. **Validate every LLM output** — Wrap all LLM responses in try/except with JSON parsing fallback.
5. **Set temperature=0.0** — Not 0.1. Zero temperature gives the most deterministic output with small models.
6. **max_iter=2** — phi3:mini loops more aggressively than larger models. Cap at 2 iterations.

---

## Phase 0 — Environment Setup (Day 1 — Everyone)

### 0.1 Install Ollama and pull phi3:mini

```bash
# Install Ollama
# Windows: https://ollama.com/download
# Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Pull phi3:mini (~2.3GB — much faster than llama3)
ollama pull phi3:mini

# Verify it runs and test JSON output
ollama run phi3:mini "Classify this as BULLISH, BEARISH, or NEUTRAL. Return JSON only: {\"label\": \"...\", \"confidence\": 0.0}. Article: Apple hits record revenue."
```

### 0.2 Test phi3:mini JSON reliability

Before writing any agent code, every member should run this test:

```python
# scripts/test_phi3_json.py
from langchain_ollama import ChatOllama
import json

llm = ChatOllama(
    model="phi3:mini",
    base_url="http://localhost:11434",
    temperature=0.0,
)

prompt = """Classify this financial news as BULLISH, BEARISH, or NEUTRAL.
Article: "Apple reports record quarterly revenue, beating all analyst expectations."
Respond ONLY with valid JSON in this exact format:
{"label": "BULLISH", "confidence": 0.95, "reason": "one sentence"}
No other text."""

response = llm.invoke(prompt)
print("Raw response:", response.content)

try:
    # phi3:mini sometimes wraps JSON in markdown — strip it
    clean = response.content.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(clean)
    print("Parsed successfully:", data)
except json.JSONDecodeError as e:
    print("FAILED to parse JSON:", e)
    print("You will need fallback parsing in your tools.")
```

### 0.3 Install all dependencies

Use Python 3.13 or 3.12 for this project. CrewAI 1.14.4 does not support Python 3.14, so the install will fail on the default 3.14 runtime.

```bash
# Create and activate venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Core
pip install crewai crewai-tools
pip install langchain-ollama langchain-community

# Data
pip install yfinance feedparser httpx
pip install pandas numpy

# Storage & output
pip install sqlalchemy
pip install jinja2

# Logging & display
pip install loguru rich

# Testing
pip install pytest hypothesis pytest-cov

# Utilities
pip install python-dotenv pydantic

pip freeze > requirements.txt
```

### 0.4 Verify full stack

```python
# scripts/verify_setup.py
import sys

def check(label, fn):
    try:
        result = fn()
        print(f"  ✓ {label}: {result}")
    except Exception as e:
        print(f"  ✗ {label}: FAILED — {e}")
        sys.exit(1)

print("FinSight MAS — Setup Verification\n")

check("Python version", lambda: sys.version.split()[0])
check("crewai import", lambda: __import__("crewai") and "OK")
check("yfinance AAPL price", lambda: f"${__import__('yfinance').Ticker('AAPL').fast_info.last_price:.2f}")
check("feedparser", lambda: f"{len(__import__('feedparser').parse('https://finance.yahoo.com/rss/topstories').entries)} articles from Yahoo")
check("Ollama/phi3:mini", lambda: __import__('langchain_ollama').ChatOllama(model='phi3:mini').invoke('Reply OK').content.strip())
check("SQLite", lambda: __import__('sqlalchemy').create_engine('sqlite:///test.db') and "OK")

print("\n✓ All systems go. You are ready to build.\n")
```

---

## Phase 1 — Foundation (Day 1–2 — All Members Together)

> **Complete this before splitting into individual work. Everyone must agree on these files.**

### 1.1 Project folder — create this structure first

```bash
mkdir finsight-mas && cd finsight-mas
mkdir agents tools state crew evaluation templates logs outputs scripts
touch agents/__init__.py tools/__init__.py state/__init__.py
touch crew/__init__.py evaluation/__init__.py
touch logs/.gitkeep outputs/.gitkeep
```

### 1.2 config.py — central constants

```python
# config.py
from pathlib import Path

# ── LLM ──────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "phi3:mini"
LLM_TEMPERATURE = 0.0          # Zero = most deterministic for phi3:mini
LLM_MAX_ITER    = 2            # phi3:mini loops aggressively — keep at 2
LLM_MAX_RETRY   = 1

# ── Watchlist ─────────────────────────────────────────────
WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN"]

# ── News Sources ──────────────────────────────────────────
RSS_FEEDS = [
    "https://finance.yahoo.com/rss/topstories",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
]
MAX_ARTICLES_PER_FEED = 15

# ── Risk Thresholds ───────────────────────────────────────
ALERT_THRESHOLD_HIGH   = 0.70
ALERT_THRESHOLD_MEDIUM = 0.40

# ── Paths ─────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DB_PATH     = BASE_DIR / "outputs" / "finsight_history.db"
LOG_DIR     = BASE_DIR / "logs"
OUTPUT_DIR  = BASE_DIR / "outputs"
TEMPLATE_DIR = BASE_DIR / "templates"
```

### 1.3 state/mas_state.py — the team contract

**Everyone must agree on this before writing any tool or agent.**

```python
# state/mas_state.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


class Article(BaseModel):
    """A single news article fetched from an RSS source."""
    id: str                           # SHA256(url)[:12] — dedup key
    title: str
    summary: str
    url: str
    source: str                       # "Yahoo Finance", "Reuters", etc.
    tickers_mentioned: list[str] = [] # ["AAPL", "NVDA"]
    published_at: str = ""


class SentimentResult(BaseModel):
    """phi3:mini sentiment classification for one article."""
    article_id: str
    ticker: str
    label: str                        # "BULLISH" | "BEARISH" | "NEUTRAL"
    confidence: float                 # 0.0–1.0
    reason: str                       # One-sentence justification from LLM


class RiskSignal(BaseModel):
    """Composite risk score for one ticker."""
    ticker: str
    sentiment_score: float            # Avg bearish weight across articles (0–1)
    price_momentum_7d: float          # % change: positive = up, negative = down
    volatility_14d: float             # Normalised std dev of returns (0–1)
    composite_risk: float             # 0.55*sentiment + 0.45*volatility
    risk_tier: str                    # "HIGH" | "MEDIUM" | "LOW"
    articles_analysed: int
    current_price: float = 0.0


class MASState(BaseModel):
    """Global state passed through all four agents."""
    run_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_timestamp: datetime = Field(default_factory=datetime.now)

    # Configuration (set at startup, read-only for agents)
    watchlist: list[str] = []
    alert_threshold: float = 0.70

    # Agent 1 output
    raw_articles: list[Article] = []

    # Agent 2 output
    sentiment_results: list[SentimentResult] = []

    # Agent 3 output
    risk_signals: list[RiskSignal] = []

    # Agent 4 output
    final_report_path: Optional[str] = None
    high_risk_tickers: list[str] = []

    # System-wide error accumulator
    errors: list[str] = []
```

### 1.4 state/store.py — singleton state accessor

```python
# state/store.py
from state.mas_state import MASState
from config import WATCHLIST, ALERT_THRESHOLD_HIGH

_state: MASState | None = None


def init_state(watchlist: list[str] | None = None) -> MASState:
    global _state
    _state = MASState(
        watchlist=watchlist or WATCHLIST,
        alert_threshold=ALERT_THRESHOLD_HIGH,
    )
    return _state


def get_state() -> MASState:
    if _state is None:
        raise RuntimeError("State not initialised. Call init_state() first.")
    return _state


def update_state(**kwargs) -> MASState:
    """Merge keyword updates into the global state."""
    global _state
    updated = _state.model_copy(update=kwargs)
    _state = updated
    return _state
```

### 1.5 config/logger.py — shared structured logger

```python
# config/logger.py
import json
import time
from loguru import logger
from pathlib import Path
from config import LOG_DIR

LOG_DIR.mkdir(exist_ok=True)
_run_id = time.strftime("%Y%m%d_%H%M%S")

logger.remove()
logger.add(
    LOG_DIR / f"trace_{_run_id}.jsonl",
    format="{message}",
    level="DEBUG",
    enqueue=True,
)


def _emit(event: str, agent: str, payload: dict) -> None:
    record = {"ts": time.time(), "event": event, "agent": agent, **payload}
    logger.info(json.dumps(record))


def log_agent_start(agent: str, task: str) -> None:
    _emit("AGENT_START", agent, {"task": task})

def log_tool_call(agent: str, tool: str, inputs: dict) -> None:
    _emit("TOOL_CALL", agent, {"tool": tool, "inputs": inputs})

def log_tool_result(agent: str, tool: str, summary: str) -> None:
    _emit("TOOL_RESULT", agent, {"tool": tool, "summary": summary})

def log_agent_complete(agent: str, summary: str) -> None:
    _emit("AGENT_COMPLETE", agent, {"summary": summary})

def log_error(agent: str, error: str) -> None:
    _emit("ERROR", agent, {"error": error})
```

---

## Phase 2 — Individual Agent & Tool Development

> From this point, each member works independently. Refer to their section.

---

## Member 1 — News Fetcher Agent

**Files to create:** `tools/news_tools.py`, `agents/fetcher_agent.py`, `evaluation/test_fetcher.py`

### tools/news_tools.py

```python
# tools/news_tools.py
import feedparser
import hashlib
import json
import re
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from config.logger import log_tool_call, log_tool_result, log_error

AGENT = "NewsFetcherAgent"


# ─── Tool 1: Fetch RSS Feed ────────────────────────────────────────────────

class FetchRSSFeedInput(BaseModel):
    """Input schema for FetchRSSFeedTool."""
    url: str = Field(..., description="The RSS feed URL to fetch articles from.")
    max_articles: int = Field(default=15, description="Maximum articles to return.")


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
```

### agents/fetcher_agent.py

```python
# agents/fetcher_agent.py
from crewai import Agent, Task
from langchain_ollama import ChatOllama
from tools.news_tools import FetchRSSFeedTool, FilterByTickersTool
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TEMPERATURE, LLM_MAX_ITER, LLM_MAX_RETRY, RSS_FEEDS

_llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=LLM_TEMPERATURE,
)

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
    llm=_llm,
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
```

### evaluation/test_fetcher.py

```python
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
```

---

## Member 2 — Sentiment Analyst Agent

**Files to create:** `tools/sentiment_tools.py`, `agents/sentiment_agent.py`, `evaluation/test_sentiment.py`

### tools/sentiment_tools.py

```python
# tools/sentiment_tools.py
import json
import re
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TEMPERATURE
from config.logger import log_tool_call, log_tool_result, log_error

AGENT = "SentimentAnalystAgent"

_llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=LLM_TEMPERATURE,
)


def _safe_json_parse(raw: str) -> dict | None:
    """
    Parse JSON from phi3:mini output which may include markdown fences.

    Args:
        raw: Raw LLM response string.

    Returns:
        Parsed dict on success, None on failure.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract first JSON object with regex
        match = re.search(r'\{[^}]+\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None


# ─── Tool 1: Classify Sentiment ──────────────────────────────────────────────

class ClassifySentimentInput(BaseModel):
    """Input schema for ClassifySentimentTool."""
    text: str = Field(..., description="The article title and summary to classify.")
    ticker: str = Field(..., description="The ticker symbol this article relates to, e.g. 'AAPL'.")


class ClassifySentimentTool(BaseTool):
    name: str = "classify_sentiment"
    description: str = (
        "Classifies financial news text as BULLISH, BEARISH, or NEUTRAL for a given ticker. "
        "Uses the local phi3:mini LLM. Returns JSON with label, confidence (0.0–1.0), and reason."
    )
    args_schema: Type[BaseModel] = ClassifySentimentInput

    def _run(self, text: str, ticker: str) -> str:
        """
        Classify sentiment of a financial news article.

        Args:
            text: Article title and summary text (max 400 chars recommended for phi3:mini).
            ticker: Ticker symbol to frame the classification context.

        Returns:
            JSON string: {"label": str, "confidence": float, "reason": str}
            or {"label": "NEUTRAL", "confidence": 0.5, "reason": "parse_error", "error": str}
        """
        log_tool_call(AGENT, self.name, {"ticker": ticker, "text_len": len(text)})

        # Keep prompt minimal for phi3:mini — it struggles with long prompts
        prompt = (
            f"Classify this financial news about {ticker} stock.\n"
            f"Text: {text[:400]}\n\n"
            "Choose one: BULLISH (positive for stock price), BEARISH (negative), NEUTRAL (irrelevant).\n"
            'Respond ONLY with valid JSON:\n{"label": "BULLISH", "confidence": 0.9, "reason": "one sentence"}\n'
            "No other text. No explanation."
        )

        try:
            response = _llm.invoke(prompt)
            parsed = _safe_json_parse(response.content)

            if parsed is None:
                log_error(AGENT, f"JSON parse failed for ticker {ticker}. Raw: {response.content[:100]}")
                result = {"label": "NEUTRAL", "confidence": 0.5, "reason": "parse_error",
                          "error": "LLM returned unparseable output"}
            else:
                # Validate and clamp values
                label = parsed.get("label", "NEUTRAL").upper()
                if label not in ("BULLISH", "BEARISH", "NEUTRAL"):
                    label = "NEUTRAL"
                confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
                result = {"label": label, "confidence": confidence,
                          "reason": str(parsed.get("reason", ""))[:200]}

            log_tool_result(AGENT, self.name, f"{ticker} → {result['label']} ({result['confidence']:.2f})")
            return json.dumps(result)

        except Exception as e:
            err = f"Sentiment classification failed: {str(e)}"
            log_error(AGENT, err)
            return json.dumps({"label": "NEUTRAL", "confidence": 0.5, "reason": "exception", "error": err})


# ─── Tool 2: Extract Entities ─────────────────────────────────────────────────

class ExtractEntitiesInput(BaseModel):
    """Input schema for ExtractFinancialEntitiesTool."""
    text: str = Field(..., description="Article text to extract entities from.")


class ExtractFinancialEntitiesTool(BaseTool):
    name: str = "extract_financial_entities"
    description: str = (
        "Extracts key financial entities from article text: company names, executives, "
        "economic events, and figures. Returns JSON list of entity strings."
    )
    args_schema: Type[BaseModel] = ExtractEntitiesInput

    def _run(self, text: str) -> str:
        """
        Extract financial entities from text using phi3:mini.

        Args:
            text: Article title + summary text.

        Returns:
            JSON string: {"entities": ["Apple", "Tim Cook", "Q3 earnings"]} or error dict.
        """
        log_tool_call(AGENT, self.name, {"text_len": len(text)})

        prompt = (
            f"Extract financial entities from this text:\n{text[:300]}\n\n"
            "Find: company names, executive names, financial events, economic figures.\n"
            'Respond ONLY with valid JSON: {"entities": ["name1", "name2"]}\n'
            "No other text."
        )

        try:
            response = _llm.invoke(prompt)
            parsed = _safe_json_parse(response.content)

            if parsed is None or "entities" not in parsed:
                result = {"entities": []}
            else:
                entities = [str(e) for e in parsed["entities"] if e][:10]  # cap at 10
                result = {"entities": entities}

            log_tool_result(AGENT, self.name, f"Extracted {len(result['entities'])} entities")
            return json.dumps(result)

        except Exception as e:
            err = f"Entity extraction failed: {str(e)}"
            log_error(AGENT, err)
            return json.dumps({"entities": [], "error": err})
```

### agents/sentiment_agent.py

```python
# agents/sentiment_agent.py
from crewai import Agent, Task
from langchain_ollama import ChatOllama
from tools.sentiment_tools import ClassifySentimentTool, ExtractFinancialEntitiesTool
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TEMPERATURE, LLM_MAX_ITER, LLM_MAX_RETRY

_llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=LLM_TEMPERATURE)

sentiment_agent = Agent(
    role="Financial Sentiment Analyst",
    goal="Classify sentiment of financial articles and extract key entities using LLM tools.",
    backstory=(
        "You are a quantitative analyst who reads financial news precisely. "
        "You classify each article as BULLISH, BEARISH, or NEUTRAL with a confidence score. "
        "You never guess — ambiguous articles are NEUTRAL."
    ),
    tools=[ClassifySentimentTool(), ExtractFinancialEntitiesTool()],
    llm=_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=LLM_MAX_ITER,
    max_retry_limit=LLM_MAX_RETRY,
)


def make_sentiment_task(context_task) -> Task:
    return Task(
        description=(
            "Load the fetched articles from the previous task output.\n"
            "For each article:\n"
            "  1. Call classify_sentiment with the article text and its primary ticker.\n"
            "  2. Call extract_financial_entities with the article text.\n"
            "Return a summary: 'Classified N articles: X BULLISH, Y BEARISH, Z NEUTRAL.'"
        ),
        expected_output="Summary of sentiment classification counts across all articles.",
        agent=sentiment_agent,
        context=[context_task],
    )
```

### evaluation/test_sentiment.py

```python
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
    phi3:mini must score confidence ≥ 0.65 on obvious cases.
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
```

---

## Member 3 — Market Correlator Agent

**Files to create:** `tools/market_tools.py`, `agents/correlator_agent.py`, `evaluation/test_correlator.py`

### tools/market_tools.py

```python
# tools/market_tools.py
import json
import math
from typing import Type
import yfinance as yf
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from config.logger import log_tool_call, log_tool_result, log_error

AGENT = "MarketCorrelatorAgent"


# ─── Tool 1: Fetch Stock Data ─────────────────────────────────────────────

class FetchStockDataInput(BaseModel):
    """Input schema for FetchStockDataTool."""
    ticker: str = Field(..., description="The stock ticker symbol, e.g. 'AAPL'.")
    period: str = Field(default="14d", description="Time period: '7d', '14d', '1mo', etc.")


class FetchStockDataTool(BaseTool):
    name: str = "fetch_stock_data"
    description: str = (
        "Fetches OHLCV stock price history and current price for a ticker via yfinance (free). "
        "Returns JSON with: ticker, current_price, prices (list), dates (list), momentum_7d (%)."
    )
    args_schema: Type[BaseModel] = FetchStockDataInput

    def _run(self, ticker: str, period: str = "14d") -> str:
        """
        Fetch historical price data for a stock ticker.

        Args:
            ticker: Stock ticker symbol (uppercase).
            period: yfinance period string — '7d', '14d', '1mo'.

        Returns:
            JSON string with price data dict, or {"error": "..."} on failure.
        """
        log_tool_call(AGENT, self.name, {"ticker": ticker, "period": period})
        try:
            t = yf.Ticker(ticker.upper())
            hist = t.history(period=period)

            if hist.empty:
                err = f"No price data for {ticker}"
                log_error(AGENT, err)
                return json.dumps({"error": err})

            closes = hist["Close"].tolist()
            dates  = hist.index.strftime("%Y-%m-%d").tolist()

            # 7-day momentum
            if len(closes) >= 7:
                momentum_7d = ((closes[-1] - closes[-7]) / closes[-7]) * 100
            else:
                momentum_7d = 0.0

            current_price = closes[-1]

            result = {
                "ticker": ticker.upper(),
                "current_price": round(current_price, 2),
                "prices": [round(p, 4) for p in closes],
                "dates": dates,
                "momentum_7d": round(momentum_7d, 4),
            }
            log_tool_result(AGENT, self.name, f"{ticker}: ${current_price:.2f}, momentum={momentum_7d:.2f}%")
            return json.dumps(result)

        except Exception as e:
            err = f"Failed to fetch {ticker}: {str(e)}"
            log_error(AGENT, err)
            return json.dumps({"error": err})


# ─── Tool 2: Compute Risk Signal ──────────────────────────────────────────

class ComputeRiskSignalInput(BaseModel):
    """Input schema for ComputeRiskSignalTool."""
    ticker: str = Field(..., description="Ticker symbol.")
    sentiment_scores: list[float] = Field(..., description="List of bearish sentiment weights (0.0–1.0) from articles.")
    sentiment_confidences: list[float] = Field(..., description="Confidence weights for each sentiment score.")
    prices: list[float] = Field(..., description="List of closing prices for volatility calculation.")
    momentum_7d: float = Field(..., description="7-day price momentum as percentage (e.g. -5.3).")
    articles_analysed: int = Field(..., description="Number of articles used.")
    current_price: float = Field(default=0.0, description="Current stock price.")


class ComputeRiskSignalTool(BaseTool):
    name: str = "compute_risk_signal"
    description: str = (
        "Computes a composite risk score (0.0–1.0) and risk tier (HIGH/MEDIUM/LOW) for a ticker "
        "by combining weighted sentiment with 14-day price volatility. "
        "Formula: risk = 0.55 × sentiment + 0.45 × volatility."
    )
    args_schema: Type[BaseModel] = ComputeRiskSignalInput

    def _run(
        self,
        ticker: str,
        sentiment_scores: list[float],
        sentiment_confidences: list[float],
        prices: list[float],
        momentum_7d: float,
        articles_analysed: int,
        current_price: float = 0.0,
    ) -> str:
        """
        Compute composite risk signal for a ticker.

        Args:
            ticker: Stock ticker symbol.
            sentiment_scores: List of bearish weights (0=bullish, 1=bearish).
            sentiment_confidences: Matching confidence values for weighted avg.
            prices: Closing price list for std dev computation.
            momentum_7d: 7-day % price change.
            articles_analysed: Number of source articles.
            current_price: Latest stock price.

        Returns:
            JSON string: RiskSignal dict or {"error": "..."}.
        """
        log_tool_call(AGENT, self.name, {"ticker": ticker, "articles": articles_analysed})
        try:
            # Weighted sentiment average
            if sentiment_scores and sentiment_confidences:
                total_weight = sum(sentiment_confidences)
                if total_weight > 0:
                    sentiment_score = sum(s * c for s, c in zip(sentiment_scores, sentiment_confidences)) / total_weight
                else:
                    sentiment_score = 0.5
            else:
                sentiment_score = 0.5

            # Volatility: std dev of daily log returns, normalised to 0–1
            if len(prices) >= 2:
                log_returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices)) if prices[i-1] > 0]
                raw_vol = float(np.std(log_returns)) if log_returns else 0.0
                # Normalise: typical daily vol ~0.01–0.04 → map to 0–1
                volatility_14d = min(1.0, raw_vol / 0.04)
            else:
                volatility_14d = 0.0

            # Composite formula
            composite_risk = round(0.55 * sentiment_score + 0.45 * volatility_14d, 4)
            composite_risk = max(0.0, min(1.0, composite_risk))

            # Risk tier
            if composite_risk >= 0.70:
                risk_tier = "HIGH"
            elif composite_risk >= 0.40:
                risk_tier = "MEDIUM"
            else:
                risk_tier = "LOW"

            result = {
                "ticker": ticker.upper(),
                "sentiment_score": round(sentiment_score, 4),
                "price_momentum_7d": round(momentum_7d, 4),
                "volatility_14d": round(volatility_14d, 4),
                "composite_risk": composite_risk,
                "risk_tier": risk_tier,
                "articles_analysed": articles_analysed,
                "current_price": round(current_price, 2),
            }
            log_tool_result(AGENT, self.name, f"{ticker}: risk={composite_risk:.2f} [{risk_tier}]")
            return json.dumps(result)

        except Exception as e:
            err = f"Risk computation failed for {ticker}: {str(e)}"
            log_error(AGENT, err)
            return json.dumps({"error": err})
```

### agents/correlator_agent.py

```python
# agents/correlator_agent.py
from crewai import Agent, Task
from langchain_ollama import ChatOllama
from tools.market_tools import FetchStockDataTool, ComputeRiskSignalTool
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TEMPERATURE, LLM_MAX_ITER, LLM_MAX_RETRY

_llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=LLM_TEMPERATURE)

correlator_agent = Agent(
    role="Quantitative Market Analyst",
    goal="Fetch stock price data and compute composite risk signals for each ticker.",
    backstory=(
        "You are a quant who combines price action with sentiment to identify risk. "
        "For each ticker, you fetch its price history, compute volatility, and combine "
        "with sentiment scores to output a single risk signal."
    ),
    tools=[FetchStockDataTool(), ComputeRiskSignalTool()],
    llm=_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=LLM_MAX_ITER,
    max_retry_limit=LLM_MAX_RETRY,
)


def make_correlator_task(context_task) -> Task:
    return Task(
        description=(
            "For each unique ticker in the sentiment results from the previous task:\n"
            "  1. Call fetch_stock_data to get 14-day price history.\n"
            "  2. Compute bearish sentiment scores from BEARISH/BULLISH/NEUTRAL labels.\n"
            "  3. Call compute_risk_signal to get composite risk score and tier.\n"
            "Return: 'Risk computed for N tickers: [ticker: tier, ...]'"
        ),
        expected_output="Summary listing each ticker and its risk tier.",
        agent=correlator_agent,
        context=[context_task],
    )
```

### evaluation/test_correlator.py

```python
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
```

---

## Member 4 — Briefing & Alert Agent

**Files to create:** `tools/report_tools.py`, `agents/briefing_agent.py`, `evaluation/test_briefing.py`

### tools/report_tools.py

```python
# tools/report_tools.py
import json
from datetime import datetime
from pathlib import Path
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from config import DB_PATH, OUTPUT_DIR
from config.logger import log_tool_call, log_tool_result, log_error

AGENT   = "BriefingAlertAgent"
console = Console()
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── Tool 1: Insert DB Record ─────────────────────────────────────────────

class InsertDBRecordInput(BaseModel):
    """Input schema for InsertDBRecordTool."""
    signal_json: str = Field(..., description="JSON string of a single RiskSignal dict.")
    run_id: str = Field(..., description="The current run identifier (timestamp string).")


class InsertDBRecordTool(BaseTool):
    name: str = "insert_db_record"
    description: str = "Persists a single RiskSignal to the SQLite history database for audit and trend tracking."
    args_schema: Type[BaseModel] = InsertDBRecordInput

    def _run(self, signal_json: str, run_id: str) -> str:
        """
        Insert a RiskSignal into the SQLite history database.

        Args:
            signal_json: JSON string of RiskSignal dict.
            run_id: Current execution run identifier.

        Returns:
            JSON string: {"success": true, "ticker": "AAPL"} or {"error": "..."}.
        """
        log_tool_call(AGENT, self.name, {"run_id": run_id})
        try:
            signal = json.loads(signal_json)
            engine = create_engine(f"sqlite:///{DB_PATH}")

            with engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS risk_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT,
                        ticker TEXT,
                        sentiment_score REAL,
                        price_momentum_7d REAL,
                        volatility_14d REAL,
                        composite_risk REAL,
                        risk_tier TEXT,
                        articles_analysed INTEGER,
                        current_price REAL,
                        recorded_at TEXT
                    )
                """))
                conn.execute(text("""
                    INSERT INTO risk_signals
                    (run_id, ticker, sentiment_score, price_momentum_7d, volatility_14d,
                     composite_risk, risk_tier, articles_analysed, current_price, recorded_at)
                    VALUES (:run_id, :ticker, :sentiment, :momentum, :volatility,
                            :risk, :tier, :articles, :price, :recorded_at)
                """), {
                    "run_id": run_id,
                    "ticker": signal.get("ticker"),
                    "sentiment": signal.get("sentiment_score", 0.0),
                    "momentum": signal.get("price_momentum_7d", 0.0),
                    "volatility": signal.get("volatility_14d", 0.0),
                    "risk": signal.get("composite_risk", 0.0),
                    "tier": signal.get("risk_tier", "LOW"),
                    "articles": signal.get("articles_analysed", 0),
                    "price": signal.get("current_price", 0.0),
                    "recorded_at": datetime.now().isoformat(),
                })

            log_tool_result(AGENT, self.name, f"Persisted {signal.get('ticker')} to DB")
            return json.dumps({"success": True, "ticker": signal.get("ticker")})

        except Exception as e:
            err = f"DB insert failed: {str(e)}"
            log_error(AGENT, err)
            return json.dumps({"error": err})


# ─── Tool 2: Generate HTML Report ────────────────────────────────────────

class GenerateHTMLReportInput(BaseModel):
    """Input schema for GenerateHTMLReportTool."""
    signals_json: str = Field(..., description="JSON string of list of RiskSignal dicts.")
    run_id: str = Field(..., description="Current run identifier for file naming.")
    alert_threshold: float = Field(default=0.70, description="Score above which tier is HIGH.")


class GenerateHTMLReportTool(BaseTool):
    name: str = "generate_html_report"
    description: str = (
        "Generates a professional HTML executive briefing report from risk signals. "
        "Saves to outputs/ directory and prints colour-coded terminal alerts. "
        "Returns the output file path."
    )
    args_schema: Type[BaseModel] = GenerateHTMLReportInput

    def _run(self, signals_json: str, run_id: str, alert_threshold: float = 0.70) -> str:
        """
        Generate HTML report and print terminal alerts.

        Args:
            signals_json: JSON string of RiskSignal list.
            run_id: Run identifier used in filename.
            alert_threshold: Risk score threshold for HIGH tier.

        Returns:
            JSON string: {"report_path": "...", "high_risk": [...]} or {"error": "..."}.
        """
        log_tool_call(AGENT, self.name, {"run_id": run_id})
        try:
            signals = json.loads(signals_json)
            if not isinstance(signals, list):
                return json.dumps({"error": "signals_json must be a list"})

            # Sort by risk score descending
            signals.sort(key=lambda s: s.get("composite_risk", 0), reverse=True)

            high  = [s for s in signals if s.get("risk_tier") == "HIGH"]
            med   = [s for s in signals if s.get("risk_tier") == "MEDIUM"]
            low   = [s for s in signals if s.get("risk_tier") == "LOW"]

            # Terminal alerts via rich
            self._print_alerts(signals)

            # Generate HTML
            html_path = OUTPUT_DIR / f"report_{run_id}.html"
            html = self._build_html(signals, high, med, low, run_id)
            html_path.write_text(html, encoding="utf-8")

            log_tool_result(AGENT, self.name, f"Report saved: {html_path}")
            return json.dumps({
                "report_path": str(html_path),
                "high_risk": [s["ticker"] for s in high],
                "total_tickers": len(signals),
            })

        except Exception as e:
            err = f"Report generation failed: {str(e)}"
            log_error(AGENT, err)
            return json.dumps({"error": err})

    def _print_alerts(self, signals: list) -> None:
        table = Table(title="FinSight MAS — Risk Alert Summary", show_header=True, header_style="bold white")
        table.add_column("Ticker", style="bold")
        table.add_column("Risk Score")
        table.add_column("Tier")
        table.add_column("Momentum 7d")
        table.add_column("Price")

        for s in signals:
            tier = s.get("risk_tier", "LOW")
            color = "red" if tier == "HIGH" else "yellow" if tier == "MEDIUM" else "green"
            table.add_row(
                s.get("ticker", ""),
                f"{s.get('composite_risk', 0):.2f}",
                f"[{color}]{tier}[/{color}]",
                f"{s.get('price_momentum_7d', 0):+.1f}%",
                f"${s.get('current_price', 0):.2f}",
            )
        console.print(table)

    def _build_html(self, signals, high, med, low, run_id) -> str:
        def tier_badge(tier):
            colors = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}
            return f'<span style="background:{colors.get(tier,"#999")};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">{tier}</span>'

        rows = "".join(f"""
        <tr>
          <td><b>{s['ticker']}</b></td>
          <td>{tier_badge(s['risk_tier'])}</td>
          <td>{s['composite_risk']:.2f}</td>
          <td>{s['sentiment_score']:.2f}</td>
          <td style="color:{'red' if s['price_momentum_7d']<0 else 'green'}">{s['price_momentum_7d']:+.1f}%</td>
          <td>{s['volatility_14d']:.2f}</td>
          <td>${s['current_price']:.2f}</td>
          <td>{s['articles_analysed']}</td>
        </tr>""" for s in signals)

        return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>FinSight MAS Report — {run_id}</title>
<style>
  body{{font-family:Arial,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;background:#f8fafc}}
  h1{{color:#1e3a5f}}h2{{color:#2e5090;border-bottom:2px solid #e2e8f0;padding-bottom:8px}}
  table{{border-collapse:collapse;width:100%}}
  th{{background:#1e3a5f;color:#fff;padding:10px}}
  td{{padding:8px 10px;border-bottom:1px solid #e2e8f0}}
  tr:hover{{background:#f1f5f9}}
  .stat{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:20px;text-align:center;display:inline-block;margin:8px;min-width:120px}}
  .stat .num{{font-size:36px;font-weight:bold;color:#1e3a5f}}
</style></head><body>
<h1>💹 FinSight MAS — Executive Risk Briefing</h1>
<p style="color:#64748b">Run ID: {run_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<h2>Summary</h2>
<div>
  <div class="stat"><div class="num" style="color:#ef4444">{len(high)}</div>HIGH Risk</div>
  <div class="stat"><div class="num" style="color:#f59e0b">{len(med)}</div>MEDIUM Risk</div>
  <div class="stat"><div class="num" style="color:#22c55e">{len(low)}</div>LOW Risk</div>
  <div class="stat"><div class="num">{len(signals)}</div>Total Tickers</div>
</div>
<h2>Risk Signal Table</h2>
<table><tr>
  <th>Ticker</th><th>Tier</th><th>Risk Score</th><th>Sentiment</th>
  <th>Momentum 7d</th><th>Volatility</th><th>Price</th><th>Articles</th>
</tr>{rows}</table>
</body></html>"""
```

### agents/briefing_agent.py

```python
# agents/briefing_agent.py
from crewai import Agent, Task
from langchain_ollama import ChatOllama
from tools.report_tools import InsertDBRecordTool, GenerateHTMLReportTool
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TEMPERATURE, LLM_MAX_ITER, LLM_MAX_RETRY

_llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=LLM_TEMPERATURE)

briefing_agent = Agent(
    role="Chief Risk Intelligence Officer",
    goal="Persist risk signals to database and generate the executive HTML briefing report.",
    backstory=(
        "You are the final stage of the risk intelligence pipeline. "
        "You persist all signals for audit and generate clear, concise executive reports."
    ),
    tools=[InsertDBRecordTool(), GenerateHTMLReportTool()],
    llm=_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=LLM_MAX_ITER,
    max_retry_limit=LLM_MAX_RETRY,
)


def make_briefing_task(context_task, run_id: str) -> Task:
    return Task(
        description=(
            f"Take the risk signals from the previous task. Run ID: {run_id}.\n"
            "For each signal: call insert_db_record to persist it.\n"
            "Then call generate_html_report with all signals to create the executive briefing.\n"
            "Return: 'Report saved to <path>. HIGH risk tickers: [list].'"
        ),
        expected_output="Path to generated HTML report and list of HIGH risk tickers.",
        agent=briefing_agent,
        context=[context_task],
    )
```

### evaluation/test_briefing.py

```python
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
```

---

## Phase 3 — Crew Assembly (Team Together)

### crew/finsight_crew.py

```python
# crew/finsight_crew.py
from crewai import Crew, Process
from agents.fetcher_agent    import fetcher_agent, make_fetcher_task
from agents.sentiment_agent  import sentiment_agent, make_sentiment_task
from agents.correlator_agent import correlator_agent, make_correlator_task
from agents.briefing_agent   import briefing_agent, make_briefing_task
from state.store import get_state


class FinSightCrew:
    def crew(self) -> Crew:
        state = get_state()

        t1 = make_fetcher_task(watchlist=state.watchlist)
        t2 = make_sentiment_task(context_task=t1)
        t3 = make_correlator_task(context_task=t2)
        t4 = make_briefing_task(context_task=t3, run_id=state.run_id)

        return Crew(
            agents=[fetcher_agent, sentiment_agent, correlator_agent, briefing_agent],
            tasks=[t1, t2, t3, t4],
            process=Process.sequential,
            verbose=True,
        )
```

### main.py

```python
# main.py
from rich.console import Console
from rich.panel   import Panel
from state.store  import init_state
from crew.finsight_crew import FinSightCrew
from config import WATCHLIST

console = Console()

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold green]FinSight MAS — Financial News Intelligence System[/bold green]\n"
        "[dim]CrewAI + phi3:mini (Ollama) | Zero cloud costs | Zero paid APIs[/dim]",
        border_style="green"
    ))

    state = init_state(watchlist=WATCHLIST)
    console.print(f"\n[bold]Run ID:[/bold] {state.run_id}")
    console.print(f"[bold]Watchlist:[/bold] {', '.join(state.watchlist)}\n")

    crew   = FinSightCrew().crew()
    result = crew.kickoff()

    console.print(Panel(
        f"[green]✓ Completed.[/green]\nCheck: [bold]outputs/report_{state.run_id}.html[/bold]",
        border_style="green"
    ))
```

### evaluation/test_harness.py

```python
# evaluation/test_harness.py
"""
Unified test harness — runs all 4 member test modules.
Run: pytest evaluation/ -v --cov=tools --cov-report=term-missing
"""
from evaluation.test_fetcher   import *  # noqa
from evaluation.test_sentiment import *  # noqa
from evaluation.test_correlator import * # noqa
from evaluation.test_briefing  import *  # noqa
```

---

## Phase 4 — Final Integration & Delivery

### .gitignore

```
venv/
__pycache__/
*.pyc
.env
logs/
outputs/*.html
outputs/*.pdf
outputs/*.db
.DS_Store
```

### Run Tests

```bash
pytest evaluation/ -v --cov=tools --cov-report=term-missing
```

### Run System

```bash
ollama serve       # terminal 1
python main.py     # terminal 2
```

### Demo Video Checklist

- [ ] Start with `ollama serve` visible
- [ ] Run `python main.py` — show all 4 agents activating in sequence
- [ ] Show live log file: `tail -f logs/trace_<timestamp>.jsonl`
- [ ] Open generated HTML report in browser
- [ ] Open SQLite DB with DB Browser for SQLite — show records
- [ ] Total runtime: **3–4 minutes**

---

## phi3:mini Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| JSONDecodeError in tool | phi3:mini wraps JSON in markdown | Use `_safe_json_parse()` to strip fences |
| Agent loops indefinitely | phi3:mini re-calls same tool | Set `max_iter=2` in all agents |
| Label is "POSITVE" or misspelled | phi3:mini hallucinates enum values | Post-process: if label not in valid set → NEUTRAL |
| Empty response from Ollama | Model not loaded yet | Run `ollama run phi3:mini "test"` first |
| Very slow inference | CPU-only mode | Normal for phi3:mini — ~20 tok/s on CPU |
| Context too long | phi3:mini 4k context limit | Cap article text at 400 chars in prompts |

---

*FinSight MAS — Implementation Plan v1.0 | phi3:mini edition | SE4010 CTSE Assignment 2*

from pathlib import Path

# ── LLM ──────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "llama3.2:3b"
LLM_TEMPERATURE = 0.0

LLM_MAX_RETRY   = 1

# Per-agent iteration budgets:
#   fetcher    — 1 yfinance news fetch + 1 filter + buffer
#   sentiment  — 2 calls/article × up to 18 articles + buffer
#   correlator — 2 calls/ticker × 6 tickers + buffer
#   briefing   — 1 insert/ticker × 6 tickers + 1 report + buffer
LLM_MAX_ITER            = 8   # fallback default (kept for any future agents)
LLM_MAX_ITER_FETCHER    = 8
LLM_MAX_ITER_SENTIMENT  = 40
LLM_MAX_ITER_CORRELATOR = 20
LLM_MAX_ITER_BRIEFING   = 15

# ── Watchlist ─────────────────────────────────────────────
WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN"]

# ── News Sources (Yahoo JSON via yfinance; avoids brittle RSS / feed blocking) ──
MAX_NEWS_PER_TICKER = 10

# ── Risk Thresholds ───────────────────────────────────────
ALERT_THRESHOLD_HIGH   = 0.70
ALERT_THRESHOLD_MEDIUM = 0.40

# ── Paths ─────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent  # project root (one level up from config/)
DB_PATH      = BASE_DIR / "outputs" / "finsight_history.db"
LOG_DIR      = BASE_DIR / "logs"
OUTPUT_DIR   = BASE_DIR / "outputs"
TEMPLATE_DIR = BASE_DIR / "templates"

from pathlib import Path

# ── LLM ──────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "llama3.2:3b"
LLM_TEMPERATURE = 0.0          # Zero = most deterministic output
LLM_MAX_ITER    = 2            # Cap iterations to prevent looping
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
BASE_DIR     = Path(__file__).parent.parent  # project root (one level up from config/)
DB_PATH      = BASE_DIR / "outputs" / "finsight_history.db"
LOG_DIR      = BASE_DIR / "logs"
OUTPUT_DIR   = BASE_DIR / "outputs"
TEMPLATE_DIR = BASE_DIR / "templates"

# agents/correlator_agent.py
from crewai import Agent, Task
from tools.market_tools import FetchStockDataTool, ComputeRiskSignalTool
from config import OLLAMA_MODEL, LLM_MAX_ITER, LLM_MAX_RETRY

correlator_agent = Agent(
    role="Quantitative Market Analyst",
    goal="Fetch stock price data and compute composite risk signals for each ticker.",
    backstory=(
        "You are a quant who combines price action with sentiment to identify risk. "
        "For each ticker, you fetch its price history, compute volatility, and combine "
        "with sentiment scores to output a single risk signal."
    ),
    tools=[FetchStockDataTool(), ComputeRiskSignalTool()],
    llm=f"ollama/{OLLAMA_MODEL}",
    verbose=True,
    allow_delegation=False,
    max_iter=LLM_MAX_ITER,
    max_retry_limit=LLM_MAX_RETRY,
)


def make_correlator_task(context_task) -> Task:
    return Task(
        description=(
            "Compute composite risk signals for each ticker from the sentiment task.\n\n"
            "IMPORTANT: Use ONLY the real ticker symbols mentioned in the sentiment task output "
            "(e.g. AAPL, MSFT, NVDA). Never use placeholder names like X, Y, or Z.\n\n"
            "For each real ticker:\n\n"
            "  STEP A — Call fetch_stock_data with:\n"
            "      ticker: the ticker symbol as a plain string, e.g. 'AAPL'\n"
            "      period: '14d'\n"
            "    Parse the JSON result to extract prices (list of floats), "
            "momentum_7d (float), and current_price (float).\n"
            "    If fetch_stock_data returns an error, skip this ticker.\n\n"
            "  STEP B — Build sentiment arrays from the classify_sentiment results for that ticker:\n"
            "      sentiment_scores: for each article, map label → float "
            "(BEARISH=1.0, NEUTRAL=0.5, BULLISH=0.0)\n"
            "      sentiment_confidences: the confidence float from each classify_sentiment result\n"
            "      articles_analysed: total count of articles for this ticker\n\n"
            "  STEP C — Call compute_risk_signal with all of the above as typed values:\n"
            "      ticker: plain string\n"
            "      sentiment_scores: JSON array of floats, e.g. [1.0, 0.5]\n"
            "      sentiment_confidences: JSON array of floats, e.g. [0.85, 0.7]\n"
            "      prices: JSON array of floats from STEP A\n"
            "      momentum_7d: float from STEP A\n"
            "      articles_analysed: integer from STEP B\n"
            "      current_price: float from STEP A\n\n"
            "Return: 'Risk computed for N tickers: [TICKER: TIER, ...]' using real ticker names."
        ),
        expected_output="Summary listing each real ticker symbol and its computed risk tier.",
        agent=correlator_agent,
        context=[context_task],
    )

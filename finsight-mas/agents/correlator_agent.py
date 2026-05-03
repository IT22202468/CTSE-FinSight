# agents/correlator_agent.py
from crewai import Agent, Task
from tools.market_tools import FetchStockDataTool, ComputeRiskSignalTool
from config import OLLAMA_MODEL, LLM_MAX_ITER_CORRELATOR, LLM_MAX_RETRY

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
    max_iter=LLM_MAX_ITER_CORRELATOR,
    max_retry_limit=LLM_MAX_RETRY,
)


def make_correlator_task(context_task) -> Task:
    return Task(
        description=(
            "Compute composite risk signals for each ticker from the sentiment task.\n\n"
            "IMPORTANT: Use ONLY the real ticker symbols from the sentiment task "
            "(e.g. AAPL, MSFT, NVDA). Never use placeholder names like X, Y, or Z.\n\n"
            "For each real ticker, execute these two steps IN ORDER and DO NOT call "
            "compute_risk_signal until you have the fetch_stock_data result in hand:\n\n"
            "  STEP A — Call fetch_stock_data(ticker='AAPL', period='14d').\n"
            "    The tool returns a JSON string. Parse it and note these three values:\n"
            "      prices_from_fetch    = the 'prices' array (list of floats)\n"
            "      momentum_from_fetch  = the 'momentum_7d' float\n"
            "      price_from_fetch     = the 'current_price' float\n"
            "    If the result contains 'error', skip this ticker entirely.\n\n"
            "  STEP B — Call compute_risk_signal with ALL of these arguments together "
            "in a single call (never split across multiple calls):\n"
            "      ticker:                 plain string, e.g. 'AAPL'\n"
            "      sentiment_scores:       array of floats mapped from labels "
            "(BEARISH=1.0, NEUTRAL=0.5, BULLISH=0.0)\n"
            "      sentiment_confidences:  array of confidence floats from classify_sentiment\n"
            "      prices:                 prices_from_fetch  ← from STEP A\n"
            "      momentum_7d:            momentum_from_fetch ← from STEP A\n"
            "      current_price:          price_from_fetch   ← from STEP A\n"
            "      articles_analysed:      integer count of articles for this ticker\n\n"
            "    Example of a CORRECT call:\n"
            "      compute_risk_signal(\n"
            "        ticker='AAPL',\n"
            "        sentiment_scores=[1.0, 0.5],\n"
            "        sentiment_confidences=[0.85, 0.7],\n"
            "        prices=[180.1, 181.3, 179.8],\n"
            "        momentum_7d=-1.2,\n"
            "        current_price=179.8,\n"
            "        articles_analysed=2\n"
            "      )\n\n"
            "Return: 'Risk computed for N tickers: [TICKER: TIER, ...]' using real ticker names."
        ),
        expected_output="Summary listing each real ticker symbol and its computed risk tier.",
        agent=correlator_agent,
        context=[context_task],
    )

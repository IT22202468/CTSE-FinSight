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

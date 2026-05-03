# agents/sentiment_agent.py
from crewai import Agent, Task
from tools.sentiment_tools import ClassifySentimentTool, ExtractFinancialEntitiesTool
from config import OLLAMA_MODEL, LLM_MAX_ITER_SENTIMENT, LLM_MAX_RETRY

sentiment_agent = Agent(
    role="Financial Sentiment Analyst",
    goal="Classify sentiment of financial articles and extract key entities using LLM tools.",
    backstory=(
        "You are a quantitative analyst who reads financial news precisely. "
        "You classify each article as BULLISH, BEARISH, or NEUTRAL with a confidence score. "
        "You never guess — ambiguous articles are NEUTRAL."
    ),
    tools=[ClassifySentimentTool(), ExtractFinancialEntitiesTool()],
    llm=f"ollama/{OLLAMA_MODEL}",
    verbose=True,
    allow_delegation=False,
    max_iter=LLM_MAX_ITER_SENTIMENT,
    max_retry_limit=LLM_MAX_RETRY,
)


def make_sentiment_task(context_task) -> Task:
    return Task(
        description=(
            "Classify the sentiment of each fetched article from the previous task.\n\n"
            "For each article in the fetched list:\n"
            "  1. Call classify_sentiment with:\n"
            "       text: article title + space + summary (plain string)\n"
            "       ticker: the first entry of tickers_mentioned as a plain string, e.g. 'NVDA'\n"
            "             (NOT a list like ['NVDA'] — must be a bare string)\n"
            "  2. Call extract_financial_entities with:\n"
            "       text: same article title + summary string\n\n"
            "IMPORTANT RULES:\n"
            "  - 'ticker' must be a plain string like 'AAPL', never a list.\n"
            "  - Only process articles that have at least one entry in tickers_mentioned.\n"
            "  - Use ONLY real tickers from tickers_mentioned — never placeholder names.\n\n"
            "After processing all articles, return EXACTLY this format (replace counts with real numbers\n"
            "and replace the ticker list with the actual tickers you processed):\n"
            "  Classified N articles: B BULLISH, R BEARISH, U NEUTRAL. "
            "Tickers processed: TICK1, TICK2, TICK3."
        ),
        expected_output=(
            "Plain-English summary with real counts and the actual ticker symbols that were classified."
        ),
        agent=sentiment_agent,
        context=[context_task],
    )

# agents/sentiment_agent.py
from crewai import Agent, Task
from tools.sentiment_tools import ClassifySentimentTool, ExtractFinancialEntitiesTool
from config import OLLAMA_MODEL, LLM_MAX_ITER, LLM_MAX_RETRY

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

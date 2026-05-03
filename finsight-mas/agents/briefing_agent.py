# agents/briefing_agent.py
from crewai import Agent, Task
from tools.report_tools import InsertDBRecordTool, GenerateHTMLReportTool
from config import OLLAMA_MODEL, LLM_MAX_ITER, LLM_MAX_RETRY

briefing_agent = Agent(
    role="Chief Risk Intelligence Officer",
    goal="Persist risk signals to database and generate the executive HTML briefing report.",
    backstory=(
        "You are the final stage of the risk intelligence pipeline. "
        "You persist all signals for audit and generate clear, concise executive reports."
    ),
    tools=[InsertDBRecordTool(), GenerateHTMLReportTool()],
    llm=f"ollama/{OLLAMA_MODEL}",
    verbose=True,
    allow_delegation=False,
    max_iter=LLM_MAX_ITER,
    max_retry_limit=LLM_MAX_RETRY,
)


def make_briefing_task(context_task, run_id: str) -> Task:
    return Task(
        description=(
            f"Persist risk signals and generate the executive HTML report. Run ID: {run_id}.\n\n"
            "STEP 1 — For EACH risk signal from the previous task, call insert_db_record with:\n"
            "  signal_json: the signal serialised as a valid JSON string using double-quote keys.\n"
            "               Each signal MUST include all these fields:\n"
            '               {"ticker":"AAPL","sentiment_score":0.6,"price_momentum_7d":-1.2,\n'
            '                "volatility_14d":0.3,"composite_risk":0.65,"risk_tier":"MEDIUM",\n'
            '                "articles_analysed":3,"current_price":182.50}\n'
            "               Use only standard double quotes. No single quotes. No trailing commas.\n"
            f'  run_id: "{run_id}"\n\n'
            "STEP 2 — Call generate_html_report with:\n"
            "  signals_json: ALL signals as a JSON array string, e.g. [{...}, {...}]\n"
            "                Every signal in the array must include 'risk_tier' — do not omit it.\n"
            f'  run_id: "{run_id}"\n\n'
            "Return: 'Report saved to <path>. HIGH risk tickers: [list].'"
        ),
        expected_output="Path to generated HTML report and list of HIGH risk tickers.",
        agent=briefing_agent,
        context=[context_task],
    )

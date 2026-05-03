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

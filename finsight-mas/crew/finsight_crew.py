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

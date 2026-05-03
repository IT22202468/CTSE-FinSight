# main.py
from rich.console import Console
from rich.panel   import Panel
from state.store  import init_state
from crew.finsight_crew import FinSightCrew
from config import WATCHLIST

console = Console()

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold green]FinSight MAS — Financial News Intelligence System[/bold green]\n"
        "[dim]CrewAI + llama3.2:3b (Ollama) | Zero cloud costs | Zero paid APIs[/dim]",
        border_style="green"
    ))

    state = init_state(watchlist=WATCHLIST)
    console.print(f"\n[bold]Run ID:[/bold] {state.run_id}")
    console.print(f"[bold]Watchlist:[/bold] {', '.join(state.watchlist)}\n")

    crew   = FinSightCrew().crew()
    result = crew.kickoff()

    console.print(Panel(
        f"[green]Completed.[/green]\nCheck: [bold]outputs/report_{state.run_id}.html[/bold]",
        border_style="green"
    ))

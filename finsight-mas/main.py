# main.py
import json
from pathlib import Path
from rich.console import Console
from rich.panel   import Panel
from state.store  import init_state
from crew.finsight_crew import FinSightCrew
from config import WATCHLIST, DB_PATH, OUTPUT_DIR

console = Console()


def _ensure_report(run_id: str) -> str | None:
    """
    Guarantee the HTML report exists for this run.
    Reads risk_signals rows from SQLite and calls GenerateHTMLReportTool directly,
    bypassing the LLM.  Called after crew.kickoff() so it acts as a safety net
    when the briefing agent exhausts its iteration budget before calling the tool.
    """
    report_path = OUTPUT_DIR / f"report_{run_id}.html"
    if report_path.exists():
        return str(report_path)          # LLM already created it — nothing to do

    try:
        from sqlalchemy import create_engine, text
        from tools.report_tools import GenerateHTMLReportTool

        engine = create_engine(f"sqlite:///{DB_PATH}")
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM risk_signals WHERE run_id = :run_id"),
                {"run_id": run_id},
            ).mappings().all()

        if not rows:
            console.print("[yellow]⚠ No risk signals in DB for this run — report skipped.[/yellow]")
            return None

        signals = [
            {
                "ticker":            r["ticker"],
                "sentiment_score":   r["sentiment_score"] or 0.0,
                "price_momentum_7d": r["price_momentum_7d"] or 0.0,
                "volatility_14d":    r["volatility_14d"] or 0.0,
                "composite_risk":    r["composite_risk"] or 0.0,
                "risk_tier":         r["risk_tier"] or "LOW",
                "articles_analysed": r["articles_analysed"] or 0,
                "current_price":     r["current_price"] or 0.0,
            }
            for r in rows
        ]

        result_json = GenerateHTMLReportTool()._run(
            signals_json=json.dumps(signals),
            run_id=run_id,
        )
        result = json.loads(result_json)
        if "error" in result:
            console.print(f"[red]Report generation failed: {result['error']}[/red]")
            return None
        return result["report_path"]

    except Exception as e:
        console.print(f"[red]Could not generate fallback report: {e}[/red]")
        return None


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

    report_path = _ensure_report(state.run_id)

    if report_path:
        console.print(Panel(
            f"[green]Completed.[/green]\nReport: [bold]{report_path}[/bold]",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[yellow]Completed but no report was generated.\n"
            "Check logs/ for details.[/yellow]",
            border_style="yellow"
        ))

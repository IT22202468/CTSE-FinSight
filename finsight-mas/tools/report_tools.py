# tools/report_tools.py
import json
from datetime import datetime
from pathlib import Path
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import create_engine, text
from rich.console import Console
from rich.table import Table
from config import DB_PATH, OUTPUT_DIR
from config.logger import log_tool_call, log_tool_result, log_error
from tools.input_guards import unwrap_json_envelope

AGENT   = "BriefingAlertAgent"
console = Console()
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── Tool 1: Insert DB Record ─────────────────────────────────────────────

class InsertDBRecordInput(BaseModel):
    """Input schema for InsertDBRecordTool."""
    signal_json: str = Field(..., description="JSON string of a single RiskSignal dict.")
    run_id: str = Field(..., description="The current run identifier (timestamp string).")

    @model_validator(mode="before")
    @classmethod
    def unwrap_json_envelope(cls, values):
        return unwrap_json_envelope(values)

    @field_validator("signal_json", mode="before")
    @classmethod
    def sanitize_signal_json(cls, v):
        """Coerce dict input directly; fix common LLM JSON serialization artefacts."""
        if isinstance(v, dict):
            return json.dumps(v)
        if not isinstance(v, str):
            return json.dumps(v)
        cleaned = v.strip()
        # Replace smart/single quotes with double quotes where safe
        cleaned = cleaned.replace("'", '"')
        # Strip trailing commas before } or ]
        import re as _re
        cleaned = _re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            return v  # return original and let _run surface the error naturally


class InsertDBRecordTool(BaseTool):
    name: str = "insert_db_record"
    description: str = "Persists a single RiskSignal to the SQLite history database for audit and trend tracking."
    args_schema: Type[BaseModel] = InsertDBRecordInput

    def _run(self, signal_json: str, run_id: str) -> str:
        """
        Insert a RiskSignal into the SQLite history database.

        Args:
            signal_json: JSON string of RiskSignal dict.
            run_id: Current execution run identifier.

        Returns:
            JSON string: {"success": true, "ticker": "AAPL"} or {"error": "..."}.
        """
        log_tool_call(AGENT, self.name, {"run_id": run_id})
        try:
            signal = json.loads(signal_json)
            engine = create_engine(f"sqlite:///{DB_PATH}")

            with engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS risk_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT,
                        ticker TEXT,
                        sentiment_score REAL,
                        price_momentum_7d REAL,
                        volatility_14d REAL,
                        composite_risk REAL,
                        risk_tier TEXT,
                        articles_analysed INTEGER,
                        current_price REAL,
                        recorded_at TEXT
                    )
                """))
                conn.execute(text("""
                    INSERT INTO risk_signals
                    (run_id, ticker, sentiment_score, price_momentum_7d, volatility_14d,
                     composite_risk, risk_tier, articles_analysed, current_price, recorded_at)
                    VALUES (:run_id, :ticker, :sentiment, :momentum, :volatility,
                            :risk, :tier, :articles, :price, :recorded_at)
                """), {
                    "run_id": run_id,
                    "ticker": signal.get("ticker"),
                    "sentiment": signal.get("sentiment_score", 0.0),
                    "momentum": signal.get("price_momentum_7d", 0.0),
                    "volatility": signal.get("volatility_14d", 0.0),
                    "risk": signal.get("composite_risk", 0.0),
                    "tier": signal.get("risk_tier", "LOW"),
                    "articles": signal.get("articles_analysed", 0),
                    "price": signal.get("current_price", 0.0),
                    "recorded_at": datetime.now().isoformat(),
                })

            log_tool_result(AGENT, self.name, f"Persisted {signal.get('ticker')} to DB")
            return json.dumps({"success": True, "ticker": signal.get("ticker")})

        except Exception as e:
            err = f"DB insert failed: {str(e)}"
            log_error(AGENT, err)
            return json.dumps({"error": err})


# ─── Tool 2: Generate HTML Report ────────────────────────────────────────

class GenerateHTMLReportInput(BaseModel):
    """Input schema for GenerateHTMLReportTool."""
    signals_json: str = Field(..., description="JSON string of list of RiskSignal dicts.")
    run_id: str = Field(..., description="Current run identifier for file naming.")
    alert_threshold: float = Field(default=0.70, description="Score above which tier is HIGH.")

    @model_validator(mode="before")
    @classmethod
    def unwrap_json_envelope(cls, values):
        return unwrap_json_envelope(values)


class GenerateHTMLReportTool(BaseTool):
    name: str = "generate_html_report"
    description: str = (
        "Generates a professional HTML executive briefing report from risk signals. "
        "Saves to outputs/ directory and prints colour-coded terminal alerts. "
        "Returns the output file path."
    )
    args_schema: Type[BaseModel] = GenerateHTMLReportInput

    def _run(self, signals_json: str, run_id: str, alert_threshold: float = 0.70) -> str:
        """
        Generate HTML report and print terminal alerts.

        Args:
            signals_json: JSON string of RiskSignal list.
            run_id: Run identifier used in filename.
            alert_threshold: Risk score threshold for HIGH tier.

        Returns:
            JSON string: {"report_path": "...", "high_risk": [...]} or {"error": "..."}.
        """
        log_tool_call(AGENT, self.name, {"run_id": run_id})
        try:
            signals = json.loads(signals_json)
            if not isinstance(signals, list):
                return json.dumps({"error": "signals_json must be a list"})

            # Sort by risk score descending
            signals.sort(key=lambda s: s.get("composite_risk", 0), reverse=True)

            high  = [s for s in signals if s.get("risk_tier") == "HIGH"]
            med   = [s for s in signals if s.get("risk_tier") == "MEDIUM"]
            low   = [s for s in signals if s.get("risk_tier") == "LOW"]

            # Terminal alerts via rich
            self._print_alerts(signals)

            # Generate HTML
            html_path = OUTPUT_DIR / f"report_{run_id}.html"
            html = self._build_html(signals, high, med, low, run_id)
            html_path.write_text(html, encoding="utf-8")

            log_tool_result(AGENT, self.name, f"Report saved: {html_path}")
            return json.dumps({
                "report_path": str(html_path),
                "high_risk": [s["ticker"] for s in high],
                "total_tickers": len(signals),
            })

        except Exception as e:
            err = f"Report generation failed: {str(e)}"
            log_error(AGENT, err)
            return json.dumps({"error": err})

    def _print_alerts(self, signals: list) -> None:
        table = Table(title="FinSight MAS — Risk Alert Summary", show_header=True, header_style="bold white")
        table.add_column("Ticker", style="bold")
        table.add_column("Risk Score")
        table.add_column("Tier")
        table.add_column("Momentum 7d")
        table.add_column("Price")

        for s in signals:
            tier = s.get("risk_tier", "LOW")
            color = "red" if tier == "HIGH" else "yellow" if tier == "MEDIUM" else "green"
            table.add_row(
                s.get("ticker", ""),
                f"{s.get('composite_risk', 0):.2f}",
                f"[{color}]{tier}[/{color}]",
                f"{s.get('price_momentum_7d', 0):+.1f}%",
                f"${s.get('current_price', 0):.2f}",
            )
        console.print(table)

    def _build_html(self, signals, high, med, low, run_id) -> str:
        def tier_badge(tier):
            colors = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}
            return f'<span style="background:{colors.get(tier,"#999")};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">{tier}</span>'

        rows = "".join(f"""
        <tr>
          <td><b>{s.get('ticker', '')}</b></td>
          <td>{tier_badge(s.get('risk_tier', 'LOW'))}</td>
          <td>{s.get('composite_risk', 0.0):.2f}</td>
          <td>{s.get('sentiment_score', 0.0):.2f}</td>
          <td style="color:{'red' if s.get('price_momentum_7d', 0) < 0 else 'green'}">{s.get('price_momentum_7d', 0.0):+.1f}%</td>
          <td>{s.get('volatility_14d', 0.0):.2f}</td>
          <td>${s.get('current_price', 0.0):.2f}</td>
          <td>{s.get('articles_analysed', 0)}</td>
        </tr>""" for s in signals)

        return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>FinSight MAS Report — {run_id}</title>
<style>
  body{{font-family:Arial,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;background:#f8fafc}}
  h1{{color:#1e3a5f}}h2{{color:#2e5090;border-bottom:2px solid #e2e8f0;padding-bottom:8px}}
  table{{border-collapse:collapse;width:100%}}
  th{{background:#1e3a5f;color:#fff;padding:10px}}
  td{{padding:8px 10px;border-bottom:1px solid #e2e8f0}}
  tr:hover{{background:#f1f5f9}}
  .stat{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:20px;text-align:center;display:inline-block;margin:8px;min-width:120px}}
  .stat .num{{font-size:36px;font-weight:bold;color:#1e3a5f}}
</style></head><body>
<h1>FinSight MAS — Executive Risk Briefing</h1>
<p style="color:#64748b">Run ID: {run_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<h2>Summary</h2>
<div>
  <div class="stat"><div class="num" style="color:#ef4444">{len(high)}</div>HIGH Risk</div>
  <div class="stat"><div class="num" style="color:#f59e0b">{len(med)}</div>MEDIUM Risk</div>
  <div class="stat"><div class="num" style="color:#22c55e">{len(low)}</div>LOW Risk</div>
  <div class="stat"><div class="num">{len(signals)}</div>Total Tickers</div>
</div>
<h2>Risk Signal Table</h2>
<table><tr>
  <th>Ticker</th><th>Tier</th><th>Risk Score</th><th>Sentiment</th>
  <th>Momentum 7d</th><th>Volatility</th><th>Price</th><th>Articles</th>
</tr>{rows}</table>
</body></html>"""

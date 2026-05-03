# config/logger.py
import json
import time
from loguru import logger
from pathlib import Path
from config import LOG_DIR

LOG_DIR.mkdir(exist_ok=True)
_run_id = time.strftime("%Y%m%d_%H%M%S")

logger.remove()
logger.add(
    LOG_DIR / f"trace_{_run_id}.jsonl",
    format="{message}",
    level="DEBUG",
    enqueue=True,
)


def _emit(event: str, agent: str, payload: dict) -> None:
    record = {"ts": time.time(), "event": event, "agent": agent, **payload}
    logger.info(json.dumps(record))


def log_agent_start(agent: str, task: str) -> None:
    _emit("AGENT_START", agent, {"task": task})

def log_tool_call(agent: str, tool: str, inputs: dict) -> None:
    _emit("TOOL_CALL", agent, {"tool": tool, "inputs": inputs})

def log_tool_result(agent: str, tool: str, summary: str) -> None:
    _emit("TOOL_RESULT", agent, {"tool": tool, "summary": summary})

def log_agent_complete(agent: str, summary: str) -> None:
    _emit("AGENT_COMPLETE", agent, {"summary": summary})

def log_error(agent: str, error: str) -> None:
    _emit("ERROR", agent, {"error": error})

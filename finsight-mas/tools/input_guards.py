# tools/input_guards.py
"""
Pre-call input validation helpers used by every tool's _run method.

Responsibilities:
  - Detect when the LLM passes a JSON-schema description instead of real content
  - Provide a uniform error-return format so _run can return early gracefully
  - Expose _unwrap_json_envelope as the single canonical copy (imported by all tools)
"""

import json
import re
from typing import Any

# Keys that are only present in JSON Schema objects, never in real tool arguments
_SCHEMA_KEYS = {"type", "description", "title", "items", "properties", "required",
                "$schema", "anyOf", "allOf", "oneOf", "enum", "default", "examples"}

_TICKER_RE = re.compile(r'^[A-Z]{1,5}$')


# ── JSON-envelope unwrap ───────────────────────────────────────────────────

def _looks_like_schema(obj: Any) -> bool:
    """Return True if obj appears to be a JSON-Schema fragment, not real data."""
    if isinstance(obj, dict):
        # A dict whose keys are mostly schema keywords is a schema
        if len(obj) > 0 and len(_SCHEMA_KEYS & set(obj.keys())) / len(obj) >= 0.5:
            return True
        # Any value that is itself a schema-like dict
        return any(_looks_like_schema(v) for v in obj.values())
    return False


def unwrap_json_envelope(values: dict) -> dict:
    """
    Detect and unwrap the LLM anti-pattern of packing all args as a JSON string
    under an arbitrary single key, e.g. {'yahoo': '{"ticker":"NVDA","period":"14d"}'}.

    Only unwraps if the parsed inner dict does NOT look like a JSON-Schema object.
    """
    if len(values) != 1:
        return values

    only_value = next(iter(values.values()))
    if not isinstance(only_value, str):
        return values

    try:
        inner = json.loads(only_value)
    except (json.JSONDecodeError, ValueError):
        return values

    if isinstance(inner, dict) and not _looks_like_schema(inner):
        return inner

    return values


# ── Content guards ─────────────────────────────────────────────────────────

def is_schema_string(value: str) -> bool:
    """Return True if a string looks like a JSON-schema fragment rather than real content."""
    stripped = value.strip()
    if not stripped.startswith("{"):
        return False
    try:
        parsed = json.loads(stripped)
        return isinstance(parsed, dict) and _looks_like_schema(parsed)
    except (json.JSONDecodeError, ValueError):
        # Even if it doesn't parse cleanly, check for schema keyword density
        hits = sum(1 for kw in _SCHEMA_KEYS if f'"{kw}"' in stripped)
        return hits >= 2


def validate_text_field(text: str, min_length: int = 5) -> str | None:
    """
    Return an error message if `text` is empty or looks like a schema description.
    Return None if the value is acceptable.
    """
    if not isinstance(text, str) or len(text.strip()) < min_length:
        return f"'text' is empty or too short (got {len(text) if isinstance(text, str) else type(text).__name__!r})"
    if is_schema_string(text):
        return ("'text' contains a JSON-schema description instead of article content. "
                "Pass the article title and summary as a plain string.")
    return None


def validate_ticker_field(ticker: str) -> str | None:
    """
    Return an error message if `ticker` is not a valid symbol string.
    Return None if the value is acceptable.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        return f"'ticker' is missing or empty (got {ticker!r})"
    t = ticker.strip().upper()
    if is_schema_string(ticker):
        return ("'ticker' contains a JSON-schema description instead of a symbol. "
                "Pass a plain ticker string like 'AAPL'.")
    if not _TICKER_RE.match(t):
        return (f"'ticker' must be 1–5 uppercase letters (got {ticker!r}). "
                "Example: 'AAPL', 'NVDA'.")
    return None


def validate_json_list_field(value: str, field_name: str) -> str | None:
    """
    Return an error message if `value` is not a JSON array string.
    Return None if the value is acceptable.
    """
    if not isinstance(value, str) or not value.strip():
        return f"'{field_name}' is empty."
    stripped = value.strip()
    if is_schema_string(stripped):
        return (f"'{field_name}' contains a JSON-schema description. "
                "Pass a real JSON array string from a previous tool result.")
    try:
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            return f"'{field_name}' must be a JSON array, got {type(parsed).__name__}."
    except json.JSONDecodeError as e:
        return f"'{field_name}' is not valid JSON: {e}"
    return None


def guard_error(field_errors: list[str]) -> str:
    """Build a standard JSON error string from a list of field error messages."""
    return json.dumps({
        "error": "invalid_arguments",
        "details": field_errors,
        "hint": "Fix the listed argument(s) and call the tool again.",
    })

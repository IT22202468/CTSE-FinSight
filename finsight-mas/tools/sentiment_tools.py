# tools/sentiment_tools.py
import json
import re
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, model_validator
from langchain_ollama import ChatOllama
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TEMPERATURE
from config.logger import log_tool_call, log_tool_result, log_error

AGENT = "SentimentAnalystAgent"


def _unwrap_json_envelope(values: dict) -> dict:
    """Unwrap the LLM anti-pattern of packing all args as a JSON string under one key."""
    if len(values) == 1:
        only_value = next(iter(values.values()))
        if isinstance(only_value, str):
            try:
                inner = json.loads(only_value)
                if isinstance(inner, dict):
                    return inner
            except (json.JSONDecodeError, ValueError):
                pass
    return values

_llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=LLM_TEMPERATURE,
)


def _safe_json_parse(raw: str) -> dict | None:
    """
    Parse JSON from LLM output which may include markdown fences.

    Args:
        raw: Raw LLM response string.

    Returns:
        Parsed dict on success, None on failure.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract first JSON object with regex
        match = re.search(r'\{[^}]+\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None


# ─── Tool 1: Classify Sentiment ──────────────────────────────────────────────

class ClassifySentimentInput(BaseModel):
    """Input schema for ClassifySentimentTool."""
    text: str = Field(..., description="The article title and summary to classify.")
    ticker: str = Field(..., description="The ticker symbol this article relates to, e.g. 'AAPL'.")

    @model_validator(mode="before")
    @classmethod
    def unwrap_json_envelope(cls, values):
        return _unwrap_json_envelope(values)

    @field_validator("ticker", mode="before")
    @classmethod
    def coerce_ticker_string(cls, v):
        """Accept a single-element list by unwrapping it."""
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("ticker list is empty")
            return str(v[0])
        return v


class ClassifySentimentTool(BaseTool):
    name: str = "classify_sentiment"
    description: str = (
        "Classifies financial news text as BULLISH, BEARISH, or NEUTRAL for a given ticker. "
        "Uses the local LLM. Returns JSON with label, confidence (0.0–1.0), and reason."
    )
    args_schema: Type[BaseModel] = ClassifySentimentInput

    def _run(self, text: str, ticker: str) -> str:
        """
        Classify sentiment of a financial news article.

        Args:
            text: Article title and summary text (max 400 chars recommended for phi3:mini).
            ticker: Ticker symbol to frame the classification context.

        Returns:
            JSON string: {"label": str, "confidence": float, "reason": str}
            or {"label": "NEUTRAL", "confidence": 0.5, "reason": "parse_error", "error": str}
        """
        log_tool_call(AGENT, self.name, {"ticker": ticker, "text_len": len(text)})

        prompt = (
            f"Classify this financial news about {ticker} stock.\n"
            f"Text: {text[:600]}\n\n"
            "Choose one: BULLISH (positive for stock price), BEARISH (negative), NEUTRAL (irrelevant).\n"
            'Respond ONLY with valid JSON:\n{"label": "BULLISH", "confidence": 0.9, "reason": "one sentence"}\n'
            "No other text. No explanation."
        )

        try:
            response = _llm.invoke(prompt)
            raw_content = getattr(response, "content", None) or ""
            if not raw_content.strip():
                log_error(AGENT, f"LLM returned empty response for ticker {ticker}")
                return json.dumps({"label": "NEUTRAL", "confidence": 0.5,
                                   "reason": "empty_llm_response", "error": "LLM returned empty output"})

            parsed = _safe_json_parse(raw_content)

            if parsed is None:
                log_error(AGENT, f"JSON parse failed for ticker {ticker}. Raw: {raw_content[:100]}")
                result = {"label": "NEUTRAL", "confidence": 0.5, "reason": "parse_error",
                          "error": "LLM returned unparseable output"}
            else:
                # Validate and clamp values
                label = parsed.get("label", "NEUTRAL").upper()
                if label not in ("BULLISH", "BEARISH", "NEUTRAL"):
                    label = "NEUTRAL"
                confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
                result = {"label": label, "confidence": confidence,
                          "reason": str(parsed.get("reason", ""))[:200]}

            log_tool_result(AGENT, self.name, f"{ticker} → {result['label']} ({result['confidence']:.2f})")
            return json.dumps(result)

        except Exception as e:
            err = f"Sentiment classification failed: {str(e)}"
            log_error(AGENT, err)
            return json.dumps({"label": "NEUTRAL", "confidence": 0.5, "reason": "exception", "error": err})


# ─── Tool 2: Extract Entities ─────────────────────────────────────────────────

class ExtractEntitiesInput(BaseModel):
    """Input schema for ExtractFinancialEntitiesTool."""
    text: str = Field(..., description="Article text to extract entities from.")

    @model_validator(mode="before")
    @classmethod
    def unwrap_json_envelope(cls, values):
        return _unwrap_json_envelope(values)


class ExtractFinancialEntitiesTool(BaseTool):
    name: str = "extract_financial_entities"
    description: str = (
        "Extracts key financial entities from article text: company names, executives, "
        "economic events, and figures. Returns JSON list of entity strings."
    )
    args_schema: Type[BaseModel] = ExtractEntitiesInput

    def _run(self, text: str) -> str:
        """
        Extract financial entities from text using phi3:mini.

        Args:
            text: Article title + summary text.

        Returns:
            JSON string: {"entities": ["Apple", "Tim Cook", "Q3 earnings"]} or error dict.
        """
        log_tool_call(AGENT, self.name, {"text_len": len(text)})

        prompt = (
            f"Extract financial entities from this text:\n{text[:500]}\n\n"
            "Find: company names, executive names, financial events, economic figures.\n"
            'Respond ONLY with valid JSON: {"entities": ["name1", "name2"]}\n'
            "No other text."
        )

        try:
            response = _llm.invoke(prompt)
            raw_content = getattr(response, "content", None) or ""
            parsed = _safe_json_parse(raw_content) if raw_content.strip() else None

            if parsed is None or "entities" not in parsed:
                result = {"entities": []}
            else:
                entities = [str(e) for e in parsed["entities"] if e][:10]  # cap at 10
                result = {"entities": entities}

            log_tool_result(AGENT, self.name, f"Extracted {len(result['entities'])} entities")
            return json.dumps(result)

        except Exception as e:
            err = f"Entity extraction failed: {str(e)}"
            log_error(AGENT, err)
            return json.dumps({"entities": [], "error": err})

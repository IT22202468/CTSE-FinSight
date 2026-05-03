# tools/sentiment_tools.py
import json
import re
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, model_validator
from langchain_ollama import ChatOllama
from tenacity import RetryError
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TEMPERATURE
from config.logger import log_tool_call, log_tool_result, log_error
from tools.retry_utils import llm_retry
from tools.input_guards import (
    unwrap_json_envelope, guard_error,
    validate_text_field, validate_ticker_field,
)

AGENT = "SentimentAnalystAgent"

_llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=LLM_TEMPERATURE,
)


def _safe_json_parse(raw: str) -> dict | None:
    cleaned = raw.strip()
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
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
    text: str = Field(default="", description="The article title and summary to classify.")
    ticker: str = Field(default="", description="The ticker symbol this article relates to, e.g. 'AAPL'.")

    @model_validator(mode="before")
    @classmethod
    def unwrap_and_validate(cls, values):
        return unwrap_json_envelope(values)

    @field_validator("ticker", mode="before")
    @classmethod
    def coerce_ticker_string(cls, v):
        if isinstance(v, list):
            if len(v) == 0:
                return ""
            return str(v[0])
        return v


class ClassifySentimentTool(BaseTool):
    name: str = "classify_sentiment"
    description: str = (
        "Classifies financial news text as BULLISH, BEARISH, or NEUTRAL for a given ticker. "
        "Uses the local LLM. Returns JSON with label, confidence (0.0–1.0), and reason. "
        "Required: text (plain article string), ticker (plain symbol string e.g. 'AAPL')."
    )
    args_schema: Type[BaseModel] = ClassifySentimentInput

    def _run(self, text: str = "", ticker: str = "") -> str:
        # ── Input guards ──────────────────────────────────────────────────
        errors = []
        err = validate_text_field(text)
        if err:
            errors.append(err)
        err = validate_ticker_field(ticker)
        if err:
            errors.append(err)
        if errors:
            msg = guard_error(errors)
            log_error(AGENT, f"classify_sentiment called with bad args: {errors}")
            return msg

        ticker = ticker.strip().upper()
        log_tool_call(AGENT, self.name, {"ticker": ticker, "text_len": len(text)})

        prompt = (
            f"Classify this financial news about {ticker} stock.\n"
            f"Text: {text[:600]}\n\n"
            "Choose one: BULLISH (positive for stock price), BEARISH (negative), NEUTRAL (irrelevant).\n"
            'Respond ONLY with valid JSON:\n{"label": "BULLISH", "confidence": 0.9, "reason": "one sentence"}\n'
            "No other text. No explanation."
        )

        @llm_retry
        def _invoke_llm():
            resp = _llm.invoke(prompt)
            content = getattr(resp, "content", None) or ""
            if not content.strip():
                raise ValueError("LLM returned empty response")
            return content

        try:
            try:
                raw_content = _invoke_llm()
            except (RetryError, Exception) as e:
                err = f"LLM call failed for ticker {ticker} after retries: {e}"
                log_error(AGENT, err)
                return json.dumps({"label": "NEUTRAL", "confidence": 0.5,
                                   "reason": "llm_unavailable", "error": str(e)})

            parsed = _safe_json_parse(raw_content)
            if parsed is None:
                log_error(AGENT, f"JSON parse failed for ticker {ticker}. Raw: {raw_content[:100]}")
                result = {"label": "NEUTRAL", "confidence": 0.5, "reason": "parse_error",
                          "error": "LLM returned unparseable output"}
            else:
                label = parsed.get("label", "NEUTRAL").upper()
                if label not in ("BULLISH", "BEARISH", "NEUTRAL"):
                    label = "NEUTRAL"
                confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
                result = {"label": label, "confidence": confidence,
                          "reason": str(parsed.get("reason", ""))[:200]}

            log_tool_result(AGENT, self.name, f"{ticker} → {result['label']} ({result['confidence']:.2f})")
            return json.dumps(result)

        except Exception as e:
            err = f"Sentiment classification failed: {e}"
            log_error(AGENT, err)
            return json.dumps({"label": "NEUTRAL", "confidence": 0.5, "reason": "exception", "error": err})


# ─── Tool 2: Extract Entities ─────────────────────────────────────────────────

class ExtractEntitiesInput(BaseModel):
    """Input schema for ExtractFinancialEntitiesTool."""
    text: str = Field(default="", description="Article text to extract entities from.")

    @model_validator(mode="before")
    @classmethod
    def unwrap_and_validate(cls, values):
        return unwrap_json_envelope(values)


class ExtractFinancialEntitiesTool(BaseTool):
    name: str = "extract_financial_entities"
    description: str = (
        "Extracts key financial entities from article text: company names, executives, "
        "economic events, and figures. Returns JSON list of entity strings. "
        "Required: text (plain article string)."
    )
    args_schema: Type[BaseModel] = ExtractEntitiesInput

    def _run(self, text: str = "") -> str:
        # ── Input guard ───────────────────────────────────────────────────
        err = validate_text_field(text)
        if err:
            log_error(AGENT, f"extract_financial_entities called with bad args: {err}")
            return json.dumps({"entities": [], "error": err,
                               "hint": "Pass the article title and summary as a plain string."})

        log_tool_call(AGENT, self.name, {"text_len": len(text)})

        prompt = (
            f"Extract financial entities from this text:\n{text[:500]}\n\n"
            "Find: company names, executive names, financial events, economic figures.\n"
            'Respond ONLY with valid JSON: {"entities": ["name1", "name2"]}\n'
            "No other text."
        )

        @llm_retry
        def _invoke_llm():
            resp = _llm.invoke(prompt)
            content = getattr(resp, "content", None) or ""
            if not content.strip():
                raise ValueError("LLM returned empty response")
            return content

        try:
            try:
                raw_content = _invoke_llm()
            except (RetryError, Exception) as e:
                log_error(AGENT, f"LLM call failed for entity extraction after retries: {e}")
                return json.dumps({"entities": [], "error": str(e)})

            parsed = _safe_json_parse(raw_content)
            if parsed is None or "entities" not in parsed:
                result = {"entities": []}
            else:
                entities = [str(e) for e in parsed["entities"] if e][:10]
                result = {"entities": entities}

            log_tool_result(AGENT, self.name, f"Extracted {len(result['entities'])} entities")
            return json.dumps(result)

        except Exception as e:
            err = f"Entity extraction failed: {e}"
            log_error(AGENT, err)
            return json.dumps({"entities": [], "error": err})

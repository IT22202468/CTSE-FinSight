# tools/retry_utils.py
import time
from functools import wraps
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)
from loguru import logger


def _log_retry(retry_state) -> None:
    """Log each retry attempt through the existing logger."""
    exc = retry_state.outcome.exception()
    logger.warning(
        f"[retry] attempt {retry_state.attempt_number} failed "
        f"({type(exc).__name__}: {exc}); retrying in "
        f"{retry_state.next_action.sleep:.1f}s …"
    )


# ── Prebuilt retry policies ────────────────────────────────────────────────

def network_retry(fn=None, *, attempts: int = 3, min_wait: float = 1.0, max_wait: float = 8.0):
    """
    Retry decorator for external HTTP / network calls.
    3 attempts, exponential backoff 1 → 2 → 4 s (capped at 8 s).
    """
    decorator = retry(
        reraise=True,
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
    )
    if fn is not None:
        return decorator(fn)
    return decorator


def llm_retry(fn=None, *, attempts: int = 3, min_wait: float = 2.0, max_wait: float = 10.0):
    """
    Retry decorator for local LLM calls.
    3 attempts, slightly longer waits to let the model settle.
    """
    decorator = retry(
        reraise=True,
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
    )
    if fn is not None:
        return decorator(fn)
    return decorator

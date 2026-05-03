# state/store.py
from state.mas_state import MASState
from config import WATCHLIST, ALERT_THRESHOLD_HIGH

_state: MASState | None = None


def init_state(watchlist: list[str] | None = None) -> MASState:
    global _state
    _state = MASState(
        watchlist=watchlist or WATCHLIST,
        alert_threshold=ALERT_THRESHOLD_HIGH,
    )
    return _state


def get_state() -> MASState:
    if _state is None:
        raise RuntimeError("State not initialised. Call init_state() first.")
    return _state


def update_state(**kwargs) -> MASState:
    """Merge keyword updates into the global state."""
    global _state
    updated = _state.model_copy(update=kwargs)
    _state = updated
    return _state

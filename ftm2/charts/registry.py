import os
import time
from typing import Optional

from ftm2.config.settings import load_env_chain

SETTINGS = load_env_chain()
THROTTLE_SEC = getattr(SETTINGS, "CHART_THROTTLE_SEC", 60)

_LAST_FP: dict[str, Optional[str]] = {}
_LAST_TS: dict[str, float] = {}


def should_render(symbol: str, fingerprint: Optional[str]) -> bool:
    """Return whether to render chart given fingerprint and throttle."""
    now = time.monotonic()
    last_ts = _LAST_TS.get(symbol, 0.0)
    if (now - last_ts) < THROTTLE_SEC:
        return False
    if fingerprint is not None and _LAST_FP.get(symbol) == fingerprint:
        return False
    _LAST_TS[symbol] = now
    _LAST_FP[symbol] = fingerprint
    return True


def render_ready(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except Exception:
        return False

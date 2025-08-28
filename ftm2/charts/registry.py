import os
import time
from typing import Optional

from ftm2.config.settings import load_env_chain

SETTINGS = load_env_chain()
THROTTLE_SEC = getattr(SETTINGS, "CHART_THROTTLE_SEC", 60)
FORCE_FIRST_RENDER = getattr(SETTINGS, "CHART_FORCE_FIRST_RENDER", False)

def reset_cache() -> None:
    """Clear cached fingerprints/throttle timestamps."""
    _LAST_FP.clear()
    _LAST_TS.clear()

if getattr(SETTINGS, "CHART_FINGERPRINT_RESET", False):
    reset_cache()


_LAST_FP: dict[str, Optional[str]] = {}
_LAST_TS: dict[str, float] = {}


def should_render(symbol: str, fingerprint: Optional[str]) -> bool:
    """Return whether to render chart given fingerprint and throttle."""
    now = time.monotonic()
    if FORCE_FIRST_RENDER and symbol not in _LAST_TS:
        _LAST_TS[symbol] = now
        _LAST_FP[symbol] = fingerprint
        return True

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


"""Chart rendering gate utilities."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Tuple


_LAST_FP: Dict[str, Tuple[str | None, float]] = {}
_FORCE_LEFT: Dict[str, int] = {}


def reset_cache() -> None:
    _LAST_FP.clear()
    _FORCE_LEFT.clear()


def compute_fingerprint(snapshot) -> str:

    """Compute a simple fingerprint for a snapshot."""
    direction = getattr(snapshot, "direction", "").upper()
    total = float(getattr(snapshot, "total_score", 0.0))
    tf_scores = getattr(snapshot, "scores", {}) or getattr(snapshot, "tf_scores", {}) or {}
    mtf_hash = hashlib.md5(str(sorted(tf_scores.items())).encode("utf-8")).hexdigest()[:4]

    last_ts = 0
    indicators = getattr(snapshot, "indicators", {}) or {}
    try:
        for df in indicators.values():
            if hasattr(df, "iloc") and len(df) > 0:
                ts = float(df.iloc[-1].get("ts", 0.0))
                if ts > last_ts:
                    last_ts = ts
    except Exception:
        pass
    payload = f"{direction}|{total:.1f}|{mtf_hash}|{int(last_ts)}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]



def should_render(cfg, snapshot) -> Tuple[bool, Dict[str, Any]]:
    """Decide whether a chart should be rendered for the snapshot."""
    sym = getattr(snapshot, "symbol", "UNK")
    if sym not in _FORCE_LEFT:
        _FORCE_LEFT[sym] = int(getattr(cfg, "CHART_FORCE_N_CYCLES", 2))

    now = time.time()
    last_fp, last_ts = _LAST_FP.get(sym, (None, 0.0))
    min_interval = int(getattr(cfg, "CHART_MIN_INTERVAL_S", 10))
    if now - last_ts < min_interval:
        return False, {"reason": "interval"}

    fp = compute_fingerprint(snapshot)
    if _FORCE_LEFT[sym] > 0:
        _FORCE_LEFT[sym] -= 1
        _LAST_FP[sym] = (fp, now)
        return True, {"reason": "force"}

    if fp == last_fp:
        return False, {"reason": "same_fp"}

    _LAST_FP[sym] = (fp, now)
    return True, {"reason": "changed"}



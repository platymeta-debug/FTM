from __future__ import annotations
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from .score import _compute_trend_state


def to_viewdict(snapshot) -> Dict[str, Any]:
    """Normalize snapshot object or dict into view-friendly dict."""
    if snapshot is None:
        return {}
    if isinstance(snapshot, dict):
        data = dict(snapshot)
    elif is_dataclass(snapshot):
        data = asdict(snapshot)
    else:
        data = snapshot.__dict__.copy()

    tf_scores = (
        data.get("tf_scores")
        or data.get("tfs")
        or data.get("scores")
        or data.get("mtf_summary")
        or {}
    )
    data["tf_scores"] = tf_scores


    # expose both weighted decision score and raw total score
    decision = float(data.get("decision_score", data.get("total_score", 0.0)))
    decision = max(min(decision, 100.0), -100.0)
    total = 0.0
    if isinstance(tf_scores, dict):
        try:
            total = sum(float(v) for v in tf_scores.values())
        except Exception:
            total = 0.0
    data["decision_score"] = decision
    data["total_score"] = total


    if not data.get("trend_state"):
        indicators = data.get("indicators", {}) or {}
        primary = None
        if isinstance(tf_scores, dict) and tf_scores:
            primary = list(tf_scores.keys())[0]
        df = indicators.get(primary) if isinstance(indicators, dict) else None
        try:
            data["trend_state"] = _compute_trend_state(df)
        except Exception:
            data["trend_state"] = "UNKNOWN"

    data.setdefault("plan", {})
    return data

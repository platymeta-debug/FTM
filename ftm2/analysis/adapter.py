from __future__ import annotations

"""Adapter converting strategy score snapshots to analysis snapshots."""

from ftm2.analysis.state import AnalysisSnapshot


def to_analysis_snapshot(sym: str, s) -> AnalysisSnapshot:
    """Convert a strategy.score.Snapshot to AnalysisSnapshot."""

    mtf_summary: dict[str, dict] = {}
    tf_scores = getattr(s, "tf_scores", {}) or {}
    contribs_map = getattr(s, "contribs", {}) or {}
    for tf, score in tf_scores.items():
        reasons = []
        for c in contribs_map.get(tf, []) or []:
            try:
                reasons.append(c.text)
            except AttributeError:
                reasons.append(c.get("text", ""))
        mtf_summary[tf] = {
            "score": float(score),
            "dir": "NEUTRAL",
            "reasons": reasons,
        }

    return AnalysisSnapshot(
        symbol=sym,
        tfs=list(tf_scores.keys()),
        ts=__import__("time").time(),
        total_score=float(getattr(s, "total_score", 0.0)),
        direction=getattr(s, "direction", "NEUTRAL"),
        confidence=float(getattr(s, "confidence", 0.0)),
        scores={tf: float(score) for tf, score in tf_scores.items()},
        mtf_summary=mtf_summary,
        rules=getattr(s, "rules", {}) or {},
        risk_tier="N/A",
        plan=getattr(s, "plan", {}) or {},
        trend_state=getattr(s, "trend_state", "N/A"),
        indicators=getattr(s, "indicators", {}) or {},
    )


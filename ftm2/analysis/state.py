from __future__ import annotations
from dataclasses import dataclass

@dataclass
class AnalysisSnapshot:
    symbol: str
    tfs: list[str]
    ts: float
    total_score: float
    direction: str
    confidence: float
    scores: dict
    mtf_summary: dict
    rules: dict
    risk_tier: str
    plan: dict
    trend_state: str
    indicators: dict

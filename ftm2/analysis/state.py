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


# [ANCHOR:SETUP_TICKET_DTO]
@dataclass
class SetupTicket:
    id: str
    symbol: str
    side: str          # LONG | SHORT
    tf: str
    score: int         # -100..+100
    entry_px: float    # 진입 기준(중심) 또는 현재가
    stop_px: float     # 무효화(손절)
    tps: list[float]   # 목표들
    rr: float          # min RR
    created_ts: float
    expire_ts: float
    reason: list[str]

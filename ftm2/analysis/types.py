from dataclasses import dataclass
from typing import List


# [SETUP_TICKET_DTO]
@dataclass
class SetupTicket:
    id: str
    symbol: str
    side: str          # "LONG" | "SHORT"
    tf: str
    score: int         # -100..+100
    entry_px: float
    stop_px: float
    tps: List[float]
    rr: float
    created_ts: float
    expire_ts: float
    reasons: List[str]
    confidence: float = 0.8
    regime: str | None = None


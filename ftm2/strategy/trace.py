from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
from time import time

@dataclass
class DecisionTrace:
    symbol: str
    ts: float = field(default_factory=time)
    direction: str = "FLAT"  # LONG/SHORT/EXIT/FLAT
    decision_score: float = 0.0
    total_score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    gates: Dict[str, Any] = field(default_factory=dict)

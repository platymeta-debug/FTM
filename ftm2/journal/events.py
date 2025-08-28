from dataclasses import dataclass, asdict
from typing import Optional, Literal
import time

EventKind = Literal[
  "ORDER_SUBMIT","FILL","SL_SET","TP_SET","SL_MOVE","TP_HIT","SL_HIT",
  "CLOSE","PNL_REALIZED","WS","GUARD_TRIP","RISK","SETUP","ERROR","INFO"
]

@dataclass
class JEvent:
    ts: float
    kind: EventKind
    symbol: str
    side: Optional[str] = None          # LONG/SHORT
    qty: Optional[float] = None
    price: Optional[float] = None
    order_id: Optional[str] = None
    ticket_id: Optional[str] = None
    message: Optional[str] = None
    entry: Optional[float] = None
    mark: Optional[float] = None
    sl: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    upnl: Optional[float] = None
    roe: Optional[float] = None
    realized: Optional[float] = None
    lev: Optional[int] = None
    mode: Optional[str] = None          # isolated/cross
    session: Optional[str] = None

    @staticmethod
    def now(kind: EventKind, **kw):
        return JEvent(ts=time.time(), kind=kind, **kw)

    def to_row(self):
        return asdict(self)

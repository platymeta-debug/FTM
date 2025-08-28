from dataclasses import dataclass

@dataclass
class PosSnap:
    qty: float
    entry_price: float
    mark_price: float
    leverage: int
    margin_mode: str  # "isolated" | "cross"
    notional: float
    margin_used: float
    upnl: float
    roe: float

def from_positionRisk(r, mark_px) -> PosSnap:
    qty = float(r.get("positionAmt", 0))
    entry = float(r.get("entryPrice", 0))
    lev = int(float(r.get("leverage", 1) or 1))
    iso = "isolated" if r.get("isolated") in (True, "true", "TRUE", "1") or r.get("isolatedWallet", "0") != "0" else "cross"
    notional = abs(qty) * mark_px
    if iso == "isolated":
        margin_used = float(r.get("isolatedWallet") or 0.0) or (notional / lev if lev else 0.0)
    else:
        margin_used = notional / lev if lev else 0.0
    upnl = (mark_px - entry) * qty
    roe = (upnl / margin_used) if margin_used > 0 else 0.0
    return PosSnap(qty, entry, mark_px, lev, iso, notional, margin_used, upnl, roe)

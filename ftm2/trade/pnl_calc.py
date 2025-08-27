# [ANCHOR:M5_PNL_CALC]
from __future__ import annotations


def upnl_usdm(entry: float, mark: float, qty: float, side: str, contract_size: float=1.0) -> float:
    if qty == 0 or entry <= 0 or mark <= 0: return 0.0
    return ((mark - entry) if side=="LONG" else (entry - mark)) * qty * contract_size



def initial_margin(entry: float, qty: float, lev: float) -> float:
    if entry <= 0 or qty == 0 or lev <= 0: return 0.0
    return abs(entry * qty) / lev


def roe_pct(upnl: float, init_margin: float) -> float:
    if init_margin <= 0: return 0.0
    return upnl / init_margin * 100.0


from dataclasses import dataclass
import time

@dataclass
class OpsSnap:
    ts: float
    autotrade: bool
    pipeline: str
    ws_mkt_ok: bool
    ws_user_ok: bool
    daily_realized: float
    loss_streak: int
    guard: str | None
    symbols: list   # [(sym, pos_dict, ticket_dict, open_orders, risk_state)]
    notes: list     # freeform

def collect(rt, cfg, market, bracket, guard=None):
    syms = []
    for sym, snap in rt.positions.items():
        sl, tps = None, []
        try:
            sl, tps = rt.bracket.current_brackets_sync(sym)  # 비동기라면 호출부에서 await
        except: pass
        tk = rt.active_ticket.get(sym)
        syms.append((
            sym,
            {
                "side": "LONG" if snap.qty>0 else ("SHORT" if snap.qty<0 else "FLAT"),
                "qty": abs(snap.qty),
                "entry": snap.entry_price,
                "mark": snap.mark_price,
                "lev": snap.leverage, "mode": snap.margin_mode,
                "upnl": snap.upnl, "roe": snap.roe,
                "sl": sl, "tps": tps
            },
            ({
                "side": tk.side, "score": getattr(tk,"score",None), "rr": getattr(tk,"rr",None),
                "ttl": max(0,int(tk.expire_ts - time.time()))
            } if tk else None),
            rt.open_orders.get(sym, 0),
            rt.risk.state.get(sym, {}) if getattr(rt, 'risk', None) else {}
        ))
    ops = OpsSnap(
        ts=time.time(),
        autotrade=bool(getattr(cfg,"AUTOTRADE_SWITCH", True)),
        pipeline=getattr(cfg,"PIPELINE_MODE","tickets"),
        ws_mkt_ok=rt.ws.get("mkt_ok",False),
        ws_user_ok=rt.ws.get("user_ok",False),
        daily_realized=getattr(rt,"daily_realized",0.0),
        loss_streak=getattr(rt,"loss_streak",0),
        guard=getattr(rt,"guard_reason",None),
        symbols=syms,
        notes=[]
    )
    return ops

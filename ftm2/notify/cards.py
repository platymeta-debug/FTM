import time
from dataclasses import dataclass
from ftm2.runtime.positions import PosSnap

@dataclass
class CardRef:
    message_id: int
    created_at: float
    last_edit_at: float

class TradeCards:
    def __init__(self, cfg, dc):
        self.cfg = cfg
        self.dc = dc
        self.cards: dict[str, CardRef] = {}
        self.prev_qty: dict[str, float] = {}
        self.prev_upnl: dict[str, float] = {}

    def render_pos_card(self, sym: str, snap: PosSnap, sl: float | None, tps: list[tuple[float, float]], prev_qty: float | None):
        side = "LONG" if snap.qty > 0 else "SHORT" if snap.qty < 0 else "FLAT"
        qty = abs(snap.qty)
        delta = qty - (prev_qty or 0.0)
        tps_txt = " / ".join(f"{tp:.2f}×{q:.6f}" for tp, q in (tps or [])) if tps else "0.00"
        sl_txt = f"{sl:.2f}" if sl else "0.00"
        lines = [
            f"**{sym} — ● {side} × {qty:.6f}** ({'격리' if snap.margin_mode=='isolated' else '교차'}x{snap.leverage})",
            f"진입/마크 {snap.entry_price:.2f} / {snap.mark_price:.2f}",
            f"실투/명목 {snap.margin_used:.2f} / {snap.notional:.2f} USDT",
            f"UPNL/ROE {snap.upnl:.2f} / {snap.roe*100:.2f}%",
            f"SL/TP  {sl_txt} / {tps_txt}",
        ]
        if abs(delta) > 1e-12:
            lines.append(f"Δ수량 {delta:+.6f}")
        return "\n".join(lines)

    async def upsert_trade_card(self, sym: str, snap: PosSnap, sl: float | None, tps: list[tuple[float, float]]):
        now = time.time()
        card = self.cards.get(sym)
        txt = self.render_pos_card(sym, snap, sl, tps, self.prev_qty.get(sym))
        if card and (now - card.last_edit_at) < (self.cfg.TRADE_CARD_EDIT_MIN_MS / 1000):
            return
        if card and (now - card.created_at) > (self.cfg.TRADE_CARD_LIFETIME_MIN * 60):
            card = None
        if card:
            await self.dc.edit(card.message_id, txt)
            card.last_edit_at = now
        else:
            mid = await self.dc.send(self.cfg.CHANNEL_TRADES, txt)
            card = CardRef(message_id=mid, created_at=now, last_edit_at=now)
            self.cards[sym] = card
        self.prev_qty[sym] = abs(snap.qty)

    async def maybe_update(self, sym: str, snap: PosSnap, sl: float | None, tps: list[tuple[float, float]]):
        change = abs((snap.upnl - self.prev_upnl.get(sym, 0.0)) / (snap.margin_used or 1))
        if change >= (self.cfg.PNL_CHANGE_BPS / 10000):
            await self.upsert_trade_card(sym, snap, sl, tps)
        self.prev_upnl[sym] = snap.upnl

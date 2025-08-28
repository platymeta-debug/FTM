import time
from dataclasses import dataclass
from ftm2.runtime.positions import PosSnap

@dataclass
class CardRef:
    message_id: int
    created_at: float

class TradeCards:
    def __init__(self, cfg, dc):
        self.cfg = cfg
        self.dc = dc
        self.cards: dict[str, CardRef] = {}
        self.prev_qty: dict[str, float] = {}

    def render_pos_card(self, sym: str, snap: PosSnap, sl: float | None, tp_list: list[tuple[float, float]]):
        side = "LONG" if snap.qty > 0 else "SHORT" if snap.qty < 0 else "FLAT"
        qty = abs(snap.qty)
        delta = qty - self.prev_qty.get(sym, 0.0)
        self.prev_qty[sym] = qty
        tps = " / ".join(f"{tp:.2f}×{q:.6f}" for tp, q in tp_list) if tp_list else "0.00"
        sls = f"{sl:.2f}" if sl else "0.00"
        lines = [
            f"**{sym} — ● {side} × {qty:.6f}** (격리x{snap.leverage})" if snap.margin_mode == "isolated" else f"**{sym} — ● {side} × {qty:.6f}** (교차x{snap.leverage})",
            f"진입가 / 마크가   {snap.entry_price:.2f} / {snap.mark_price:.2f}",
            f"U P N L / R O E   {snap.upnl:.2f} / {snap.roe*100:.2f}%",
            f"실투(마진) / 명목  {snap.margin_used:.2f} / {snap.notional:.2f} USDT",
            f"SL / TP           {sls} / {tps}",
        ]
        if abs(delta) > 1e-12:
            lines.append(f"Δ수량 {delta:+.6f}")
        return "\n".join(lines)

    async def upsert_trade_card(self, sym: str, snap: PosSnap, sl: float | None, tps: list[tuple[float, float]]):
        now = time.time()
        text = self.render_pos_card(sym, snap, sl, tps)
        card = self.cards.get(sym)
        if card and (now - card.created_at < self.cfg.CARD_MAX_EDIT_MIN * 60):
            await self.dc.edit(card.message_id, text)
        else:
            mid = await self.dc.send(self.cfg.CHANNEL_TRADES, text)
            self.cards[sym] = CardRef(message_id=mid, created_at=now)

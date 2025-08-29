import time
from dataclasses import dataclass
from ftm2.runtime.positions import PosSnap
from ftm2.notify.dispatcher import discord_safe_send

# [ANCHOR:DISCORD_TRADE_CARD]
def _safe_margin_mode(obj) -> str:
    mm = getattr(obj, "margin_mode", None)
    if mm:
        return str(mm)
    iso = getattr(obj, "isolated", None)
    return "isolated" if iso is True else "cross"

@dataclass
class CardRef:
    message_id: int
    created_at: float
    last_edit_at: float

class TradeCards:
    def __init__(self, cfg, dc, rt=None, analysis_views=None):
        self.cfg = cfg
        self.dc = dc
        self.rt = rt
        self.analysis_views = analysis_views
        self.cards: dict[str, CardRef] = {}
        self.prev_qty: dict[str, float] = {}
        self.prev_upnl: dict[str, float] = {}

    def render_pos_card(
        self,
        sym: str,
        snap: PosSnap,
        sl: float | None,
        tps: list[tuple[float, float]],
        prev_qty: float | None,
    ):
        qty_val = getattr(snap, "qty", 0.0)
        side = "LONG" if qty_val > 0 else "SHORT" if qty_val < 0 else "FLAT"
        qty = abs(qty_val)
        delta = qty - (prev_qty or 0.0)
        tps_txt = " / ".join(f"{tp:.2f}×{q:.6f}" for tp, q in (tps or [])) if tps else "0.00"
        sl_txt = f"{sl:.2f}" if sl else "0.00"
        mm = _safe_margin_mode(snap)
        lev = getattr(snap, "leverage", 1)
        mm_txt = "격리" if mm == "isolated" else "교차"
        entry_price = getattr(snap, "entry_price", 0.0)
        mark_price = getattr(snap, "mark_price", 0.0)
        margin_used = getattr(snap, "margin_used", 0.0)
        notional = getattr(snap, "notional", 0.0)
        upnl = getattr(snap, "upnl", 0.0)
        roe = getattr(snap, "roe", 0.0) * 100
        lines = [
            f"**{sym} — ● {side} × {qty:.6f}** ({mm_txt}x{lev})",
            f"진입/마크 {entry_price:.2f} / {mark_price:.2f}",
            f"실투/명목 {margin_used:.2f} / {notional:.2f} USDT",
            f"UPNL/ROE {upnl:.2f} / {roe:.2f}%",
            f"SL/TP  {sl_txt} / {tps_txt}",
        ]
        if abs(delta) > 1e-12:
            lines.append(f"Δ수량 {delta:+.6f}")
        # [ANCHOR:CARD_WHY_ONELINE]
        why = getattr(self.rt, "last_reasons", {}).get(sym) if self.rt else None
        if (not why) and self.analysis_views and hasattr(self.analysis_views, "last_ticket") and self.analysis_views.last_ticket.get(sym):
            why = (self.analysis_views.last_ticket[sym].reasons or [])[:1]
        if why:
            lines.append("Why: " + " · ".join(why if isinstance(why, list) else [why]))
        return "\n".join(lines)

    async def upsert_trade_card(
        self,
        sym: str,
        snap: PosSnap,
        sl: float | None,
        tps: list[tuple[float, float]],
        force: bool = False,
    ):
        now = time.time()
        card = self.cards.get(sym)
        txt = self.render_pos_card(sym, snap, sl, tps, self.prev_qty.get(sym))
        if card and not force and (now - card.last_edit_at) < (
            self.cfg.TRADE_CARD_EDIT_MIN_MS / 1000
        ):
            return
        if card and (now - card.created_at) > (
            self.cfg.TRADE_CARD_LIFETIME_MIN * 60
        ):
            card = None
        components = [
            [
                {"type": 2, "label": "BE", "style": 2, "custom_id": f"btn_be_{sym}"},
                {
                    "type": 2,
                    "label": "Close 50%",
                    "style": 4,
                    "custom_id": f"btn_half_{sym}",
                },
                {
                    "type": 2,
                    "label": "Flatten",
                    "style": 4,
                    "custom_id": f"btn_flat_{sym}",
                },
                {
                    "type": 2,
                    "label": "Cancel TP1",
                    "style": 2,
                    "custom_id": f"btn_ctp1_{sym}",
                },
            ]
        ]
        if card:
            await discord_safe_send(
                self.dc.edit,
                message_id=card.message_id,
                text=txt,
                components=components,
            )
            card.last_edit_at = now
        else:
            mid = await discord_safe_send(
                self.dc.send,
                channel_key_or_name=self.cfg.CHANNEL_TRADES,
                text=txt,
                components=components,
            )
            card = CardRef(message_id=mid, created_at=now, last_edit_at=now)
            self.cards[sym] = card
        self.prev_qty[sym] = abs(getattr(snap, "qty", 0.0))

    async def maybe_update(self, sym: str, snap: PosSnap, sl: float | None, tps: list[tuple[float, float]]):
        cur_upnl = getattr(snap, "upnl", 0.0)
        margin_used = getattr(snap, "margin_used", 1) or 1
        change = abs((cur_upnl - self.prev_upnl.get(sym, 0.0)) / margin_used)
        if change >= (self.cfg.PNL_CHANGE_BPS / 10000):
            await self.upsert_trade_card(sym, snap, sl, tps)
        self.prev_upnl[sym] = cur_upnl

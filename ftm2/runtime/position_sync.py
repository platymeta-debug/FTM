import time
from ftm2.runtime.positions import from_positionRisk

class PositionSync:
    def __init__(self, cfg, client, market, rt, bracket, cards, risk=None, guard=None):
        self.cfg = cfg
        self.client = client
        self.market = market
        self.rt = rt
        self.bracket = bracket
        self.cards = cards
        self.risk = risk
        self.guard = guard

    async def hydrate_all(self):
        risks = await self.client.get_position_risk(symbols=self.cfg.SYMBOLS)
        marks = self.market.get_marks()
        for r in risks:
            sym = r["symbol"]
            mpx = float(marks.get(sym) or r.get("markPrice") or r.get("entryPrice"))
            prev = self.rt.positions.get(sym)
            snap = from_positionRisk(r, mpx)
            self.rt.positions[sym] = snap
            self.rt.last_position_update = time.time()
            sl, tps = (await self.bracket.current_brackets(sym)) if self.bracket else (None, [])
            if self.bracket and self.cards:
                await self.cards.maybe_update(sym, snap, sl, tps)
            if self.guard and prev and prev.qty != 0 and abs(snap.qty) < 1e-12:
                self.guard.on_trade_close(prev.upnl)
            if self.risk and abs(snap.qty) > 1e-12:
                await self.risk.maybe_move_to_breakeven(sym, snap, snap.entry_price, sl or snap.entry_price)
                await self.risk.maybe_trail(sym, snap)
                await self.risk.maybe_timeout_close(sym, snap)

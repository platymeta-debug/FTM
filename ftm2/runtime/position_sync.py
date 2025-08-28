import time
from ftm2.runtime.positions import from_positionRisk

class PositionSync:
    def __init__(self, cfg, client, market, rt, bracket, cards):
        self.cfg = cfg
        self.client = client
        self.market = market
        self.rt = rt
        self.bracket = bracket
        self.cards = cards

    async def hydrate_all(self):
        risks = await self.client.get_position_risk(symbols=self.cfg.SYMBOLS)
        marks = self.market.get_marks()
        for r in risks:
            sym = r["symbol"]
            mpx = float(marks.get(sym) or r.get("markPrice") or r.get("entryPrice"))
            snap = from_positionRisk(r, mpx)
            self.rt.positions[sym] = snap
            self.rt.last_position_update = time.time()
            if self.bracket and self.cards:
                sl, tps = await self.bracket.current_brackets(sym)
                await self.cards.maybe_update(sym, snap, sl, tps)

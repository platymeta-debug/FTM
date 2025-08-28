import asyncio
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
        async def _req():
            params = {"symbols": ",".join(self.cfg.SYMBOLS)}
            r = self.client.trade_rest_signed("GET", "/fapi/v2/positionRisk", params)
            r.raise_for_status()
            return r.json()
        risks = await asyncio.to_thread(_req)
        marks = getattr(self.market, "get_marks", lambda: {})()
        for r in risks:
            sym = r.get("symbol")
            mpx = float(marks.get(sym) or r.get("markPrice") or r.get("entryPrice") or 0)
            snap = from_positionRisk(r, mpx)
            self.rt.positions[sym] = snap
            self.rt.last_position_update = time.time()
            if self.bracket and self.cards:
                sl, tps = await self.bracket.current_brackets(sym)
                await self.cards.upsert_trade_card(sym, snap, sl, tps)

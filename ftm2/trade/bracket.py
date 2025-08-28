from __future__ import annotations
from decimal import Decimal
from dataclasses import dataclass
import asyncio

@dataclass
class BracketPlan:
    sl_px: Decimal | None
    tps: list[tuple[Decimal, Decimal]]  # [(tp_px, qty)]
    reason: str = ""

class Bracket:
    def __init__(self, cfg, client, filters):
        self.cfg = cfg
        self.client = client
        self.filters = filters

    def _pct_plan(self, symbol: str, side: str, entry_px: Decimal, pos_qty: Decimal) -> BracketPlan:
        sl_pct = Decimal(str(self.cfg.SL_PCT))
        sl_px = entry_px * (Decimal("1") - sl_pct) if side == "LONG" else entry_px * (Decimal("1") + sl_pct)

        tp1 = Decimal(str(self.cfg.TP1_PCT))
        tp2 = Decimal(str(self.cfg.TP2_PCT))
        s1 = Decimal(str(self.cfg.TP_SPLIT))
        step = self.filters.step_size(symbol)
        q1 = (pos_qty * s1).quantize(step)
        q2 = pos_qty - q1
        tp1_px = entry_px * (Decimal("1") + tp1) if side == "LONG" else entry_px * (Decimal("1") - tp1)
        tp2_px = entry_px * (Decimal("1") + tp2) if side == "LONG" else entry_px * (Decimal("1") - tp2)
        return BracketPlan(sl_px=sl_px, tps=[(tp1_px, q1), (tp2_px, q2)], reason="percent")

    def build(self, symbol: str, side: str, entry_px: float, pos_qty: float) -> BracketPlan:
        px = Decimal(str(entry_px))
        qty = Decimal(str(pos_qty))
        if self.cfg.BRACKET_MODE == "percent":
            return self._pct_plan(symbol, side, px, qty)
        return self._pct_plan(symbol, side, px, qty)

    async def place(self, symbol: str, side: str, entry_px: float, pos_qty: float) -> BracketPlan:
        plan = self.build(symbol, side, entry_px, pos_qty)
        opp = "SELL" if side == "LONG" else "BUY"

        async def _order(**params):
            return await asyncio.to_thread(self.client.new_order, **params)

        if plan.sl_px:
            await _order(
                symbol=symbol,
                side=opp,
                type="STOP_MARKET",
                stopPrice=str(plan.sl_px),
                closePosition=True,
                reduceOnly=True,
                workingType="MARK_PRICE",
                timeInForce="GTC",
            )

        for tp_px, tp_qty in plan.tps:
            q = self.filters.q_qty(symbol, float(tp_qty))
            if q <= 0:
                continue
            await _order(
                symbol=symbol,
                side=opp,
                type="TAKE_PROFIT_MARKET",
                stopPrice=str(tp_px),
                quantity=str(q),
                reduceOnly=True,
                workingType="MARK_PRICE",
                timeInForce="GTC",
            )
        return plan

    async def current_brackets(self, symbol: str):
        async def _req():
            r = self.client.trade_rest_signed("GET", "/fapi/v1/openOrders", {"symbol": symbol})
            r.raise_for_status()
            return r.json()

        od = await asyncio.to_thread(_req)
        sl = None
        tps: list[tuple[float, float]] = []
        for o in od:
            if o.get("type") == "STOP_MARKET":
                sl = float(o.get("stopPrice", 0))
            elif o.get("type") == "TAKE_PROFIT_MARKET":
                qty = float(o.get("origQty") or o.get("cumQty") or 0)
                tps.append((float(o.get("stopPrice", 0)), qty))
        tps.sort(key=lambda x: x[0], reverse=True)
        return sl, tps

    # [ANCHOR:BRACKET_FROM_TICKET]
    async def place_from_ticket(self, symbol: str, ticket, qty: float):
        opp = "SELL" if ticket.side == "LONG" else "BUY"
        await self.client.new_order(
            symbol=symbol,
            side=opp,
            type="STOP_MARKET",
            stopPrice=str(ticket.stop_px),
            closePosition=True,
            reduceOnly=True,
            workingType="MARK_PRICE",
            timeInForce="GTC",
        )
        split = float(self.cfg.TP_SPLIT)
        tp_qty1 = self.filters.q_qty(symbol, qty * split)
        tp_qty2 = self.filters.q_qty(symbol, qty - tp_qty1)
        tps = [(ticket.tps[0], tp_qty1)]
        if len(ticket.tps) > 1 and tp_qty2 > 0:
            tps.append((ticket.tps[1], tp_qty2))
        for tp_px, q in tps:
            if q <= 0:
                continue
            await self.client.new_order(
                symbol=symbol,
                side=opp,
                type="TAKE_PROFIT_MARKET",
                stopPrice=str(tp_px),
                quantity=str(q),
                reduceOnly=True,
                workingType="MARK_PRICE",
                timeInForce="GTC",
            )
        return tps

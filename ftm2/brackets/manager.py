# [ANCHOR:SET_BRACKETS]
from __future__ import annotations
import asyncio
from typing import Literal

from ftm2.notify import dispatcher


class BracketManager:
    def __init__(self, client, cfg):
        self.client = client
        self.cfg = cfg

    async def set_brackets(
        self, sym: str, side: Literal["LONG", "SHORT"], entry_px: float, filled_qty: float, *, atr: float
    ) -> None:
        opp = "SELL" if side == "LONG" else "BUY"
        sl = entry_px - atr * self.cfg.STOP_ATR if side == "LONG" else entry_px + atr * self.cfg.STOP_ATR
        tp1 = entry_px + atr * self.cfg.TP1_ATR if side == "LONG" else entry_px - atr * self.cfg.TP1_ATR
        tp2 = entry_px + atr * self.cfg.TP2_ATR if side == "LONG" else entry_px - atr * self.cfg.TP2_ATR

        async def _order(**params):
            return await self.client.new_order(**params)

        try:
            await _order(
                symbol=sym,
                side=opp,
                type="STOP_MARKET",
                stopPrice=str(sl),
                closePosition=True,
                reduceOnly=True,
            )
            for tp in (tp1, tp2):
                await _order(
                    symbol=sym,
                    side=opp,
                    type="TAKE_PROFIT_MARKET",
                    stopPrice=str(tp),
                    quantity=str(filled_qty / 2),
                    reduceOnly=True,
                )
        except Exception as e:
            dispatcher.emit("error", f"bracket set failed for {sym}: {e}")

    # [ANCHOR:MOVE_BE]
    async def move_to_be(self, sym: str, side: Literal["LONG", "SHORT"], entry_px: float, filled_qty: float) -> None:
        opp = "SELL" if side == "LONG" else "BUY"
        try:
            await self.client.new_order(
                symbol=sym,
                side=opp,
                type="STOP_MARKET",
                stopPrice=str(entry_px),
                closePosition=True,
                reduceOnly=True,
            )
        except Exception as e:
            dispatcher.emit("error", f"move BE failed for {sym}: {e}")

    # [ANCHOR:SYNC_GUARD]
    async def sync_guard(self, sym: str) -> None:
        dispatcher.emit_once(f"bracket_sync_{sym}", "system", f"sync_guard check for {sym}")


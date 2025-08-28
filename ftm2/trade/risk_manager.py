from __future__ import annotations
import time


class RiskManager:
    def __init__(self, cfg, client, filters, notify, guardrails=None, atr_source=None):
        self.cfg = cfg
        self.client = client
        self.filters = filters
        self.notify = notify
        self.guard = guardrails
        self.atr_source = atr_source
        self.state = {}  # {sym: {"armed_be":False, "trail_anchor":None, "opened_ts":0}}

    def on_open(self, sym, entry_px):
        st = self.state.setdefault(sym, {})
        st["opened_ts"] = time.time()
        st["armed_be"] = False
        st["trail_anchor"] = entry_px

    async def maybe_move_to_breakeven(self, sym, pos, entry_px, stop_order_px):
        if not self.cfg.BE_ENABLED:
            return
        R = abs(entry_px - stop_order_px)
        if R <= 0:
            return
        mpx = pos.mark_price
        moved = (
            (pos.qty > 0 and mpx - entry_px >= self.cfg.BE_ARM_R_MULT * R)
            or (pos.qty < 0 and entry_px - mpx >= self.cfg.BE_ARM_R_MULT * R)
        )
        if moved:
            be_px = entry_px * (1 + (self.cfg.BE_FEE_BUFFER_PCT / 100.0) * (1 if pos.qty > 0 else -1))
            await self._modify_stop_market(sym, be_px)
            self.notify.emit_once(f"be_{sym}", "fill", f"💹 {sym} BE 이동: {be_px:.2f}", 60000)

    async def maybe_trail(self, sym, pos):
        if not self.cfg.TRAILING_ENABLED:
            return
        st = self.state.setdefault(sym, {})
        atr = self._atr(sym)
        mpx = pos.mark_price
        if pos.qty > 0:
            anchor = max(st.get("trail_anchor", mpx), mpx)
            new_sl = anchor - self.cfg.TRAILING_ATR * atr
            st["trail_anchor"] = anchor
        else:
            anchor = min(st.get("trail_anchor", mpx), mpx)
            new_sl = anchor + self.cfg.TRAILING_ATR * atr
            st["trail_anchor"] = anchor
        await self._modify_stop_market(sym, new_sl)

    async def maybe_timeout_close(self, sym, pos):
        if not self.cfg.TIMEOUT_ENABLED:
            return
        st = self.state.get(sym, {})
        if time.time() - st.get("opened_ts", 0) >= self.cfg.TIMEOUT_SEC:
            await self.client.new_order(
                symbol=sym,
                side=("SELL" if pos.qty > 0 else "BUY"),
                type="MARKET",
                reduceOnly=True,
            )
            if self.guard:
                self.guard.on_trade_close(pos.upnl)
            self.notify.emit("close", f"💹 {sym} 타임아웃 청산")

    async def _modify_stop_market(self, sym, new_px):
        od = await self.client.get_open_orders(symbol=sym)
        for o in od:
            if o["type"] == "STOP_MARKET" and o.get("closePosition"):
                await self.client.cancel_order(symbol=sym, orderId=o["orderId"])
                await self.client.new_order(
                    symbol=sym,
                    side=("SELL" if o["side"] == "SELL" else "BUY"),
                    type="STOP_MARKET",
                    stopPrice=str(new_px),
                    closePosition=True,
                    reduceOnly=True,
                    workingType="MARK_PRICE",
                )
                return

    def _atr(self, sym: str) -> float:
        if self.atr_source and hasattr(self.atr_source, "atr"):
            try:
                return float(self.atr_source.atr(sym, self.cfg.ENTRY_TF))
            except Exception:
                return 0.0
        return 0.0

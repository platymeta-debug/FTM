# [ANCHOR:GUARDRAILS]
from __future__ import annotations
import time

class GuardRails:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cooldowns = {}           # symbol->ts
        self.daily_loss = 0.0         # 누적 실현손익 추적은 별도에서 주입
        self.kill_switch = cfg.KILL_SWITCH_ON

    def set_realized_pnl(self, pnl: float):
        self.daily_loss = -min(0.0, pnl) * 1.0

    def cooldown_ok(self, symbol: str) -> bool:
        now = time.time()
        ts = self.cooldowns.get(symbol, 0)
        return (now - ts) >= self.cfg.COOLDOWN_S

    def arm_cooldown(self, symbol: str):
        self.cooldowns[symbol] = time.time()

    def daily_ok(self) -> bool:
        if not self.cfg.KILL_SWITCH_ON: return True
        return self.daily_loss < self.cfg.MAX_DAILY_LOSS_USDT

    def slippage_ok(self, ref_price: float, exec_price: float) -> bool:
        if ref_price<=0: return True
        bps = abs(exec_price - ref_price)/ref_price*10000
        return bps <= self.cfg.MAX_SLIPPAGE_BPS

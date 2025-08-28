from __future__ import annotations
import time
from ftm2.journal.events import JEvent


class GuardRails:
    def __init__(self, cfg, notify):
        self.cfg = cfg
        self.notify = notify
        self.reset_daily()

    def reset_daily(self):
        self.day = time.strftime("%Y-%m-%d")
        self.realized_pnl = 0.0
        self.loss_streak = 0

    def on_trade_close(self, pnl):
        self.realized_pnl += pnl
        self.loss_streak = (self.loss_streak + 1) if pnl < 0 else 0
        if self.realized_pnl <= -abs(self.cfg.DAILY_LOSS_CAP_USDT):
            self._trip("DAILY_LOSS_CAP")
        if self.loss_streak >= self.cfg.MAX_LOSS_STREAK:
            self._trip("LOSS_STREAK")

    def _trip(self, reason):
        if hasattr(self.cfg, "AUTOTRADE_SWITCH"):
            self.cfg.AUTOTRADE_SWITCH.set(False)  # type: ignore[attr-defined]
        self.notify.emit("error", f"⛔ AutoTrade OFF: {reason}")
        if getattr(self, "rt", None) and getattr(self.rt, "journal", None):
            self.rt.journal.write(JEvent.now("GUARD_TRIP", symbol="", message=reason))
            self.rt.guard_reason = reason

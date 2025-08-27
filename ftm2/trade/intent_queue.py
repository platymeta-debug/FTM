from __future__ import annotations
import asyncio, time
from ftm2.strategy.trace import DecisionTrace
from ftm2.trade.order_router import log_decision

class IntentQueue:
    """Placeholder intent queue managing trade intents based on analysis snapshots."""
    def __init__(self, cfg, divergence, router, csv, notify):
        self.cfg = cfg
        self.divergence = divergence
        self.router = router
        self.csv = csv
        self.notify = notify
        self.intents: dict[str, dict] = {}

    def on_snapshot(self, snap):
        sym = snap["symbol"] if isinstance(snap, dict) else snap.symbol
        direction = snap["direction"] if isinstance(snap, dict) else snap.direction
        score = snap["total_score"] if isinstance(snap, dict) else snap.total_score
        trace = DecisionTrace(symbol=sym, decision_score=score, total_score=score, direction=direction)
        trace.gates["ENTER_TH"] = self.cfg.ENTRY_TH
        if self.divergence and self.divergence.too_wide(sym):
            trace.reasons.append("divergence too wide")
            log_decision(trace)
            return
        if abs(score) < self.cfg.ENTRY_TH:
            trace.reasons.append("below enter threshold")
            log_decision(trace)
            return
        self.intents[sym] = {
            "state": "pending",
            "dir": direction,
            "score": score,
            "created": time.time(),
            "expire": time.time() + self.cfg.CONFIRM_TIMEOUT_S,
        }
        if self.csv:
            self.csv.log("TRADE_INTENT_NEW", symbol=sym, score=score)
        trace.reasons.append("INTENT")
        log_decision(trace)

    async def tick(self):
        while True:
            now = time.time()
            expired = [s for s, it in self.intents.items() if it["expire"] <= now]
            for s in expired:
                if self.csv:
                    self.csv.log("TRADE_INTENT_CANCELLED", symbol=s)
                self.intents.pop(s, None)
            await asyncio.sleep(1)

    async def confirm(self, symbol: str):
        if symbol in self.intents:
            if self.csv:
                self.csv.log("TRADE_INTENT_CONFIRMED", symbol=symbol)
            self.intents.pop(symbol, None)

    async def cancel(self, symbol: str, reason: str = "manual"):
        if symbol in self.intents:
            if self.csv:
                self.csv.log("TRADE_INTENT_CANCELLED", symbol=symbol, reason=reason)
            self.intents.pop(symbol, None)

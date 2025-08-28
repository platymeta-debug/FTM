from __future__ import annotations

import asyncio
from dataclasses import dataclass
from ftm2.trade.position_sizer import SizingDecision


@dataclass
class _Intent:
    side: str
    score: float
    mark: float
    sizing: SizingDecision
    gates_ok: bool = True
    autofire: bool = False


class IntentQueue:
    """Manage trade intents produced by analysis snapshots."""

    def __init__(self, cfg, divergence, router, csv, notify):
        self.cfg = cfg
        self.divergence = divergence
        self.router = router
        self.csv = csv
        self.notify = notify
        self.intents: dict[str, _Intent] = {}
        self._runner_started = False

    def _mark_from_snapshot(self, snap) -> float:
        inds = getattr(snap, "indicators", {}) or {}
        tf = snap.tfs[0] if getattr(snap, "tfs", None) else None
        if tf and tf in inds and hasattr(inds[tf], "iloc") and len(inds[tf]) > 0:
            try:
                return float(inds[tf].iloc[-1].get("close", 0.0))
            except Exception:
                return 0.0
        return 0.0

    def on_snapshot(self, snap):
        sym = getattr(snap, "symbol", "")
        direction = getattr(snap, "direction", "NEUTRAL")
        score = float(getattr(snap, "total_score", 0.0))

        # notify intent
        try:
            self.notify(f"{sym} 의도만: {direction} / {score:+.1f}")
        except Exception:
            pass

        mark = self._mark_from_snapshot(snap)
        qty = 0.0 if mark <= 0 else 0.001
        sl = mark * 0.99 if direction == "LONG" else mark * 1.01
        tp = mark * 1.02 if direction == "LONG" else mark * 0.98
        sizing = SizingDecision(
            side=direction,
            qty=qty,
            entry_type=self.cfg.ENTRY_ORDER,
            limit_offset_ticks=self.cfg.LIMIT_OFFSET_TICKS,
            sl=sl,
            tp=tp,
            reason="auto",
        )

        it = _Intent(side=direction, score=score, mark=mark, sizing=sizing)
        if self.cfg.LIVE_CONFIRM_MODE == "auto":
            it.autofire = True
        self.intents[sym] = it

    async def run(self):
        if self._runner_started:
            return
        self._runner_started = True
        while True:
            try:
                for sym, it in list(self.intents.items()):
                    if not it.gates_ok:
                        continue
                    if it.autofire:
                        ok = self.router.place_entry(sym, it.sizing, it.mark)
                        if ok:
                            try:
                                self.notify(
                                    f"{sym} 진입: {it.side} x{it.sizing.qty} @~{it.mark:.2f}"
                                )
                            except Exception:
                                pass
                            self.intents.pop(sym, None)
                await asyncio.sleep(0.2)
            except Exception as e:
                try:
                    self.notify(f"[INTENT][ERR] {e}")
                except Exception:
                    pass
                await asyncio.sleep(1.0)


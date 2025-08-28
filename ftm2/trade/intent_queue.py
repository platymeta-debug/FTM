from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from ftm2.trade.position_sizer import SizingDecision
from ftm2.notify import dispatcher


@dataclass
class _Intent:
    side: str
    score: float
    mark: float
    sizing: SizingDecision
    gates_ok: bool = True
    autofire: bool = False
    attempts: int = 0
    next_try_ts: float = 0.0


class IntentQueue:
    """Manage trade intents produced by analysis snapshots."""

    def __init__(self, cfg, divergence, router, csv, notify=dispatcher):
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
            self.notify.send_log(f"{sym} 의도만: {direction} / {score:+.1f}")
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
                    filters = self.router.filters
                    filters.use(sym)
                    for need in ("q_price", "q_qty", "min_ok", "min_qty_for"):
                        if not hasattr(filters, need):
                            raise RuntimeError(f"ExchangeFilters has no {need}")
                    if not it.gates_ok:
                        continue
                    # 백오프
                    now = time.time()
                    if it.next_try_ts and now < it.next_try_ts:
                        continue

                    # 사전 필터 검증
                    q = it.sizing.qty
                    px = it.mark
                    if not filters.min_ok(px, q):
                        if self.cfg.ORDER_SCALE_TO_MIN and self.cfg.INTENT_AUTOFIRE_SCALE_TO_MIN:
                            q_min = filters.min_qty_for(px, symbol=sym)
                            it.sizing.qty = float(q_min)
                        else:
                            self.intents.pop(sym, None)
                            try:
                                self.notify.send_once(
                                    f"intent_skip_{sym}",
                                    f"{sym} 의도 취소: 최소 명목 미달",
                                    "logs",
                                    self.cfg.NOTIFY_THROTTLE_MS,
                                )
                            except Exception:
                                pass
                            continue

                    if it.autofire:
                        ok = self.router.place_entry(sym, it.sizing, it.mark)
                        if ok:
                            try:
                                self.notify.send_trade(
                                    f"{sym} 진입: {it.side} x{it.sizing.qty} @~{it.mark:.2f}"
                                )
                            except Exception:
                                pass
                            self.intents.pop(sym, None)
                        else:
                            it.attempts += 1
                            it.next_try_ts = time.time() + (self.cfg.INTENT_BACKOFF_MS / 1000.0)
                            if it.attempts >= self.cfg.INTENT_MAX_RETRY:
                                self.intents.pop(sym, None)
                                try:
                                    self.notify.send_log(f"{sym} 의도 취소: 재시도 초과")
                                except Exception:
                                    pass
                await asyncio.sleep(0.2)
            except Exception as e:
                try:
                    self.notify.send_log(f"[INTENT][ERR] {e}")
                except Exception:
                    pass
                await asyncio.sleep(1.0)


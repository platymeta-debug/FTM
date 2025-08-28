import time


class AnalysisRunner:
    def __init__(self, cfg, scoring, engine, notify, views, rt, symbols):
        self.cfg = cfg
        self.scoring = scoring
        self.engine = engine
        self.notify = notify
        self.views = views
        self.rt = rt
        self.symbols = symbols
        self.prev_score: dict[str, int] = {}
        self.prev_has_ticket: dict[str, bool] = {}

    def _all_tf_ready(self) -> bool:
        """Placeholder for timeframe readiness check."""
        return True

    def run_cycle(self):
        for sym in self.symbols:
            score = self.scoring.mtf_score(sym)
            ticket = self.engine.build_ticket(sym, score)

            changed = (
                abs(score - self.prev_score.get(sym, 0)) >= self.cfg.ANALYSIS_SCORE_DELTA_MIN
            ) or (bool(ticket) != self.prev_has_ticket.get(sym, False))
            if changed and getattr(self.notify, "edit_ok", lambda *a, **k: True)(
                f"analysis_{sym}", self.cfg.ANALYSIS_EDIT_MIN_MS
            ):
                text = self.views.render(sym, score=score, ticket=ticket)
                upsert = getattr(self.notify, "upsert_sticky", self.notify.emit)
                upsert(
                    self.cfg.CHANNEL_SIGNALS,
                    f"analysis_{sym}",
                    text,
                    lifetime_min=self.cfg.ANALYSIS_LIFETIME_MIN,
                )
                if ticket:
                    self.notify.emit(
                        "intent",
                        f"📡 SETUP: {sym} {ticket.side} score={ticket.score} RR≈{ticket.rr:.2f} SL={ticket.stop_px:.2f} TP1={ticket.tps[0]:.2f}",
                    )

            if ticket:
                self.rt.active_ticket[sym] = ticket

            self.prev_score[sym] = score
            self.prev_has_ticket[sym] = bool(ticket)

        # [ANCHOR:ANALYSIS_READY_FLAG]
        if not self.rt.analysis_ready and self._all_tf_ready():
            self.rt.analysis_ready = True


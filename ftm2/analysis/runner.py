from ftm2.analysis.notify import AnalysisNotify

class AnalysisRunner:
    def __init__(self, cfg, scoring, engine, notify, views, rt, symbols):
        self.cfg = cfg
        self.scoring = scoring
        self.engine = engine
        self.notify = notify
        self.views = views
        self.rt = rt
        self.symbols = symbols
        self.anotify = AnalysisNotify(cfg, views, notify)

    def _all_tf_ready(self) -> bool:
        """Placeholder for timeframe readiness check."""
        return True

    def run_cycle(self):
        for sym in self.symbols:
            score, conf, regime = self.scoring.mtf_score(sym)
            trend = self.engine.trend(sym, self.cfg.ENTRY_TF)
            ticket = self.engine.build_ticket(sym, score, confidence=conf, regime=regime)
            self.anotify.upsert_analysis(sym, score, trend, ticket, confidence=conf, regime=regime)
            if ticket:
                self.notify.emit(
                    "intent",
                    f"📡 SETUP: {sym} {ticket.side} score={ticket.score} conf={conf:.2f} reg={regime} RR≈{ticket.rr:.2f} SL={ticket.stop_px:.2f} TP1={ticket.tps[0]:.2f}",
                )
                self.rt.active_ticket[sym] = ticket
        if not self.rt.analysis_ready and self._all_tf_ready():
            self.rt.analysis_ready = True

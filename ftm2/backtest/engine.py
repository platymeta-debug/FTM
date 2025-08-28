class BacktestEngine:
    """Minimal ticket-driven backtest engine."""
    def __init__(self, cfg, data, analysis, router, fees_bps=2):
        self.cfg, self.data, self.analysis, self.router = cfg, data, analysis, router
        self.fees = fees_bps / 10000.0
        self.trades = []

    def step(self, sym, t):
        # 1) feed bar to analysis -> score/ticket
        snap = self.analysis.feed_bar(sym, t, self.data[sym][t]) if hasattr(self.analysis, "feed_bar") else None
        score, conf, reg = self.analysis.scoring.mtf_score(sym)
        ticket = self.analysis.engine.build_ticket(sym, score, confidence=conf, regime=reg)
        if ticket:
            self.analysis.rt.active_ticket[sym] = ticket

        # 2) router dry-run
        self.router.dry_run = True
        try:
            self.router.try_from_ticket(sym)
        except Exception:
            pass

        # 3) bracket & pnl checks (placeholder)
        if hasattr(self, "_check_brackets_and_pnl"):
            self._check_brackets_and_pnl(sym, t)

    def run(self, start, end):
        for t in range(start, end):
            for sym in self.cfg.SYMBOLS:
                self.step(sym, t)
        if hasattr(self, "_metrics"):
            return self._metrics()
        return {}

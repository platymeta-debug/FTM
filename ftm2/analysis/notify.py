# [ANCHOR:ANALYSIS_STICKY]
class AnalysisNotify:
    def __init__(self, cfg, views, notify):
        self.cfg = cfg
        self.views = views
        self.notify = notify
        self.prev: dict[str, dict] = {}

    def _edit_ok(self, key: str, ms: int) -> bool:
        fn = getattr(self.notify, "edit_ok", None)
        if callable(fn):
            return fn(key, ms)
        return True

    def _upsert_sticky(self, ch: str, key: str, text: str, lifetime_min: int):
        upsert = getattr(self.notify, "upsert_sticky", None)
        if callable(upsert):
            upsert(ch, key, text, lifetime_min=lifetime_min)
        else:
            self.notify.emit("system", text)

    def upsert_analysis(self, sym: str, score: int, trend: str, ticket, confidence: float | None = None, regime: str | None = None):
        changed = (
            abs(score - self.prev.get(sym, {}).get("score", 0)) >= self.cfg.ANALYSIS_SCORE_DELTA_MIN
            or bool(ticket) != self.prev.get(sym, {}).get("has_ticket", False)
            or trend != self.prev.get(sym, {}).get("trend")
        )
        if (not changed) or (not self._edit_ok(f"analysis_{sym}", self.cfg.ANALYSIS_EDIT_MIN_MS)):
            return
        text = self.views.render(sym, score=score, trend=trend, ticket=ticket, confidence=confidence, regime=regime)
        self._upsert_sticky(self.cfg.CHANNEL_SIGNALS, f"analysis_{sym}", text,
                             lifetime_min=self.cfg.ANALYSIS_LIFETIME_MIN)
        self.prev[sym] = {"score": score, "trend": trend, "has_ticket": bool(ticket)}

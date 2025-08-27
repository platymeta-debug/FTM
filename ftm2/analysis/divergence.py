from __future__ import annotations

class DivergenceMonitor:
    """Track mark price divergence between live and testnet feeds."""
    def __init__(self, max_bps: int):
        self.max_bps = max_bps
        self.live: dict[str, float] = {}
        self.test: dict[str, float] = {}
        self.bps: dict[str, float] = {}

    def update_live(self, sym: str, mark: float):
        self.live[sym] = mark
        self._recalc(sym)

    def update_test(self, sym: str, mark: float):
        self.test[sym] = mark
        self._recalc(sym)

    def _recalc(self, sym: str):
        l = self.live.get(sym)
        t = self.test.get(sym)
        if l and t and l > 0:
            self.bps[sym] = abs(t - l) / l * 1e4

    def get_bps(self, sym: str) -> float:
        return float(self.bps.get(sym, 0.0))

    def too_wide(self, sym: str) -> bool:
        return self.get_bps(sym) >= self.max_bps

import time

class RuntimeState:
    """Track runtime flags and startup gating."""
    def __init__(self, cfg):
        self.cfg = cfg
        # [ANCHOR:RUNTIME_STARTUP]
        self.boot_ts = time.time()
        self.startup_until = self.boot_ts + self.cfg.STARTUP_HOLD_SEC
        self.autotrade_enabled = (self.cfg.AUTOTRADE_DEFAULT)
        self.analysis_ready = False  # 모든 TF 분석 1회 완료 후 True
        self.idem_hit: dict[tuple[str, str], float] = {}
        self.cooldown_until: dict[str, float] = {}
        self.positions: dict[str, object] = {}
        self.last_position_update: float = 0.0
        # [RUNTIME_TICKETS]
        self.active_ticket = {}     # {symbol: SetupTicket}

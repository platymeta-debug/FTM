import time
from typing import List, Tuple

def pre_trade_gates(rt, cfg, market, sym: str, reasons: List[Tuple[str, str]] | None = None):
    """Check startup and autotrade gates."""
    reasons = reasons or []
    # [ANCHOR:GATE_STARTUP]
    now = time.time()
    if now < rt.startup_until:
        reasons.append(("gate", "startup_hold"))
        return False, reasons
    if not rt.analysis_ready and cfg.STARTUP_REQUIRE_ANALYSIS:
        reasons.append(("gate", "wait_analysis"))
        return False, reasons
    if cfg.STARTUP_REQUIRE_NEW_BAR:
        ok = True
        if market and hasattr(market, "new_major_bar_after_boot"):
            ok = market.new_major_bar_after_boot(sym)
        if not ok:
            reasons.append(("gate", "wait_new_bar"))
            return False, reasons
    if not rt.autotrade_enabled:
        reasons.append(("gate", "autotrade_off"))
        return False, reasons
    return True, reasons

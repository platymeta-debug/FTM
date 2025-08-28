import time
from typing import List, Tuple


def pre_trade_gates(rt, cfg, market, sym: str, dec, reasons: List[Tuple[str, str]] | None = None):
    """Check startup and autotrade gates."""
    reasons = reasons or []
    # [ANCHOR:GATE_REQUIRE_TICKET]
    ticket = getattr(rt, "active_ticket", {}).get(sym) if hasattr(rt, "active_ticket") else None
    if not ticket:
        reasons.append(("gate", "no_setup_ticket"))
        return False, reasons
    now = time.time()
    if now > ticket.expire_ts:
        reasons.append(("gate", "ticket_expired"))
        if hasattr(rt, "active_ticket"):
            rt.active_ticket.pop(sym, None)
        return False, reasons
    if market and hasattr(market, "mark"):
        mpx = market.mark(sym)
        buf = cfg.SETUP_INVALIDATION_BUFFER_PCT / 100.0 if hasattr(cfg, "SETUP_INVALIDATION_BUFFER_PCT") else 0.0
        if (ticket.side == "LONG" and mpx <= ticket.stop_px * (1 + buf)) or \
           (ticket.side == "SHORT" and mpx >= ticket.stop_px * (1 - buf)):
            reasons.append(("gate", "ticket_invalidated"))
            if hasattr(rt, "active_ticket"):
                rt.active_ticket.pop(sym, None)
            return False, reasons

    # 동일 바/방향 중복 진입 금지
    if hasattr(rt, "idem_hit") and hasattr(market, "bar_open_ts"):
        key = (sym, dec.side)
        bar_ts = market.bar_open_ts(sym, cfg.ENTRY_TF) if market else 0
        last_hit = rt.idem_hit.get(key) if hasattr(rt, "idem_hit") else None
        if last_hit == bar_ts:
            reasons.append(("gate", "idem_same_bar"))
            return False, reasons

    # 체결 후 쿨다운
    cool_until = rt.cooldown_until.get(sym, 0) if hasattr(rt, "cooldown_until") else 0
    if time.time() < cool_until:
        reasons.append(("gate", "cooldown"))
        return False, reasons
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

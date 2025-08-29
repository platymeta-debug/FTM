import time
from typing import List, Tuple


def pre_trade_gates(rt, cfg, market, sym: str, dec, reasons: List[Tuple[str, str]] | None = None):
    """Check startup and autotrade gates."""
    reasons = reasons or []
    # [ANCHOR:ANALYSIS_READY_GATE]
    if cfg.STARTUP_REQUIRE_ANALYSIS and not getattr(rt, "analysis_ready", False):
        from ftm2.notify import dispatcher

        dispatcher.emit("gate_skip", f"{sym} analysis not ready")
        reasons.append(("gate", "analysis_not_ready"))
        return False, reasons
    # [GATE_REQUIRE_TICKET]
    tk = rt.active_ticket.get(sym)
    if not tk:
        reasons.append(("gate", "no_setup_ticket"))
        return False, reasons
    # [TICKET_VALIDITY]
    now = time.time()
    if now > tk.expire_ts:
        reasons.append(("gate", "ticket_expired"))
        rt.active_ticket.pop(sym, None)
        return False, reasons
    if market:
        mpx = market.mark(sym)
        buf = 1.0 + (cfg.SETUP_INVALIDATION_BUFFER_PCT / 100.0)
        if (tk.side == "LONG" and mpx <= tk.stop_px * buf) or (
            tk.side == "SHORT" and mpx >= tk.stop_px * (2 - buf)
        ):
            reasons.append(("gate", "ticket_invalidated"))
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

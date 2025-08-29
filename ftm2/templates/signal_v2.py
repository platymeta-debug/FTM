
"""Signal card templates (v2).

These templates render position and analysis information.
"""



def position_card(
    sym: str,
    side: str,
    qty: float,

    leverage: int,
    margin_mode: str,
    entry_px: float,
    sl: float,
    tp1: float,
    tp2: float,
) -> str:
    notional = qty * entry_px
    return (
        f"{sym} | {side} x{qty:.3f} (Lvg {leverage}x {margin_mode.lower()})\n"
        f"AvgPx {entry_px:.2f} Notional={notional:.2f}\n"
        f"SL {sl:.2f} TP1 {tp1:.2f} TP2 {tp2:.2f}"
    )


def analysis_card(**info) -> str:
    return (
        f"Score {info.get('score')} Trend {info.get('trend')} ATR {info.get('atr')}\n"
        f"RR {info.get('rr')} Invalidation {info.get('invalid')}"
    )


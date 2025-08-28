"""Simple text renderers for analysis outputs."""


def render(sym, score: int, trend: str, ticket=None, confidence: float | None = None, regime: str | None = None, reasons=None):
    lines = [f"{sym} 분석", f"Score {score} / Trend {trend}"]
    if confidence is not None or regime is not None:
        cf = confidence if confidence is not None else 0.0
        lines.append(f"Conf {cf:.2f} / Regime {regime}")
    if ticket:
        lines.append(f"Ticket: {ticket.side} RR {getattr(ticket,'rr',0):.2f}")
    if reasons:
        lines.append("Why: " + " · ".join(reasons))
    return "\n".join(lines)


def render_active(sym, side: str, entry: float, stop: float, tps: list[float], reasons=None):
    tps_txt = ", ".join(f"{tp:.2f}" for tp in tps) if tps else "—"
    lines = [
        f"{sym} {side} @ {entry:.2f}",
        f"SL {stop:.2f} / TP {tps_txt}",
    ]
    if reasons:
        lines.append("Why: " + " · ".join(reasons))
    return "\n".join(lines)

def render_ops_board(ops):
    lines = []
    lines.append(f"**FTM v2 — Ops Board**  ({'ON' if ops.autotrade else 'OFF'})  |  mode: {ops.pipeline}")
    lines.append(f"WS  mkt={'✅' if ops.ws_mkt_ok else '⚠️'}  user={'✅' if ops.ws_user_ok else '⚠️'}")
    lines.append(f"Risk  realized(Today): {ops.daily_realized:.2f} USDT  |  loss_streak: {ops.loss_streak}")
    if ops.guard: lines.append(f"⛔ Guard: {ops.guard}")
    lines.append("---")
    for sym, pos, tk, oo, risk in ops.symbols:
        hdr = f"**{sym}**  {pos['mode']}x{pos['lev']}  |  {pos['side']} × {pos['qty']:.6f}"
        pl  = f"P&L {pos['upnl']:.2f} / {pos['roe']*100:.2f}%"
        px  = f"Entry/Mark {pos['entry']:.2f} / {pos['mark']:.2f}"
        br  = f"SL/TP {pos['sl']:.2f if pos['sl'] else 0.0:.2f} / " + ( " / ".join(f"{tp:.2f}×{q:.6f}" for tp,q in (pos['tps'] or [])) if pos['tps'] else "0.00" )
        tkx = f"🎫 {tk['side']} score={tk['score']} RR≈{tk['rr']:.2f} ttl={tk['ttl']}s" if tk else "🎫 -"
        lines += [hdr, px, br, pl, tkx, f"OpenOrders: {oo}", ""]
    return "\n".join(lines)

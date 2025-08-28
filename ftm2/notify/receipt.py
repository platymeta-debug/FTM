# [ANCHOR:RECEIPT_MAKER]

def build_receipt(sym, pos, tk, sl, tps):
    data = {
        "symbol": sym,
        "side": ("LONG" if pos.qty > 0 else "SHORT"),
        "qty": abs(pos.qty),
        "entry": pos.entry_price,
        "mark": pos.mark_price,
        "mode": pos.margin_mode,
        "lev": pos.leverage,
        "sl": sl,
        "tps": [px for px, _ in (tps or [])],
        "rr": getattr(tk, "rr", None),
        "score": getattr(tk, "score", None),
        "confidence": getattr(tk, "confidence", None),
        "regime": getattr(tk, "regime", None),
        "reasons": getattr(tk, "reasons", [])[:3],
    }
    txt = (
        f"🎟️ {sym} {data['side']} ×{data['qty']:.6f} @~{data['entry']:.2f} "
        f"(lev {data['mode']}x{data['lev']})\n"
        f"SL {sl:.2f} / TP {', '.join(f'{x:.2f}' for x in data['tps']) or '—'} | "
        f"Score {data['score']} | Conf {data['confidence']:.2f} | Regime {data['regime']}\n"
        f"Why: " + " · ".join(data['reasons'])
    )
    return data, txt

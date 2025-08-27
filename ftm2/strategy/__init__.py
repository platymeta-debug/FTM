from __future__ import annotations

CSV = None

def log_signal(symbol, side, why, score, inds=None, mtf=None, trend_state=None, strategy_ver="v1"):
    if not CSV:
        return
    payload = {
        "symbol": symbol,
        "side": side,
        "reason": why,
        "score_total": getattr(score, "total", score),
        "score": {
            "rsi": getattr(score, "rsi", ""),
            "adx": getattr(score, "adx", ""),
            "atr": getattr(score, "atr", ""),
            "ema": getattr(score, "ema", ""),
            "ichimoku": getattr(score, "ichimoku", ""),
            "kama": getattr(score, "kama", ""),
            "vwap": getattr(score, "vwap", ""),
            "cci": getattr(score, "cci", ""),
            "obv": getattr(score, "obv", ""),
            "corr": getattr(score, "corr", ""),
            "mtf": getattr(score, "mtf", ""),
        },
        "ind": inds or {},
        "mtf": mtf or {},
        "trend_state": trend_state or "",
        "strategy_ver": strategy_ver,
    }
    CSV.log("SIGNAL", **payload)

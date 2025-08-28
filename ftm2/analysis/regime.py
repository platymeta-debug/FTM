from statistics import median

# [ANCHOR:REGIME_CLASSIFIER]
def atr_percentile(atr_series, lookback=500):
    xs = atr_series[-lookback:] if len(atr_series) >= lookback else atr_series
    if not xs:
        return 0.5
    rank = sum(1 for v in xs if v <= xs[-1]) / len(xs)
    return rank  # 0..1

def classify_regime(pct: float):
    if pct < 0.25:
        return "LOW"
    if pct < 0.60:
        return "NORMAL"
    if pct < 0.85:
        return "HIGH"
    return "EXTREME"

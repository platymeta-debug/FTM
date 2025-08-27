from typing import Dict


DEFAULT_WEIGHTS: Dict[str, float] = {
    "rsi": 1.0,
    "ema": 1.0,
    "atr": 1.0,
    "adx": 1.0,
    "corr": 1.0,
}


def score_row(row: Dict[str, float], weights: Dict[str, float] | None = None) -> float:
    """Convert indicator readings into a 0-100 composite score."""
    weights = weights or DEFAULT_WEIGHTS
    scores: Dict[str, float] = {}
    scores["rsi"] = max(0.0, min(100.0, row.get("rsi", 0.0)))
    scores["ema"] = 100.0 if row.get("close", 0.0) > row.get("ema", 0.0) else 0.0
    atr = row.get("atr", 0.0)
    scores["atr"] = 100.0 / (1.0 + atr) if atr else 50.0
    scores["adx"] = max(0.0, min(100.0, row.get("adx", 0.0)))
    if "corr" in row:
        scores["corr"] = 50.0 * (row["corr"] + 1)  # -1..1 -> 0..100
    total_weight = sum(weights.get(k, 0.0) for k in scores)
    if total_weight == 0:
        return 0.0
    return sum(scores[k] * weights.get(k, 0.0) for k in scores) / total_weight

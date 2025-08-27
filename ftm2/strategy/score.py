from __future__ import annotations

"""간단한 스냅샷 점수화 로직."""

from dataclasses import dataclass
from typing import Dict, List, Any

import pandas as pd

from ftm2.indicators.all import add_indicators
from . import reasons


@dataclass
class Contribution:
    name: str
    score: float
    text: str


@dataclass
class Snapshot:
    symbol: str
    total_score: float
    direction: str
    confidence: float
    tf_scores: Dict[str, float]
    contribs: Dict[str, List[Contribution]]
    indicators: Dict[str, pd.DataFrame]
    rules: Dict[str, Any]

    # ✅ 과거 코드 호환용 alias
    @property
    def tfs(self):
        return self.tf_scores

def _parse_tf_weights(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in s.split(","):
        if ":" not in part:
            continue
        tf, w = part.split(":", 1)
        try:
            out[tf.strip()] = float(w)
        except ValueError:
            continue
    return out


def _score_row(row: pd.Series) -> List[Contribution]:
    c: List[Contribution] = []
    rsi_v = float(row.get("rsi", 50.0))
    c.append(Contribution("RSI(14)", rsi_v - 50, reasons.interpret_rsi(rsi_v)))

    close = float(row.get("close", 0.0))
    ema_fast = float(row.get("ema_fast", 0.0))
    ema_slow = float(row.get("ema_slow", 0.0))
    ema_score = 20.0 if close > ema_fast > ema_slow else (-20.0 if close < ema_fast < ema_slow else 0.0)
    c.append(Contribution("EMA", ema_score, reasons.interpret_ema(close, ema_fast, ema_slow)))

    bb_mid = float(row.get("bb_mid", 0.0))
    bb_up = float(row.get("bb_up", 0.0))
    bb_dn = float(row.get("bb_dn", 0.0))
    bb_score = 10.0 if close >= bb_up else (-10.0 if close <= bb_dn else 0.0)
    c.append(Contribution("Bollinger", bb_score, reasons.interpret_bb(close, bb_mid, bb_up, bb_dn)))

    adx_v = float(row.get("adx", 0.0))
    plus_di = float(row.get("plus_di", 0.0))
    minus_di = float(row.get("minus_di", 0.0))
    adx_score = adx_v - 25
    c.append(Contribution("ADX(14)", adx_score, reasons.interpret_adx(adx_v, plus_di, minus_di)))

    cci_v = float(row.get("cci", 0.0))
    c.append(Contribution("CCI(20)", cci_v / 2.0, reasons.interpret_cci(cci_v)))

    return c


def score_snapshot(symbol: str, cache: Dict[str, pd.DataFrame], tfs: List[str], tf_weights: Dict[str, float]) -> Snapshot:
    tf_scores: Dict[str, float] = {}
    contribs: Dict[str, List[Contribution]] = {}
    indicators: Dict[str, pd.DataFrame] = {}

    for tf in tfs:
        df = cache.get(tf)
        if df is None or len(df) == 0:
            continue
        df = add_indicators(df.copy())
        row = df.iloc[-1]
        cons = _score_row(row)
        score = sum(c.score for c in cons)
        tf_scores[tf] = score
        contribs[tf] = cons
        indicators[tf] = df

    total = 0.0
    total_w = 0.0
    for tf, s in tf_scores.items():
        w = tf_weights.get(tf, 0.0)
        total += s * w
        total_w += w
    if total_w > 0:
        total /= total_w

    direction = "NEUTRAL"
    if total > 0:
        direction = "LONG"
    elif total < 0:
        direction = "SHORT"

    confidence = min(1.0, abs(total) / 100.0)

    snap = Snapshot(
        symbol=symbol,
        total_score=total,
        direction=direction,
        confidence=confidence,
        tf_scores=tf_scores,
        contribs=contribs,
        indicators=indicators,
        rules={},
    )
    return snap

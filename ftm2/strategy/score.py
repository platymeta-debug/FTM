from __future__ import annotations
import pandas as pd

from ftm2.analysis.state import AnalysisSnapshot


def score_snapshot(symbol: str, df: pd.DataFrame, cache: dict, tfs: list[str]) -> AnalysisSnapshot:
    """간단한 스냅샷 점수화. 실제 프로젝트에서는 복잡한 로직이 들어갈 수 있다."""
    last = df.iloc[-1]
    total = float(last.get("rsi", 0.0))
    direction = "LONG" if last.get("ema_fast", 0.0) >= last.get("ema_slow", 0.0) else "SHORT"
    snap = AnalysisSnapshot(
        symbol=symbol,
        tfs=tfs,
        ts=float(last.get("ts", 0.0)),
        total_score=total,
        direction=direction,
        confidence=1.0,
        scores={},
        mtf_summary={},
        rules={},
        risk_tier="T1",
        plan={},
        trend_state="",
        indicators={tfs[0]: df},
    )
    return snap


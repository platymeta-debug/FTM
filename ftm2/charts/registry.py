# ftm2/charts/registry.py
import os
import hashlib
import time
from typing import Dict, Tuple, Any

# --- 전역 캐시(먼저 선언!) ---
_LAST_FP: Dict[str, str] = {}       # 심볼별 최근 fingerprint
_LAST_SCORE: Dict[str, float] = {}  # 심볼별 최근 총점(가중)
_LAST_RENDER: Dict[str, float] = {} # 심볼별 최근 렌더 타임스탬프

def reset_cache() -> None:
    """분석 차트 렌더 판단 캐시 초기화"""
    _LAST_FP.clear()
    _LAST_SCORE.clear()
    _LAST_RENDER.clear()

# ---- 유틸 ----
def _get_score(snapshot) -> float:
    # Snapshot.total_score 혹은 .score 중 있는 값을 사용
    if hasattr(snapshot, "total_score"):
        return float(snapshot.total_score)
    return float(getattr(snapshot, "score", 0.0))

def _get_conf(snapshot) -> float:
    return float(getattr(snapshot, "confidence", 0.0))

def _get_price(snapshot) -> float:
    return float(getattr(snapshot, "last_price", 0.0))

def _get_tfs(snapshot) -> str:
    # ['15m','4h',...] → "15m,4h"
    if hasattr(snapshot, "tfs") and snapshot.tfs:
        return ",".join(snapshot.tfs)
    return ""

def compute_fingerprint(snapshot) -> str:
    """
    스냅샷 핵심 요소로 간단한 지문 생성.
    (dir, total_score, mtf_hash, last_close_ts)
    """
    direction = getattr(snapshot, "direction", "").upper()
    total = _get_score(snapshot)
    tf_scores = getattr(snapshot, "tf_scores", {}) or {}
    mtf_hash = hashlib.md5(str(sorted(tf_scores.items())).encode("utf-8")).hexdigest()[:4]
    # indicators에서 마지막 ts 추출
    last_ts = 0
    indicators = getattr(snapshot, "indicators", {}) or {}
    try:
        for df in indicators.values():
            if hasattr(df, "iloc") and len(df) > 0:
                ts = float(df.iloc[-1].get("ts", 0.0))
                if ts > last_ts:
                    last_ts = ts
    except Exception:
        pass
    payload = f"{direction}|{total:.1f}|{mtf_hash}|{int(last_ts)}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]
def render_ready(snapshot, cfg) -> Tuple[bool, Dict[str, Any]]:
    """차트 렌더 여부 판단."""
    sym = getattr(snapshot, "symbol", "UNK")
    new_fp = compute_fingerprint(snapshot)
    last_fp = _LAST_FP.get(sym)

    force_first = bool(getattr(cfg, "CHART_FORCE_FIRST_RENDER", False))
    min_delta = float(getattr(cfg, "CHART_MIN_SCORE_DELTA", 0.5))
    min_div   = float(getattr(cfg, "CHART_MIN_DIVERGENCE_BPS", 1.0))
    div_bps   = float(getattr(snapshot, "divergence_bps", 0.0))
    cooldown  = int(getattr(cfg, "CHART_COOLDOWN_S", 0))

    score     = _get_score(snapshot)
    last_score = _LAST_SCORE.get(sym, score)
    delta     = abs(score - last_score)

    now = time.time()
    last_r = _LAST_RENDER.get(sym, 0.0)
    if cooldown > 0 and (now - last_r) < cooldown:
        return False, {"cause": "cooldown", "remain": cooldown - (now - last_r)}

    if last_fp is None:
        _LAST_FP[sym] = new_fp
        _LAST_SCORE[sym] = score
        if force_first:
            _LAST_RENDER[sym] = now
            return True, {"cause": "first-render", "fp": new_fp}
        return False, {"cause": "first-seen", "fp": new_fp}

    if new_fp == last_fp:
        return False, {"cause": "fingerprint-same", "fp": new_fp}

    if (delta >= min_delta) or (div_bps >= min_div):
        _LAST_FP[sym] = new_fp
        _LAST_SCORE[sym] = score
        _LAST_RENDER[sym] = now
        return True, {"cause": "changed", "delta": delta, "div_bps": div_bps, "fp": new_fp}

    _LAST_FP[sym] = new_fp
    _LAST_SCORE[sym] = score
    return False, {"cause": "below-threshold", "delta": delta, "div_bps": div_bps, "fp": new_fp}

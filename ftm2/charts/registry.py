# ftm2/charts/registry.py
import os
import hashlib
from typing import Dict, Tuple, Any

# --- 전역 캐시(먼저 선언!) ---
_LAST_FP: Dict[str, str] = {}       # 심볼별 최근 fingerprint
_LAST_SCORE: Dict[str, float] = {}  # 심볼별 최근 총점(가중)

def reset_cache() -> None:
    """분석 차트 렌더 판단 캐시 초기화"""
    _LAST_FP.clear()
    _LAST_SCORE.clear()

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
    (심볼/라스트가격(소수2)/총점(소수1)/신뢰도(소수2)/TF묶음)
    """
    sym = getattr(snapshot, "symbol", "UNK")
    payload = f"{sym}|{_get_price(snapshot):.2f}|{_get_score(snapshot):.1f}|{_get_conf(snapshot):.2f}|{_get_tfs(snapshot)}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]

def should_render(cfg: Any, snapshot) -> Tuple[bool, Dict[str, Any]]:
    """
    차트 렌더 여부 판단.
      - 최초 1회 강제 렌더 (CHART_FORCE_FIRST_RENDER)
      - 점수 변화(CHART_MIN_SCORE_DELTA) 또는 괴리(CHART_MIN_DIVERGENCE_BPS) 기준 충족
      - fingerprint 변화가 있어야 함
    """
    sym = getattr(snapshot, "symbol", "UNK")
    new_fp = compute_fingerprint(snapshot)
    last_fp = _LAST_FP.get(sym)

    # ENV/설정 파라미터
    force_first = bool(getattr(cfg, "CHART_FORCE_FIRST_RENDER", False))
    min_delta = float(getattr(cfg, "CHART_MIN_SCORE_DELTA", 0.5))
    min_div   = float(getattr(cfg, "CHART_MIN_DIVERGENCE_BPS", 1.0))
    div_bps   = float(getattr(snapshot, "divergence_bps", 0.0))

    score     = _get_score(snapshot)
    last_score = _LAST_SCORE.get(sym, score)
    delta     = abs(score - last_score)

    # 최초 렌더 강제
    if last_fp is None:
        _LAST_FP[sym] = new_fp
        _LAST_SCORE[sym] = score
        if force_first:
            return True, {"cause": "first-render", "fp": new_fp}
        # 강제 렌더가 아니라면 다음 변화까지 대기
        return False, {"cause": "first-seen-no-force", "fp": new_fp}

    # fingerprint가 바뀌지 않으면 스킵
    if new_fp == last_fp:
        return False, {"cause": "fingerprint-same", "fp": new_fp}

    # 변화 기준(점수/괴리) 충족 시 렌더
    if (delta >= min_delta) or (div_bps >= min_div):
        _LAST_FP[sym] = new_fp
        _LAST_SCORE[sym] = score
        return True, {"cause": "changed", "delta": delta, "div_bps": div_bps, "fp": new_fp}

    # 기준 미충족 → 스킵 (fingerprint는 갱신해도 무방)
    _LAST_FP[sym] = new_fp
    return False, {"cause": "changed-but-below-threshold", "delta": delta, "div_bps": div_bps, "fp": new_fp}

def render_ready() -> bool:
    """필요시 사용할 수 있는 간단한 게이트 함수 (현재는 항상 True)"""
    return True

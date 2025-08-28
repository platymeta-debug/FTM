# ftm2/charts/registry.py

from __future__ import annotations
import time

# --- global caches ---
_LAST_FP: dict[str, tuple[str, float]] = {}
_FORCE_LEFT: dict[str, int] = {}

def reset_cache():
    _LAST_FP.clear()
    _FORCE_LEFT.clear()

def _cfg_int(cfg, name, default):
    try:
        return int(getattr(cfg, name, default))
    except Exception:
        return default

def _cfg_float(cfg, name, default):
    try:
        return float(getattr(cfg, name, default))
    except Exception:
        return default

def compute_fingerprint(snapshot):
    """
    스냅샷의 핵심 특징으로 지문을 만든다.
    방향/총점(소수1)/TF키/분 단위 타임슬라이스를 조합해 과도한 중복 렌더를 방지.
    """
    try:
        direction = getattr(snapshot, "direction", "NEUTRAL")
        score = round(float(getattr(snapshot, "total_score", 0.0)), 1)
        scores = getattr(snapshot, "scores", {}) or {}
        mtf_keys = ",".join(sorted(list(scores.keys())))
        last_ts = int(getattr(snapshot, "ts", time.time())) // 60  # 분단위
        return f"{direction}:{score}:{mtf_keys}:{last_ts}"
    except Exception:
        return f"unknown:{int(time.time())}"

def should_render(cfg, snapshot):
    """
    차트 렌더 여부 판정.
    - CHART_MIN_INTERVAL_S: 최소 간격
    - CHART_FORCE_N_CYCLES: 부팅 후 강제 렌더 횟수
    - 기본 정책: 지문이 바뀌었을 때만 렌더
    """
    sym = getattr(snapshot, "symbol", "UNKNOWN")
    now = time.time()

    min_interval = _cfg_int(cfg, "CHART_MIN_INTERVAL_S", 10)
    last = _LAST_FP.get(sym)
    if last and (now - last[1]) < min_interval:
        return False, {"reason": "interval"}

    # 부팅 후 N회 강제 렌더
    force_n = _cfg_int(cfg, "CHART_FORCE_N_CYCLES", 0)
    if sym not in _FORCE_LEFT:
        _FORCE_LEFT[sym] = force_n

    fp = compute_fingerprint(snapshot)

    if _FORCE_LEFT[sym] > 0:
        _FORCE_LEFT[sym] -= 1
        _LAST_FP[sym] = (fp, now)
        return True, {"reason": "force"}

    prev_fp = last[0] if last else None
    if prev_fp == fp:
        return False, {"reason": "same_fp"}

    _LAST_FP[sym] = (fp, now)
    return True, {"reason": "changed"}

# -----------------------------
# 🔁 호환 레이어(이 줄들만 추가해도 OK)
# 기존 코드가 기대하는 이름을 그대로 제공
def render_ready(cfg, snapshot):
    ready, _ = should_render(cfg, snapshot)
    return ready

def render_meta(cfg, snapshot):
    _, meta = should_render(cfg, snapshot)
    return meta
# -----------------------------

# [ANCHOR:M6_CHART_REGISTRY]
import json, time, os, hashlib
from typing import Dict, Any

REG_PATH = "storage/charts/registry.json"


def _now() -> float:
    return time.time()


def _load() -> Dict[str, Any]:
    if os.path.exists(REG_PATH):
        try:
            return json.load(open(REG_PATH, "r", encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save(reg: Dict[str, Any]):
    os.makedirs(os.path.dirname(REG_PATH), exist_ok=True)
    tmp = REG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, REG_PATH)


def compute_fingerprint(snapshot) -> str:
    """
    값 몇 개만 취합해서 가벼운 지문 생성(그림을 그리기 전 중복 판단).
    총점, 마지막 종가/EMA/RSI/ADX/CCI 등.
    """
    x = {
        "sym": snapshot.symbol,
        "score": round(snapshot.total_score, 2),
        "dir": snapshot.direction,
        "conf": round(snapshot.confidence, 3),
    }
    df = snapshot.indicators.get("1m") or snapshot.indicators.get("main")
    if df is not None and len(df) > 0:
        last = df.iloc[-1]
        for k in ("close", "ema_fast", "ema_slow", "rsi", "adx", "cci", "obv"):
            if k in df.columns:
                x[k] = float(last.get(k, 0.0))
    raw = json.dumps(x, sort_keys=True).encode()
    return hashlib.sha1(raw).hexdigest()


def should_render(cfg, snapshot) -> tuple[bool, Dict[str, Any]]:
    """
    - 최소 간격/점수 변화/강제 주기 체크
    - True면 렌더, False면 스킵
    returns: (ok, meta) where meta contains 'reason' / 'counter'
    """
    reg = _load()
    sym = snapshot.symbol
    now = _now()
    ent = reg.get(sym, {"last_ts": 0, "last_fp": "", "counter": 0})
    # 최소 간격
    if now - ent["last_ts"] < cfg.CHART_MIN_INTERVAL_S:
        ent["counter"] += 1
        reg[sym] = ent
        _save(reg)
        return False, {"reason": "interval", "counter": ent["counter"]}

    # 점수 변화
    prev_score = ent.get("last_score", 0.0)
    if abs(snapshot.total_score - prev_score) < cfg.CHART_MIN_SCORE_DELTA:
        # 강제 주기 조건
        ent["counter"] = ent.get("counter", 0) + 1
        if ent["counter"] < cfg.CHART_FORCE_N_CYCLES:
            reg[sym] = ent
            _save(reg)
            return False, {"reason": "score-delta", "counter": ent["counter"]}

    # 지문(fingerprint) 중복
    fp = compute_fingerprint(snapshot)
    if fp == ent.get("last_fp", ""):
        ent["counter"] += 1
        if ent["counter"] < cfg.CHART_FORCE_N_CYCLES:
            reg[sym] = ent
            _save(reg)
            return False, {"reason": "fingerprint", "counter": ent["counter"]}

    # 렌더 승인
    ent.update({
        "last_ts": now,
        "last_fp": fp,
        "last_score": float(snapshot.total_score),
        "counter": 0,
    })
    reg[sym] = ent
    _save(reg)
    return True, {"reason": "ok", "counter": 0}


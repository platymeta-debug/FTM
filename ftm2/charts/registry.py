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

    경량 메타(심볼/총점/방향/신뢰도 + 마지막 캔들 몇 값)로 지문 생성.
    DataFrame의 진릿값 평가(or/and) 절대 사용하지 않음.
    """
    x = {
        "sym": snapshot.symbol,
        "score": round(float(snapshot.total_score), 2),
        "dir": snapshot.direction,
        "conf": round(float(snapshot.confidence), 3),
    }

    indicators = getattr(snapshot, "indicators", {}) or {}
    df = indicators.get("1m")
    if df is None:
        df = indicators.get("main")


    if df is not None and len(df) > 0:
        last = df.iloc[-1]
        for k in ("close", "ema_fast", "ema_slow", "rsi", "adx", "cci", "obv"):
            if k in df.columns:
                try:
                    x[k] = float(last[k])
                except Exception:
                    pass

    raw = json.dumps(x, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha1(raw).hexdigest()


def should_render(cfg, snapshot) -> tuple[bool, Dict[str, Any]]:
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
    if abs(float(snapshot.total_score) - float(prev_score)) < cfg.CHART_MIN_SCORE_DELTA:
        ent["counter"] = ent.get("counter", 0) + 1
        if ent["counter"] < cfg.CHART_FORCE_N_CYCLES:
            reg[sym] = ent
            _save(reg)
            return False, {"reason": "score-delta", "counter": ent["counter"]}


    # 지문 비교 (여기서도 DataFrame 진릿값을 절대 쓰지 않음)
    try:
        fp = compute_fingerprint(snapshot)
    except Exception:
        # 어떤 이유로든 fingerprint 실패하면 렌더 1회 허용해서 진행
        fp = f"{sym}:{time.time():.0f}"


    if fp == ent.get("last_fp", ""):
        ent["counter"] += 1
        if ent["counter"] < cfg.CHART_FORCE_N_CYCLES:
            reg[sym] = ent
            _save(reg)
            return False, {"reason": "fingerprint", "counter": ent["counter"]}


    # 통과 → 상태 갱신
    ent.update({
        "last_ts": now,
        "last_fp": fp,
        "last_score": float(snapshot.total_score),
        "counter": 0,
    })
    reg[sym] = ent
    _save(reg)
    return True, {"reason": "ok", "counter": 0}


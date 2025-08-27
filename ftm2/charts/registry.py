# [ANCHOR:M6_CHART_REGISTRY]
import json, time, os
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
    """Return lightweight fingerprint from snapshot data."""
    if isinstance(snapshot, dict):
        snap = snapshot
    else:
        snap = snapshot.__dict__
    indicators = snap.get("indicators", {}) or {}
    df = indicators.get("1m") or indicators.get("main")
    ts = 0
    bb = 0.0
    rsi = 0.0
    if df is not None and len(df) > 0:
        last = df.iloc[-1]
        try:
            ts = int(last.get("timestamp", 0))
        except Exception:
            try:
                ts = int(df.index[-1])
            except Exception:
                ts = 0
        bb = float(last.get("bb_pos", last.get("bbp", 0.0))) if hasattr(last, "get") else 0.0
        rsi = float(last.get("rsi", 0.0)) if hasattr(last, "get") else 0.0
    score = round(float(snap.get("total_score", 0.0)), 1)
    return f"{ts}:{score}:{int(bb*100)}:{int(rsi)}"


def _get(snapshot, key, default=0.0):
    if isinstance(snapshot, dict):
        return snapshot.get(key, default)
    return getattr(snapshot, key, default)


def should_render(cfg, snapshot) -> tuple[bool, Dict[str, Any]]:
    reg = _load()
    sym = _get(snapshot, "symbol")
    now = _now()
    ent = reg.get(sym, {"last_ts": 0, "last_fp": "", "counter": 0, "last_score": 0.0})

    rules = getattr(snapshot, "rules", {}) if not isinstance(snapshot, dict) else snapshot.get("rules", {})
    divergence_bps = float(rules.get("divergence_bps", 0.0))

    # 괴리도 게이트
    if abs(divergence_bps) < cfg.CHART_MIN_DIVERGENCE_BPS:
        ent["counter"] = ent.get("counter", 0) + 1
        if ent["counter"] < cfg.CHART_FORCE_N_CYCLES:
            reg[sym] = ent
            _save(reg)
            reason = (
                f"divergence {divergence_bps:.2f}bps < {cfg.CHART_MIN_DIVERGENCE_BPS}"
            )
            return False, {"reason": reason, "counter": ent["counter"]}

    # 최소 간격
    if now - ent["last_ts"] < cfg.CHART_MIN_INTERVAL_S:
        ent["counter"] = ent.get("counter", 0) + 1
        reg[sym] = ent
        _save(reg)
        reason = f"interval {now - ent['last_ts']:.1f}s < {cfg.CHART_MIN_INTERVAL_S}"
        return False, {"reason": reason, "counter": ent["counter"]}

    # 점수 변화
    prev_score = ent.get("last_score", 0.0)
    cur_score = float(_get(snapshot, "total_score"))
    if abs(cur_score - float(prev_score)) < cfg.CHART_MIN_SCORE_DELTA:
        ent["counter"] = ent.get("counter", 0) + 1
        if ent["counter"] < cfg.CHART_FORCE_N_CYCLES:
            reg[sym] = ent
            _save(reg)
            reason = (
                f"score Δ{cur_score - float(prev_score):.2f} < {cfg.CHART_MIN_SCORE_DELTA}"
            )
            return False, {"reason": reason, "counter": ent["counter"]}

    # 지문 비교
    try:
        fp = compute_fingerprint(snapshot)
    except Exception:
        fp = f"{sym}:{time.time():.0f}"

    if fp == ent.get("last_fp", ""):
        ent["counter"] = ent.get("counter", 0) + 1
        if ent["counter"] < cfg.CHART_FORCE_N_CYCLES:
            reg[sym] = ent
            _save(reg)
            reason = f"fingerprint same {fp}"
            return False, {"reason": reason, "counter": ent["counter"]}

    # 통과 → 상태 갱신
    ent.update({
        "last_ts": now,
        "last_fp": fp,
        "last_score": cur_score,
        "counter": 0,
    })
    reg[sym] = ent
    _save(reg)
    return True, {"reason": "render", "counter": 0}


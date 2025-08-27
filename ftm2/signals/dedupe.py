import time
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable

# 기본값은 settings에서 오버라이드 가능 (없으면 이 값 사용)
DEFAULT_ENTER_TH = 60
DEFAULT_COOLDOWN_SEC = 300
DEFAULT_SCORE_BUCKET = 5


@dataclass
class LastState:
    last_emit_ts: float = 0.0
    last_above: bool = False  # ENTER_TH 이상 구간에 있었는지


_last_state: Dict[str, LastState] = {}
_seen: Dict[Tuple, float] = {}


def _bucket(v: float, step: int) -> int:
    return int(round(v / step) * step)


def should_emit(
    symbol: str,
    side: str,
    score: float,
    reasons: Iterable[str],
    candle_open_ts: int,
    *,
    enter_th: int = DEFAULT_ENTER_TH,
    cooldown_sec: int = DEFAULT_COOLDOWN_SEC,
    score_bucket: int = DEFAULT_SCORE_BUCKET,
    edge_trigger: bool = True,
) -> bool:
    """엣지 트리거 + 디듀프/쿨다운."""
    key = f"{symbol}:{side}"
    st = _last_state.get(key, LastState())

    above = abs(score) >= enter_th
    emit_edge = (not st.last_above) and above if edge_trigger else above

    now = time.monotonic()
    rtuple = tuple(sorted(reasons or []))
    dedupe_key = (symbol, side, candle_open_ts, _bucket(abs(score), score_bucket), rtuple)
    last_seen = _seen.get(dedupe_key, 0.0)
    in_cooldown = (now - last_seen) < cooldown_sec

    st.last_above = above
    _last_state[key] = st

    if emit_edge and not in_cooldown:
        _seen[dedupe_key] = now
        return True
    return False

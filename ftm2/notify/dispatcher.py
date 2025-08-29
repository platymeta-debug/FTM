import os
import time
import asyncio
import re
from ftm2.config.settings import load_env_chain
from ftm2.notify import discord_bot

# [NOTIFY_MAP]
# 조용한 시그널: 실제 액션만 signals로, 나머지는 logs로
ROUTE_MAP = {
    "intent": "logs",         # \ud83d\udcc9/\ud83d\udcc8 \uc810\uc218 \uc54c\ub9bc(\uc758\ub3c4\ub9cc) -> logs
    "gate_skip": "logs",      # \uc9c4\uc785 \uae08\uc9c0 \uc0ac\uc720 -> logs
    "intent_cancel": "logs",  # \uc7ac\uc2dc\ub3c4 \ucd08\uacfc/\ucde8\uc18c -> logs
    "order_submitted": "signals",  # \uc2e4\uc81c \uc8fc\ubb38 \uc2dc\uae00\ub85c\ub9cc \uc2e0\ud638 \ucc44\ub110
    "order_failed": "logs",
    "fill": "trades",
    "close": "trades",
    "pnl": "trades",
    "system": "logs",
    "error": "logs",
    "chart": "logs",          # \ucc28\ud2b8\ub294 VS \ub85c\uae45\uc쪾\ub85c\ub9cc
}


class _DiscordAdapter:
    def __init__(self, cfg):
        self.cfg = cfg

    def send(self, channel: str, text: str):
        if channel == self.cfg.CHANNEL_TRADES:
            discord_bot.send_trade(text)
        elif channel == self.cfg.CHANNEL_SIGNALS:
            discord_bot.send_signal(text)
        else:
            discord_bot.send_log(text)


_cfg = load_env_chain()
notifier = type("Notifier", (), {"dc": _DiscordAdapter(_cfg), "route": ROUTE_MAP})()

async def send(channel_key_or_name: str, text: str):
    """Bridge to actual send implementation."""
    return await dc.send(channel_key_or_name, text)

async def edit(message_id, text: str):
    return None


# [ANCHOR:DISPATCHER_DC_ADAPTER_V2]

# 채널 별칭 → 실제 타겟(채널ID나 '#이름') 매핑
# 실제 프로젝트에서 init 시점 또는 env에서 재설정됨을 가정
CHANNELS = {
    "signals": os.getenv("CHANNEL_SIGNALS", "signals"),
    "trades":  os.getenv("CHANNEL_TRADES", "trades"),
    "logs":    os.getenv("CHANNEL_LOGS", "logs"),
}

def _resolve_channel(key_or_name: str):
    """
    'signals' 같은 별칭, '#포지션신호' 같은 디스코드 채널명, '1234567890' 같은 ID 모두 허용.
    매칭 실패 시 'signals'로 폴백.
    """
    if not key_or_name:
        return CHANNELS.get("signals", "signals")

    k = str(key_or_name).strip()
    # 1) 별칭이면 매핑
    if k in CHANNELS:
        return CHANNELS[k]
    # 2) '#이름' 그대로 허용
    if k.startswith("#"):
        return k
    # 3) 숫자(ID)면 그대로 반환
    if k.isdigit():
        return k
    # 4) 값으로 '#이름' 저장된 경우 역탐색
    for alias, val in CHANNELS.items():
        if val == k:
            return val
    # 5) 폴백
    return CHANNELS.get("signals", "signals")

async def _send_impl(channel_key_or_name: str, text: str):
    """
    실제 전송 함수에 연결. DRY 모드면 콘솔/로그만.
    """
    target = _resolve_channel(channel_key_or_name)
    if hasattr(notifier, "dc") and hasattr(notifier.dc, "send"):
        result = notifier.dc.send(target, text)
        if asyncio.iscoroutine(result):
            return await result
        return result
    # DRY/no-op fallback
    if 'emit' in globals():
        emit("system", f"[DRY][send->{target}] {text}")
    return None

async def _edit_impl(message_id, text: str):
    if 'edit' in globals():
        return await edit(message_id, text)
    if 'emit' in globals():
        emit("system", f"[DRY][edit->{message_id}] {text}")
    return None


class _DCUseCtx:
    def __init__(self, parent, channel_key_or_name):
        self.parent = parent
        self.target = channel_key_or_name
    async def send(self, text: str):
        return await _send_impl(self.target, text)
    async def edit(self, message_id, text: str):
        return await _edit_impl(message_id, text)

class _DCAdapter:
    async def send(self, channel_key_or_name: str, text: str):
        return await _send_impl(channel_key_or_name, text)
    async def edit(self, message_id, text: str):
        return await _edit_impl(message_id, text)
    def use(self, channel_key_or_name: str):
        """
        notify.dc.use('signals').send('...') 형태 지원
        """
        return _DCUseCtx(self, channel_key_or_name)

# 항상 dc를 노출(초기화 실패/DRY 상황에서도 None이 되지 않게)
dc = _DCAdapter()


# [EMIT_RATE_LIMIT]
_LAST_EMIT: dict[str, int] = {}

# \uc774\ubca4\ud2b8\ubcc4 \uae30\ubcf8 TTL(ms)
EMIT_TTL = {
    "gate_skip": 60_000,     # 1\ubd84\uc5d0 1\ubc88\ub9cc
    "intent": 60_000,
    "intent_cancel": 60_000,
    "order_failed": 10_000,
    "system": 5_000,
    "error": 5_000,
    "chart": 30_000,        # \ucc28\ud2b8\ub3c4 \uc7a5\ucc28\uac00 \uc548 \uc62c\ub9bc
}

def _normalize(kind: str, text: str) -> str:
    """\ub3d9\uc77c \uba54\uc2dc\uc9c0\ub85c \ucde8\uae09\ud558\uae30 \uc704\ud55c \uc815\uaddc\ud654"""
    t = re.sub(r"\s+", " ", str(text)).strip()
    t = re.sub(r"@~\d+(\.\d+)?", "@~PX", t)
    t = re.sub(r"\d+(\.\d+)?", "N", t)
    return f"{kind}:{t}"

async def _emit(kind: str, text: str, route: str | None = None, *, ttl_ms: int | None = None):
    route = route or ROUTE_MAP.get(kind, "logs")
    ttl = EMIT_TTL.get(kind, 0) if ttl_ms is None else max(0, int(ttl_ms))
    key = _normalize(kind, text)
    now = int(time.time() * 1000)
    last = _LAST_EMIT.get(key, 0)
    if ttl and now - last < ttl:
        return None
    _LAST_EMIT[key] = now

    from ftm2.notify import dispatcher as dp  # self import safe
    try:
        return await dp.dc.send(route, text)
    except Exception as e:
        try:
            return await dp.dc.send("logs", f"[EMIT_FAIL->{route}] {type(e).__name__}: {e}\n{text}")
        except Exception:
            return None

def emit(kind: str, text: str, route: str | None = None, *, ttl_ms: int | None = None):
    return asyncio.create_task(_emit(kind, text, route, ttl_ms=ttl_ms))

_THROTTLE: dict[str, int] = {}

def emit_once(key: str, kind: str, text: str, ttl_ms: int | None = None):
    ttl = ttl_ms or 0
    now = int(time.time() * 1000)
    last = _THROTTLE.get(key, 0)
    if ttl and now - last < ttl:
        return None
    _THROTTLE[key] = now
    return emit(kind, text)


def configure_channels(**kw):
    """
    런타임에서 CHANNELS 갱신(예: env 반영).
    """
    CHANNELS.update({k: v for k, v in kw.items() if v})
    if 'emit' in globals():
        emit("system", f"[NOTIFY_CHANNELS] {CHANNELS}")

def ensure_dc():
    """외부에서 보증 호출 가능(이미 객체면 그대로 둠)"""
    global dc
    if dc is None or not hasattr(dc, "send") or not hasattr(dc, "use"):
        dc = _DCAdapter()
    return dc


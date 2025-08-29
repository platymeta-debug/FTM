# --- dispatcher.py: discord_bot 바인딩(재귀 방지 + 폴백 포함) ---
from types import SimpleNamespace
from ftm2.notify import discord_bot as _bot

async def _missing(*args, **kwargs):
    raise RuntimeError("discord_bot API(send/edit/upsert) 가 구현되어 있지 않습니다.")

def _noop_use(*args, **kwargs):
    # 과거 코드 호환용: 외부에서 dispatcher.dc.use(...)를 부르는 경우가 있어도 NOP
    return None

dc = SimpleNamespace(
    # discord_bot에 함수가 있으면 그것을, 없으면 폴백을 사용
    send=getattr(_bot, "send", None) or _missing,
    edit=getattr(_bot, "edit", None) or _missing,
    upsert=getattr(_bot, "upsert", None) or _missing,
    # 과거 인터페이스 호환(없어도 되지만 호출하는 코드가 있을 수 있어 NOP 제공)
    use=getattr(_bot, "use", None) or _noop_use,
)


# channel key/name to real channel ID mapping
_CHANNELS: dict[str, int | str] = {}


def configure_channels(chmap: dict[str, int | str]) -> None:
    """Configure channel aliases used by the dispatcher."""
    global _CHANNELS
    _CHANNELS = dict(chmap)


def _resolve_channel(key_or_name: int | str) -> int | str:
    """Resolve channel alias or ID to the target used by discord_bot."""
    k = key_or_name
    if isinstance(k, int):
        return k
    if isinstance(k, str):
        if k in _CHANNELS:
            return _CHANNELS[k]
        if k.startswith("#") and k in _CHANNELS:
            return _CHANNELS[k]
        if k.isdigit():
            return int(k)
    return k


# ---------------------------------------------------------------------------
# Boot queue handling
# ---------------------------------------------------------------------------
_BOOT_QUEUE: list[tuple[str, str, str | None, int]] = []
_BOOT_READY = False


async def flush_boot_queue() -> None:
    """Flush queued emits once the event loop is ready."""
    global _BOOT_READY
    _BOOT_READY = True
    while _BOOT_QUEUE:
        kind, text, route, ttl_ms = _BOOT_QUEUE.pop(0)
        await _emit(kind, text, route, ttl_ms=ttl_ms)


# ---------------------------------------------------------------------------
# Rate limiting / dedupe
# ---------------------------------------------------------------------------
_EMIT_TTL_MS = {
    "intent": 60_000,
    "gate_skip": 60_000,
    "intent_cancel": 60_000,
    "chart": 30_000,
    "system": 5_000,
    "error": 5_000,
}

_LAST_EMIT: dict[tuple[str, str], int] = {}
_ONCE_LAST_TS: dict[str, int] = {}


def _norm_text(s: str) -> str:
    return " ".join(s.split())[:500]


def emit(kind: str, text: str, route: str | None = None, ttl_ms: int = 0):
    """Emit a notification; queue if the event loop is not ready."""
    ts = int(time.time() * 1000)
    norm = _norm_text(text)
    ttl = ttl_ms or _EMIT_TTL_MS.get(kind, 0)
    if ttl:
        last = _LAST_EMIT.get((kind, norm), 0)
        if ts - last < ttl:
            return None
        _LAST_EMIT[(kind, norm)] = ts

    if not _BOOT_READY:
        _BOOT_QUEUE.append((kind, text, route, ttl_ms))
        return None

    return asyncio.create_task(_emit(kind, text, route, ttl_ms=ttl_ms))


def emit_once(key: str, kind: str, text: str, ttl_ms: int = 60_000, route: str | None = None):
    """Emit only once per key within the TTL window."""
    ts = int(time.time() * 1000)
    last = _ONCE_LAST_TS.get(key, 0)
    if ts - last < ttl_ms:
        return None
    _ONCE_LAST_TS[key] = ts
    return emit(kind, text, route=route, ttl_ms=0)


# ---------------------------------------------------------------------------
# Sending helpers
# ---------------------------------------------------------------------------
async def _send_impl(target: int | str, text: str):
    """Internal send helper that directly calls discord_bot.send."""
    chan = _resolve_channel(target)
    return await dc.send(chan, text)


async def send(channel_key_or_name: int | str, text: str):
    """Public send wrapper."""
    return await _send_impl(channel_key_or_name, text)


# ---------------------------------------------------------------------------
# Routing map and emit implementation
# ---------------------------------------------------------------------------
ROUTE_MAP = {
    "intent": "logs",
    "gate_skip": "logs",
    "intent_cancel": "logs",
    "order_submitted": "signals",
    "order_failed": "logs",
    "fill": "trades",
    "close": "trades",
    "pnl": "trades",
    "system": "logs",
    "error": "logs",
    "chart": "logs",
}


async def _emit(kind: str, text: str, route: str | None = None, ttl_ms: int = 0):
    target = route or ROUTE_MAP.get(kind, "logs")
    try:
        return await _send_impl(target, text)
    except Exception as e:  # pragma: no cover - best effort logging
        print(f"[DISPATCHER][ERR] {kind} route={target} -> {e}", flush=True)


# [NOTIFY_MAP]
ROUTE_MAP = {
    "intent": "logs",
    "gate_skip": "logs",
    "intent_cancel": "logs",
    "order_submitted": "signals",
    "order_failed": "logs",
    "fill": "trades",
    "close": "trades",
    "pnl": "trades",
    "system": "logs",
    "error": "logs",
    "chart": "logs",
}

# [ANCHOR:EMIT_RATE_LIMIT]
EMIT_RATE_LIMIT = {
    "intent": 60000,
    "gate_skip": 60000,
    "intent_cancel": 60000,
    "chart": 30000,
    "error": 5000,
    "system": 5000,
}

_emit_history: dict[str, float] = {}
_BOOT_QUEUE: list[tuple[str, str]] = []
_ONCE_CACHE: dict[str, float] = {}


def _should_emit(kind: str) -> bool:
    ttl = EMIT_RATE_LIMIT.get(kind)
    if not ttl:
        return True
    now = time.time() * 1000
    last = _emit_history.get(kind, 0)
    if now - last < ttl:
        return False
    _emit_history[kind] = now
    return True


def emit(kind: str, text: str, route: str | None = None) -> None:
    if not _should_emit(kind):
        return
    ch = route or ROUTE_MAP.get(kind, "logs")
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            _BOOT_QUEUE.append((ch, text))
            return
        loop.create_task(dc.send(ch, text))
    except RuntimeError:
        _BOOT_QUEUE.append((ch, text))


def emit_once(key: str, kind: str, text: str, ttl_ms: int = 60000, route: str | None = None) -> None:
    now = time.time() * 1000
    exp = _ONCE_CACHE.get(key)
    if exp and now < exp:
        return
    _ONCE_CACHE[key] = now + ttl_ms
    emit(kind, text, route=route)


def flush_boot_queue() -> None:
    while _BOOT_QUEUE:
        ch, text = _BOOT_QUEUE.pop(0)
        try:
            asyncio.get_event_loop().create_task(dc.send(ch, text))
        except Exception:
            pass


# 외부에서 직접 호출 가능하도록 별도 함수 유지
async def send(channel_key_or_name: str, text: str):
    await dc.send(channel_key_or_name, text)


async def edit(message_id, text: str):
    return await dc.edit(message_id, text)



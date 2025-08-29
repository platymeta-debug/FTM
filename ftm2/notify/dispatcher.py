
# ==== dispatcher.py: 안정화 공통 import ====
import asyncio, time, inspect, os
from typing import Optional
from types import SimpleNamespace

# ==== discord_bot 바인딩 (항상 외부 구현만 호출; 재귀 금지) ====
try:
    from ftm2.notify import discord_bot as _bot
except Exception as e:
    _bot = None

async def _missing(*args, **kwargs):
    raise RuntimeError("discord_bot API(send/edit/upsert)가 구현되어 있지 않습니다.")

def _noop_use(*args, **kwargs):
    # 예전 코드 호환: dispatcher.dc.use(...) 호출이 남아있어도 무해하게 처리
    return None

# [ANCHOR:NOTIFY_DISPATCH]
DISCORD_ALLOW_COMPONENTS = os.getenv("DISCORD_ALLOW_COMPONENTS", "off").lower() in (
    "1",
    "true",
    "on",
)

# [ANCHOR:NOTIFY_DISPATCH]
async def maybe_await(x):
    if inspect.isawaitable(x):
        return await x
    return x

def _supports_kw(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except Exception:
        return False

def _to_view_from_components(components):
    """
    components(JSON 유사) → discord.ui.View 변환.
    변환 실패 시 None 반환하여 조용히 드랍.
    """
    try:
        import discord
        from discord.ui import View, Button, Select

        v = View()
        for row in components or []:
            for comp in row or []:
                t = comp.get("type")
                if t == "button":
                    btn = Button(
                        label=comp.get("label"),
                        custom_id=comp.get("custom_id"),
                        url=comp.get("url"),
                        disabled=comp.get("disabled", False),
                        style=getattr(
                            discord.ButtonStyle,
                            comp.get("style", "secondary"),
                            discord.ButtonStyle.secondary,
                        ),
                        emoji=comp.get("emoji"),
                    )
                    v.add_item(btn)
                elif t == "select":
                    sel = Select(
                        custom_id=comp.get("custom_id"),
                        placeholder=comp.get("placeholder"),
                        min_values=comp.get("min_values", 1),
                        max_values=comp.get("max_values", 1),
                        options=comp.get("options", []),
                        disabled=comp.get("disabled", False),
                    )
                    v.add_item(sel)
        return v
    except Exception:
        return None

async def discord_safe_send(target, **kwargs):
    """
    target: Channel/Context/Webhook 등 .send 를 가진 객체
    kwargs: content, embed/embeds, components, view, files 등
    - discord.py: view= 사용, components 미지원
    - disnake/py-cord: components= 가능
    - webhook: 대부분 components/view 미지원
    """
    send_fn = getattr(target, "send", None) or target
    if not callable(send_fn):
        return None

    comps = kwargs.pop("components", None)

    if "embed" in kwargs and "embeds" not in kwargs:
        pass
    elif "embeds" in kwargs and "embed" in kwargs:
        kwargs.pop("embed", None)

    if comps and DISCORD_ALLOW_COMPONENTS:
        if _supports_kw(send_fn, "components"):
            kwargs["components"] = comps
        elif _supports_kw(send_fn, "view"):
            view = _to_view_from_components(comps)
            if view is not None:
                kwargs["view"] = view

    try:
        return await maybe_await(send_fn(**kwargs))
    except TypeError:
        minimal = {k: kwargs[k] for k in ("content", "embed", "embeds", "text") if k in kwargs}
        return await maybe_await(send_fn(**minimal))

dc = SimpleNamespace(
    send=(getattr(_bot, "send", None) or _missing),
    edit=(getattr(_bot, "edit", None) or _missing),
    upsert=(getattr(_bot, "upsert", None) or _missing),
    use=(getattr(_bot, "use", None) or _noop_use),
)

# ==== 채널 라우팅 테이블과 해석 ====
_CHANNELS: dict[str, int | str] = {}  # 예: {'signals': 1234, 'logs': 5678}

def configure_channels(chmap: dict[str, int | str]) -> None:
    global _CHANNELS
    _CHANNELS = dict(chmap)

def _resolve_channel(key_or_name):
    k = key_or_name
    if isinstance(k, int):
        return k
    if isinstance(k, str):
        if k in _CHANNELS:
            return _CHANNELS[k]
        if k.startswith("#") and k[1:] in _CHANNELS:
            return _CHANNELS[k[1:]]
        if k.isdigit():
            return int(k)
    return k  # 마지막 방어

# ==== 부팅 큐 & 중복 억제 ====
_BOOT_QUEUE: list[tuple[str, str, Optional[str], int]] = []  # (kind, text, route, ttl_ms)
_BOOT_READY = False

_EMIT_TTL_MS = {
    "intent": 60_000,
    "gate_skip": 60_000,
    "intent_cancel": 60_000,
    "chart": 30_000,
    "system": 5_000,
    "error": 5_000,

}
_LAST_EMIT: dict[tuple[str, str], int] = {}   # (kind, normalized) -> ts
_ONCE_LAST_TS: dict[str, int] = {}            # key -> ts

def _norm(s: str) -> str:
    return " ".join(s.split())[:600]


# ==== 퍼블릭 API: emit / emit_once / flush_boot_queue ====
def emit(kind: str, text: str, route: Optional[str] = None, ttl_ms: int = 0):
    """
    루프 전에는 부팅 큐에 저장, 루프 준비 후에는 비동기 태스크 생성.
    """
    ts = int(time.time() * 1000)
    norm = _norm(text)
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

def emit_once(key: str, kind: str, text: str, ttl_ms: int = 60_000, route: Optional[str] = None):
    ts = int(time.time() * 1000)
    last = _ONCE_LAST_TS.get(key, 0)
    if ts - last < ttl_ms:
        return None
    _ONCE_LAST_TS[key] = ts
    return emit(kind, text, route=route, ttl_ms=0)

async def flush_boot_queue():
    """
    반드시 async여야 app.py에서 await 가능.
    """
    global _BOOT_READY
    _BOOT_READY = True
    while _BOOT_QUEUE:
        kind, text, route, ttl_ms = _BOOT_QUEUE.pop(0)
        await _emit(kind, text, route, ttl_ms=ttl_ms)

# ==== 라우팅 테이블 ====
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

# ==== 내부 전송(재귀 금지: 반드시 dc.send만 호출) ====
async def _send_impl(target, text: str):
    chan = _resolve_channel(target)
    return await discord_safe_send(dc.send, channel_key_or_name=chan, text=text)

async def send(channel_key_or_name, text: str):
    return await _send_impl(channel_key_or_name, text)

async def edit(message_id, text: str):
    return await discord_safe_send(dc.edit, message_id=message_id, text=text)


async def _emit(kind: str, text: str, route: Optional[str] = None, ttl_ms: int = 0):
    target = route or ROUTE_MAP.get(kind, "logs")
    try:
        return await _send_impl(target, text)
    except Exception as e:
        print(f"[DISPATCHER][ERR] kind={kind} route={target} -> {e}", flush=True)

__all__ = [
    "emit",
    "emit_once",
    "flush_boot_queue",
    "send",
    "edit",
    "discord_safe_send",
    "configure_channels",
    "dc",
]

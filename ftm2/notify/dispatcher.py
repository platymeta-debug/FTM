
import os
import time
from ftm2.config.settings import load_env_chain
from ftm2.notify import discord_bot


class Notifier:
    # [ANCHOR:NOTIFIER_INIT]
    def __init__(self, cfg, discord_client):
        self.cfg = cfg
        self.dc = discord_client
        self._throttle: dict[str, float] = {}
        # 채널 바인딩 (이름 또는 ID 지원)
        self.ch_signals = cfg.CHANNEL_SIGNALS
        self.ch_trades = cfg.CHANNEL_TRADES
        self.ch_logs = cfg.CHANNEL_LOGS

        # 이벤트 → 채널 맵(기본)
        self.route = {
            "intent": "logs",
            "gate_skip": "logs",
            "intent_cancel": "logs",
            "order_submitted": "signals",
            "order_failed": "signals",
            "fill": "trades",
            "close": "trades",
            "pnl": "trades",
            "system": "logs",
            "error": "logs",
            "chart": "logs",
        }

        # 이벤트별 스팸 억제 TTL(ms)
        self.ttl_ms = {
            "intent": 60000,
            "gate_skip": 60000,
            "intent_cancel": 60000,
        }

    def _send(self, which: str, text: str):
        ch = {
            "signals": self.ch_signals,
            "trades": self.ch_trades,
            "logs": self.ch_logs,
        }[which]
        self.dc.send(ch, text)

    def push_signal(self, text: str):
        """Directly send to signal channel."""
        self._send("signals", text)

    def push_trade(self, text: str):
        """Directly send to trade channel."""
        self._send("trades", text)

    def push_log(self, text: str):
        """Directly send to log channel."""
        self._send("logs", text)


    def emit(self, event: str, text: str):
        ttl = self.ttl_ms.get(event)
        if ttl:
            now = time.time() * 1000
            key = f"{event}:{text}"
            if now - self._throttle.get(key, 0) < ttl:
                return
            self._throttle[key] = now

        which = self.route.get(event, "logs")
        if self.cfg.NOTIFY_STRICT:
            if event in ("intent", "order_submitted", "order_failed", "gate_skip") and text.startswith("💹"):
                text = text.replace("💹", "📡", 1)
            if event in ("fill", "close", "pnl") and text.startswith("📡"):
                text = text.replace("📡", "💹", 1)
        self._send(which, text)

    def emit_once(self, key: str, event: str, text: str, ttl_ms: int | None = None):
        ttl = ttl_ms or self.cfg.NOTIFY_THROTTLE_MS
        now = time.time() * 1000
        last = self._throttle.get(key, 0)
        if now - last < ttl:
            return
        self._throttle[key] = now
        self.emit(event, text)


    def send_once(self, key: str, text: str, to: str = "logs"):
        now = time.time() * 1000
        if now - self._throttle.get(key, 0) < self.cfg.NOTIFY_THROTTLE_MS:
            return
        self._throttle[key] = now
        if to == "signals":
            self.push_signal(text)
        elif to == "trades":
            self.push_trade(text)
        else:
            self.push_log(text)



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

# 원래의 Notifier 기반 구현은 제거되고, 아래의 단일 라우터 함수들로 대체된다.



import os, re, time, asyncio
from ftm2.notify import discord_bot

# 별칭 → 실제 대상(채널ID, '#이름', 별칭 그대로)

CHANNELS = {
    "signals": os.getenv("CHANNEL_SIGNALS", "signals"),
    "trades": os.getenv("CHANNEL_TRADES", "trades"),
    "logs": os.getenv("CHANNEL_LOGS", "logs"),
}

def configure_channels(**kw):
    """런타임에서 채널 매핑 갱신 (부팅 중에도 호출 안전)"""
    for k, v in kw.items():
        if v:
            CHANNELS[k] = v
    # 부팅 시점엔 루프가 없을 수 있으므로 emit은 그냥 큐에 쌓임
    emit("system", f"[NOTIFY_CHANNELS] {CHANNELS}")

def _resolve_channel(key_or_name: str):
    if not key_or_name:
        return CHANNELS.get("signals", "signals")
    k = str(key_or_name).strip()
    if k in CHANNELS:
        return CHANNELS[k]
    if k.startswith("#") or k.isdigit():
        return k
    for _, v in CHANNELS.items():
        if v == k:
            return v
    return CHANNELS.get("signals", "signals")


# ---------- 라우팅 맵(시그널 조용하게) ----------
ROUTE_MAP = {
    "intent": "logs",  # 의도만 → logs
    "gate_skip": "logs",  # 진입 금지 → logs
    "intent_cancel": "logs",  # 의도 취소/재시도 초과 → logs
    "order_submitted": "signals",  # 실제 주문만 시그널
    "order_failed": "logs",
    "fill": "trades",
    "close": "trades",
    "pnl": "trades",
    "system": "logs",
    "error": "logs",
    "chart": "logs",
}

# ---------- 스팸 억제 ----------
_LAST_EMIT: dict[str, int] = {}
EMIT_TTL = {
    "gate_skip": 60_000,
    "intent": 60_000,
    "intent_cancel": 60_000,
    "order_failed": 10_000,
    "system": 5_000,
    "error": 5_000,
    "chart": 30_000,
}

def _norm(kind: str, text: str) -> str:
    t = re.sub(r"\s+", " ", str(text)).strip()
    t = re.sub(r"@~\d+(\.\d+)?", "@~PX", t)
    t = re.sub(r"\d+(\.\d+)?", "N", t)
    return f"{kind}:{t}"

# ---------- 부팅 큐 ----------
_BOOT_QUEUE: list[tuple[str, str, str | None, int | None]] = []

async def send(channel_key_or_name: str, text: str):
    alias = None
    for k, v in CHANNELS.items():
        if v == channel_key_or_name or k == channel_key_or_name:
            alias = k
            break
    if alias == "trades":
        discord_bot.send_trade(text)
    elif alias == "signals":
        discord_bot.send_signal(text)
    else:
        discord_bot.send_log(text)
    return None

async def edit(message_id, text: str):
    return None

async def _send_impl(target_key_or_name: str, text: str):
    target = _resolve_channel(target_key_or_name)
    if 'send' in globals():
        return await send(target, text)
    if 'emit' in globals():  # DRY/no-op
        try:
            emit("system", f"[DRY][send->{target}] {text}", route="logs")
        except Exception:
            pass

    return None

async def _edit_impl(message_id, text: str):
    if 'edit' in globals():
        return await edit(message_id, text)
    if 'emit' in globals():
        try:
            emit("system", f"[DRY][edit->{message_id}] {text}", route="logs")
        except Exception:
            pass
    return None


async def _emit(kind: str, text: str, route: str | None, *, ttl_ms: int | None):
    route = route or ROUTE_MAP.get(kind, "logs")
    ttl = EMIT_TTL.get(kind, 0) if ttl_ms is None else max(0, int(ttl_ms))
    key = _norm(kind, text)
    now = int(time.time() * 1000)
    last = _LAST_EMIT.get(key, 0)
    if ttl and now - last < ttl:
        return None
    _LAST_EMIT[key] = now
    return await _send_impl(route, text)

def emit(kind: str, text: str, route: str | None = None, *, ttl_ms: int | None = None):
    """
    부팅 전(루프 없음)에도 안전. 루프가 없으면 부팅 큐에 저장,
    루프가 있으면 현재 루프에 task로 올림.
    """
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(_emit(kind, text, route, ttl_ms=ttl_ms))
    except RuntimeError:
        _BOOT_QUEUE.append((kind, text, route, ttl_ms))
        return None

async def flush_boot_queue():
    """루프가 돌기 시작하면 한 번 호출해서 부팅 큐를 비워 전송"""
    while _BOOT_QUEUE:
        kind, text, route, ttl = _BOOT_QUEUE.pop(0)
        try:
            await _emit(kind, text, route, ttl_ms=ttl)
        except Exception:
            pass

_ONCE_LAST_TS = {}  # key -> last_ts(ms)

def emit_once(key: str, kind: str, text: str, ttl_ms: int = 60000, route: str | None = None):
    """
    key     : 이벤트 식별자(예: 'ws_user_ok', 'ws_user_re', 'ws_mkt_ok' 등)
    kind    : 'system' | 'error' | 'intent' ... (ROUTE_MAP에 따라 채널 자동 라우팅)
    text    : 보낼 메시지
    ttl_ms  : 같은 key로 중복 전송 억제 기간(ms)
    route   : 강제 라우팅이 필요하면 지정(없으면 ROUTE_MAP 사용)
    """
    try:
        now = int(time.time() * 1000)
        last = _ONCE_LAST_TS.get(key, 0)
        if ttl_ms and now - last < int(ttl_ms):
            return None  # TTL 내면 무시(중복 방지)
        _ONCE_LAST_TS[key] = now
    except Exception:
        # 시간/캐시 에러가 나더라도 전송은 시도
        pass

    # 실제 전송은 부팅-안전 emit()에 위임 (루프 전이면 부팅큐에 쌓임)
    return emit(kind, text, route=route, ttl_ms=0)  # 이중 TTL 방지 위해 여기선 0

# ---------- dc 어댑터(항상 객체 보장) ----------

class _DCUseCtx:
    def __init__(self, target):
        self.target = target

    async def send(self, text):
        return await _send_impl(self.target, text)

    async def edit(self, mid, text):
        return await _edit_impl(mid, text)

class _DCAdapter:
    def use(self, target: str):
        return _DCUseCtx(target)

    async def send(self, target: str, text: str):
        return await _send_impl(target, text)

    async def edit(self, message_id, text: str):
        return await _edit_impl(message_id, text)

dc = _DCAdapter()



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



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
            "intent": "signals",
            "gate_skip": "signals",
            "order_submitted": "signals",
            "order_failed": "signals",
            "fill": "trades",
            "close": "trades",
            "pnl": "trades",
            "system": "logs",
            "error": "logs",
            "chart": "logs",
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
notifier = Notifier(_cfg, _DiscordAdapter(_cfg))

emit = notifier.emit
emit_once = notifier.emit_once
push_signal = notifier.push_signal
push_trade = notifier.push_trade
push_log = notifier.push_log
send_once = notifier.send_once

async def send(channel_key_or_name: str, text: str):
    """Bridge to actual send implementation."""
    notifier.dc.send(channel_key_or_name, text)

async def edit(message_id, text: str):
    return None

# ================== DISPATCHER DC ADAPTER (HARDENED) ==================
import os, asyncio

# 별칭 → 실제 대상(채널ID, '#이름', 별칭 그대로) 매핑
CHANNELS = {
    "signals": os.getenv("CHANNEL_SIGNALS", "signals"),
    "trades":  os.getenv("CHANNEL_TRADES",  "trades"),
    "logs":    os.getenv("CHANNEL_LOGS",    "logs"),
}

def configure_channels(**kw):
    """런타임에서 채널 매핑 갱신"""
    for k, v in kw.items():
        if v:
            CHANNELS[k] = v
    if 'emit' in globals():
        try:
            emit("system", f"[NOTIFY_CHANNELS] {CHANNELS}")
        except Exception:
            pass

def _resolve_channel(key_or_name: str):
    """
    'signals'(별칭) / '#포지션신호'(이름) / '123456789...' (ID) 모두 허용.
    매칭 실패 시 'signals'로 폴백.
    """
    if not key_or_name:
        return CHANNELS.get("signals", "signals")
    k = str(key_or_name).strip()
    if k in CHANNELS:                # 별칭
        return CHANNELS[k]
    if k.startswith("#"):            # 채널명
        return k
    if k.isdigit():                  # 채널ID
        return k
    for _, val in CHANNELS.items():  # 역탐색
        if val == k:
            return val
    return CHANNELS.get("signals", "signals")

async def _send_impl(channel_key_or_name: str, text: str):
    target = _resolve_channel(channel_key_or_name)
    if 'send' in globals():
        return await send(target, text)   # 프로젝트 내 실제 전송 함수명으로 연결됨
    # DRY/no-op fallback
    if 'emit' in globals():
        try:
            emit("system", f"[DRY][send->{target}] {text}")
        except Exception:
            pass
    return None

async def _edit_impl(message_id, text: str):
    if 'edit' in globals():
        return await edit(message_id, text)  # 실제 수정 함수명으로 연결
    if 'emit' in globals():
        try:
            emit("system", f"[DRY][edit->{message_id}] {text}")
        except Exception:
            pass
    return None

class _DCUseCtx:
    def __init__(self, channel_key_or_name):
        self.target = channel_key_or_name
    async def send(self, text: str):
        return await _send_impl(self.target, text)
    async def edit(self, message_id, text: str):
        return await _edit_impl(message_id, text)

class _DCAdapter:
    def use(self, channel_key_or_name: str):
        return _DCUseCtx(channel_key_or_name)
    async def send(self, channel_key_or_name: str, text: str):
        return await _send_impl(channel_key_or_name, text)
    async def edit(self, message_id, text: str):
        return await _edit_impl(message_id, text)

class _NoopDC:
    """최후방 안전망: dc가 None이어도 .use/.send/.edit가 존재하도록 보장"""
    def use(self, channel_key_or_name: str):
        return self
    async def send(self, *_a, **_k):
        if 'emit' in globals():
            try: emit("system", f"[NOOP][send] {_a} {_k}")
            except Exception: pass
        return None
    async def edit(self, *_a, **_k):
        if 'emit' in globals():
            try: emit("system", f"[NOOP][edit] {_a} {_k}")
            except Exception: pass
        return None

# 전역 dc를 항상 객체로 보장
dc = _DCAdapter()

def ensure_dc():
    """외부에서 보증 호출 가능(이미 객체면 그대로 둠)"""
    global dc
    if dc is None or not hasattr(dc, "send") or not hasattr(dc, "use"):
        dc = _DCAdapter()
    return dc
# =====================================================================


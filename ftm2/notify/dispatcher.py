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


# [ANCHOR:DISPATCHER_DC_ADAPTER_V2]
import asyncio

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
    if 'send' in globals():
        # 프로젝트의 실제 전송 함수명으로 맞추세요.
        return await send(target, text)
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

=
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


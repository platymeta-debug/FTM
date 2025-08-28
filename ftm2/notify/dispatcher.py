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

# [ANCHOR:DISPATCHER_DC_ADAPTER]
async def _send(channel_key: str, text: str):
    notifier._send(channel_key, text)

async def _edit(message_id, text: str):
    return None

class _DCAdapter:
    async def send(self, channel_key: str, text: str):
        return await _send(channel_key, text)

    async def edit(self, message_id, text: str):
        return await _edit(message_id, text)

dc = _DCAdapter()


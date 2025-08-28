import os
import time

from ftm2.config.settings import load_env_chain
from ftm2.notify.discord_bot import (
    send_trade as _send_trade,
    send_log as _send_log,
    send_signal as _send_signal,
)


class Notifier:
    """Dispatch notifications to pre-configured channels."""

    def __init__(self, cfg=None):
        self.cfg = cfg or load_env_chain()
        self.ch_signals = os.getenv("CHANNEL_SIGNALS", "#포지션신호")
        self.ch_trades = os.getenv("CHANNEL_TRADES", "#트레이딩")
        self.ch_logs = os.getenv("CHANNEL_LOGS", "#로그")
        self._throttle: dict[str, float] = {}

    def _send(self, channel: str, text: str, embed=None):
        if channel == self.ch_trades:
            _send_trade(text, embed)
        elif channel == self.ch_signals:
            _send_signal(text, embed)
        else:
            _send_log(text, embed)

    # 채널별 푸시 인터페이스 -------------------------------------------------
    def push_signal(self, text: str, embed=None):
        self._send(self.ch_signals, text, embed)

    def push_trade(self, text: str, embed=None):
        self._send(self.ch_trades, text, embed)

    def push_log(self, text: str, embed=None):
        self._send(self.ch_logs, text, embed)

    # [ANCHOR:NOTIFY_THROTTLE]
    def send_once(self, key: str, text: str, to: str = "logs", embed=None):
        now = time.time() * 1000
        last = self._throttle.get(key, 0)
        if now - last < self.cfg.NOTIFY_THROTTLE_MS:
            return
        self._throttle[key] = now
        if to == "signals":
            self.push_signal(text, embed)
        elif to == "trades":
            self.push_trade(text, embed)
        else:
            self.push_log(text, embed)


# 기본 인스턴스와 편의 함수 -------------------------------------------------
notifier = Notifier()

push_signal = notifier.push_signal
push_trade = notifier.push_trade
push_log = notifier.push_log
send_once = notifier.send_once


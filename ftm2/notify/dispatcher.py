import time
from ftm2.notify.discord_bot import send_trade as _send_trade, send_log as _send_log, send_signal as _send_signal

# [ANCHOR:NOTIFY_THROTTLE]
_throttle = {}

def send_trade(text: str, embed=None):
    _send_trade(text, embed)

def send_log(text: str, embed=None):
    _send_log(text, embed)

def send_signal(text: str, embed=None):
    _send_signal(text, embed)

def send_once(key: str, text: str, channel: str, ttl_ms: int, embed=None):
    now = time.time() * 1000
    last = _throttle.get(key, 0)
    if now - last < ttl_ms:
        return
    _throttle[key] = now
    if channel == "trades":
        send_trade(text, embed)
    else:
        send_log(text, embed)

# ==== discord_bot.py: 어댑터 API 보장 ====
"""Minimal Discord adapter.
실제 Discord 연동은 외부에서 주입되며, 채널 바인딩 실패 시에도 예외를 발생시키지 않는다.
"""

from __future__ import annotations
import time
from typing import Any, Dict, Optional
from types import SimpleNamespace

# 메시지 ID 시뮬레이션용 카운터
_LAST_MSG_ID = 0

# sticky 메시지 캐시
_STICKY: Dict[str, Dict[str, Any]] = {}

async def send(channel_key_or_name: int | str, text: str) -> int:
    """Send a message to a Discord channel.
    실제 전송 대신 콘솔에 [DRY] 로그를 남긴다.
    Returns a pseudo message id.
    """
    global _LAST_MSG_ID
    _LAST_MSG_ID += 1
    print(f"[DISCORD][send][DRY] {channel_key_or_name}: {text}")
    return _LAST_MSG_ID

async def edit(message_id: int, text: str) -> None:
    """Edit a previously sent message."""
    print(f"[DISCORD][edit][DRY] {message_id}: {text}")

async def upsert(
    channel_key_or_name: int | str,
    text: str,
    *,
    dedupe_ms: int = 3000,
    max_age_edit_s: int = 3300,
    sticky_key: Optional[str] = None,
):
    """Dedupe/edit-or-send helper.
    - If `sticky_key` is provided, we remember the last message per key and
      edit it if possible within ``max_age_edit_s``.
    - When text is unchanged within ``dedupe_ms`` the call is ignored.
    """
    now = int(time.time() * 1000)
    key = sticky_key
    if key:
        item = _STICKY.get(key)
        if item:
            if item.get("text") == text and (now - item.get("ts", 0)) < dedupe_ms:
                return {"ok": True, "deduped": True, "id": item.get("id")}
            # 편집 가능 시간 체크
            if (now - item.get("ts", 0)) < max_age_edit_s * 1000:
                try:
                    await edit(item["id"], text)
                    item.update({"text": text, "ts": now})
                    return {"ok": True, "edited": True, "id": item.get("id")}
                except Exception as e:
                    print("[DISCORD][edit][ERR]", e)
        # 새 메시지 발송
        mid = await send(channel_key_or_name, text)
        _STICKY[key] = {"id": mid, "text": text, "ts": now}
        return {"ok": True, "posted": True, "id": mid}
    else:
        # sticky key 없으면 단순 send
        mid = await send(channel_key_or_name, text)
        return {"ok": True, "posted": True, "id": mid}

async def edit_trade_card(symbol, tracker, cfg, force: bool = False):
    """Proxy to TradeCards helper to keep card in sync."""
    try:
        from .cards import TradeCards
    except Exception as e:
        print(f"[DISCORD][trade_card][ERR] import cards failed: {e}")
        return
    global _TRADE_CARDS
    try:
        _TRADE_CARDS
    except NameError:
        _TRADE_CARDS = None
    if _TRADE_CARDS is None:
        _TRADE_CARDS = TradeCards(cfg, dc := SimpleNamespace(send=send, edit=edit))
    snap = None
    if tracker and hasattr(tracker, "get_symbol_view"):
        snap = tracker.get_symbol_view(symbol)
    if snap is None:
        return
    try:
        await _TRADE_CARDS.upsert_trade_card(symbol, snap, None, [], force=force)
    except Exception as e:
        print(f"[DISCORD][trade_card][ERR] {e}")

# 과거 인터페이스 호환
def use(*args, **kwargs):
    return None

# 로그용 더미 함수들 (과거 코드 호환)
def send_log(text: str, embed=None):
    print(f"[DISCORD][log] {text}")

def send_trade(text: str, embed=None):
    print(f"[DISCORD][trade] {text}")

def send_signal(text: str, embed=None):
    print(f"[DISCORD][signal] {text}")

async def update_analysis(*args, **kwargs):
    return None

async def start_notifier(cfg):
    return None

def register_hooks(**kwargs):
    return None

def register_tracker(tracker):
    return None

__all__ = [
    "send",
    "edit",
    "upsert",
    "edit_trade_card",
    "use",
    "send_log",
    "send_trade",
    "send_signal",
    "update_analysis",
    "start_notifier",
    "register_hooks",
    "register_tracker",
]

# ==== discord_bot.py: 최소 인터페이스 ====
# 실제 라이브러리 호출 부분은 기존 구현을 사용하고, 이름만 맞게 래핑하세요.

# 이 모듈은 async이어야 합니다.
# 채널 ID(int) 또는 이름(str)과 text(str)를 받아 디스코드로 보내는 역할.

async def send(channel, text: str):
    # TODO: 실제 Discord SDK/HTTP 호출로 교체
    print(f"[DISCORD][send] {channel}: {text}")
    return {"ok": True}

async def edit(message_or_channel, text: str):
    # TODO: 실제 편집 구현
    print(f"[DISCORD][edit] {message_or_channel}: {text}")
    return {"ok": True}

_STICKY: dict[str, dict] = {}

async def upsert(channel, text: str, *, sticky_key: str, dedupe_ms: int = 2000, max_age_edit_s: int = 3300):
    import time
    now = int(time.time() * 1000)
    item = _STICKY.get(sticky_key)
    if item:
        # dedupe
        if item.get("text") == text and (now - item.get("ts", 0) < dedupe_ms):
            return {"ok": True, "deduped": True}
        # 편집(실패시 새 메시지 발행)

        try:
            await edit(item["channel"], text)
            item["text"] = text
            item["ts"] = now
            return {"ok": True, "edited": True}
        except Exception:
            pass

    # 신규 발행
    await send(channel, text)
    _STICKY[sticky_key] = {"channel": channel, "text": text, "ts": now}
    return {"ok": True, "posted": True}

def use(*args, **kwargs):
    # 필요 없으면 빈 함수로 두어도 됨(과거 인터페이스 호환용)
    return None

# 추가 호환 함수들
def send_log(text: str, embed=None):
    print(f"[DISCORD][log] {text}")

def send_trade(text: str, embed=None):
    print(f"[DISCORD][trade] {text}")

def send_signal(text: str, embed=None):
    print(f"[DISCORD][signal] {text}")

async def update_analysis(*args, **kwargs):
    return None

# 기존 코드와의 호환을 위한 더미 함수들
async def start_notifier(cfg):
    return None

def register_hooks(**kwargs):
    return None

def register_tracker(tracker):
    return None

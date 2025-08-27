# [ANCHOR:DISCORD_BOT]
import asyncio, os, traceback
from typing import Optional, Callable
import discord

_send_queue: "asyncio.Queue[tuple[str,str]]" = asyncio.Queue()
_cfg = None
_client: Optional[discord.Client] = None
_ch_logs = _ch_trades = _ch_signals = None


# 외부에서 주입할 훅(한국어 명령용)
_hooks = {
    "force_flat": None,   # def func(symbol:str) -> str
    "kill_switch": None,  # def func(on:bool) -> str
    "get_status": None,   # def func() -> str
    "get_signal": None,   # def func(symbol:str) -> str
}

def register_hooks(**kwargs):
    _hooks.update({k:v for k,v in kwargs.items() if k in _hooks})

# 동기 호출 가능: 내부 큐에 적재
def send_log(text: str):    _send_queue.put_nowait(("logs", text))
def send_trade(text: str):  _send_queue.put_nowait(("trades", text))
def send_signal(text: str): _send_queue.put_nowait(("signals", text))

async def _sender_loop():
    global _ch_logs, _ch_trades, _ch_signals
    while True:
        chan, text = await _send_queue.get()
        try:
            if chan=="logs" and _ch_logs:    await _ch_logs.send(f"🧩 {text}")
            elif chan=="trades" and _ch_trades: await _ch_trades.send(f"💹 {text}")
            elif chan=="signals" and _ch_signals: await _ch_signals.send(f"📡 {text}")
            else:
                print(f"[DISCORD][DRY] {chan}: {text}")  # 채널 미설정 시 콘솔로
        except Exception:
            traceback.print_exc()

async def _resolve_guild_and_channels(client: discord.Client):
    global _ch_logs, _ch_trades, _ch_signals
    gid = _cfg.DISCORD_GUILD_ID
    guild = client.get_guild(gid) if gid else None
    if not guild and gid:
        try:
            guild = await client.fetch_guild(gid)
        except discord.Forbidden:
            print(f"[DISCORD][ERR] 길드({gid}) 접근 권한 없음."); return False
        except discord.NotFound:
            print(f"[DISCORD][ERR] 길드({gid})를 찾을 수 없음."); return False
        except Exception as e:
            print(f"[DISCORD][ERR] 길드({gid}) 조회 실패: {e}"); return False
    if not guild:
        print("[DISCORD][ERR] 길드 ID 미설정 또는 미접근. .env.notify의 DISCORD_GUILD_ID 확인")
        return False

    async def bind_channel(cid: int, label: str):
        ch = client.get_channel(cid)
        if ch is None:
            try:
                ch = await client.fetch_channel(cid)
            except discord.Forbidden:
                print(f"[DISCORD][ERR] {label} 채널({cid}) 권한 없음."); return None
            except discord.NotFound:
                print(f"[DISCORD][ERR] {label} 채널({cid}) 없음(잘못된 ID)."); return None
            except Exception as e:
                print(f"[DISCORD][ERR] {label} 채널({cid}) 조회 실패: {e}"); return None
        return ch

    _ch_logs = await bind_channel(_cfg.DISCORD_CHANNEL_LOGS, "logs") if _cfg.DISCORD_CHANNEL_LOGS else None
    _ch_trades = await bind_channel(_cfg.DISCORD_CHANNEL_TRADES, "trades") if _cfg.DISCORD_CHANNEL_TRADES else None
    _ch_signals = await bind_channel(_cfg.DISCORD_CHANNEL_SIGNALS, "signals") if _cfg.DISCORD_CHANNEL_SIGNALS else None

    print(f"[DISCORD] 연결 완료. guild={guild.name} "
          f"logs={'OK' if _ch_logs else 'X'} trades={'OK' if _ch_trades else 'X'} signals={'OK' if _ch_signals else 'X'}")
    return True

async def _on_ready(client: discord.Client):
    ok = await _resolve_guild_and_channels(client)
    if ok and _cfg.DISCORD_TEST_ON_BOOT:
        # 이제 채널 바인딩이 끝났으니 반드시 채널로 전송됨
        send_log("FTM2 봇이 부팅되었습니다. (테스트 메시지)")
        send_log(f"환경: MODE={_cfg.MODE}, 심볼={_cfg.SYMBOLS}, 인터벌={_cfg.INTERVAL}, 프로파일={os.getenv('ENV_PROFILE','-')}")


async def _handle_message(msg: discord.Message):
    if msg.author.bot: return
    if not msg.content.startswith(_cfg.DISCORD_PREFIX): return
    args = msg.content[len(_cfg.DISCORD_PREFIX):].strip().split()
    if not args: return
    cmd = args[0]
    # --- 한국어 명령 ---
    if cmd == "상태":
        f = _hooks.get("get_status")
        text = f() if f else "상태 조회 훅이 아직 연결되지 않았습니다."
        await msg.channel.send(f"🧭 {text}")
    elif cmd == "전량청산" and len(args)>=2:
        sym = args[1].upper()
        f = _hooks.get("force_flat")
        text = f(sym) if f else f"{sym} 전량청산 훅 미연결"
        await msg.channel.send(f"🔻 {text}")
    elif cmd == "킬스위치" and len(args)>=2:
        on = args[1] in ("켜","on","ON","true","TRUE")
        f = _hooks.get("kill_switch")
        text = f(on) if f else "킬스위치 훅 미연결"
        await msg.channel.send(f"🛑 {text}")
    elif cmd == "신호" and len(args)>=2:
        sym = args[1].upper()
        f = _hooks.get("get_signal")
        text = f(sym) if f else f"{sym} 신호 훅 미연결"
        await msg.channel.send(f"📡 {text}")
    elif cmd == "로그테스트":
        send_log("이것은 로그 테스트입니다.")
        await msg.add_reaction("✅")
    elif cmd == "채널테스트":
        send_log("logs 채널 테스트 메시지")
        send_trade("trades 채널 테스트 메시지")
        send_signal("signals 채널 테스트 메시지")
        await msg.channel.send("📨 테스트 메시지 전송 시도 완료 (권한/바인딩 확인은 콘솔 로그 참조)")
    elif cmd == "디버그":
        await msg.channel.send(f"🔧 디버그: "
                               f"GUILD_ID={_cfg.DISCORD_GUILD_ID}, "
                               f"LOGS={_cfg.DISCORD_CHANNEL_LOGS}, "
                               f"TRADES={_cfg.DISCORD_CHANNEL_TRADES}, "
                               f"SIGNALS={_cfg.DISCORD_CHANNEL_SIGNALS}")
    else:
        await msg.channel.send("❓ 지원하지 않는 명령입니다. (상태, 전량청산 심볼, 킬스위치 켜|꺼, 신호 심볼, 로그테스트)")

async def start_notifier(cfg):
    """앱 루프 내에서 호출: 디스코드 클라이언트 + 송신 루프 실행"""
    global _cfg, _client
    _cfg = cfg
    # 송신 루프는 항상 가동 (토큰 없어도 콘솔 출력)
    asyncio.create_task(_sender_loop())

    token = (cfg.DISCORD_TOKEN or os.getenv("DISCORD_TOKEN") or "").strip()
    if not token:
        print("[DISCORD] 토큰 없음. 콘솔 로그만 출력합니다.")
        return
    print(f"[DISCORD] 토큰 감지: {token[:6]}… (길이={len(token)})")


    intents = discord.Intents.default()
    intents.message_content = True  # 포털에서 Message Content Intent 켜야 함
    _client = discord.Client(intents=intents)

    @_client.event
    async def on_ready():
        await _on_ready(_client)

    @_client.event
    async def on_message(message: discord.Message):
        await _handle_message(message)

    try:
        await _client.start(token)

    except Exception:
        traceback.print_exc()

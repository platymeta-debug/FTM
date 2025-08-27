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

async def _on_ready(client: discord.Client):
    global _ch_logs, _ch_trades, _ch_signals
    gid = _cfg.DISCORD_GUILD_ID
    guild = client.get_guild(gid) if gid else None
    if not guild:
        print("[DISCORD] 길드 ID가 없거나 접근 불가. 콘솔 모드로 동작합니다.")
        return
    if _cfg.DISCORD_CHANNEL_LOGS:
        _ch_logs = guild.get_channel(_cfg.DISCORD_CHANNEL_LOGS)
    if _cfg.DISCORD_CHANNEL_TRADES:
        _ch_trades = guild.get_channel(_cfg.DISCORD_CHANNEL_TRADES)
    if _cfg.DISCORD_CHANNEL_SIGNALS:
        _ch_signals = guild.get_channel(_cfg.DISCORD_CHANNEL_SIGNALS)
    print("[DISCORD] 연결 완료.")
    if _cfg.DISCORD_TEST_ON_BOOT:
        send_log("FTM2 봇이 부팅되었습니다. (테스트 메시지)")
        # 환경 요약
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
    else:
        await msg.channel.send("❓ 지원하지 않는 명령입니다. (상태, 전량청산 심볼, 킬스위치 켜|꺼, 신호 심볼, 로그테스트)")

async def start_notifier(cfg):
    """앱 루프 내에서 호출: 디스코드 클라이언트 + 송신 루프 실행"""
    global _cfg, _client
    _cfg = cfg
    # 송신 루프는 항상 가동 (토큰 없어도 콘솔 출력)
    asyncio.create_task(_sender_loop())
    if not cfg.DISCORD_TOKEN:
        print("[DISCORD] 토큰 없음. 콘솔 로그만 출력합니다.")
        return

    intents = discord.Intents.default()
    intents.message_content = True
    _client = discord.Client(intents=intents)

    @_client.event
    async def on_ready():
        await _on_ready(_client)

    @_client.event
    async def on_message(message: discord.Message):
        await _handle_message(message)

    try:
        await _client.start(cfg.DISCORD_TOKEN)
    except Exception:
        traceback.print_exc()

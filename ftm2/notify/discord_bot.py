# [ANCHOR:DISCORD_KR]
import asyncio, os

KR_PREFIX = "!"


def send_log(text: str):
    # logs 채널로 한국어 알림
    print("[LOG]", text)
    # TODO: discord.py로 채널 전송
    # await channel.send(f"🧩 {text}")


def send_trade_update(text: str):
    print("[TRADE]", text)
    # TODO: #trades 편집/전송


# 한국어 명령들
# !상태  !전량청산 BTCUSDT  !킬스위치 켜/꺼  !모드  !포지션  !신호 BTCUSDT
async def handle_command(message):
    content = message.content.strip()
    if not content.startswith(KR_PREFIX):
        return
    cmd = content[len(KR_PREFIX):].split()
    if not cmd:
        return
    if cmd[0] == "상태":
        send_log("상태: 연결 정상, 테스트넷, 지표/전략 활성")
    elif cmd[0] == "전량청산" and len(cmd) >= 2:
        sym = cmd[1].upper()
        # TODO: router 통해 시장가 reduceOnly 청산
        send_trade_update(f"🔻 전량 청산 지시: {sym}")
    elif cmd[0] == "킬스위치" and len(cmd) >= 2:
        on = cmd[1] in ("켜", "on", "ON")
        # TODO: guard.kill_switch = on
        send_log(f"🛑 킬스위치 {'켜짐' if on else '꺼짐'}")
    elif cmd[0] == "신호" and len(cmd) >= 2:
        sym = cmd[1].upper()
        # TODO: 최신 점수 계산 후 임베드 전송
        send_log(f"📡 {sym} 최신 신호 조회")
    # ... 필요 명령 추가


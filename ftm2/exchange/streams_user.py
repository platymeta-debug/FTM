
import asyncio, json, websockets, contextlib

from ftm2.config.settings import load_env_chain
from ftm2.exchange.binance_client import BinanceClient

CFG = load_env_chain()

WS_BASE = "wss://fstream.binance.com" if CFG.MODE == "live" else "wss://fstream.binancefuture.com"  # docs


async def user_stream(on_event):
    bx = BinanceClient()
    listen_key = bx.create_listen_key()
    print(f"[USER_WS] listenKey={listen_key[:8]}… created")
    url = f"{WS_BASE}/ws/{listen_key}"

    async def keepalive_task():
        while True:
            await asyncio.sleep(1800)  # 30분마다 연장 (유효기간 60분)
            bx.keepalive_listen_key(listen_key)
            print("[USER_WS] keepalive sent")

    ka = asyncio.create_task(keepalive_task())
    try:
        print(f"[USER_WS] connecting → {url}")
        async with websockets.connect(url, ping_interval=150) as ws:
            print("[USER_WS] connected")
            first = True
            async for raw in ws:
                data = json.loads(raw)
                if first:
                    et = data.get("e") or data.get("eventType") or "unknown"
                    print(f"[USER_WS] first event: {et}")
                    first = False
                await on_event(data)
    finally:
        ka.cancel()
        with contextlib.suppress(Exception):
            bx.delete_listen_key(listen_key)
            print("[USER_WS] listenKey deleted")




import asyncio, json, websockets
from ftm2.config.settings import load_env_chain
CFG = load_env_chain()
WS_BASE = "wss://fstream.binance.com" if CFG.MODE == "live" else "wss://fstream.binancefuture.com"


def kline_stream(symbol: str, interval: str) -> str:
    return f"{symbol.lower()}@kline_{interval}"


def mark_stream(symbol: str) -> str:
    return f"{symbol.lower()}@markPrice@1s"



async def market_stream(symbols, interval, on_msg):
    names = [kline_stream(s, interval) for s in symbols] + [mark_stream(s) for s in symbols]

    url = f"{WS_BASE}/stream?streams={'/'.join(names)}"
    print(f"[MKT_WS] connecting → {url}")
    async with websockets.connect(url, ping_interval=150) as ws:
        print("[MKT_WS] connected")
        first = True
        async for raw in ws:
            data = json.loads(raw)
            if first:
                stream = data.get("stream")
                print(f"[MKT_WS] first msg on {stream}")
                first = False
            await on_msg(data)


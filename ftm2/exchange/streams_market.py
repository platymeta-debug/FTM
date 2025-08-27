
import asyncio, json, websockets
from ftm2.config.settings import load_env_chain
from ftm2.notify.discord_bot import edit_trade_card
from ftm2.trade.position_tracker import PositionTracker
CFG = load_env_chain()
WS_BASE = "wss://fstream.binance.com" if CFG.MODE == "live" else "wss://fstream.binancefuture.com"

TRACKER: PositionTracker | None = None

async def on_mark_price(symbol: str, mark: float, cfg):
    if not TRACKER: return
    for side in ("LONG","SHORT"):
        TRACKER.update_mark(symbol, side, mark)
    await edit_trade_card(symbol, TRACKER, cfg, force=False)


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


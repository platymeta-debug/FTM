
import asyncio, json, websockets
from ftm2.config.settings import load_env_chain
from ftm2.notify.discord_bot import edit_trade_card
from ftm2.trade.position_tracker import PositionTracker
from ftm2.analysis.divergence import DivergenceMonitor

CFG = load_env_chain()
WS_LIVE = "wss://fstream.binance.com"
WS_TEST = "wss://fstream.binancefuture.com"


TRACKER_REF: PositionTracker | None = None
CSV = None
LEDGER = None
DIVERGENCE: DivergenceMonitor | None = None

async def on_mark_live(symbol: str, mark: float, cfg):
    global TRACKER_REF, DIVERGENCE
    if DIVERGENCE:
        DIVERGENCE.update_live(symbol, mark)
    if not TRACKER_REF:
        return
    for side in ("LONG","SHORT"):
        TRACKER_REF.update_mark(symbol, side, mark)
    await edit_trade_card(symbol, TRACKER_REF, cfg, force=False)

def on_mark_test(symbol: str, mark: float):
    global DIVERGENCE
    if DIVERGENCE:
        DIVERGENCE.update_test(symbol, mark)



def kline_stream(symbol: str, interval: str) -> str:
    return f"{symbol.lower()}@kline_{interval}"


def mark_stream(symbol: str) -> str:
    return f"{symbol.lower()}@markPrice@1s"



async def _test_mark_loop(symbols):
    names = [mark_stream(s) for s in symbols]
    url = f"{WS_TEST}/stream?streams={'/'.join(names)}"
    print(f"[MKT_WS][TEST] connecting → {url}")
    async with websockets.connect(url, ping_interval=150) as ws:
        async for raw in ws:
            data = json.loads(raw)
            stream = data.get("stream", "")
            sym = stream.split("@")[0].upper()
            mark = float(data.get("data", {}).get("p", 0) or 0)
            on_mark_test(sym, mark)

async def market_stream(symbols, interval, on_msg):
    names = [kline_stream(s, interval) for s in symbols] + [mark_stream(s) for s in symbols]
    base = WS_LIVE if CFG.DATA_FEED == "live" else WS_TEST
    url = f"{base}/stream?streams={'/'.join(names)}"
    print(f"[MKT_WS] connecting → {url}")
    test_task = None
    if CFG.DATA_FEED == "live" and DIVERGENCE:
        test_task = asyncio.create_task(_test_mark_loop(symbols))
    async with websockets.connect(url, ping_interval=150) as ws:
        print("[MKT_WS] connected")
        first = True
        try:
            async for raw in ws:
                data = json.loads(raw)
                if first:
                    stream = data.get("stream")
                    print(f"[MKT_WS] first msg on {stream}")
                    first = False
                await on_msg(data)
        finally:
            if test_task:
                test_task.cancel()


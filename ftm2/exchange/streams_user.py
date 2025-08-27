import asyncio
import json
import websockets

from ..config.settings import load_env_chain
from .binance_client import BinanceClient

CFG = load_env_chain()
WS_BASE = (
    "wss://fstream.binance.com" if CFG.MODE == "live" else "wss://fstream.binancefuture.com"
)


async def user_stream(client: BinanceClient, on_msg):
    """Connect to user data stream and forward messages to callback."""
    listen_key = client.account.new_listen_key().to_dict()["listenKey"]
    url = f"{WS_BASE}/ws/{listen_key}"

    async def keepalive():
        while True:
            await asyncio.sleep(30 * 60)
            client.account.keepalive_listen_key(listen_key=listen_key)

    asyncio.create_task(keepalive())

    async with websockets.connect(url, ping_interval=300) as ws:
        try:
            async for raw in ws:
                data = json.loads(raw)
                await on_msg(data)
        finally:
            client.account.close_listen_key(listen_key=listen_key)

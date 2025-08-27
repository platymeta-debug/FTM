import asyncio

from config.settings import load_env_chain
from exchange.binance_client import BinanceClient
from exchange.streams_market import run_market_stream

CFG = load_env_chain()


async def main():
    print(f"[FTM2] MODE={CFG.MODE}")
    bx = BinanceClient()
    bx.sync_time()
    bx.load_exchange_info()

    async def on_msg(data):
        pass  # Placeholder for processing market data

    await run_market_stream(CFG.SYMBOLS, CFG.INTERVAL, on_msg)


if __name__ == "__main__":
    asyncio.run(main())

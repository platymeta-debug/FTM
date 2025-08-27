import asyncio
from config.settings import load_env_chain
from exchange.binance_client import BinanceClient
from exchange.streams_market import market_stream
from exchange.streams_user import user_stream

CFG = load_env_chain()


async def on_market(msg):
    # TODO: kline/markPrice 라우팅 → 인디케이터 파이프라인으로 전달
    if "stream" in msg and "data" in msg:
        pass


async def on_user(msg):
    # TODO: 주문/체결/포지션 이벤트 라우팅 → position_tracker/discord
    pass


async def main():
    print(f"[FTM2][BOOT_ENV_SUMMARY] MODE={CFG.MODE}, SYMBOLS={CFG.SYMBOLS}, INTERVAL={CFG.INTERVAL}")
    bx = BinanceClient()
    t = bx.server_time()
    print(f"[FTM2] serverTime={t.get('serverTime')} REST_BASE OK")
    info = bx.load_exchange_info()
    print(f"[FTM2] exchangeInfo symbols={len(info.get('symbols', []))} FILTERS OK")

    # 동시에 WS 시작
    tasks = [
        asyncio.create_task(market_stream(CFG.SYMBOLS, CFG.INTERVAL, on_market)),
        asyncio.create_task(user_stream(on_user)),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())


from datetime import datetime, timezone
import pandas as pd
import asyncio

from ftm2.analysis.state import AnalysisSnapshot

BOOT_LIMIT = 500  # 초기 캔들 개수 (필요시 .env로 빼도 됨)


def _klines_to_df(klines):
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "qav",
        "n_trades",
        "taker_base",
        "taker_quote",
        "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["ts"] = df["close_time"] / 1000.0
    return df[["ts", "open", "high", "low", "close", "volume"]]


async def bootstrap_candles(bx, symbol: str, intervals: list[str]):
    out = {}
    for iv in intervals:
        r = await asyncio.to_thread(
            bx.market_rest,
            "GET",
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": iv, "limit": BOOT_LIMIT},
        )
        df = _klines_to_df(r.json())
        out[iv] = df
    return out


def _need_more(df: pd.DataFrame, min_len: int = 200) -> bool:
    return (df is None) or (len(df) < min_len)


async def run_analysis_loop(cfg, symbols, market_cache, divergence, on_snapshot):
    """
    cfg.ANALYZE_INTERVAL_S 마다 스냅샷 생성.
    market_cache: 심볼/TF별 df를 저장/업데이트하는 dict 같은 간단 캐시 객체면 충분.
    """
    tfs = [s.strip() for s in cfg.ANALYSIS_TF.split(",")]
    from ftm2.exchange.binance_client import BinanceClient

    bx = BinanceClient()

    # 1) 처음에 캔들 부트스트랩
    for sym in symbols:
        market_cache.setdefault(sym, {})
        for iv in tfs:
            if _need_more(market_cache[sym].get(iv)):
                boot = await bootstrap_candles(bx, sym, [iv])
                market_cache[sym][iv] = boot[iv]
                print(f"[ANALYSIS][BOOT] {sym} {iv} bars={len(boot[iv])}")

    # 2) 루프: 매 주기 스냅샷
    while True:
        started = datetime.now(timezone.utc).timestamp()
        for sym in symbols:
            from ftm2.indicators import add_indicators

            iv_main = tfs[0]
            df = market_cache[sym][iv_main].copy()

            # 안전장치: 데이터가 너무 적으면 스킵하지 말고 재부트
            if _need_more(df, 100):
                boot = await bootstrap_candles(bx, sym, tfs)
                for iv in tfs:
                    market_cache[sym][iv] = boot[iv]
                df = market_cache[sym][iv_main].copy()
                print(f"[ANALYSIS][REBOOT] {sym} bars={len(df)}")

            df = add_indicators(df)

            from ftm2.strategy.score import score_snapshot as _score_snap

            snap = _score_snap(sym, df, market_cache[sym], tfs)

            bps = divergence.get_bps(sym)
            snap.rules["divergence_bps"] = bps

            await on_snapshot(sym, snap)

        elapsed = datetime.now(timezone.utc).timestamp() - started
        wait = max(1, cfg.ANALYZE_INTERVAL_S - int(elapsed))
        await asyncio.sleep(wait)


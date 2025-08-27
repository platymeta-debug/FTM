from datetime import datetime, timezone
import pandas as pd
import asyncio

from ftm2.strategy.score import score_snapshot, _parse_tf_weights

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
    """심볼별 멀티타임프레임 분석 루프."""
    tfs = [s.strip() for s in cfg.ANALYSIS_TF.split(",") if s.strip()]
    tf_weights = _parse_tf_weights(getattr(cfg, "TF_WEIGHTS", ""))
    from ftm2.exchange.binance_client import BinanceClient

    bx = BinanceClient()

    # [ANCHOR:M6_ENGINE_BOOT] 최초 캔들 부트스트랩
    for sym in symbols:
        cache = market_cache.setdefault(sym, {})
        for tf in tfs:
            df = cache.get(tf)
            if _need_more(df):
                boot = await bootstrap_candles(bx, sym, [tf])
                cache[tf] = boot[tf]
                print(f"[ANALYSIS][BOOT] {sym} {tf} bars={len(boot[tf])}")

    # [ANCHOR:M6_ENGINE_LOOP] 분석 루프
    while True:
        started = datetime.now(timezone.utc).timestamp()
        for sym in symbols:
            cache = market_cache.setdefault(sym, {})
            for tf in tfs:
                if _need_more(cache.get(tf), 100):
                    boot = await bootstrap_candles(bx, sym, [tf])
                    cache[tf] = boot[tf]
            snap = score_snapshot(sym, cache, tfs, tf_weights)
            bps = divergence.get_bps(sym)
            snap.rules["divergence_bps"] = bps
            await on_snapshot(sym, snap)

        elapsed = datetime.now(timezone.utc).timestamp() - started
        wait = max(1, cfg.ANALYZE_INTERVAL_S - int(elapsed))
        await asyncio.sleep(wait)


# [ANCHOR:APP_MARKET_PIPELINE]
import asyncio, os
import pandas as pd
from collections import defaultdict
from ftm2.indicators.core import add_indicators
from ftm2.strategy.scorer import score_row
from datetime import timezone


from ftm2.config.settings import load_env_chain
from ftm2.exchange.binance_client import BinanceClient
from ftm2.exchange.streams_market import market_stream
from ftm2.exchange.streams_user import user_stream
from ftm2.trade.position_sizer import sizing_decision
from ftm2.trade.order_router import OrderRouter
from ftm2.risk.guardrails import GuardRails


CFG = load_env_chain()

BUFFERS: dict[str, pd.DataFrame] = {}
ROUTER: OrderRouter | None = None
GUARD: GuardRails | None = None
BX: BinanceClient | None = None



def _append_kline(symbol, k):
    # close-only bar on close (x=True)
    ts = pd.to_datetime(int(k["T"]), unit="ms", utc=True)
    row = {
        "open":  float(k["o"]),
        "high":  float(k["h"]),
        "low":   float(k["l"]),
        "close": float(k["c"]),
        "volume":float(k["v"]),
    }
    df = BUFFERS.get(symbol)
    if df is None:
        df = pd.DataFrame(columns=["open","high","low","close","volume"])
    df.loc[ts] = row
    # tail keep
    df = df.iloc[-CFG.LOOKBACK:]
    BUFFERS[symbol] = df
    return df

def _mtf_context(df_1m: pd.DataFrame, cfg):
    """
    1m OHLCV DataFrame(UTC index) -> higher TF contexts dict
    keys from cfg.MTF_HIGHERS e.g. ["5m","15m"]
    returns {tf: {close, ema_fast, ema_slow, senkou_a, senkou_b}}
    """
    out = {}
    if not cfg.MTF_USE or df_1m is None or df_1m.empty:
        return out
    base = df_1m.copy()
    for tf in cfg.MTF_HIGHERS:
        try:
            r = base.resample(tf, label="right", closed="right").agg({
                "open":"first","high":"max","low":"min","close":"last","volume":"sum"
            }).dropna()
            if len(r) < max(cfg.EMA_SLOW, cfg.ICHI_KIJUN)+2:
                out[tf] = None; continue
            feat = add_indicators(r, cfg)
            last = feat.iloc[-1]
            out[tf] = dict(
                close=float(last["close"]),
                ema_fast=float(last.get("ema_fast", float("nan"))),
                ema_slow=float(last.get("ema_slow", float("nan"))),
                senkou_a=float(last.get("senkou_a", float("nan"))),
                senkou_b=float(last.get("senkou_b", float("nan"))),
            )
        except Exception:
            out[tf] = None
    return out

async def on_market(msg):
    if "stream" not in msg: return
    st = msg["stream"]
    data = msg.get("data", {})
    if st.endswith("kline_"+CFG.INTERVAL):
        k = data.get("k", {})
        sym = k.get("s")
        if not k.get("x"):  # only on closed candle
            return
        df = _append_kline(sym, k)
        if len(df) < max(CFG.EMA_SLOW, CFG.EMA_TREND, CFG.BB_LEN, CFG.ATR_LEN, CFG.ADX_LEN, CFG.DONCHIAN_LEN, CFG.ICHI_SENKOUB)+2:
            return
        feat = add_indicators(df, CFG)
        # prevs for slope-like features
        feat['macd_hist_prev'] = feat['macd_hist'].shift(1)
        feat['kama_prev'] = feat['kama'].shift(1)
        last = feat.iloc[-1].to_dict()
        mtf_ctx = _mtf_context(df, CFG)
        L,S,is_trend = score_row(last, CFG, mtf_ctx)
        print(f"[SCORE][{sym}] close={last['close']:.2f} | Long={L} Short={S} | ADX={last['adx']:.1f} RSI={last['rsi']:.1f} Z={last['z']:.2f} MTF={mtf_ctx and 'ON' or 'OFF'} TREND={is_trend}")
        # --- 라우팅: 임계값/대칭/쿨다운/데일리컷 체크 후 진입 ---
        if ROUTER and GUARD and BX:
            side = "LONG" if L>=S else "SHORT"
            sc = L if side=="LONG" else S
            sc_opp = S if side=="LONG" else L
            if sc >= CFG.ENTRY_SCORE and sc_opp <= CFG.OPPOSITE_MAX and GUARD.cooldown_ok(sym) and GUARD.daily_ok():
                dec = sizing_decision(sym, side, L, S, last['close'], last['atr'], BX.filters,
                                      pos_state=None, cfg=CFG, is_trend=is_trend,
                                      mtf_bias=(1 if (mtf_ctx and True) else 0))
                if dec and dec.qty>0:
                    ROUTER.place_entry(sym, dec, mark_price=last['close'])
                    GUARD.arm_cooldown(sym)


async def on_user(msg):
    # TODO: 주문/체결/포지션 이벤트 라우팅 → position_tracker/discord
    pass


async def main():
    print(f"[FTM2][BOOT_ENV_SUMMARY] MODE={CFG.MODE}, SYMBOLS={CFG.SYMBOLS}, INTERVAL={CFG.INTERVAL}")
    print(f"[FTM2] APIKEY={(CFG.BINANCE_API_KEY[:4] + '…') if CFG.BINANCE_API_KEY else 'EMPTY'}")
    bx = BinanceClient()
    t = bx.server_time()
    print(f"[FTM2] serverTime={t.get('serverTime')} REST_BASE OK")
    info = bx.load_exchange_info()
    print(f"[FTM2] exchangeInfo symbols={len(info.get('symbols', []))} FILTERS OK")
    # 라우터/가드 초기화
    global ROUTER, GUARD, BX
    ROUTER = OrderRouter(CFG, bx.filters)
    GUARD = GuardRails(CFG)
    BX = bx

    # 동시에 WS 시작
    tasks = [
        asyncio.create_task(market_stream(CFG.SYMBOLS, CFG.INTERVAL, on_market)),
        asyncio.create_task(user_stream(on_user)),
    ]
    smoke = int(os.getenv("SMOKE_SECONDS", "0"))
    if smoke > 0:
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=smoke)
        except asyncio.TimeoutError:
            print("[SMOKE] timed out; exiting")
    else:
        await asyncio.gather(*tasks)



if __name__ == "__main__":
    asyncio.run(main())


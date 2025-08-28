# [ANCHOR:APP_MARKET_PIPELINE]
import asyncio, os
import pandas as pd
from collections import defaultdict
from ftm2.indicators.core import add_indicators
from ftm2.strategy.scorer import score_row
from datetime import timezone
import asyncio
import pandas as pd

from ftm2.config.settings import load_env_chain
from ftm2.exchange.binance_client import BinanceClient
from ftm2.exchange import streams_market
from ftm2.exchange.streams_user import user_stream
from ftm2.trade.position_sizer import sizing_decision
from ftm2.trade.order_router import OrderRouter, log_decision
from ftm2.trade.bracket import Bracket
from ftm2.risk.guardrails import GuardRails
from ftm2.trade.position_tracker import PositionTracker
from ftm2.reconcile.reconciler import resync_loop
from ftm2.notify.discord_bot import start_notifier, register_hooks, register_tracker
from ftm2.trade import order_router
from ftm2.indicators.all import add_indicators
from ftm2.strategy.scorer import score_row
from ftm2.storage.csv_logger import CsvLogger
from ftm2.risk.ledger import DailyLedger, LossCutController
from ftm2.reconcile.income_poll import income_poll_loop
from ftm2.analysis.divergence import DivergenceMonitor
from ftm2.analysis.engine import run_analysis_loop
from ftm2.notify.analysis_views import build_analysis_embed  # 내부에서 사용
from ftm2.charts.janitor import run_chart_janitor
from ftm2.trade.intent_queue import IntentQueue
from ftm2.analysis.adapter import to_analysis_snapshot
from ftm2.storage.analysis_persistence import load_analysis_cards, save_analysis_cards
from ftm2.strategy.trace import DecisionTrace

# 전역 주입 포인트(간단)
from ftm2.notify import discord_bot as DB
from ftm2.exchange import streams_user as US
from ftm2.exchange import streams_market as MS
from ftm2 import strategy as ST



CFG = load_env_chain()

BUFFERS: dict[str, pd.DataFrame] = {}
ROUTER: OrderRouter | None = None
GUARD: GuardRails | None = None
BX: BinanceClient | None = None
CSV = None
LEDGER = None
div: DivergenceMonitor | None = None
INTQ: IntentQueue | None = None



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
            side = "LONG" if L >= S else "SHORT"
            sc = L if side == "LONG" else S
            sc_opp = S if side == "LONG" else L
            trace = DecisionTrace(symbol=sym, decision_score=sc if side == "LONG" else -sc,
                                  total_score=sc, direction=side)
            cd_ok = GUARD.cooldown_ok(sym)
            daily_ok = GUARD.daily_ok()
            trace.gates.update({
                "ENTRY_SCORE": CFG.ENTRY_SCORE,
                "OPPOSITE_MAX": CFG.OPPOSITE_MAX,
                "abs(score)": sc,
                "opp_score": sc_opp,
                "cooldown_ok": cd_ok,
                "daily_ok": daily_ok,
            })
            if sc >= CFG.ENTRY_SCORE and sc_opp <= CFG.OPPOSITE_MAX and cd_ok and daily_ok:
                dec = sizing_decision(
                    sym,
                    side,
                    L,
                    S,
                    last['close'],
                    last['atr'],
                    BX.filters,
                    pos_state=None,
                    cfg=CFG,
                    is_trend=is_trend,
                    mtf_bias=(1 if (mtf_ctx and True) else 0),
                )
                if dec and dec.qty > 0:
                    ROUTER.place_entry(sym, dec, mark_price=last['close'], trace=trace)
                    GUARD.arm_cooldown(sym)
                else:
                    trace.reasons.append("no sizing")
                    log_decision(trace)
            else:
                if sc < CFG.ENTRY_SCORE:
                    trace.reasons.append("below entry score")
                if sc_opp > CFG.OPPOSITE_MAX:
                    trace.reasons.append("opp above max")
                if not cd_ok:
                    trace.reasons.append("in cooldown")
                if not daily_ok:
                    trace.reasons.append("daily loss lock")
                log_decision(trace)
    elif st.endswith("@markPrice@1s"):
        sym = data.get("s")
        mark = float(data.get("p", 0) or 0)
        if CFG.DATA_FEED == "live":
            await streams_market.on_mark_live(sym, mark, CFG)
        else:
            streams_market.on_mark_test(sym, mark)


async def main():
    print(f"[FTM2][BOOT_ENV_SUMMARY] MODE={CFG.MODE}, SYMBOLS={CFG.SYMBOLS}, INTERVAL={CFG.INTERVAL}")
    print(f"[FTM2] APIKEY={(CFG.BINANCE_API_KEY[:4] + '…') if CFG.BINANCE_API_KEY else 'EMPTY'}")
    bx = BinanceClient()
    t = bx.server_time()
    print(f"[FTM2] serverTime={t.get('serverTime')} REST_BASE OK")
    info = bx.load_exchange_info()
    print(f"[FTM2] exchangeInfo symbols={len(info.get('symbols', []))} FILTERS OK")

    # 라우터/가드/트래커 초기화
    global ROUTER, GUARD, BX, CSV, LEDGER, div, INTQ
    brkt = Bracket(CFG, bx, bx.filters)
    ROUTER = OrderRouter(CFG, bx.filters, bracket=brkt)
    GUARD = GuardRails(CFG)
    BX = bx
    tracker = PositionTracker()
    register_tracker(tracker)
    register_hooks(
        close_all=lambda sym: order_router.close_position_all(sym),
        get_status=lambda: f"심볼={CFG.SYMBOLS}, 모드={CFG.MODE}",
    )
    streams_market.TRACKER_REF = tracker

    CSV = CsvLogger(CFG)
    await CSV.start()
    LEDGER = DailyLedger(CFG, CSV)

    DB.CSV = CSV; DB.LEDGER = LEDGER
    US.CSV = CSV; US.LEDGER = LEDGER
    MS.CSV = CSV; MS.LEDGER = LEDGER
    ST.CSV = CSV

    div = DivergenceMonitor(CFG.MAX_DIVERGENCE_BPS)
    MS.DIVERGENCE = div

    def _notify(text):
        try:
            DB.send_log(text)
        except Exception:
            print("[NOTIFY_FALLBACK]", text)

    async def _close_all(sym):
        from ftm2.trade import order_router as OR
        return await OR.close_position_all(sym)

    from ftm2.notify import dispatcher as NOTIFY
    INTQ = IntentQueue(CFG, div, ROUTER, CSV, NOTIFY)
    NOTIFY.emit("system", f"[NOTIFY_MAP] {NOTIFY.notifier.route}")
    NOTIFY.emit("intent", "📡 [테스트] 신호 채널 확인")
    NOTIFY.emit("fill", "💹 [테스트] 트레이드 채널 확인")

    LC = LossCutController(CFG, LEDGER, tracker, router=type("R",(),{"close_all":_close_all}), notify=_notify, csv_logger=CSV)

    # 동시에 WS 시작
    tasks = [
        asyncio.create_task(start_notifier(CFG)),
        asyncio.create_task(streams_market.market_stream(CFG.SYMBOLS, CFG.INTERVAL, on_market)),
        asyncio.create_task(user_stream(bx, tracker, CFG)),
        asyncio.create_task(resync_loop(bx, tracker, CFG, CFG.SYMBOLS)),
    ]
    tasks.append(asyncio.create_task(income_poll_loop(bx, LEDGER, CSV, CFG)))

    market_cache = {}

    async def on_snapshot(sym, snap_raw):
        snap = to_analysis_snapshot(sym, snap_raw)
        view = {
            "symbol": snap.symbol,
            "decision_score": snap.total_score,
            "total_score": snap.total_score,
            "direction": snap.direction,
            "confidence": snap.confidence,
            "tf_scores": snap.scores,
            "contribs": {},
        }
        contribs_map = getattr(snap_raw, "contribs", {}) or {}
        for tf, cons in contribs_map.items():
            view["contribs"][tf] = [
                {"name": getattr(c, "name", ""), "score": getattr(c, "score", 0.0), "text": getattr(c, "text", "")}
                for c in cons
            ]

        await DB.update_analysis(
            sym,
            snap,
            div.get_bps(sym),
            CFG.ANALYZE_INTERVAL_S,
            view=view,
        )

        INTQ.on_snapshot(snap)
        CSV.log(
            "ANALYSIS_SNAPSHOT",
            symbol=sym,
            data_feed=CFG.DATA_FEED,
            trade_mode=CFG.TRADE_MODE,
            divergence_bps=div.get_bps(sym),
            analysis={"tfs": snap.tfs, "confidence": snap.confidence},
            rule=snap.rules,
            plan=snap.plan,
            score_total=snap.total_score,
            scores=snap.scores,
            mtf=snap.mtf_summary,
            trend_state=snap.trend_state,
        )

    tasks.append(asyncio.create_task(run_analysis_loop(CFG, CFG.SYMBOLS, market_cache, div, on_snapshot)))
    tasks.append(asyncio.create_task(INTQ.run()))
    tasks.append(asyncio.create_task(run_chart_janitor(CFG)))

    async def snapshot_loop():
        while True:
            if CFG.CSV_MARK_SNAPSHOT_SEC and any(p.qty!=0 for p in tracker.pos.values()):
                for k, ps in tracker.pos.items():
                    if ps.qty==0: continue
                    CSV.log("SNAPSHOT", symbol=ps.symbol, side=ps.side, entry=ps.entry_price,
                            mark=ps.mark_price, upnl=ps.upnl, roe=ps.roe,
                            wallet=tracker.account.wallet_balance, equity=tracker.account.equity, avail=tracker.account.available_balance)
            await asyncio.sleep(max(5, int(CFG.CSV_MARK_SNAPSHOT_SEC) or 60))
    tasks.append(asyncio.create_task(snapshot_loop()))

    async def ledger_guard_loop():
        while True:
            LEDGER.rollover_if_needed(tracker.account.equity)
            LEDGER.on_equity_tick(tracker.account.equity)
            await LC.check_and_fire()
            await asyncio.sleep(5)
    tasks.append(asyncio.create_task(ledger_guard_loop()))
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


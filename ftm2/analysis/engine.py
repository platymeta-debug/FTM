from datetime import datetime, timezone
import pandas as pd
import asyncio

STATE: dict[str, str] = {}

from ftm2.strategy.score import score_snapshot, _parse_tf_weights
from time import time
from ftm2.analysis.types import SetupTicket


class AnalysisEngine:
    def __init__(self, cfg, market=None):
        self.cfg = cfg
        self.market = market
        self.snapshots = {}
        self.scoring = None

    def trend(self, sym: str, tf: str) -> str:
        return "FLAT"

    def atr(self, sym: str, tf: str) -> float:
        return 0.0

    def mark(self, sym: str) -> float:
        if self.market and hasattr(self.market, "mark"):
            return self.market.mark(sym)
        return 0.0

    def series(self, sym: str, tf: str, key: str):
        """Placeholder for price/indicator series lookup."""
        return []

    # [BUILD_TICKET]
    def build_ticket(self, sym: str, score: int, confidence: float | None = None, regime: str | None = None):

        side = None
        reasons = []
        if (
            score >= self.cfg.LONG_MIN_SCORE
            and self.trend(sym, "1h") == "UP"
            and self.trend(sym, "4h") == "UP"
        ):
            side = "LONG"
            reasons.append("score>=LONG_MIN & HTF=UP")
        elif (
            score <= -self.cfg.SHORT_MIN_SCORE
            and self.trend(sym, "1h") == "DOWN"
            and self.trend(sym, "4h") == "DOWN"
        ):
            side = "SHORT"
            reasons.append("score<=-SHORT_MIN & HTF=DOWN")
        else:
            return None

        from ftm2.analysis.divergence import rsi_bear_div, rsi_bull_div  # [ANCHOR:DIVERGENCE_FILTER]
        hi = self.series(sym, self.cfg.ENTRY_TF, "high")
        lo = self.series(sym, self.cfg.ENTRY_TF, "low")
        rsi = self.series(sym, self.cfg.ENTRY_TF, "rsi")
        if self.cfg.DIV_FILTER:
            if side == "LONG" and rsi_bear_div(hi, rsi):
                return None
            if side == "SHORT" and rsi_bull_div(lo, rsi):
                return None


        atr = self.atr(sym, self.cfg.ENTRY_TF)
        px = float(self.mark(sym))
        stop = px - self.cfg.STOP_ATR * atr if side == "LONG" else px + self.cfg.STOP_ATR * atr
        tp1 = px + self.cfg.TP1_ATR * atr if side == "LONG" else px - self.cfg.TP1_ATR * atr
        tp2 = px + self.cfg.TP2_ATR * atr if side == "LONG" else px - self.cfg.TP2_ATR * atr
        rr = abs((tp1 - px) / (px - stop)) if px != stop else 0.0
        if rr < self.cfg.MIN_RR:
            return None

        tk = SetupTicket(
            id=f"{sym}_{int(time()*1000)}",
            symbol=sym,
            side=side,
            tf=self.cfg.ENTRY_TF,
            score=score,
            entry_px=px,
            stop_px=stop,
            tps=[tp1, tp2],
            rr=rr,
            created_ts=time(),
            expire_ts=time() + self.cfg.SETUP_TICKET_TTL_SEC,
            reasons=reasons,
            confidence=confidence if confidence is not None else 0.8,
            regime=regime,
        )

        # [ANCHOR:TICKET_REASONS_ATTACH]
        from ftm2.analysis.reasons import top_reasons
        tk.rr = rr
        tk.confidence = getattr(self.scoring, "last_conf", 0.8)
        tk.regime = getattr(self.scoring, "last_regime", "NORMAL")
        tk.reasons = top_reasons(self.snapshots.get(sym), score, tk.confidence, tk.regime)
        return tk

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
        if bx.filters:
            bx.filters.use(sym)
        cache = market_cache.setdefault(sym, {})
        for tf in tfs:
            df = cache.get(tf)
            if _need_more(df):
                boot = await bootstrap_candles(bx, sym, [tf])
                cache[tf] = boot[tf]
                print(f"[ANALYSIS][BOOT] {sym} {tf} bars={len(boot[tf])}")

    # [ANCHOR:M6_ENGINE_LOOP] 분석 루프
    while True:
        try:
            started = datetime.now(timezone.utc).timestamp()
            for sym in symbols:
                if bx.filters:
                    bx.filters.use(sym)
                cache = market_cache.setdefault(sym, {})
                for tf in tfs:
                    if _need_more(cache.get(tf), 100):
                        boot = await bootstrap_candles(bx, sym, [tf])
                        cache[tf] = boot[tf]
                snap = score_snapshot(sym, cache, tfs, tf_weights)
                bps = divergence.get_bps(sym)
                snap.rules["divergence_bps"] = bps

                fp_key = f"fp:{sym}"
                fp = getattr(snap, "fingerprint", None)
                last_fp = STATE.get(fp_key)
                if fp and last_fp == fp:
                    continue
                STATE[fp_key] = fp
                await on_snapshot(sym, snap)

            elapsed = datetime.now(timezone.utc).timestamp() - started
            wait = max(1, cfg.ANALYZE_INTERVAL_S - int(elapsed))
            await asyncio.sleep(wait)
        except Exception as e:
            try:
                from ftm2.notify import dispatcher
                dispatcher.emit("error", f"[ANALYSIS][ERR] {e}")
            except Exception:
                print(f"[ANALYSIS][ERR] {e}")
            await asyncio.sleep(1.0)


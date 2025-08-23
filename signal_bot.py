# signal_bot.py
import ccxt
import pandas as pd
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ★ 비대화형 백엔드 (파일 저장 전용)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import platform
import os, sys, logging
import discord
import json, uuid
import asyncio  # ✅ 이 줄을 꼭 추가

# [ANCHOR: DEBUG_FLAG_BEGIN]
def _env_on(k: str, default="0") -> bool:
    """Return True if env var looks like on: 1/true/yes/on (case-insensitive)."""
    import os as _os
    v = _os.getenv(k, default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")

# Global debug flag (safe default)
DEBUG = _env_on("DEBUG")
# [ANCHOR: DEBUG_FLAG_END]

# [ANCHOR: ENV_CHAIN_BEGIN]
def load_env_chain(paths=("key.env", "key.advanced.env", "token.env")):
    """
    Load .env files in order; later files override earlier ones.
    If ENV_CHAIN_ORDER is set (e.g., "key,advanced,token"), use that mapping.
    Support "*.local.env" overlays loaded immediately after their base.
    """
    import os as _os
    from pathlib import Path as _Path

    # map short tokens -> filenames
    _map = {"key": "key.env", "advanced": "key.advanced.env", "token": "token.env"}

    # 1) resolve order
    _order_raw = (_os.getenv("ENV_CHAIN_ORDER") or "").strip()
    if _order_raw:
        toks = [t.strip() for t in _order_raw.split(",") if t.strip()]
        paths = tuple(_map.get(t, t) for t in toks)  # allow direct filenames too

    # 2) expand .local overlays
    def _expand_with_local(seq):
        out = []
        for p in seq:
            out.append(p)
            if p.endswith(".env"):
                loc = p.replace(".env", ".local.env")
                if _Path(loc).exists():
                    out.append(loc)
        return out

    seq = _expand_with_local(paths)

    # 3) load in order
    for p in seq:
        try:
            fp = _Path(p)
            if not fp.exists():
                continue
            with open(fp, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    _os.environ[k.strip()] = v.strip()
            print(f"[ENV] loaded: {fp}")
        except Exception as e:
            print(f"[ENV] warn: {p}: {e}")

def _boot_env_summary():
    keys = [
        "AUTO_TRADE","TRADE_MODE","EXCHANGE_ID","HEDGE_MODE",
        "CAPITAL_SOURCE","CAPITAL_BASE","CAPITAL_INCLUDE_UPNL",
        "ENV_CHAIN_ORDER",
        "DASH_LOCALE","DASHBOARD_CHANNEL_ID","TRADE_CHANNEL_ID",
        "DASHBOARD_EQUITY_MODE","DASHBOARD_UPDATE_SEC",
        "DASH_SHOW_FEES","UPNL_INCLUDE_FEES","FEE_SOURCE",
        "FEE_TAKER_RATE","FEE_MAKER_RATE",
        "TF_LEVERAGE","TF_MARGIN",
    ]
    vals = {k: os.getenv(k, "") for k in keys}
    print(f"[BOOT_ENV_SUMMARY] {vals}")
# [ANCHOR: ENV_CHAIN_END]

# ENV chain: key.env → key.advanced.env → token.env
load_env_chain()
_boot_env_summary()

# [ANCHOR: LOCKS_BEGIN]
import asyncio
ENABLE_POS_LOCK   = int(os.getenv("ENABLE_POS_LOCK","1") or 1)
ENABLE_STATE_LOCK = int(os.getenv("ENABLE_STATE_LOCK","1") or 1)
_POS_LOCK   = asyncio.Lock() if ENABLE_POS_LOCK else None
_STATE_LOCK = asyncio.Lock() if ENABLE_STATE_LOCK else None
# [ANCHOR: LOCKS_END]

import traceback
import re
from datetime import datetime, timezone, timedelta
from collections import deque
from matplotlib import rcParams
from collections import defaultdict



# [ANCHOR: RUNTIME_CFG_DECL]
RUNTIME_CFG = {}  # overlay store: key -> raw string

def cfg_get(key: str, default: str | None = None) -> str | None:
    if key in RUNTIME_CFG:
        return RUNTIME_CFG[key]
    return os.getenv(key, default)

def cfg_set(key: str, value: str) -> None:
    RUNTIME_CFG[key] = value

def cfg_del(key: str) -> None:
    if key in RUNTIME_CFG:
        del RUNTIME_CFG[key]

# [ANCHOR: VALIDATE_ON_BOOT]
def _validate_tf_map(name: str, raw: str):
    ok = all(k in ("15m", "1h", "4h", "1d") for k in re.findall(r"(15m|1h|4h|1d)", raw or ""))
    if not ok:
        logging.warning(f"[CFG_WARN] {name} has unknown TF keys: {raw}")

for KEY in ("SCALE_UP_SCORE_DELTA", "SCALE_DOWN_SCORE_DELTA", "SCALE_STEP_PCT", "SCALE_REDUCE_PCT"):
    _validate_tf_map(KEY, cfg_get(KEY, ""))
for KEY in ("SLIPPAGE_BY_SYMBOL", "TP_PCT_BY_SYMBOL", "SL_PCT_BY_SYMBOL", "TRAIL_PCT_BY_SYMBOL"):
    if not cfg_get(KEY):
        logging.warning(f"[CFG_WARN] empty {KEY}")

# === 전역 심볼 상수 ===
symbol_eth = 'ETH/USDT'
symbol_btc = 'BTC/USDT'

# 최근 계산된 지표 점수/이유를 분봉·심볼별로 캐시
LATEST_WEIGHTS = defaultdict(dict)          # key: (symbol, tf) -> {indicator: score}
LATEST_WEIGHTS_DETAIL = defaultdict(dict)   # key: (symbol, tf) -> {indicator: reason}

# [ANCHOR: PAUSE_GLOBALS]
KST = timezone(timedelta(hours=9))
PAUSE_UNTIL = {}  # (symbol, tf) -> epoch_ms; "__ALL__" -> epoch_ms
DEFAULT_PAUSE = (cfg_get("DEFAULT_PAUSE", "1") == "1")
AFTER_CLOSE_PAUSE = (cfg_get("AFTER_CLOSE_PAUSE", "1") == "1")
DAILY_RESUME_HOUR_KST = int(cfg_get("DAILY_RESUME_HOUR_KST", "11"))
_LAST_RESUME_YMD = None

# [ANCHOR: PAUSE_BOOT_INIT]

# Apply DEFAULT_PAUSE only once on boot (global), not per-loop
try:
    if (cfg_get("DEFAULT_PAUSE","1") == "1") and ("__ALL__" not in PAUSE_UNTIL):
        PAUSE_UNTIL["__ALL__"] = 2**62
        logging.info("[PAUSE] default all paused on boot (until resume)")
except Exception:
    pass


# === [ANCHOR: OBS_COOLDOWN_CFG] Gatekeeper/Observe/Cooldown Config ===
def _parse_kv_numbers(val: str | None, default: dict[str, float]) -> dict[str, float]:
    d = {}
    try:
        for part in str(val or "").split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                continue
            k, v = part.split(":", 1)
            k = k.strip()
            v = v.strip()
            d[k] = float(v)
    except Exception:
        pass
    return d or default

ENABLE_OBSERVE = os.getenv("ENABLE_OBSERVE", "1") == "1"
ENABLE_COOLDOWN = os.getenv("ENABLE_COOLDOWN", "1") == "1"
STRONG_BYPASS_SCORE = float(os.getenv("STRONG_BYPASS_SCORE", "0.80"))
GK_TTL_HOLD_SEC = float(os.getenv("GK_TTL_HOLD_SEC", "0.8"))
GATEKEEPER_OBS_SEC = _parse_kv_numbers(os.getenv("GATEKEEPER_OBS_SEC"), {"15m": 20.0, "1h": 25.0, "4h": 40.0, "1d": 60.0})

OBS_WINDOW_SEC = _parse_kv_numbers(os.getenv("OBS_WINDOW_SEC"), {"15m": 1.5, "1h": 2.0, "4h": 2.5, "1d": 3.0})
POST_EXIT_COOLDOWN_SEC = _parse_kv_numbers(os.getenv("POST_EXIT_COOLDOWN_SEC"), {"15m": 10.0, "1h": 30.0, "4h": 60.0, "1d": 120.0})

WAIT_TARGET_ENABLE = os.getenv("WAIT_TARGET_ENABLE", "1") == "1"
TARGET_SCORE_BY_TF = _parse_kv_numbers(os.getenv("TARGET_SCORE_BY_TF"), {"15m": 0.0, "1h": 0.0, "4h": 0.0, "1d": 0.0})
WAIT_TARGET_SEC = _parse_kv_numbers(os.getenv("WAIT_TARGET_SEC"), {"15m": 0.0, "1h": 0.0, "4h": 0.0, "1d": 0.0})
TARGET_WAIT_MODE = os.getenv("TARGET_WAIT_MODE", "SOFT").upper()
GK_DEBUG = os.getenv("GK_DEBUG", "0") == "1"

EXIT_DEBUG = os.getenv("EXIT_DEBUG", "0") == "1"
STRICT_EXIT_NOTIFY = os.getenv("STRICT_EXIT_NOTIFY", "1") == "1"
PAPER_STRICT_NONZERO = os.getenv("PAPER_STRICT_NONZERO", "0") == "1"
PAPER_CSV_OPEN_LOG = (cfg_get("PAPER_CSV_OPEN_LOG", "1") == "1")
PAPER_CSV_CLOSE_LOG = (cfg_get("PAPER_CSV_CLOSE_LOG", "1") == "1")
FUTURES_CSV_CLOSE_LOG = (cfg_get("FUTURES_CSV_CLOSE_LOG", "1") == "1")
CLEAR_IDEMP_ON_CLOSEALL = (cfg_get("CLEAR_IDEMP_ON_CLOSEALL", "1") == "1")

# Risk interpretation mode (default = MARGIN_RETURN)
# PRICE_PCT: tp/sl/tr are price-change %
# MARGIN_RETURN: tp/sl/tr are target returns on margin; effective price % = pct / leverage
RISK_INTERPRET_MODE = cfg_get("RISK_INTERPRET_MODE", "MARGIN_RETURN").upper()
APPLY_LEV_TO_TRAIL  = (cfg_get("APPLY_LEV_TO_TRAIL", "1") == "1")

# === Unified exit evaluation config (applies to ALL TFs) ===

EXIT_RESOLUTION = cfg_get("EXIT_RESOLUTION", "1m").lower()  # must be "1m"
EXIT_EVAL_MODE  = cfg_get("EXIT_EVAL_MODE", "TOUCH").upper()  # TOUCH | CLOSE (on 1m)
# Which price feed to read before clamping. 'mark' is not allowed to trigger directly (we clamp anyway).
EXIT_PRICE_SOURCE = cfg_get("EXIT_PRICE_SOURCE", "last").lower()  # last | index | mark(→forced last)
TRIGGER_PRICE_SOURCE = cfg_get("STOP_TRIGGER_PRICE", os.getenv("EXIT_PRICE_SOURCE", "mark")).lower()
# 1m outlier spike guard (fraction, e.g., 0.03=3% vs 1m open/close); 0 disables

OUTLIER_MAX_1M = float(cfg_get("OUTLIER_MAX_1M", "0.03"))

# Global last price cache
LAST_PRICE = {}  # symbol -> last/mark price
def set_last_price(symbol: str, price: float) -> None:
    try: LAST_PRICE[str(symbol).upper()] = float(price)
    except Exception: pass
def get_last_price(symbol: str, default_price: float = 0.0) -> float:
    try:
        v = LAST_PRICE.get(str(symbol).upper())
        return float(v) if v is not None else float(default_price)
    except Exception:
        return float(default_price)

# Context cache: symbol -> {ts, regime, regime_strength, r2, adx, ma_slope,
#                           structure_bias, channel_z, avwap_dist, hhhl_tag, summary}
CTX_STATE: dict[str, dict] = {}

# 구조 컨텍스트/오버레이 캐시: key=(symbol, tf) -> {"ctx": dict, "img": str|None, "ts": int, "mtime": int}
STRUCT_CACHE: dict = {}

# 상위TF 구조 알림/상태 저장
STRUCT_ALERT_STATE: dict = {}

def _load_ohlcv(symbol: str, tf: str, limit: int = 300):
    """Try multiple loaders; ALWAYS return list of [ts, o, h, l, c, v]."""
    providers = []
    if 'get_ohlcv' in globals(): providers.append(lambda: get_ohlcv(symbol, tf, limit=limit))
    if 'fetch_ohlcv' in globals(): providers.append(lambda: fetch_ohlcv(symbol, tf, limit=limit))
    if 'get_recent_ohlc_series' in globals(): providers.append(lambda: get_recent_ohlc_series(symbol, tf, limit=limit))
    for fn in providers:
        try:
            raw = fn()
            rows = _ensure_ohlcv_list(raw)
            if rows: return rows
        except Exception:
            continue
    return []

# === Exit resolution helpers (1m bar fetch + sanitize/clamp/guard) ===
def _fetch_recent_bar_1m(symbol: str):
    """
    Return dict {open, high, low, close} for the latest 1m bar.
    Reuse existing OHLCV cache/fetchers if available; safe fallback to last price.
    """
    try:
        # Try common helpers first
        if 'get_ohlcv' in globals():
            ohlc = get_ohlcv(symbol, "1m", limit=1)[-1]
            return {"open": float(ohlc[1]), "high": float(ohlc[2]), "low": float(ohlc[3]), "close": float(ohlc[4])}
        if 'fetch_ohlcv' in globals():
            ohlc = fetch_ohlcv(symbol, "1m", limit=1)[-1]
            return {"open": float(ohlc[1]), "high": float(ohlc[2]), "low": float(ohlc[3]), "close": float(ohlc[4])}
        if 'get_recent_ohlc' in globals():
            b = get_recent_ohlc(symbol, "1m")
            return {"open": float(b["open"]), "high": float(b["high"]), "low": float(b["low"]), "close": float(b["close"]) }
    except Exception:
        pass
    # Fallback: use last price cache to emulate a flat bar
    lp = get_last_price(symbol, 0.0)
    return {"open": lp, "high": lp, "low": lp, "close": lp}

def _raw_exit_price(symbol: str, last_hint: float|None = None) -> float:
    # honor EXIT_PRICE_SOURCE, but never let 'mark' directly drive exits (we'll clamp anyway)
    try:
        src = EXIT_PRICE_SOURCE
        if src == "index" and 'get_index_price' in globals():
            return float(get_index_price(symbol))
    except Exception:
        pass
    # 'mark' → force to last (will be clamped to current 1m H/L)
    return float(last_hint if last_hint is not None else get_last_price(symbol, 0.0))

def _sanitize_exit_price(symbol: str, last_hint: float|None = None):
    """
    Returns (clamped, bar), where bar is dict(open,high,low,close) of the *current* 1m bar.
    We clamp price to [low, high] to avoid unseen spikes.
    """
    bar = _fetch_recent_bar_1m(symbol)
    last_raw = _raw_exit_price(symbol, last_hint)
    hi, lo = float(bar["high"]), float(bar["low"])
    clamped = max(min(float(last_raw), hi), lo)
    return clamped, bar

def _sanitize_exit_price_with_bar(symbol: str, price: float, bar_1m):
    """
    Same clamp/outlier logic as _sanitize_exit_price(), but DOES NOT fetch OHLCV.
    'bar_1m' should be a single 1m candle tuple/list or None.
    """
    if bar_1m is None:
        return _sanitize_exit_price(symbol, price)
    try:
        if isinstance(bar_1m, dict):
            bar = {k: float(bar_1m[k]) for k in ("open", "high", "low", "close")}
        else:
            bar = {
                "open": float(bar_1m[1]),
                "high": float(bar_1m[2]),
                "low": float(bar_1m[3]),
                "close": float(bar_1m[4]),
            }
    except Exception:
        return _sanitize_exit_price(symbol, price)
    last_raw = _raw_exit_price(symbol, price)
    hi, lo = float(bar["high"]), float(bar["low"])
    clamped = max(min(float(last_raw), hi), lo)
    return clamped, bar

def _outlier_guard(clamped: float, bar: dict) -> bool:
    """
    True → skip this minute as outlier (|Δ| > OUTLIER_MAX_1M vs 1m open/close).
    """
    try:
        if OUTLIER_MAX_1M <= 0:
            return False
        ref = float(bar.get("open") or bar.get("close") or clamped)
        if ref <= 0:
            return False
        delta = abs(clamped - ref) / ref
        return delta > OUTLIER_MAX_1M
    except Exception:
        return False


# [ANCHOR: EXIT_EVAL_HELPERS_BEGIN]
EXIT_FILL_MODE = cfg_get("EXIT_FILL_MODE", "bar_bound").lower()  # bar_bound | threshold

def _eff_pct_from_env(tf: str, pct: float, apply_leverage: bool = True) -> float:
    """PRICE_PCT vs MARGIN_RETURN 모드 해석"""
    try:
        if RISK_INTERPRET_MODE == "MARGIN_RETURN" and apply_leverage:
            lev = float(TF_LEVERAGE.get(tf, 1))
            return float(pct) / max(lev, 1.0)
        return float(pct)
    except Exception:
        return float(pct)

def _eval_exit_touch(side: str, entry: float, tf: str, bar: dict):
    """
    보호체크 전용의 간이 TOUCH 로직 (TRAIL 제외).
    returns: (hit:bool, reason:str|None, trigger_price:float|None)
    """
    hi, lo = float(bar["high"]), float(bar["low"])
    tp_pct = _eff_pct_from_env(tf, (take_profit_pct or {}).get(tf, 0.0))
    sl_pct = _eff_pct_from_env(tf, (HARD_STOP_PCT or {}).get(tf, 0.0))
    if entry <= 0:
        return False, None, None
    if str(side).upper() == "LONG":
        tp_price = entry * (1.0 + tp_pct / 100.0)
        sl_price = entry * (1.0 - sl_pct / 100.0)
        if lo <= sl_price:
            return True, "SL", sl_price
        if hi >= tp_price:
            return True, "TP", tp_price
    elif str(side).upper() == "SHORT":
        tp_price = entry * (1.0 - tp_pct / 100.0)
        sl_price = entry * (1.0 + sl_pct / 100.0)
        if hi >= sl_price:
            return True, "SL", sl_price
        if lo <= tp_price:
            return True, "TP", tp_price
    return False, None, None

async def _choose_exec_price(symbol: str, tf: str, reason: str, side: str, trig_px: float, bar: dict) -> float:
    """
    EXIT_FILL_MODE에 따라 실제 기록할 '종결가' 선택 후 공통 실행가격 모델 적용.
    """
    hi, lo = float(bar["high"]), float(bar["low"])
    if EXIT_FILL_MODE == "threshold":
        base_px = float(trig_px)
    else:
        side = str(side).upper()
        reasonU = (reason or "").upper()
        if reasonU == "TRAIL":
            base_px = lo if side == "LONG" else hi
        elif reasonU == "SL":
            base_px = max(trig_px, lo) if side == "LONG" else min(trig_px, hi)
        elif reasonU == "TP":
            base_px = min(trig_px, hi) if side == "LONG" else max(trig_px, lo)
        else:
            base_px = max(min(trig_px, hi), lo)

    snap = await get_price_snapshot(symbol)
    bar1m = _fetch_recent_bar_1m(symbol)
    _ex_guard = FUT_EXCHANGE or PUB_FUT_EXCHANGE
    final_px = _exec_price_model(_ex_guard, symbol, tf, side, "exit", snap, bar1m, ref_override=float(base_px))


    if _env_on("DEBUG") or _env_on("FILL_MODEL_DEBUG"):

        try:
            log(f"[FILL_MODEL] EXIT {symbol} {tf} side={side} reason={reason} ref={os.getenv('EXIT_PRICE_SOURCE','mark')} slp={_resolve_slippage_pct(symbol, tf, 'exit')}")
        except Exception:
            pass
    return final_px
# [ANCHOR: EXIT_EVAL_HELPERS_END]


# --- Normalize OHLCV to list-of-lists: [ts, open, high, low, close, volume] ---
def _ensure_ohlcv_list(data):
    """
    Accepts list/tuple of rows OR pandas.DataFrame.
    Returns [] if cannot normalize.
    """
    try:
        # 1) Already sequence of rows
        if isinstance(data, (list, tuple)):
            if len(data) == 0: return []
            # row is list-like and first element looks like timestamp -> assume OK
            if isinstance(data[0], (list, tuple)) and len(data[0]) >= 5:
                return [list(r[:6]) + [0.0] * max(0, 6-len(r)) for r in data]
        # 2) Pandas DataFrame-like
        if hasattr(data, "empty") and hasattr(data, "columns"):
            if getattr(data, "empty", False): return []
            cols = [str(c).lower() for c in list(data.columns)]
            # try name-based mapping
            def _find(*names, default_idx=None):
                for nm in names:
                    if nm in cols: return cols.index(nm)
                return default_idx
            i_ts  = _find("timestamp","time","ts","open_time", default_idx=0)
            i_o   = _find("open",  default_idx=1)
            i_h   = _find("high",  default_idx=2)
            i_l   = _find("low",   default_idx=3)
            i_c   = _find("close", default_idx=4)
            i_v   = _find("volume","vol", default_idx=5)
            vals = data.values.tolist()
            out = []
            for r in vals:
                try:
                    ts = float(r[i_ts]); o=float(r[i_o]); h=float(r[i_h]); l=float(r[i_l]); c=float(r[i_c])
                    v  = float(r[i_v]) if i_v is not None and i_v < len(r) else 0.0
                    out.append([ts, o, h, l, c, v])
                except Exception:
                    continue
            return out
    except Exception:
        return []
    return []


def _linreg_y(x: list[float], y: list[float]):
    n = len(x)
    if n < 2:
        return (0.0, 0.0, 0.0)  # slope, intercept, r2
    sx = sum(x); sy = sum(y)
    sxx = sum(v*v for v in x); syy = sum(v*v for v in y)
    sxy = sum(x[i]*y[i] for i in range(n))
    den = n*sxx - sx*sx
    if den == 0:
        return (0.0, y[-1], 0.0)
    slope = (n*sxy - sx*sy)/den
    intercept = (sy - slope*sx)/n
    yhat = [slope*xi + intercept for xi in x]
    ss_res = sum((y[i]-yhat[i])**2 for i in range(n))
    ss_tot = sum((yi - (sy/n))**2 for yi in y)
    r2 = 0.0 if ss_tot == 0 else max(0.0, min(1.0, 1 - ss_res/ss_tot))
    return (slope, intercept, r2)


def _sma(a: list[float], w: int) -> list[float]:
    out = []
    s = 0.0
    q = []
    for v in a:
        q.append(v); s += v
        if len(q) > w:
            s -= q.pop(0)
        out.append(s/len(q))
    return out


def _zigzag_swings(closes: list[float], pct: float):
    """Return list of tuples (idx, price, 'H'|'L') with ~pct swing."""
    if not closes:
        return []
    th = pct/100.0
    swings = []
    i0 = 0
    p0 = closes[0]
    mode = None
    for i, p in enumerate(closes[1:], start=1):
        change = (p - p0)/p0
        if mode in (None,'L') and change >= th:
            swings.append((i0, closes[i0], 'L')); swings.append((i, p, 'H'))
            i0 = i; p0 = p; mode = 'H'
        elif mode in (None,'H') and change <= -th:
            swings.append((i0, closes[i0], 'H')); swings.append((i, p, 'L'))
            i0 = i; p0 = p; mode = 'L'
    return swings


def _zscore(v: float, mean: float, std: float) -> float:
    if std <= 1e-12:
        return 0.0
    return (v - mean)/std


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _compute_context(symbol: str) -> dict|None:
    if not REGIME_ENABLE:
        return None
    try:
        now = time.time()
        st = CTX_STATE.get(symbol)
        if st and (now - st.get("ts", 0) < CTX_TTL_SEC):
            return st
        rows = _load_ohlcv(symbol, REGIME_TF, limit=max(200, REGIME_LOOKBACK+5))
        # rows must be list now; guard length only
        if len(rows) < max(60, REGIME_LOOKBACK//2):
            return None
        closes = [float(r[4]) for r in rows]
        highs  = [float(r[2]) for r in rows]
        lows   = [float(r[3]) for r in rows]
        vols   = [float(r[5]) if len(r) > 5 else 0.0 for r in rows]
        n = len(closes)
        xs = list(range(n))
        slope_c, _, r2 = _linreg_y(xs[-REGIME_LOOKBACK:], closes[-REGIME_LOOKBACK:])
        sma50 = _sma(closes, 50); sma200 = _sma(closes, 200)
        ma_slope = _sign(sma50[-1]-sma50[-10]) + _sign(sma200[-1]-sma200[-10])
        rng = sum(abs(highs[i]-lows[i]) for i in range(n-20, n))
        chg = sum(abs(closes[i]-closes[i-1]) for i in range(n-20+1, n))
        adx_like = 100.0 * (chg/rng) if rng > 0 else 0.0
        trend_cond = (r2 >= REGIME_TREND_R2_MIN) and (adx_like >= REGIME_ADX_MIN)
        regime = "RANGE"
        if trend_cond:
            regime = "TREND_UP" if slope_c>0 and closes[-1]>sma200[-1] else ("TREND_DOWN" if slope_c<0 and closes[-1]<sma200[-1] else "RANGE")
        regime_strength = max(0.0, min(1.0, (r2-REGIME_TREND_R2_MIN)/(1.0-REGIME_TREND_R2_MIN)))
        slope, intercept, _ = _linreg_y(xs[-REGIME_LOOKBACK:], closes[-REGIME_LOOKBACK:])
        fit = [slope*xi + intercept for xi in xs[-REGIME_LOOKBACK:]]
        resid = [closes[-REGIME_LOOKBACK+i] - fit[i] for i in range(len(fit))]
        mean_r = sum(resid)/len(resid)
        std_r = (sum((v-mean_r)**2 for v in resid)/max(1, len(resid)-1))**0.5
        z = _zscore(closes[-1] - fit[-1], 0.0, std_r if std_r>0 else 1.0)
        swings = _zigzag_swings(closes[-REGIME_LOOKBACK:], STRUCT_ZIGZAG_PCT)
        hhhl = "-"
        if len(swings) >= 4:
            last = swings[-4:]
            tags = [t[2] for t in last]
            prices = [t[1] for t in last]
            if tags == ['L','H','L','H'] and prices[3] > prices[1] and prices[2] > prices[0]:
                hhhl = "HH/HL"
            elif tags == ['H','L','H','L'] and prices[3] < prices[1] and prices[2] < prices[0]:
                hhhl = "LH/LL"
        structure_bias = max(-1.0, min(1.0, -z/CHANNEL_BANDS_STD))
        rb = 0.0
        if regime == "TREND_UP":
            rb = +regime_strength
        elif regime == "TREND_DOWN":
            rb = -regime_strength
        ctx_bias = max(-1.0, min(1.0, rb + structure_bias*0.5))
        summary = f"REGIME={regime} ({regime_strength:.2f}, R2={r2:.2f}, ADX~{adx_like:.0f}) | STRUCT={hhhl}, ch.z={z:.2f}, bias={ctx_bias:.2f}"
        st = {"ts": now, "regime": regime, "regime_strength": regime_strength,
              "r2": r2, "adx": adx_like, "ma_slope": ma_slope,
              "structure_bias": structure_bias, "channel_z": z,
              "avwap_dist": None, "hhhl": hhhl, "ctx_bias": ctx_bias,
              "summary": summary}
        CTX_STATE[symbol] = st
        log(f"[CTX] {symbol} {summary}")
        return st
    except Exception as e:
        src = type(rows).__name__ if 'rows' in locals() else 'N/A'
        log(f"[CTX_ERR] {symbol} src={src} err={e}")
        return None


def _adjust_score_with_ctx(symbol: str, tf: str, base_score: float) -> tuple[float, dict|None]:
    st = _compute_context(symbol)
    if not st:
        return (base_score, None)
    sgn = _sign(base_score)
    ctx = float(st.get("ctx_bias", 0.0))
    adj = base_score * (1 + CTX_ALPHA*ctx*sgn) - CTX_BETA*max(0.0, -ctx*sgn)
    return (adj, st)


def _lerp(a: float, b: float, w: float) -> float:
    w = max(0.0, min(1.0, w))
    return a + (b - a) * w


def _pb_alignment(regime: str, side: str) -> int:
    """+1 aligned, -1 contra, 0 neutral (RANGE treated separately)."""
    r = (regime or "").upper(); s = (side or "").upper()
    if r == "RANGE": return 0
    if r == "TREND_UP":
        return +1 if s == "LONG" else -1
    if r == "TREND_DOWN":
        return +1 if s == "SHORT" else -1
    return 0


def _playbook_adjust_risk(symbol: str, tf: str, side: str,
                          tp_pct: float|None, sl_pct: float|None, tr_pct: float|None,
                          lev: float|None, alloc_frac: float|None) -> tuple[dict, dict|None]:
    """

    Returns ({tp, sl, tr, lev_cap, alloc_mul, alloc_abs_cap,
              scale_step_mul, scale_reduce_mul, scale_legs_add,
              scale_up_shift, scale_down_shift, label, eff_w}, ctx_state)

    Multiplies raw % (on-margin targets) BEFORE leverage→price conversion.
    Intensity scales by |ctx_bias| * PB_INTENSITY.
    """
    if not PLAYBOOK_ENABLE:
        return ({"tp": tp_pct, "sl": sl_pct, "tr": tr_pct,
                 "lev_cap": 0.0, "alloc_mul": 1.0, "label": "PB_OFF", "eff_w": 0.0}, None)
    st = CTX_STATE.get(symbol) or _compute_context(symbol)
    if not st:
        return ({"tp": tp_pct, "sl": sl_pct, "tr": tr_pct,
                 "lev_cap": 0.0, "alloc_mul": 1.0, "label": "PB_NCTX", "eff_w": 0.0}, None)
    regime = st.get("regime", "RANGE")
    bias = float(st.get("ctx_bias", 0.0))
    w = max(0.0, min(1.0, abs(bias) * PB_INTENSITY))
    sideU = (side or "").upper()
    # Choose base profile
    if regime == "RANGE":
        tp_mul = PB_RANGE_TP_MUL; sl_mul = PB_RANGE_SL_MUL; tr_mul = PB_RANGE_TR_MUL
        alloc_mul = PB_RANGE_ALLOC_MUL; lev_cap = PB_RANGE_LEV_CAP; label = "PB_RANGE"
    else:
        align = _pb_alignment(regime, sideU)
        if align >= 0:
            tp_mul = PB_ALIGN_TP_MUL; sl_mul = PB_ALIGN_SL_MUL; tr_mul = PB_ALIGN_TR_MUL
            alloc_mul = PB_ALIGN_ALLOC_MUL; lev_cap = PB_ALIGN_LEV_CAP; label = "PB_ALIGN" if align>0 else "PB_NEUTRAL"
        else:
            tp_mul = PB_CONTRA_TP_MUL; sl_mul = PB_CONTRA_SL_MUL; tr_mul = PB_CONTRA_TR_MUL
            alloc_mul = PB_CONTRA_ALLOC_MUL; lev_cap = PB_CONTRA_LEV_CAP; label = "PB_CONTRA"
    # Lerp from 1.0 toward profile by intensity w
    adj_tp = None if tp_pct is None else _lerp(float(tp_pct), float(tp_pct)*tp_mul, w)
    adj_sl = None if sl_pct is None else _lerp(float(sl_pct), float(sl_pct)*sl_mul, w)
    adj_tr = None if tr_pct is None else _lerp(float(tr_pct), float(tr_pct)*tr_mul, w)
    adj_alloc_mul = _lerp(1.0, alloc_mul, w)
    eff_lev_cap = float(lev_cap or 0.0)  # 0 means no cap

    # ------ hard caps (allocation & leverage) ------
    if PLAYBOOK_HARD_LIMITS:
        if regime == "RANGE":
            alloc_abs_cap = PB_RANGE_ALLOC_ABS_CAP
            max_lev_cap   = (PB_RANGE_MAX_LEV or eff_lev_cap or 0.0)
        else:
            align = _pb_alignment(regime, sideU)
            if align >= 0:
                alloc_abs_cap = PB_ALIGN_ALLOC_ABS_CAP
                max_lev_cap   = (PB_ALIGN_MAX_LEV or eff_lev_cap or 0.0)
            else:
                alloc_abs_cap = PB_CONTRA_ALLOC_ABS_CAP
                max_lev_cap   = (PB_CONTRA_MAX_LEV or eff_lev_cap or 0.0)
    else:
        alloc_abs_cap = 0.0
        max_lev_cap   = eff_lev_cap
    # ------ scaling overrides ------
    sc_step_mul = sc_reduce_mul = 1.0
    sc_legs_add = 0
    up_shift = down_shift = 0.0
    if PLAYBOOK_SCALE_OVERRIDE:
        if regime == "RANGE":
            sc_step_mul   = PB_RANGE_SCALE_STEP_MUL
            sc_reduce_mul = PB_RANGE_SCALE_REDUCE_MUL
            sc_legs_add   = PB_RANGE_SCALE_MAX_LEGS_ADD
            up_shift      = PB_RANGE_SCALE_UP_DELTA_SHIFT
            down_shift    = PB_RANGE_SCALE_DOWN_DELTA_SHIFT
        else:
            align = _pb_alignment(regime, sideU)
            if align >= 0:
                sc_step_mul   = PB_ALIGN_SCALE_STEP_MUL
                sc_reduce_mul = PB_ALIGN_SCALE_REDUCE_MUL
                sc_legs_add   = PB_ALIGN_SCALE_MAX_LEGS_ADD
                up_shift      = PB_ALIGN_SCALE_UP_DELTA_SHIFT
                down_shift    = PB_ALIGN_SCALE_DOWN_DELTA_SHIFT
            else:
                sc_step_mul   = PB_CONTRA_SCALE_STEP_MUL
                sc_reduce_mul = PB_CONTRA_SCALE_REDUCE_MUL
                sc_legs_add   = PB_CONTRA_SCALE_MAX_LEGS_ADD
                up_shift      = PB_CONTRA_SCALE_UP_DELTA_SHIFT
                down_shift    = PB_CONTRA_SCALE_DOWN_DELTA_SHIFT
        # intensity blends toward overrides
        sc_step_mul   = _lerp(1.0, sc_step_mul,   w)
        sc_reduce_mul = _lerp(1.0, sc_reduce_mul, w)
        sc_legs_add   = int(round(_lerp(0.0, float(sc_legs_add), w)))
        up_shift      = up_shift * w
        down_shift    = down_shift * w
    return ({"tp": adj_tp, "sl": adj_sl, "tr": adj_tr,
             "lev_cap": max_lev_cap, "alloc_mul": adj_alloc_mul,
             "alloc_abs_cap": float(alloc_abs_cap or 0.0),
             "scale_step_mul": sc_step_mul, "scale_reduce_mul": sc_reduce_mul,
             "scale_legs_add": sc_legs_add, "scale_up_shift": up_shift, "scale_down_shift": down_shift,

            "label": label, "eff_w": w}, st)


def _parse_brackets(spec: str, fallback_legs: int) -> list[float]:
    """
    spec: "N|w1,w2,...,wN" or "w1,w2,..." (N inferred).
    Returns normalized weights (sum=1.0) length = N (or fallback_legs if parse fails).
    """
    try:
        if "|" in spec:
            left, right = spec.split("|", 1)
            n = int(float(left.strip()))
            ws = [float(x) for x in right.split(",") if x.strip()!=""]
        else:
            ws = [float(x) for x in spec.split(",") if x.strip()!=""]
            n = len(ws)
        if n <= 0: n = max(1, int(fallback_legs))
        if len(ws) < n:  # pad tail equally
            remain = n - len(ws); ws += [ws[-1] if ws else 1.0]*remain
        ws = ws[:n]
        s = sum(ws) or 1.0
        return [max(0.0, w/s) for w in ws]
    except Exception:
        n = max(1, int(fallback_legs))
        return [1.0/n]*n


def _select_brackets_for(symbol: str, side: str, max_legs_eff: int) -> list[float]:
    """
    Pick a bracket shape by regime alignment; cap length to max_legs_eff.
    """
    st = CTX_STATE.get(symbol) or _compute_context(symbol)
    spec = SCALE_BRACKETS_DEFAULT
    try:
        regime = (st.get("regime") if st else "RANGE") or "RANGE"
        align = _pb_alignment(regime, (side or "").upper()) if '_pb_alignment' in globals() else 0
        if regime == "RANGE":
            spec = SCALE_BRACKETS_RANGE
        elif align >= 0:
            spec = SCALE_BRACKETS_ALIGN
        else:
            spec = SCALE_BRACKETS_CONTRA
    except Exception:
        pass
    ws = _parse_brackets(spec, max_legs_eff)
    if len(ws) > max_legs_eff:
        ws = ws[:max_legs_eff]
        s = sum(ws) or 1.0
        ws = [w/s for w in ws]
    return ws


def _should_realloc(prev_ctx: dict|None, new_ctx: dict|None, last_ts: float|None, side: str) -> bool:
    if not SCALE_REALLOCATE_BRACKETS:
        return False
    now = time.time()
    if last_ts and now - last_ts < SCALE_REALLOC_COOLDOWN_SEC:
        return False
    if not (SCALE_REALLOC_ON_ALIGN_CHANGE or SCALE_REALLOC_ON_BIAS_STEP):
        return False
    try:
        p = prev_ctx or {}
        n = new_ctx or {}
        if SCALE_REALLOC_ON_ALIGN_CHANGE:
            def _al(ctx):
                r = (ctx.get("regime") or "RANGE").upper()
                s = (side or "").upper()
                return _pb_alignment(r, s) if '_pb_alignment' in globals() else 0
            if _al(p) != _al(n):
                return True
        if SCALE_REALLOC_ON_BIAS_STEP:
            steps = [float(x) for x in str(SCALE_REALLOC_BIAS_STEPS).split(",") if x.strip()!=""]
            pb = abs(float(p.get("ctx_bias", 0.0) if p else 0.0))
            nb = abs(float(n.get("ctx_bias", 0.0) if n else 0.0))
            crossed = any((pb < t <= nb) or (nb < t <= pb) for t in steps)
            if crossed:
                return True
    except Exception:
        return False
    return False


def _plan_bracket_targets(total_notional: float, ws: list[float]) -> list[float]:
    s = sum(ws) or 1.0
    ws = [w/s for w in ws]
    return [max(0.0, total_notional*w) for w in ws]



# === Futures reallocation executors ===
def _qty_from_notional(symbol: str, notional: float, price: float) -> float:
    if price <= 0 or abs(notional) <= 0:
        return 0.0
    q = float(abs(notional) / price)
    try:
        if 'qty_round' in globals():
            q = qty_round(symbol, q)
    except Exception:
        pass
    return q


async def _futures_exec_delta(symbol: str, tf: str, side: str, delta_usdt: float, ref_price: float, note: str):
    """
    Execute a delta on futures:
      LONG:  +usdt => BUY add;  -usdt => SELL reduceOnly
      SHORT: +usdt => SELL add; -usdt => BUY  reduceOnly
    """
    if not REALLOC_FUTURES_EXECUTE:
        log(f"[REALLOC_SKIP] {symbol} {tf} side={side} Δ=${delta_usdt:.2f} (exec off)")
        return False
    if abs(delta_usdt) < max(1e-9, float(SCALE_REALLOC_MIN_USDT)):
        log(f"[REALLOC_SKIP] {symbol} {tf} tiny Δ=${delta_usdt:.2f}")
        return False
    qty = _qty_from_notional(symbol, abs(delta_usdt), max(ref_price, 1e-9))
    if REALLOC_MIN_QTY > 0 and qty < REALLOC_MIN_QTY:
        log(f"[REALLOC_SKIP] {symbol} {tf} qty<{REALLOC_MIN_QTY} ({qty})")
        return False
    sideU = (side or "").upper()
    is_reduce = (delta_usdt < 0)
    if sideU == "LONG":
        ord_side = "BUY" if not is_reduce else "SELL"
    else:
        ord_side = "SELL" if not is_reduce else "BUY"
    reduce_only = bool(is_reduce)
    ok = False; err = None
    for _ in range(max(1, REALLOC_MAX_RETRIES)):
        try:
            if 'futures_market_order' in globals():
                res = await futures_market_order(symbol, ord_side, qty, reduce_only=reduce_only, comment=f"REALLOC/{tf}/{note}")
            elif 'futures_place_order' in globals():
                res = await futures_place_order(symbol, ord_side, qty, order_type="market", reduce_only=reduce_only, comment=f"REALLOC/{tf}/{note}")
            else:
                if reduce_only and 'futures_close_symbol_tf' in globals():
                    await futures_close_symbol_tf(symbol, tf, sideU, ref_price, reason=f"REALLOC-{note}")
                    res = {"status": "closed"}
                else:
                    res = {"status": "logged"}
            ok = True
            try:
                if CSV_SCALE_EVENTS:
                    kind = "SCALE_REDUCE" if is_reduce else "SCALE_ADD"
                    _csv_log_scale_event(symbol, tf, kind, sideU, qty, ref_price, note)
            except Exception:
                pass
            log(f"[REALLOC_EXEC] {symbol} {tf} {ord_side} qty={qty} reduceOnly={reduce_only} note={note} res={res.get('status','ok') if isinstance(res,dict) else 'ok'}")
            break
        except Exception as e:
            err = e
            log(f"[REALLOC_ERR] {symbol} {tf} {e}")
            try:
                time.sleep(REALLOC_RETRY_SLEEP_SEC)
            except Exception:
                pass
    return ok


def append_line_csv(filename: str, line: str):
    try:
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", filename), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        log(f"[CSV_APPEND_ERR] {filename} {e}")


def _csv_log_scale_event(symbol: str, tf: str, kind: str, side: str, qty: float, px: float, note: str):
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"{ts},{symbol},{tf},{kind},{side},{qty:.6f},{px:.4f},mode=futures,note={note}"
        append_line_csv("futures_trades.csv", line)
    except Exception as e:
        log(f"[CSV_SCALE_ERR] {symbol} {tf} {e}")


# small helper label for scale ops
def _scale_note_label(i: int, delta: float) -> str:
    return f"leg#{i}{'+' if delta>=0 else '-'}${abs(delta):.0f}"



# === Exit resolution helpers (1m bar fetch + sanitize/clamp/guard) ===
def _fetch_recent_bar_1m(symbol: str):
    """
    Return dict {open, high, low, close} for the latest 1m bar.
    Reuse existing OHLCV cache/fetchers if available; safe fallback to last price.

    """
    try:
        # Try common helpers first
        if 'get_ohlcv' in globals():
            ohlc = get_ohlcv(symbol, "1m", limit=1)[-1]
            return {"open": float(ohlc[1]), "high": float(ohlc[2]), "low": float(ohlc[3]), "close": float(ohlc[4])}
        if 'fetch_ohlcv' in globals():
            ohlc = fetch_ohlcv(symbol, "1m", limit=1)[-1]
            return {"open": float(ohlc[1]), "high": float(ohlc[2]), "low": float(ohlc[3]), "close": float(ohlc[4])}
        if 'get_recent_ohlc' in globals():
            b = get_recent_ohlc(symbol, "1m")
            return {"open": float(b["open"]), "high": float(b["high"]), "low": float(b["low"]), "close": float(b["close"]) }
    except Exception:
        pass
    # Fallback: use last price cache to emulate a flat bar
    lp = get_last_price(symbol, 0.0)
    return {"open": lp, "high": lp, "low": lp, "close": lp}


def _raw_exit_price(symbol: str, last_hint: float|None = None) -> float:
    # honor EXIT_PRICE_SOURCE, but never let 'mark' directly drive exits (we'll clamp anyway)
    try:
        src = EXIT_PRICE_SOURCE

        if src == "index" and 'get_index_price' in globals():
            return float(get_index_price(symbol))
    except Exception:
        pass

    # 'mark' → force to last (will be clamped to current 1m H/L)
    return float(last_hint if last_hint is not None else get_last_price(symbol, 0.0))

def _sanitize_exit_price(symbol: str, last_hint: float|None = None):
    """
    Returns (clamped, bar), where bar is dict(open,high,low,close) of the *current* 1m bar.
    We clamp price to [low, high] to avoid unseen spikes.
    """
    bar = _fetch_recent_bar_1m(symbol)
    last_raw = _raw_exit_price(symbol, last_hint)

    hi, lo = float(bar["high"]), float(bar["low"])
    clamped = max(min(float(last_raw), hi), lo)
    return clamped, bar

def _outlier_guard(clamped: float, bar: dict) -> bool:
    """

    True → skip this minute as outlier (|Δ| > OUTLIER_MAX_1M vs 1m open/close).

    """
    try:
        if OUTLIER_MAX_1M <= 0:
            return False
        ref = float(bar.get("open") or bar.get("close") or clamped)
        if ref <= 0:
            return False
        delta = abs(clamped - ref) / ref
        return delta > OUTLIER_MAX_1M
    except Exception:
        return False


# === [ANCHOR: GATEKEEPER_STATE] 프레임 상태/쿨다운 ===

# {tf: {"candle_ts_ms": int, "cand": [dict], "winner": str|None,
#       "first_seen_ms": int, "obs_until_ms": int, "target_until_ms": int}}
FRAME_GATE = {}
LAST_EXIT_TS = {}     # {tf: epoch_sec}
COOLDOWN_UNTIL = {}   # {tf: epoch_sec}


# === Console/File logging (UTF-8 safe for Windows) ===
import logging, sys, os

def _force_utf8_console():
    try:
        if os.name == 'nt':
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    except Exception:
        pass

def _setup_logging():
    os.makedirs("logs", exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler("logs/bot.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    root.addHandler(ch)
    root.addHandler(fh)

_force_utf8_console()
_setup_logging()

def log(msg: str):
    logging.info(str(msg))






# === [ANCHOR: DAILY_CHANGE_UTILS] 일봉 변동률 유틸 (단일 기준) ===
DAILY_OPEN_CACHE = {}  # {symbol: {"open": float, "ts": epoch_seconds}}
DAILY_OPEN_TTL = 60  # seconds

async def get_daily_open(symbol: str) -> float | None:
    now = int(time.time())
    rec = DAILY_OPEN_CACHE.get(symbol)
    if rec and (now - rec.get("ts", 0) < DAILY_OPEN_TTL):
        return rec.get("open")

    try:
        df_1d = await safe_get_ohlcv(symbol, '1d', limit=1)
        if _len(df_1d) >= 1:
            val = float(df_1d['open'].iloc[-1])
            DAILY_OPEN_CACHE[symbol] = {"open": val, "ts": now}
            return val
    except Exception:
        pass
    return None

async def compute_daily_change_pct(symbol: str, price_ref: float | None) -> float | None:
    """하루 시작가(1d open) 대비 변동률을 단일 방식으로 산출"""
    try:
        if not isinstance(price_ref, (int, float)) or price_ref <= 0:
            return None
        dopen = await get_daily_open(symbol)
        if dopen and dopen > 0:
            return ((float(price_ref) - dopen) / dopen) * 100.0
    except Exception:
        pass
    return None

# === 안전 인덱싱 유틸 ===
def _closed_i(df):
    # 닫힌 봉 인덱스: 최소 2개 이상일 때 -2, 아니면 -1
    return -2 if len(df) >= 2 else -1

def closed_ts(df):
    i = _closed_i(df)
    try:
        return int(pd.Timestamp(df['timestamp'].iloc[i]).value // 10**9)
    except Exception:
        val = df['timestamp'].iloc[i]
        return int(val.timestamp()) if hasattr(val, 'timestamp') else 0

def closed_ohlc(df):
    i = _closed_i(df)
    return (
        float(df['open'].iloc[i]),
        float(df['high'].iloc[i]),
        float(df['low'].iloc[i]),
        float(df['close'].iloc[i]),
    )


def _closed_idx(df):
    n = _len(df)
    # 최소 2개 이상일 때만 -2(닫힌 캔들), 아니면 -1(유일한 캔들)로 폴백
    return -2 if n >= 2 else -1

def get_closed_ts(df) -> int | None:
    try:
        idx = _closed_idx(df)
        return int(pd.Timestamp(df['timestamp'].iloc[idx]).value // 10**9)
    except Exception:
        return None

def get_closed_price(df, col='close') -> float | None:
    try:
        idx = _closed_idx(df)
        v = df[col].iloc[idx]
        return float(v) if v == v else None
    except Exception:
        return None
    
def _len(df):
    try:
        return len(df) if df is not None else 0
    except Exception:
        return 0

def _s_iloc(series, idx, default=None):
    try:
        return series.iloc[idx]
    except Exception:
        return default

def _last(df, col, default=None):
    try:
        return _s_iloc(df[col], -1, default) if (df is not None and col in df) else default
    except Exception:
        return default

def _prev(df, col, default=None):
    try:
        return _s_iloc(df[col], -2, default) if (df is not None and col in df) else default
    except Exception:
        return default

def _score_bucket(score, cfg):
    try:
        if score >= cfg["buy_cut"]:
            return "BUY"
        if score <= cfg["sell_cut"]:
            return "SELL"
        return "NEUTRAL"
    except Exception:
        return None

# [ANCHOR: GATEKEEPER_V3_BEGIN]
import os, time

FRAME_GATE: dict[str, dict] = FRAME_GATE if 'FRAME_GATE' in globals() else {}

def _parse_tf_map(s: str, cast=float):
    out = {}
    for p in (s or "").split(","):
        if ":" in p:
            k, v = p.split(":", 1)
            k = k.strip(); v = v.strip()
            try:
                out[k] = cast(v)
            except Exception:
                pass
    return out

def _now_ms() -> int:
    return int(time.time() * 1000)

# === Gatekeeper helpers ===
def _abs_score(c: dict) -> float:
    try:
        for k in ("score", "total_score", "strength", "coefficient"):
            if k in c and isinstance(c[k], (int, float)):
                return abs(float(c[k]))
    except Exception:
        pass
    return 0.0
#
def _parse_mtf_map(spec: str) -> dict[str, list[str]]:
    """
    "15m:1h,4h;1h:4h,1d;4h:1d" -> {"15m":["1h","4h"], "1h":["4h","1d"], "4h":["1d"]}
    """
    out = {}
    try:
        for seg in (spec or "").split(";"):
            seg = seg.strip()
            if not seg:
                continue
            k, v = seg.split(":")
            out[k.strip()] = [x.strip() for x in v.split(",") if x.strip()]
    except Exception:
        pass
    return out

async def _struct_alert(symbol: str, tf: str, text: str):
    """트레이드 채널에 구조 경고 발송(쿨다운 적용)."""
    try:
        if not MTF_ALERT_ENABLE:
            return
        key = (symbol, tf, "ALERT_TEXT")
        last = STRUCT_ALERT_STATE.get(key, {}).get("ts", 0)
        now  = _now_ms()
        if (now - last) < MTF_ALERT_COOLDOWN_SEC * 1000:
            return
        STRUCT_ALERT_STATE[key] = {"ts": now}
        cid = _get_trade_channel_id(symbol, tf)
        ch  = client.get_channel(cid) if cid else None
        if ch:
            await ch.send(f"🧭 {symbol} · {tf} • {text}")
    except Exception as e:
        log(f"[STRUCT_ALERT_WARN] {symbol} {tf} {e}")

# [ANCHOR: GATEKEEPER_V3_BEGIN]
def gatekeeper_offer(tf: str, candle_ts_ms: int, cand: dict) -> bool:
    """
    OBS/TARGET-aware selector per TF & candle:
      - If two candidates arrive in the same candle, pick the one with larger |score| immediately.
      - If only one candidate exists, release it after OBS expiry.
      - With WAIT_TARGET_ENABLE:
          HARD: require score >= TARGET; else hold/sk ip this candle.
          SOFT: allow on TARGET hit, or once TARGET_WAIT_SEC elapses.
    """
    # Parse ENV each boot-run (kept simple & robust)
    OBS_SEC_MAP    = _parse_tf_map(os.getenv("GATEKEEPER_OBS_SEC", "15m:20,1h:25,4h:40,1d:60"), int)
    TARGET_MAP     = _parse_tf_map(os.getenv("TARGET_SCORE_BY_TF", "15m:0,1h:0,4h:0,1d:0"), float)
    WAIT_SEC_MAP   = _parse_tf_map(os.getenv("WAIT_TARGET_SEC", "15m:0,1h:0,4h:0,1d:0"), int)
    WAIT_ENABLE    = (os.getenv("WAIT_TARGET_ENABLE", "0") == "1")
    WAIT_MODE      = (os.getenv("TARGET_WAIT_MODE", "SOFT") or "SOFT").upper()
    STRONG_BYPASS  = float(os.getenv("STRONG_BYPASS_SCORE", "0") or 0.0)

    now_ms = _now_ms()
    g = FRAME_GATE.get(tf)

    # New candle → open frame
    if (not g) or int(g.get("candle_ts_ms", -1)) != int(candle_ts_ms):
        obs_ms = int(OBS_SEC_MAP.get(tf, 0)) * 1000
        tgt_ms = int(WAIT_SEC_MAP.get(tf, 0)) * 1000 if WAIT_ENABLE else 0
        g = {
            "candle_ts_ms": int(candle_ts_ms),
            "cand": [],
            "winner": None,
            "first_seen_ms": now_ms,
            "obs_until_ms": now_ms + obs_ms if obs_ms > 0 else now_ms,
            "target_until_ms": now_ms + tgt_ms if tgt_ms > 0 else now_ms,
        }
        FRAME_GATE[tf] = g
        if GK_DEBUG:
            log(f"[GK] {tf} frame-open ts={candle_ts_ms} obs={g['obs_until_ms']} tgt={g['target_until_ms']}")
    else:
        if GK_DEBUG:
            log(f"[GK] {tf} frame-update ts={candle_ts_ms} obs={g.get('obs_until_ms')} tgt={g.get('target_until_ms')}")

    # De-dup same symbol; keep latest
    g["cand"] = [c for c in g["cand"] if c.get("symbol") != cand.get("symbol")]
    g["cand"].append(cand)

    # If already decided
    if g.get("winner"):
        return cand.get("symbol") == g["winner"]

    # Two or more candidates → decide by |score|
    if len(g["cand"]) >= 2:
        best = max(g["cand"], key=_abs_score)
        g["winner"] = best.get("symbol")
        if GK_DEBUG:
            log(f"[GK] {tf} dual-candidate winner={g['winner']}")
        return cand.get("symbol") == g["winner"]

    # Single candidate path
    single = g["cand"][0]

    # Strong bypass
    if STRONG_BYPASS and _abs_score(single) >= STRONG_BYPASS:
        g["winner"] = single.get("symbol")
        if GK_DEBUG:
            log(f"[GK] {tf} single-candidate strong-bypass")
        return True

    obs_until = int(g.get("obs_until_ms") or 0)
    tgt_until = int(g.get("target_until_ms") or 0)

    if now_ms < obs_until:
        return False
    if GK_DEBUG and now_ms >= obs_until:
        log(f"[GK] {tf} timer-expired obs")

    if WAIT_ENABLE:
        target = float(TARGET_MAP.get(tf, 0.0))
        sc = _abs_score(single)
        if WAIT_MODE == "HARD":
            if sc < target:
                return False
        else:  # SOFT
            if sc < target and now_ms < tgt_until:
                return False
        if GK_DEBUG and now_ms >= tgt_until:
            log(f"[GK] {tf} timer-expired tgt")

    g["winner"] = single.get("symbol")
    if GK_DEBUG:
        log(f"[GK] {tf} single-candidate release={g['winner']}")
    return True
# [ANCHOR: GATEKEEPER_V3_END]


def gatekeeper_heartbeat(now_ms: int):
    """Auto-release pending frames once timers expire."""
    OBS_SEC_MAP  = _parse_tf_map(os.getenv("GATEKEEPER_OBS_SEC", "15m:20,1h:25,4h:40,1d:60"), int)
    TARGET_MAP   = _parse_tf_map(os.getenv("TARGET_SCORE_BY_TF", "15m:0,1h:0,4h:0,1d:0"), float)
    WAIT_ENABLE  = (os.getenv("WAIT_TARGET_ENABLE", "0") == "1")
    WAIT_MODE    = (os.getenv("TARGET_WAIT_MODE", "SOFT") or "SOFT").upper()

    for tf, g in list(FRAME_GATE.items()):
        if not isinstance(g, dict):
            continue
        cand_list = g.get("cand") or []
        if len(cand_list) != 1 or g.get("winner"):
            continue
        obs_until = int(g.get("obs_until_ms") or 0)
        tgt_until = int(g.get("target_until_ms") or 0)
        if now_ms < obs_until:
            continue
        if GK_DEBUG and (now_ms >= obs_until or now_ms >= tgt_until):
            log(f"[GK] {tf} timer-expired obs={now_ms >= obs_until} tgt={now_ms >= tgt_until}")
        cand = cand_list[0]
        score = _abs_score(cand)
        target = float(TARGET_MAP.get(tf, 0.0))
        allow = False
        if not WAIT_ENABLE:
            allow = True
        elif WAIT_MODE == "HARD":
            allow = score >= target
        else:  # SOFT
            allow = (score >= target) or (now_ms >= tgt_until)
        if allow:
            g["winner"] = cand.get("symbol")
            if GK_DEBUG:
                log(f"[GK] {tf} heartbeat release={g['winner']}")
        ttl_ms = int(GK_TTL_HOLD_SEC * 1000) if GK_TTL_HOLD_SEC else 0
        if ttl_ms and (now_ms - g.get("first_seen_ms", now_ms)) > ttl_ms and not g.get("winner"):
            FRAME_GATE.pop(tf, None)



# [ANCHOR: NORMALIZE_EXEC_SIGNAL]
def _normalize_exec_signal(sig: str) -> str:
    s = (sig or "").strip().upper()
    if s in {"BUY", "STRONG BUY", "WEAK BUY", "LONG", "STRONG LONG"}:
        return "BUY"
    if s in {"SELL", "STRONG SELL", "WEAK SELL", "SHORT", "STRONG SHORT"}:
        return "SELL"
    return "NEUTRAL"

# [ANCHOR: EVAL_PROTECTIVE_EXITS_STD]
def _eval_tp_sl(side: str, entry: float, price: float, tf: str) -> tuple[bool, str]:
    """TP/SL 충족 여부를 판단하여 (hit, reason)를 반환한다. hit=True면 reason∈{'TP','SL'}"""
    try:
        tp_pct = float((take_profit_pct or {}).get(tf, 0.0))
        sl_pct = float((HARD_STOP_PCT   or {}).get(tf, 0.0))
        if not (isinstance(entry, (int, float)) and isinstance(price, (int, float))):
            return False, ""
        side_u = str(side).upper()
        if side_u in ("LONG", "BUY"):
            if tp_pct and price >= entry * (1 + tp_pct/100.0): return True, "TP"
            if sl_pct and price <= entry * (1 - sl_pct/100.0): return True, "SL"
        else:
            if tp_pct and price <= entry * (1 - tp_pct/100.0): return True, "TP"
            if sl_pct and price >= entry * (1 + sl_pct/100.0): return True, "SL"
        return False, ""
    except Exception:
        return False, ""


def _should_notify(tf: str, score: float, price: float, curr_bucket: str, last_candle_ts: int,
                   last_sent_ts_map: dict, last_sent_bucket_map: dict,
                   last_sent_score_map: dict, last_sent_price_map: dict):
    """
    True면 전송, False면 억제. 두 번째 값은 억제/허용 사유 문자열.
    """
    prev_bucket = last_sent_bucket_map.get(tf)
    prev_score  = last_sent_score_map.get(tf)
    prev_price  = last_sent_price_map.get(tf)
    prev_ts     = last_sent_ts_map.get(tf, 0)

    # 1) 버킷이 바뀌면 무조건 전송 (중요 이벤트)
    if curr_bucket != prev_bucket:
        return True, "bucket-change"

    # 2) 쿨다운(같은 버킷 유지 시)
    min_gap = int(NOTIFY_CFG['MIN_COOLDOWN_MIN'].get(tf, 10)) * 60
    in_cooldown = (prev_ts and (last_candle_ts - prev_ts) < min_gap)

    # 3) 점수/가격 변화 계산
    dscore = abs(score - prev_score) if (score is not None and prev_score is not None) else None
    price_pct = 0.0
    if isinstance(price, (int, float)) and price > 0 and isinstance(prev_price, (int, float)) and prev_price > 0:
        price_pct = abs(price - prev_price) / price * 100.0

    need_dscore = NOTIFY_CFG['SCORE_DELTA'].get(tf, 1.0)
    need_pmove  = NOTIFY_CFG['PRICE_DELTA_PCT'].get(tf, 0.5)

    # 4) NEUTRAL 지속 억제: 더 센 기준 적용
    if NOTIFY_CFG.get('SUPPRESS_NEUTRAL') and curr_bucket == 'NEUTRAL':
        # 점수 OR 가격 둘 중 하나라도 충분히 움직여야 전송
        ok = ((dscore is not None and dscore >= need_dscore) or (price_pct >= need_pmove))
        return (ok, "neutral-filter" if not ok else "neutral-passed")

    # 5) 일반 케이스: 같은 버킷 + 쿨다운 영역 + 미미한 변화면 억제
    if in_cooldown and (dscore is not None and dscore < need_dscore) and (price_pct < need_pmove):
        return False, f"cooldown/low-change (Δscore={dscore:.2f}, Δprice={price_pct:.2f}%)"

    # 6) 그 외엔 전송 허용
    return True, "passed"

# ===== USD→KRW 환율 유틸 =====
import time
from functools import lru_cache

_FX_CACHE = {"usdkrw": (None, 0.0)}  # (rate, ts)

def _now():
    return time.time()

def _fmt_usd(v):
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return "$-"

def _fmt_krw(v):
    try:
        return f"₩{int(round(float(v))):,}"
    except Exception:
        return "₩-"

def get_usdkrw_rate(max_age_sec: int = 3600) -> float:
    """
    환율 소스 우선순위:
    1) 캐시(<1h)
    2) 업비트 BTC/KRW ÷ 바이낸스 BTC/USDT
       (실패 시 ETH로 동일 계산)
    3) exchangerate.host (HTTP)  ※ 방화벽 환경이면 실패할 수 있음
    4) 폴백 상수 1350.0
    """
    rate, ts = _FX_CACHE.get("usdkrw", (None, 0.0))
    if rate and (_now() - ts) < max_age_sec:
        return float(rate)

    # 2-1) 교차 환산(BTC)
    try:
        import ccxt
        b = ccxt.binance({'enableRateLimit': True})
        u = ccxt.upbit({'enableRateLimit': True})
        btc_usdt = b.fetch_ticker('BTC/USDT')['last']
        btc_krw  = u.fetch_ticker('BTC/KRW')['last']
        rate = float(btc_krw) / float(btc_usdt)
    except Exception:
        rate = None

    # 2-2) 교차 환산(ETH)
    if not rate:
        try:
            import ccxt
            b = ccxt.binance({'enableRateLimit': True})
            u = ccxt.upbit({'enableRateLimit': True})
            eth_usdt = b.fetch_ticker('ETH/USDT')['last']
            eth_krw  = u.fetch_ticker('ETH/KRW')['last']
            rate = float(eth_krw) / float(eth_usdt)
        except Exception:
            rate = None

    # 3) HTTP 환율(있으면)
    if not rate:
        try:
            import requests
            r = requests.get(
                "https://api.exchangerate.host/latest?base=USD&symbols=KRW",
                timeout=4
            )
            rate = float(r.json()['rates']['KRW'])
        except Exception:
            rate = None

    # 4) 폴백
    if not rate:
        rate = 1350.0  # 안전 폴백

    _FX_CACHE["usdkrw"] = (rate, _now())
    return float(rate)

def usd_to_krw(usd_price: float) -> str:
    rate = get_usdkrw_rate()
    return _fmt_krw(float(usd_price) * rate)

# === [STATE / UTILS] =========================================================
import time
from dataclasses import dataclass, field

INTERVAL_MS = {"15m": 15*60*1000, "1h": 60*60*1000, "4h": 4*60*60*1000}

def ts_ms_now() -> int:
    return int(time.time() * 1000)

def floor_to_candle(ts_ms: int, interval_ms: int) -> int:
    return ts_ms - (ts_ms % interval_ms)

@dataclass
class TFState:
    last_processed_open_ms: int | None = None   # 마지막으로 '평가'한 캔들의 open time
    open_position_side: str | None = None       # "LONG"/"SHORT"/None
    open_position_candle_ms: int | None = None  # 포지션 연 캔들의 open time

STATE: dict[tuple[str, str], TFState] = {}      # key: (symbol, tf)

def get_state(symbol: str, tf: str) -> TFState:
    return STATE.setdefault((symbol, tf), TFState())

def should_process(symbol: str, tf: str, open_ms: int) -> bool:
    st = get_state(symbol, tf)
    if st.last_processed_open_ms == open_ms:
        return False  # 같은 캔들 재평가 금지
    st.last_processed_open_ms = open_ms
    return True

def candle_price(kl_last: dict) -> tuple[float, dict]:
    # kl_last dict 구조 가정: keys: open_time, open, high, low, close
    close = float(kl_last["close"])
    high  = float(kl_last["high"])
    low   = float(kl_last["low"])
    meta = {"anomaly": False, "low": low, "high": high, "close": close}
    if not (low <= close <= high):
        meta["anomaly"] = True
        # 이상치면 '주문 금지'를 위해 meta만 True로 반환
    return close, meta

def make_clid(symbol: str, tf: str, open_ms: int, side: str) -> str:
    base = f"bot1:{symbol}:{tf}:{open_ms}:{side}".lower()
    return base[:32]  # 거래소 제약 고려(대개 32~36자)
# =============================================================================


# === 알림 게이팅(억제) 설정 ===
NOTIFY_CFG = {
    # 같은 버킷(BUY/NEUTRAL/SELL)일 때 ‘점수 변화’ 최소폭
    'SCORE_DELTA': {'15m': 0.8, '1h': 1.0, '4h': 1.2, '1d': 1.5},
    # 같은 버킷일 때 ‘가격 변화’ 최소폭(%) — 이전 발송가 기준
    'PRICE_DELTA_PCT': {'15m': 0.6, '1h': 0.6, '4h': 0.6, '1d': 0.5},
    # 같은 버킷일 때 최소 쿨다운(분) — 쿨다운 내엔 사소한 변화는 억제
    'MIN_COOLDOWN_MIN': {'15m': 10, '1h': 20, '4h': 45, '1d': 180},
    # NEUTRAL 지속일 때는 더 강하게 억제
    'SUPPRESS_NEUTRAL': True
}

# === ETH: 마지막 발송 상태 ===
last_sent_ts_eth     = {'15m': 0, '1h': 0, '4h': 0, '1d': 0}
last_sent_bucket_eth = {tf: None for tf in last_sent_ts_eth}
last_sent_score_eth  = {tf: None for tf in last_sent_ts_eth}
last_sent_price_eth  = {tf: None for tf in last_sent_ts_eth}

# BTC 타임프레임 공통 정의
TIMEFRAMES_BTC = ['15m', '1h', '4h', '1d']

# === BTC: 마지막 발송 상태 ===
last_sent_ts_btc     = {tf: 0 for tf in TIMEFRAMES_BTC}
last_sent_bucket_btc = {tf: None for tf in TIMEFRAMES_BTC}
last_sent_score_btc  = {tf: None for tf in TIMEFRAMES_BTC}
last_sent_price_btc  = {tf: None for tf in TIMEFRAMES_BTC}




# === 자동매매 설정 (환경변수로 제어) ===
AUTO_TRADE   = os.getenv("AUTO_TRADE", "0") == "1"
TRADE_MODE   = os.getenv("TRADE_MODE", "paper")   # 'paper' | 'spot' | 'futures'
EXCHANGE_ID  = os.getenv("EXCHANGE_ID", "binance")  # 'binance' | 'binanceusdm'(선물)
SANDBOX      = os.getenv("SANDBOX", "1") == "1"   # True면 테스트넷/샌드박스 모드
RISK_USDT    = float(os.getenv("RISK_USDT", "20"))  # 1회 주문에 사용할 USDT
MIN_NOTIONAL = float(os.getenv("MIN_NOTIONAL", "5"))  # 거래소 최소 체결가 대비 여유치
POS_TF_STRICT   = os.getenv("POS_TF_STRICT","1")=="1"
POS_TF_AUTOREPAIR = os.getenv("POS_TF_AUTOREPAIR","1")=="1"

# 실행 상태(중복 주문 방지)
EXEC_STATE = {}            # key: (symbol, tf) -> {'last_signal': 'BUY'/'SELL', ...}
GLOBAL_EXCHANGE = None     # ccxt 인스턴스 (라이브 모드에서만 사용)

# === Futures fee & funding config (통합/강화) ===
USE_DYNAMIC_FEE        = os.getenv("USE_DYNAMIC_FEE", "1") == "1"
INCLUDE_FEES_IN_PNL    = os.getenv("INCLUDE_FEES_IN_PNL", "0") == "1"
ESTIMATE_FUNDING_IN_PNL= os.getenv("ESTIMATE_FUNDING_IN_PNL", "0") == "1"

FUT_TAKER_FEE_BPS = float(os.getenv("FUT_TAKER_FEE_BPS", "6"))   # 폴백: 0.06%
FUT_MAKER_FEE_BPS = float(os.getenv("FUT_MAKER_FEE_BPS", "2"))   # 폴백: 0.02%

def _market_fee_bps_from_ccxt(ex, symbol, order_type="MARKET"):
    """
    CCXT 마켓 스펙에서 maker/taker 수수료를 bps로 읽기.
    실패하면 None 반환(폴백은 _fee_bps에서 처리).
    """
    try:
        if not ex or not symbol:
            return None
        typ = 'taker' if str(order_type).upper() == "MARKET" else 'maker'
        m = None
        try:
            # ccxt의 통합 인터페이스 우선
            m = ex.market(symbol)
        except Exception:
            pass
        if not m:
            m = (getattr(ex, "markets", {}) or {}).get(symbol) or {}
        fee_frac = m.get(typ)
        if fee_frac is None:
            fee_frac = (m.get('fees', {}).get('trading', {}) or {}).get(typ)
        if fee_frac is None:
            return None
        return float(fee_frac) * 10000.0
    except Exception:
        return None

def _fee_bps(order_type="MARKET", ex=None, symbol=None):
    """
    1) USE_DYNAMIC_FEE=1 이고 ex/symbol 제공되면 CCXT 마켓 수수료 우선
    2) 실패 시 .env 폴백
    """
    if USE_DYNAMIC_FEE and ex and symbol:
        bps = _market_fee_bps_from_ccxt(ex, symbol, order_type)
        if bps is not None:
            return bps
    return FUT_TAKER_FEE_BPS if str(order_type).upper()=="MARKET" else FUT_MAKER_FEE_BPS

def _fee_usdt(price, qty, fee_bps):
    try:
        return float(price) * float(qty) * (float(fee_bps) / 10000.0)
    except Exception:
        return 0.0


# === 마켓 레짐 유틸 (ETH/BTC + BTC Dominance) ===
REGIME_CACHE = {"tf": None, "ts": 0, "val": None}
REGIME_TTL = 180  # 3분 캐시
BTC_DOM_CACHE = {"value": None, "ts": 0}
BTC_DOM_TTL = 300

def get_btc_dominance():
    now = time.time()
    if BTC_DOM_CACHE["value"] and now - BTC_DOM_CACHE["ts"] < BTC_DOM_TTL:
        return BTC_DOM_CACHE["value"]
    try:
        import requests
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=5)
        j = r.json()
        dom = float(j["data"]["market_cap_percentage"]["btc"])
        BTC_DOM_CACHE.update({"value": dom, "ts": now})
        return dom
    except Exception:
        return None

def _get_ethbtc_snapshot(tf='1h'):
    try:
        ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        ex.load_markets()
        ohlcv = ex.fetch_ohlcv('ETH/BTC', timeframe=tf, limit=60)
        import pandas as pd, numpy as np
        d = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','vol'])
        d['ema50'] = d['close'].ewm(span=50, adjust=False).mean()
        slope = (d['close'].iloc[-1] - d['close'].iloc[-5]) / d['close'].iloc[-5] * 100
        bias = 0
        bias += 1 if d['close'].iloc[-1] > d['ema50'].iloc[-1] else -1
        bias += 1 if slope > 0 else -1
        return {"ethbtc": float(d['close'].iloc[-1]), "alt_bias": bias}
    except Exception:
        return {"ethbtc": None, "alt_bias": 0}

def detect_market_regime(tf='1h'):
    now = time.time()
    if REGIME_CACHE["val"] and REGIME_CACHE["tf"] == tf and now - REGIME_CACHE["ts"] < REGIME_TTL:
        return REGIME_CACHE["val"]
    snap = _get_ethbtc_snapshot(tf)
    btc_dom = get_btc_dominance()
    label = "알트 강세" if snap["alt_bias"] >= 1 else "비트코인 강세"
    ctx = {"ethbtc": snap["ethbtc"], "btc_dominance": btc_dom}
    REGIME_CACHE.update({"tf": tf, "ts": now, "val": (label, ctx)})
    return label, ctx

# === 실시간 가격(티커) 유틸 ===
 # === [ANCHOR: PRICE_SNAPSHOT_UTIL] 심볼별 라이브 프라이스 스냅샷 (공통 현재가) ===
PRICE_SNAPSHOT = {}  # {symbol: {"ts": ms, "last": float|None, "bid": float|None, "ask": float|None, "mid": float|None, "mark": float|None, "index": float|None, "chosen": float|None}}
PRICE_SNAPSHOT_TTL_MS = 500  # 동일 틱 처리용 짧은 TTL

async def get_price_snapshot(symbol: str) -> dict:
    """
    Build a unified 'live price snapshot' used across TFs.
    chosen follows PRICE_FALLBACK_ORDER (e.g., mark,last,mid,index).
    In paper mode, we still query a public futures exchange for mark/index.
    """
    now_ms = int(time.time() * 1000)
    rec = PRICE_SNAPSHOT.get(symbol)
    if rec and (now_ms - rec.get("ts", 0) < PRICE_SNAPSHOT_TTL_MS):
        return rec

    last = bid = ask = mid = mark = index = None

    # Prefer authenticated FUT_EXCHANGE; else fall back to PUB_FUT_EXCHANGE; else spot last.
    ex = FUT_EXCHANGE if FUT_EXCHANGE else PUB_FUT_EXCHANGE
    if ex:
        try:
            t = await _post(ex.fetch_ticker, symbol)
            last = float(t.get('last') or 0) or None
            bid  = float(t.get('bid')  or 0) or None
            ask  = float(t.get('ask')  or 0) or None
            try:
                info  = t.get('info', {}) if isinstance(t.get('info'), dict) else {}
                mark  = float(t.get('markPrice')  or info.get('markPrice')  or 0) or None
                index = float(t.get('indexPrice') or info.get('indexPrice') or 0) or None
            except Exception:
                pass
        except Exception:
            pass
    else:
        try:
            last = float(fetch_live_price(symbol) or 0) or None
        except Exception:
            last = None

    if bid and ask:
        try:
            mid = (bid + ask) / 2.0
        except Exception:
            mid = None

    # Decide chosen by fallback order
    chosen = None
    for k in PRICE_FALLBACK_ORDER:
        if k == "mark"  and mark  is not None: chosen = mark;  break
        if k == "last"  and last  is not None: chosen = last;  break
        if k == "mid"   and mid   is not None: chosen = mid;   break
        if k == "index" and index is not None: chosen = index; break

    # Optional: if you never want to use raw 'mark' as the final number for display
    if MARK_CLAMP_TO_LAST and (chosen is not None) and (mark is not None) and (chosen == mark) and (last is not None):
        chosen = last

    PRICE_SNAPSHOT[symbol] = {
        "ts": now_ms, "last": last, "bid": bid, "ask": ask,
        "mid": mid, "mark": mark, "index": index, "chosen": chosen
    }
    return PRICE_SNAPSHOT[symbol]

def fetch_live_price(symbol: str) -> float | None:
    try:
        ex = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}
        })
        t = ex.fetch_ticker(symbol)
        return float(t['last'])
    except Exception:
        return None
    
# [PATCH-④] 로그 기록 전 가격 위생 검사: 마지막 '닫힌' 캔들의 고/저 범위로 클램프
def sanitize_price_for_tf(symbol: str, tf: str, price: float) -> float:
    try:
        df_chk = get_ohlcv(symbol, tf, limit=2)
        if len(df_chk) >= 2:
            row = df_chk.iloc[-2]  # 닫힌 캔들
            lo = float(row['low']); hi = float(row['high'])
            p  = float(price)
            if not (lo <= p <= hi):
                return min(max(p, lo), hi)
    except Exception:
        pass
    return float(price)
    
# --- console-safe logger: ASCII to console, UTF-8 to file ---
import sys, logging, os
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/bot.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    encoding='utf-8'
)



def _symtag(symbol: str) -> str:
    # 'ETH/USDT' -> 'ethusdt', 'BTC/USDT' -> 'btcusdt'
    return symbol.replace('/', '').lower()

def log_path(symbol: str, tf: str) -> str:
    # 'ETH/USDT','1h' -> 'logs/signals_ethusdt_1h.csv'
    return f"logs/signals_{symbol.replace('/','').lower()}_{tf}.csv"

# 심볼/타임프레임 파서
SYMBOL_ALIAS = {'eth': 'ETH/USDT', 'btc': 'BTC/USDT'}
VALID_TFS = ['15m','1h','4h','1d']

def parse_symbol_tf(parts, default_symbol='ETH/USDT', default_tf='1h'):
    """
    예)
      '!상태'            -> (ETH/USDT, 1h)
      '!상태 btc'        -> (BTC/USDT, 1h)
      '!상태 eth 4h'     -> (ETH/USDT, 4h)
      '!리포트 btc 1d'   -> (BTC/USDT, 1d)
    """
    symbol = default_symbol
    tf = default_tf
    if len(parts) >= 2 and parts[1].lower() in SYMBOL_ALIAS:
        symbol = SYMBOL_ALIAS[parts[1].lower()]
        tf = parts[2] if len(parts) >= 3 else default_tf
    else:
        tf = parts[1] if len(parts) >= 2 else default_tf
    if tf not in VALID_TFS:
        raise ValueError(f"지원하지 않는 타임프레임: {tf}")
    return symbol, tf

# --- PDF 리포트 모듈 안전 임포트 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from generate_pdf_report import generate_pdf_report
except Exception as e:
    generate_pdf_report = None
    log(f"[PDF] generate_pdf_report 임포트 실패: {e}")
# -----------------------------------
# [REPORT_CALL_SCAN_BEGIN]


# === 전역 상태 저장용 ===
ENTERED_CANDLE = {}  # key: (symbol, tf) -> last_enter_open_ms

previous_signal = {}  # (symbol, tf) -> 'BUY'/'SELL'/'NEUTRAL'
previous_score = {'15m': None, '1h': None, '4h': None, '1d': None}
score_history = {
    '15m': deque(maxlen=4),
    '1h': deque(maxlen=4),
    '4h': deque(maxlen=4),
    '1d': deque(maxlen=4),
}
previous_price = {'15m': None, '1h': None, '4h': None, '1d': None}
neutral_info = {'15m': None, '1h': None, '4h': None, '1d': None}
entry_data = {}       # (symbol, tf) -> dict(e.g., {"entry": float, ...})
highest_price = {}    # (symbol, tf) -> float
lowest_price = {}     # (symbol, tf) -> float
trail_peak_roe = {}   # (symbol, tf) -> float

# [ANCHOR: LAST_PRICE_GLOBALS]
LAST_PRICE = {}  # symbol -> last/mark price cache

def set_last_price(symbol: str, price: float) -> None:
    try:
        LAST_PRICE[str(symbol).upper()] = float(price)
    except Exception:
        pass

def get_last_price(symbol: str, default_price: float = 0.0) -> float:
    try:
        v = LAST_PRICE.get(str(symbol).upper())
        return float(v) if v is not None else float(default_price)
    except Exception:
        return float(default_price)

previous_bucket = {'15m': None, '1h': None, '4h': None, '1d': None}



# === 손절 익절 하드 스탑(고정 손절) on/off 및 퍼센트(퍼 TF) ===
take_profit_pct   = {'15m':3.0,'1h':6.0,'4h':9.0,'1d':12.0}
trailing_stop_pct = {'15m':1.0,'1h':1.5,'4h':2.0,'1d':3.0}
USE_HARD_STOP  = {'15m':False,'1h':True,'4h':True,'1d':True}
HARD_STOP_PCT  = {'15m':0.0,  '1h':2.0,  '4h':2.5, '1d':4.0}

# 🔹 퍼센트 트레일링 사용 여부(표시/실행 모두 여기에 따름)
USE_TRAILING      = {'15m':False,'1h':False,'4h':True,'1d':True}


# === MA STOP 설정 (TF별 기준/버퍼 & 리밸런싱 스위치) ===
MA_STOP_CFG = {
    'enabled': True,
    # 'close'면 종가 기준, 그 외면 LONG은 저가/SHORT은 고가 기준으로 판정
    'confirm': 'close',
    # 기본 버퍼(개별 TF에 지정 없을 때만 사용)
    'buffer_pct': 0.15,
    # TF별 (MA종류, 기간, 버퍼%)
    'tf_rules': {
        '15m': ('SMA', 20, 0.10),
        '1h' : ('SMA', 20, 0.15),
        '4h' : ('SMA', 50, 0.20),
        '1d' : ('SMA', 100, 0.30),
    },
    # 가격이 MA에서 멀어지면(%) 트레일링으로 스위치
    'rebalance': {
        'switch_to_trailing_at': {'15m': 1.5, '1h': 2.0, '4h': 2.5, '1d': 3.0}
    }
}



# === 캔들 타임스탬프 게이트(중복 방지) ===
# 같은 (symbol, tf, candle_ts)에서 1번만 진입 허용
ENTERED_CANDLE = {}  # key: (symbol, tf) -> candle_ts(int)

last_candle_ts_eth = {'15m': 0, '1h': 0, '4h': 0, '1d': 0}
last_candle_ts_btc = {'15m': 0, '1h': 0, '4h': 0, '1d': 0}


# 같은 캔들에서 허용할 점수 변화 임계치
SCORE_DELTA = {'15m': 0.5, '1h': 0.6, '4h': 0.6, '1d': 0.7}

# 실행 중 에러 핸들링 예시
try:
    # 여기에 초기화나 실행 코드 작성
    pass

except Exception as e:
    log(f"⚠️ 오류 발생: {e}\n{traceback.format_exc()}")



# === Scoring/Threshold Config ===
CFG = {
    "sma_diff_strong": 0.5,   # %  (기존 1.0 → 0.5로 완화)
    "adx_trend_min": 20,      # (기존 25 → 20)
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "rsi_extreme_margin": 10, # 극단치(20/80) 계산용
    "cci_os": -100, "cci_ob": 100, "cci_ext_os": -200, "cci_ext_ob": 200,
    # 등급 경계 (완화)
    "strong_cut": 7,
    "buy_cut": 3,             # 기존 4 → 3
    "sell_cut": -3,           # 기존 -4 → -3
    "strong_sell_cut": -7,
}

# 기능1 점수 기반 전략 등급화
def classify_signal(score):
    if score >= 7:
        return "🔥 STRONG BUY"
    elif score >= 4:
        return "BUY"
    elif score > 0:
        return "WEAK BUY"
    elif score <= -7:
        return "🚨 STRONG SELL"
    elif score <= -4:
        return "SELL"
    elif score < 0:
        return "WEAK SELL"
    else:
        return "NEUTRAL"

# 기능2 누적 수익률 그래프 + 승률 계산
def analyze_performance_for(symbol, tf):
    fp = log_path(symbol, tf)
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp)
    if 'pnl' not in df.columns:
        return None
    df['pnl'] = pd.to_numeric(df['pnl'].astype(str).str.replace('%', ''), errors='coerce')
    df = df.dropna(subset=['pnl'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    # 누적 수익률 계산
    df['cumulative_return'] = (1 + df['pnl'] / 100).cumprod() - 1

    # 승률 계산
    total_trades = len(df)
    wins = len(df[df['pnl'] > 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    log(f"총 트레이드 수: {total_trades}회")
    log(f"승률: {win_rate:.2f}%")


    # 그래프
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['cumulative_return'] * 100, label='누적 수익률 (%)')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("누적 수익률 추이")
    plt.xlabel("시간")
    plt.ylabel("수익률 (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"logs/cumulative_return_{_symtag(symbol)}_{tf}.png")
    plt.close()
    #plt.show()
    return f"logs/cumulative_return_{_symtag(symbol)}_{tf}.png"

# 기능3 지표 조합별 평균 수익률 및 승률 평가
from collections import defaultdict

def analyze_reason_performance(csv_file='logs/signals.csv'):
    df = pd.read_csv(csv_file)
    if 'pnl' not in df.columns:
        return
    df['pnl'] = pd.to_numeric(df['pnl'].astype(str).str.replace('%', ''), errors='coerce')
    df = df.dropna(subset=['pnl'])

    reason_stats = defaultdict(list)

    for _, row in df.iterrows():
        for reason in str(row['reasons']).split(" | "):
            reason_stats[reason].append(row['pnl'])

    log("\n지표별 수익률 및 신호 품질 평가:")
    for reason, pnl_list in reason_stats.items():
        pnl_list = [p for p in pnl_list if abs(p) < 100]  # 이상치 제거
        count = len(pnl_list)
        if count == 0:
            win_rate = 0.0
        else:
            win_rate = sum(1 for p in pnl_list if p > 0) / count * 100  # ✅

        avg_return = sum(pnl_list) / count if count > 0 else 0
        log(f"* {reason}: 평균 수익률 {avg_return:.2f}%, 승률 {win_rate:.1f}% ({count}회)")


# ==== 지표 시너지 규칙 엔진 (리치 버전, '행동' 표기) ====
def _synergy_insights(
    df,
    *,
    # 추세/모멘텀
    adx=None, plus_di=None, minus_di=None,
    rsi=None, macd=None, macd_signal=None, st_dir=None,
    # 가격/레벨
    close=None, ema50=None, ema200=None, kijun=None,
    cloud_top=None, cloud_bot=None, bb_up=None, bb_lo=None,
    # 수급/유동성
    obv_slope=None, mfi=None,
    # 오실레이터
    cci=None,
    # 변동성
    atr_pct=None,
    # 출력 수 제한
    max_items: int = 6,
):
    """
    더 많은 지표를 결합해 '상황 설명 + 해석 + 행동'을 문장으로 만든다.
    사용 지표: ADX, +DI/-DI, EMA50/200, Ichimoku(구름/기준선), MACD, RSI, StochRSI,
             Bollinger, SuperTrend, OBV, MFI, CCI, ATR(%)
    """
    lines = []

    def _has(*xs):
        return all(x is not None for x in xs)

    # 1) 추세 컨플루언스: ADX + 구름 + EMA + DI
    if _has(adx, close, cloud_top, ema50, ema200):
        if adx >= 25 and close > cloud_top and ema50 > ema200 and (plus_di is None or plus_di > (minus_di or 0)):
            lines.append(
                "**지표**: 추세 컨플루언스 강함: ADX≥25에 구름 위·EMA50>EMA200(정배열), (+DI 우위면 더 좋음). \n"
                "- **해석**: **상방 추세**의 질이 높아 모멘텀 **매수 신호의 신뢰도가 상승**. \n"
                "- **행동**: **분할 매수·돌파 추종**, 손절은 기준선 또는 구름 하단 이탈 기준.\n"
            )
        elif adx < 20 and _has(cloud_bot) and cloud_bot <= close <= cloud_top:
            lines.append(
                "**지표**: **추세 힘 약화**: ADX<20에 구름 내부(횡보/혼조). \n"
                "- **해석**: 모멘텀 신호의 노이즈/휩쏘 위험 증가. \n"
                "- **행동**: 확정 돌파 전 **추격 자제**, 박스 상·하단 역추세(= mean-reversion)·짧은 스캘핑 위주.\n"
            )

    # 2) MACD × RSI (모멘텀 저점/고점 결합)
    if _has(macd, macd_signal, rsi):
        if macd > macd_signal and rsi < 40:
            lines.append(
                "**지표**: MACD **상방 전환** + RSI 40 **이하**. \n"
                "- **해석**: **저점 반등 초기일 가능성**. \n"
                "- **행동**: ADX↑/SuperTrend 상방 동반 시 신뢰도↑, 직전 고점/기준선 돌파 **확인 후 접근**.\n"
            )
        if macd < macd_signal and rsi > 60:
            lines.append(
                "**지표**: MACD **하방 전환** + RSI 60 **이상**. \n"
                "- **해석**: 단기 과열→**되돌림 경고**. \n"
                "- **행동**: 손절 타이트, EMA50/기준선 재확인 전 **추격 금지**.\n"
            )

    # 3) Bollinger × StochRSI (밴드 터치의 질)
    if _has(close, bb_lo) and close < bb_lo and 'STOCHRSI_K' in df and 'STOCHRSI_D' in df:
        k_prev = _s_iloc(df['STOCHRSI_K'], -2, None); d_prev = _s_iloc(df['STOCHRSI_D'], -2, None)
        k_now  = _s_iloc(df['STOCHRSI_K'], -1, None); d_now  = _s_iloc(df['STOCHRSI_D'], -1, None)
        if None not in (k_prev, d_prev, k_now, d_now) and (k_prev <= d_prev) and (k_now > d_now) and (k_now < 0.2):
            lines.append(
                "**지표**: 볼린저 **하단 터치** + StochRSI **저점 골든크로스**. \n"
                "- **해석**: 과매도 해소 **반등 신호**. \n"
                "- **행동**: 기준선/EMA50 재진입 확인 후 분할 접근.\n"
            )
    if _has(close, bb_up, rsi) and close > bb_up and rsi > 70:
        lines.append(
            "**지표**: 밴드 **상단 돌파** + RSI **과매수**. \n"
            "- **해석**: **단기 과열**. \n"
            "- **행동**: 분할 익절/트레일링으로 수익 보호, **눌림 확인 후** 재진입.\n"
        )

    # 4) SuperTrend × EMA/구름 (방향 일치성)
    if st_dir is not None and _has(close, ema50, cloud_top, cloud_bot):
        if st_dir == 1 and (close > ema50) and (close > cloud_top):
            lines.append(
                "**지표**: SuperTrend **상방** = EMA·구름 **상방과 일치**. \n"
                "- **해석**: **방향성 일관성↑**, 눌림 후 **재상승** 가능성. \n"
                "- **행동**: EMA50/기준선 **지지 확인** 시 재추세 진입.\n"
            )
        elif st_dir == -1 and (close < ema50) and (close < cloud_bot):
            lines.append(
                "**지표**: SuperTrend **하방** = EMA·구름 **하방과 일치**. \n"
                "- **해석**: **약세 추세**의 무게감 유지. \n"
                "- **행동**: 단기 반등은 저항(EMA50/기준선) 확인 전 **추격 금지**.\n"
            )

    # 5) 수급 컨펌: OBV × MFI
    if obv_slope is not None and obv_slope > 0:
        if mfi is not None and mfi >= 50:
            lines.append(
                "**지표**: OBV 상승 + MFI≥50. \n"
                "- **해석**: **실거래 유입이 추세를 지지**. \n"
                "- **행동**: 분할 추종 유효, **OBV 꺾임은 경계**.\n"
            )
        elif mfi is not None and mfi < 20:
            lines.append(
                "**지표**: OBV 상승이나 MFI<20. \n"
                "- **해석**: 반등 대비 **실제 매수자금 약함**(유동성 취약). \n"
                "- **행동**: 단타 위주·**엄격한 손절**.\n"
            )

    # 6) CCI 극단 + MACD 방향
    if cci is not None and macd is not None and macd_signal is not None:
        if cci < -100 and macd > macd_signal:
            lines.append(
                "**지표**: CCI **침체권** + MACD **상방**. \n"
                "- **해석**: **침체 탈출형 반등**. \n"
                "- **행동**: EMA50 복귀/구름 상단 돌파 동반 시 중기 신뢰도↑.\n"
            )
        if cci > 100 and macd < macd_signal:
            lines.append(
                "**지표**: CCI **과열권** + MACD **하방**. \n"
                "- **해석**: 의미 있는 되돌림 가능. \n"
                "- **행동**: 현물 **익절**·레버리지 **축소/헤지**.\n"
            )

    # 7) ATR(%)로 리스크 톤 조절
    if atr_pct is not None:
        if atr_pct >= 1.5:
            lines.append(
                f"**지표**: 변동성 **고조**(ATR≈{atr_pct:.2f}%). \n"
                "- **해석**: 휩쏘 위험↑. \n"
                "- **행동**: 포지션 **축소·손절** 여유/트레일링 폭 확대.\n"
            )
        elif atr_pct <= 0.6:
            lines.append(
                f"**지표**: 변동성 **저하**(ATR≈{atr_pct:.2f}%). \n"
                "- **해석**: **돌파 실패(페이크) 위험**. \n"
                "- **행동**: 거래량 동반 돌파 확인 전 **진입 지양**.\n"
            )

    return lines[:max_items] if lines else []




# ===== Top 지표 선택 유틸 =====
from typing import Dict, List, Any

# 지표 이름 기본 후보(데이터 컬럼/계산 유무에 맞춰 조정)
DEFAULT_TOP_INDS: List[str] = [
    "RSI", "MACD", "ADX", "StochRSI", "MFI", "OBV", "Bollinger", "EMA"
]

def select_top_indicators(score_map: Dict[str, Any], k: int = 4) -> List[str]:
    """
    score_map: {"RSI":  +2.0, "MACD": +1.5, "ADX": +1.0, ...} 형태(부호/크기로 강도 판단)
               값이 (score, reason) 튜플/리스트여도 됨 -> 첫 번째 값을 점수로 간주
    k: 상위 몇 개 지표를 뽑을지
    """
    if not isinstance(score_map, dict) or not score_map:
        return DEFAULT_TOP_INDS[:k]

    items = []
    for name, v in score_map.items():
        try:
            score = v[0] if isinstance(v, (list, tuple)) else float(v)
        except Exception:
            # 숫자 변환 안되면 스킵
            continue
        items.append((name, abs(float(score))))  # 절대값 큰 순서 = 강도

    if not items:
        return DEFAULT_TOP_INDS[:k]

    items.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in items[:k]]


# ===== 통합 폰트 설정 (한 번만 설정) =====
from matplotlib import font_manager

def _pick_korean_font():
    # OS에 따라 설치됐을 확률이 높은 순서
    candidates = [
        'Malgun Gothic',        # Windows 기본 한글
        'AppleGothic',          # macOS 기본 한글
        'NanumGothic',          # Linux 자주 사용 (설치되어 있어야 함)
        'Noto Sans CJK KR',     # 구글 Noto CJK
        'Noto Sans KR',
        'DejaVu Sans',          # 최후 폴백
    ]
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if any(name.lower() == a.lower() for a in avail):
            return name
    return 'DejaVu Sans'

KOREAN_FONT = _pick_korean_font()

rcParams['font.family'] = [KOREAN_FONT]             # 한글 우선
rcParams['font.sans-serif'] = [KOREAN_FONT]         # sans-serif도 고정
rcParams['axes.unicode_minus'] = False              # 마이너스 부호 깨짐 방지
# ✅ Emoji를 여기 넣지 않습니다. (Emoji를 같이 넣으면 Matplotlib가 통 문자열을 Emoji 폰트로 처리하려고 하면서 경고 발생)

# === 채널 ID 로딩 유틸 (대/소문자 폴백) ===
def _env_int_first(*keys, default=0):
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip() != "":
            try:
                return int(str(v).strip())
            except Exception:
                pass
    return default

# 설정
TOKEN = os.getenv("DISCORD_TOKEN")
# ETH 채널 맵
CHANNEL_IDS = {
    '15m': _env_int_first('CHANNEL_eth_15M', 'CHANNEL_eth_15m', default=0),
    '1h' : _env_int_first('CHANNEL_eth_1H',  'CHANNEL_eth_1h',  default=0),
    '4h' : _env_int_first('CHANNEL_eth_4H',  'CHANNEL_eth_4h',  default=0),
    '1d' : _env_int_first('CHANNEL_eth_1D',  'CHANNEL_eth_1d',  default=0),
}

# BTC 채널 맵
CHANNEL_BTC = {
    '15m': _env_int_first('CHANNEL_btc_15M', 'CHANNEL_btc_15m', default=0),
    '1h' : _env_int_first('CHANNEL_btc_1H',  'CHANNEL_btc_1h',  default=0),
    '4h' : _env_int_first('CHANNEL_btc_4H',  'CHANNEL_btc_4h',  default=0),
    '1d' : _env_int_first('CHANNEL_btc_1D',  'CHANNEL_btc_1d',  default=0),
}
if not TOKEN:
    raise RuntimeError("DISCORD_TOKEN 환경변수가 없습니다. key.env에 DISCORD_TOKEN=... 를 넣어주세요.")

intents = discord.Intents.default()
intents.message_content = True  # ✅ 메시지 읽기 권한 켜기
client = discord.Client(intents=intents)

previous_score_btc  = {tf: None for tf in TIMEFRAMES_BTC}
previous_price_btc  = {tf: None for tf in TIMEFRAMES_BTC}
neutral_info_btc    = {tf: None for tf in TIMEFRAMES_BTC}
score_history_btc   = {tf: deque(maxlen=4) for tf in TIMEFRAMES_BTC}
previous_bucket_btc = {tf: None for tf in TIMEFRAMES_BTC}
last_candle_ts_btc  = {tf: 0 for tf in TIMEFRAMES_BTC}

TRIGGER_STATE = defaultdict(lambda: 'FLAT')  # key: (symbol, tf) -> FLAT/ARMED/CONFIRMED
ARMED_SIGNAL = {}
ARMED_TS = {}

os.makedirs("logs", exist_ok=True)
os.makedirs("images", exist_ok=True)


def get_ohlcv(symbol='ETH/USDT', timeframe='1h', limit=300):
    # CCXT 최신과 바이낸스 응답 포맷 이슈 회피
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',          # 선물/마진 말고 스팟 고정
            'adjustForTimeDifference': True
        },
        # 'proxies': {'http': 'http://...', 'https': 'http://...'},  # 네트워크 필요시
    })
    # 안전 장치: 시장 로딩 실패 시 재시도/대체 엔드포인트
    exchange.load_markets()

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# --- async bridges for blocking ccxt calls ---
# ENV: AIO_CCXT_POOL=1 enables offloading to thread pool
import functools

async def _to_thread(func, *args, **kwargs):
    if os.getenv("AIO_CCXT_POOL", "1") != "1":
        # fallback (sync) – not recommended
        return func(*args, **kwargs)
    loop = asyncio.get_running_loop()
    bound = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, bound)

async def aio_get_ohlcv(symbol: str, timeframe: str, limit: int = 1):
    # wrap existing sync get_ohlcv() with executor
    return await _to_thread(get_ohlcv, symbol, timeframe, limit)

async def _fetch_recent_bar_1m_async(symbol: str):
    """Return last 1m bar (ts, open, high, low, close, vol) without blocking event loop."""
    try:
        ohlc = await aio_get_ohlcv(symbol, "1m", limit=1)
        if hasattr(ohlc, "iloc"):
            row = ohlc.iloc[-1].tolist()
            return row
        return ohlc[-1] if ohlc else None
    except Exception:
        return None



# ===== Fees helpers (env or exchange) =====
def _fee_rates_for(symbol: str):
    """
    Return (maker, taker) as floats (e.g., 0.0002, 0.0004).
    Prefers exchange metadata if FEE_SOURCE=exchange and available.
    """
    if os.getenv("FEE_SOURCE", "env") == "exchange":
        try:
            ex = globals().get("exchange") or globals().get("GLOBAL_EXCHANGE")
            if ex:
                m = ex.market(symbol)
                mk = float(m.get("maker") or os.getenv("FEE_MAKER_RATE", "0.0002"))
                tk = float(m.get("taker") or os.getenv("FEE_TAKER_RATE", "0.0004"))
                return mk, tk
        except Exception:
            pass
    return (
        float(os.getenv("FEE_MAKER_RATE", "0.0002")),
        float(os.getenv("FEE_TAKER_RATE", "0.0004")),
    )


def _role_rate(role: str, maker: float, taker: float) -> float:
    return maker if str(role).lower() == "maker" else taker


def _estimate_fees_usdt(symbol: str, qty: float, entry: float, last: float):
    """
    Returns dict:
      {
        'entry_fee': <USDT>,      # 체결 시점 수수료(가정)
        'exit_fee_est': <USDT>,   # 현재가 기준 청산 수수료(추정)
        'notional_entry': <USDT>,
        'notional_exit': <USDT>,
      }
    """
    mk, tk = _fee_rates_for(symbol)
    rr_in = _role_rate(os.getenv("FEE_ENTRY_ROLE", "taker"), mk, tk)
    rr_out = _role_rate(os.getenv("FEE_EXIT_ROLE", "taker"), mk, tk)
    notional_in = abs(qty) * float(entry)
    notional_out = abs(qty) * float(last or entry)
    return {
        "entry_fee": notional_in * rr_in,
        "exit_fee_est": notional_out * rr_out,
        "notional_entry": notional_in,
        "notional_exit": notional_out,
    }



# === [UTIL] calc_daily_change_pct — 퍼포먼스 스냅샷과 동일식 ===
def calc_daily_change_pct(symbol: str, current_price: float | None) -> float | None:
    """
    퍼포먼스 스냅샷과 동일한 방식으로 1일 변동률을 계산한다.
    식: (현재가 - 전일 종가) / 전일 종가 * 100
    """
    try:
        d1 = get_ohlcv(symbol, '1d', limit=3)
        if d1 is None or len(d1) < 2:
            return None
        prev_close = float(d1['close'].iloc[-2])   # 전일 종가
        curr = float(current_price) if isinstance(current_price, (int, float)) else float(d1['close'].iloc[-1])
        return ((curr - prev_close) / prev_close) * 100.0 if prev_close else None
    except Exception:
        return None


def add_indicators(df):

    # ✅ 이동평균선 (SMA)
    df['SMA5'] = df['close'].rolling(window=5).mean()
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()  # 🔹 MA 스탑 기준선

    # ✅ RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ✅ MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ✅ 볼린저 밴드
    df['BB_MID'] = df['close'].rolling(window=20).mean()
    df['BB_STD'] = df['close'].rolling(window=20).std()
    df['BB_UPPER'] = df['BB_MID'] + (df['BB_STD'] * 2)
    df['BB_LOWER'] = df['BB_MID'] - (df['BB_STD'] * 2)

    # ✅ Ichimoku Cloud
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2  # 전환선

    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2  # 기준선

    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)  # 선행스팬1

    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)  # 선행스팬2

    df['chikou_span'] = df['close'].shift(26)  # 후행스팬

    # ✅ ADX (Average Directional Index)
    df['tr'] = np.maximum(df['high'] - df['low'], 
                   np.maximum(abs(df['high'] - df['close'].shift()), 
                              abs(df['low'] - df['close'].shift())))
    df['plus_dm'] = np.where(
        (df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
        np.maximum(df['high'] - df['high'].shift(), 0), 0)
    df['minus_dm'] = np.where(
        (df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
        np.maximum(df['low'].shift() - df['low'], 0), 0)
    tr14 = df['tr'].rolling(window=14).sum()
    plus_dm14 = df['plus_dm'].rolling(window=14).sum()
    minus_dm14 = df['minus_dm'].rolling(window=14).sum()
    plus_di = 100 * (plus_dm14 / tr14)
    minus_di = 100 * (minus_dm14 / tr14)
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * abs(plus_di - minus_di) / denom
    df['ADX'] = dx.rolling(window=14).mean()
    df['PLUS_DI'] = plus_di
    df['MINUS_DI'] = minus_di

    # ✅ CCI (Commodity Channel Index)
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(window=20).mean()
    md = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['CCI'] = (tp - ma) / (0.015 * md)

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift()),
                               abs(df['low'] - df['close'].shift())))
    df['ATR14'] = tr.rolling(14).mean()

    # === 추가 지표들 ===
    # EMA
    df['EMA50']  = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()

    # Stochastic RSI
    rsi = df['RSI']
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    stoch = (rsi - rsi_min) / (rsi_max - rsi_min)
    df['STOCHRSI'] = stoch.clip(0, 1)
    df['STOCHRSI_K'] = df['STOCHRSI'].rolling(3).mean()
    df['STOCHRSI_D'] = df['STOCHRSI_K'].rolling(3).mean()

    # MFI
    tp2 = (df['high'] + df['low'] + df['close']) / 3
    mf = tp2 * df['volume']
    pos_flow = mf.where(tp2 > tp2.shift(), 0.0).rolling(14).sum()
    neg_flow = mf.where(tp2 < tp2.shift(), 0.0).rolling(14).sum()
    mfr = pos_flow / neg_flow.replace(0, np.nan)
    df['MFI'] = 100 - (100 / (1 + mfr))

    # OBV
    direction = np.sign(df['close'].diff()).fillna(0)
    df['OBV'] = (direction * df['volume']).cumsum()

    # SuperTrend (기본 period=10, multiplier=3)
    period = 10; mult = 3
    hl2 = (df['high'] + df['low']) / 2
    df['_basic_ub'] = hl2 + mult * df['ATR14']
    df['_basic_lb'] = hl2 - mult * df['ATR14']

    # 최종 밴드 계산
    final_ub = df['_basic_ub'].copy()
    final_lb = df['_basic_lb'].copy()
    for i in range(1, len(df)):
        final_ub.iat[i] = (df['_basic_ub'].iat[i] if (df['_basic_ub'].iat[i] < final_ub.iat[i-1]) or (df['close'].iat[i-1] > final_ub.iat[i-1]) else final_ub.iat[i-1])
        final_lb.iat[i] = (df['_basic_lb'].iat[i] if (df['_basic_lb'].iat[i] > final_lb.iat[i-1]) or (df['close'].iat[i-1] < final_lb.iat[i-1]) else final_lb.iat[i-1])

    st = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        if np.isnan(st[i-1]):
            st[i] = final_ub.iat[i] if df['close'].iat[i] <= final_ub.iat[i] else final_lb.iat[i]
        else:
            if (st[i-1] == final_ub.iat[i-1]) and (df['close'].iat[i] <= final_ub.iat[i]):
                st[i] = final_ub.iat[i]
            elif (st[i-1] == final_ub.iat[i-1]) and (df['close'].iat[i] > final_ub.iat[i]):
                st[i] = final_lb.iat[i]
            elif (st[i-1] == final_lb.iat[i-1]) and (df['close'].iat[i] >= final_lb.iat[i]):
                st[i] = final_lb.iat[i]
            elif (st[i-1] == final_lb.iat[i-1]) and (df['close'].iat[i] < final_lb.iat[i]):
                st[i] = final_ub.iat[i]
    df['SUPERTREND'] = np.where(df['close'] >= st, 1, -1)
    df['SUPERTREND_LINE'] = st
    df['SUPERTREND_UB'] = final_ub
    df['SUPERTREND_LB'] = final_lb

    # 마무리
    df = df.ffill().bfill()
    # 시프트로 생기는 뒤쪽 26개 NaN 제거 (Ichimoku)
    if len(df) > 30:
        df = df.iloc[:-26]  # 선행스팬 시프트 여유분 컷
    # 꼭 필요한 지표만 기준으로 NaN 드랍 (과도한 전부 드랍 방지)
    required = [
        'close','SMA5','SMA20','RSI','MACD','MACD_SIGNAL',
        'BB_UPPER','BB_LOWER','tenkan_sen','kijun_sen','senkou_span_a','senkou_span_b','chikou_span',
        'ATR14'
    ]
    existing = [c for c in required if c in df.columns]
    df = df.dropna(subset=existing).reset_index(drop=True)
    
    return df

# === [DOC] 분석 점수 산출 기준 =========================================
# - 기본 가격: 닫힌 캔들 종가(close_for_calc) 사용 (intrabar_confirm 모드에서도 신호/로그는 닫힌 캔들)
# - 지표/가중(예시):
#   • Ichimoku: 구름 위치(+/-1), 전환/기준 교차(+/-0.5), 종가 vs 기준선(+0.5), 치코우 vs 과거가(+/-0.5)
#   • RSI: 과매수/과매도 존, 극단 마진 보정(타임프레임별 임계치 보정)
#   • MACD: 시그널 교차/히스토그램 기여
#   • ADX(+DI/-DI): 추세 강도/방향
#   • StochRSI(K/D): 모멘텀
#   • MFI/OBV/Bollinger/SuperTrend: 보조 기여
# - 버킷 컷오프(CFG):
#   STRONG BUY/BUY/NEUTRAL/SELL/STRONG SELL 경계값은 CFG["strong_cut"], ["buy_cut"], ["sell_cut"], ["strong_sell_cut"] 사용
# - agree_long/agree_short: 상위TF 정렬은 close 값 기준(닫힌 캔들)
# ======================================================================

# === [SCE] Structure Context Engine — A) 수평레벨 ================================
def _sce_atr(df, n):
    try:
        if 'ATR14' in df.columns and n == 14:
            v = float(df['ATR14'].iloc[-1])
            if pd.notna(v):
                return max(1e-8, v)
        tr = np.maximum(df['high'] - df['low'],
                        np.maximum((df['high'] - df['close'].shift()).abs(),
                                   (df['low'] - df['close'].shift()).abs()))
        return max(1e-8, tr.rolling(n).mean().iloc[-1])
    except Exception:
        return 1.0


def _sce_pivots(df, left=2, right=2):
    """Fractal pivots: return (high_pivots, low_pivots) where each pivot=(idx, price)."""
    hp, lp = [], []
    highs = df['high'].values
    lows  = df['low'].values
    n = len(df)
    for i in range(left, n-right):
        hi_seg = highs[i-left:i+right+1]
        lo_seg = lows[i-left:i+right+1]
        if np.argmax(hi_seg) == left:  # local max
            hp.append((i, highs[i]))
        if np.argmin(lo_seg) == left:  # local min
            lp.append((i, lows[i]))
    return hp, lp


def _sce_horizontal_levels(df, tf, atr_len, max_levels=6):
    """ATH/ATL + 최근 피벗H/L(각 3개까지)으로 수평 레벨 구성."""
    levels = []
    try:
        price = float(df['close'].iloc[-1])
        ath = float(df['high'].max()); atl = float(df['low'].min())
        levels.append(('ATH', ath)); levels.append(('ATL', atl))
        hp, lp = _sce_pivots(df, left=2, right=2)
        for t, piv in [('PH', hp), ('PL', lp)]:
            for idx, pr in piv[-3:]:
                levels.append((t, float(pr)))
        # 0.1 ATR 이내 중복 레벨 제거
        uniq, used = [], []
        tol = _sce_atr(df, atr_len) * 0.1
        for t, lv in levels:
            if any(abs(lv-u) <= tol for u in used):
                continue
            uniq.append((t, lv)); used.append(lv)
        levels = uniq[-max_levels:]
    except Exception:
        pass
    return levels


def _sce_best_trendlines(df):
    """최근 피벗 2점으로 상승/하락 추세선 산출. 반환 {'up':(i1,p1,i2,p2), 'down':(...)}"""
    hp, lp = _sce_pivots(df, left=2, right=2)
    up   = lp[-2:] if len(lp) >= 2 else None
    down = hp[-2:] if len(hp) >= 2 else None
    tl = {}
    tl['up']   = (up[0][0], up[0][1], up[1][0], up[1][1])   if up   and len(up)==2   else None
    tl['down'] = (down[0][0], down[0][1], down[1][0], down[1][1]) if down and len(down)==2 else None
    return tl


def _sce_value_on_line(tl, x):
    (i1, p1, i2, p2) = tl
    if i2 == i1: return p2
    m = (p2 - p1) / (i2 - i1); b = p1 - m * i1
    return m * x + b


def _sce_build_df_from_ohlcv(rows):
    return pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])


def _df_last_ts(df) -> int:
    try:
        return int(df['ts'].iloc[-1])
    except Exception:
        return int(time.time()*1000)


def _struct_cache_key(symbol: str, tf: str) -> tuple[str,str]:
    return (symbol, tf)


def _struct_cache_get(symbol: str, tf: str, ts: int):
    if not (os.getenv("STRUCT_CACHE_ENABLE","1") in ("1","true","on")):
        return None
    ent = STRUCT_CACHE.get(_struct_cache_key(symbol, tf))
    if not ent:
        return None
    # 동일 캔들(같은 ts)만 재사용
    if int(ent.get("ts", -1)) != int(ts):
        return None
    # TTL 체크(선택)
    ttl = int(float(os.getenv("STRUCT_CACHE_TTL_SEC","7200")))
    if ttl > 0 and (time.time() - ent.get("mtime", 0)) > ttl:
        return None
    return ent


def _struct_cache_put(symbol: str, tf: str, ts: int, ctx: dict|None, img_path: str|None):
    if not (os.getenv("STRUCT_CACHE_ENABLE","1") in ("1","true","on")):
        return
    STRUCT_CACHE[_struct_cache_key(symbol, tf)] = {
        "ctx": ctx, "img": img_path, "ts": int(ts), "mtime": time.time()
    }

def _mtf_struct_guard(symbol: str, tf: str, side_signal: str):
    """
    상위TF 구조 근접/돌파 컨텍스트를 계산해 'BLOCK/SCOUT/NONE' 과 메시지를 돌려준다.
    반환: {"action": "BLOCK|SCOUT|NONE", "reason": str}
    """
    try:
        if not MTF_STRUCT_BIAS:
            return {"action":"NONE", "reason":""}

        mtf_map = _parse_mtf_map(MTF_STRUCT_MAP_STR)
        higher_list = mtf_map.get(tf, [])
        if not higher_list:
            return {"action":"NONE", "reason":""}

        # 신호 방향
        is_buy  = (side_signal.upper() == "BUY")
        is_sell = (side_signal.upper() == "SELL")

        # 상위TF들 검사
        for htf in higher_list:
            rows = _load_ohlcv(symbol, htf, limit=400)
            if not rows or len(rows) < 50:
                continue
            hdf = _sce_build_df_from_ohlcv(rows)
            ctx = build_struct_context_basic(hdf, htf)
            atr = ctx.get("atr", 0.0) or 0.0
            near = ctx.get("nearest") or {}
            res  = near.get("res"); sup = near.get("sup")

            # 사전 경고(저항/지지까지 남은 거리)
            try:
                if MTF_ALERT_ENABLE and atr > 0:
                    if res and (abs(res[2]) <= MTF_ALERT_PREWARN_ATR):
                        asyncio.create_task(_struct_alert(symbol, htf, f"{htf} 저항까지 {res[2]:.2f}×ATR 남음 — 분할익절/레버 축소 고려"))
                    if sup and (abs(sup[2]) <= MTF_ALERT_PREWARN_ATR):
                        asyncio.create_task(_struct_alert(symbol, htf, f"{htf} 지지까지 {sup[2]:.2f}×ATR 남음 — 분할매수/스탑 여유 고려"))
            except Exception:
                pass

            # 돌파 컨펌/리테스트 대기 알림
            try:
                break_buf = float(cfg_get("STRUCT_BREAK_CLOSE_ATR", "0.2"))
                h_close   = float(hdf['close'].iloc[-1])
                if res and (h_close > float(res[1]) + break_buf*atr):
                    asyncio.create_task(_struct_alert(symbol, htf, f"{htf} 전고/저항 종가 돌파(+{break_buf}×ATR) — 리테스트 대기"))
                if sup and (h_close < float(sup[1]) - break_buf*atr):
                    asyncio.create_task(_struct_alert(symbol, htf, f"{htf} 지지 종가 이탈(−{break_buf}×ATR) — 리바운드 여부 관찰"))
            except Exception:
                pass

            # 진입 바이어스/게이트
            if atr > 0:
                if is_buy and res and (res[2] <= MTF_NEAR_THR_ATR):
                    if MTF_BLOCK_NEAR:
                        return {"action":"BLOCK", "reason":f"{htf} 상단저항 근접({res[2]:.2f}×ATR)"}
                    if MTF_SCOUT_ONLY_NEAR:
                        return {"action":"SCOUT", "reason":f"{htf} 상단저항 근접({res[2]:.2f}×ATR)"}
                if is_sell and sup and (sup[2] <= MTF_NEAR_THR_ATR):
                    if MTF_BLOCK_NEAR:
                        return {"action":"BLOCK", "reason":f"{htf} 하단지지 근접({sup[2]:.2f}×ATR)"}
                    if MTF_SCOUT_ONLY_NEAR:
                        return {"action":"SCOUT", "reason":f"{htf} 하단지지 근접({sup[2]:.2f}×ATR)"}
        return {"action":"NONE", "reason":""}
    except Exception as e:
        log(f"[MTF_GUARD_WARN] {symbol} {tf} {e}")
        return {"action":"NONE", "reason":""}



def build_struct_context_basic(df, tf, atr_len=None,
                               near_thr_atr=None, max_levels=None):
    """PART A: 수평 레벨 근접도 계산. reasons=[(reason, score, key)]."""
    n = len(df)
    atr_len     = int(cfg_get("STRUCT_ATR_LEN", "14")) if atr_len is None else atr_len
    near_thr_atr= float(cfg_get("STRUCT_NEAR_THR_ATR", "0.8")) if near_thr_atr is None else near_thr_atr
    max_levels  = int(cfg_get("STRUCT_MAX_LEVELS", "6")) if max_levels is None else max_levels

    if n < max(50, atr_len+10):
        return {"reasons": [], "levels": [], "nearest": None}

    price = float(df['close'].iloc[-1])
    atr   = _sce_atr(df, atr_len)
    levels = _sce_horizontal_levels(df, tf, atr_len, max_levels=max_levels)

    # 근접도(ATR 배수) 계산
    near_sup = None; near_res = None
    for typ, lv in levels:
        d_atr = abs(price - lv)/max(1e-8, atr)
        if lv <= price:
            if (near_sup is None) or (d_atr < near_sup[2]): near_sup = (typ, lv, d_atr)
        else:
            if (near_res is None) or (d_atr < near_res[2]): near_res = (typ, lv, d_atr)

    reasons = []
    if near_res and near_res[2] <= near_thr_atr:
        reasons.append((f"구조: 상단 저항({near_res[0]} {near_res[1]:.2f})까지 {near_res[2]:.2f}×ATR", -1.0, 'STRUCT_NEAR'))
    if near_sup and near_sup[2] <= near_thr_atr:
        reasons.append((f"구조: 하단 지지({near_sup[0]} {near_sup[1]:.2f})까지 {near_sup[2]:.2f}×ATR", +1.0, 'STRUCT_NEAR'))

    # [B] Trendline proximity & break/이탈
    break_close_atr = float(cfg_get("STRUCT_BREAK_CLOSE_ATR","0.2"))
    tls    = _sce_best_trendlines(df)
    last_x = len(df)-1
    price  = float(df['close'].iloc[-1])
    atr    = _sce_atr(df, atr_len)

    for dirn in ('up','down'):
        if not tls.get(dirn):
            continue
        val   = _sce_value_on_line(tls[dirn], last_x)
        d_atr = abs(price - val)/max(1e-8, atr)

        if d_atr <= near_thr_atr:
            reasons.append((f"추세선({dirn}) 접근: 선가 {val:.2f}, 거리 {d_atr:.2f}×ATR", 0.5, f"TREND_{dirn.upper()}"))

        # 종가 기준 돌파/이탈 컨펌 + ATR 버퍼
        if dirn == 'down' and price > val + break_close_atr*atr:
            reasons.append((f"하락추세선 종가 돌파(+{break_close_atr}×ATR 버퍼) — 리테스트 대기", 1.0, "TREND_BREAK"))
        if dirn == 'up' and price < val - break_close_atr*atr:
            reasons.append((f"상승추세선 종가 하향 이탈(−{break_close_atr}×ATR 버퍼)", -1.0, "TREND_BREAK"))

    return {
        "reasons": reasons,
        "levels": levels,
        "nearest": {"res": near_res, "sup": near_sup},
        "atr": atr,
    }


def calculate_signal(df, tf, symbol):

    # 데이터 길이 체크
    if len(df) < 50:
        close_for_calc = df['close'].iloc[-1] if len(df) > 0 else 0
        live_price = fetch_live_price(symbol)
        if live_price is None:
            live_price = float(close_for_calc) if len(df) > 0 else None
        return 'NEUTRAL', live_price, 50, 0, [], 0, {}, 0, 0, {}

    # === [PATCH-②] 닫힌 캔들만 사용 ===
    # ccxt의 OHLCV는 맨 끝 행이 '진행 중' 캔들이라서 항상 -2(직전 캔들)를 본다.
    idx = -2 if len(df) >= 2 else -1
    row = df.iloc[idx]

    # 신호/로그용 가격은 닫힌 캔들의 종가로 고정
    close_for_calc = float(row['close'])
    hi_for_check   = float(row['high'])
    lo_for_check   = float(row['low'])

    # (표시용 실시간 가격은 별도로 쓸 수 있지만, 신호·로그에는 close_for_calc만 사용)
    price_for_signal = close_for_calc


    score = 0
    weights = {}
    weights_detail = {}
    strength = []

    # (합의 조건)
    agree_long = 0
    agree_short = 0

    # 타임프레임별 조건
    if tf in ["15m", "1h"]:
        rsi_buy_th = CFG["rsi_oversold"] + 5
        rsi_sell_th = CFG["rsi_overbought"] - 5
        rsi_extreme_margin = CFG["rsi_extreme_margin"] - 2
        adx_trend_min = CFG["adx_trend_min"] - 5
    else:
        rsi_buy_th = CFG["rsi_oversold"]
        rsi_sell_th = CFG["rsi_overbought"]
        rsi_extreme_margin = CFG["rsi_extreme_margin"]
        adx_trend_min = CFG["adx_trend_min"]

    # 지표값 추출
    sma5 = df['SMA5'].iloc[-1]
    sma20 = df['SMA20'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_SIGNAL'].iloc[-1]
    bb_upper = df['BB_UPPER'].iloc[-1]
    bb_lower = df['BB_LOWER'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    plus_di = df['PLUS_DI'].iloc[-1]
    minus_di = df['MINUS_DI'].iloc[-1]
    cci = df['CCI'].iloc[-1]
    senkou_a = df['senkou_span_a'].iloc[-1]
    senkou_b = df['senkou_span_b'].iloc[-1]
    chikou = df['chikou_span'].iloc[-1]

    # === [SCE:A] 구조 컨텍스트(수평 레벨) 반영 ===
    try:
        if STRUCT_ENABLE:
            sce = build_struct_context_basic(
                df, tf,
                atr_len=STRUCT_ATR_LEN,
                near_thr_atr=STRUCT_NEAR_THR_ATR,
                max_levels=STRUCT_MAX_LEVELS,
            )
            for reason, sc, key in sce.get("reasons", []):
                strength.append(reason)
                score += sc
                weights[key] = weights.get(key, 0) + sc
                weights_detail[key] = (weights[key], reason)
    except Exception as _e:
        log(f"[SCE_ERR:A] {symbol} {tf} {type(_e).__name__}: {_e}")

    past_price = df['close'].iloc[-26] if len(df) >= 27 else close_for_calc

    # ===== SMA =====
    sma_diff = (sma5 - sma20) / sma20 * 100 if sma20 else 0
    if sma_diff > CFG["sma_diff_strong"]:
        reason = "SMA 강한 골든크로스"
        sc = 1.5
    elif sma_diff > 0:
        reason = "SMA 골든크로스"
        sc = 1.0
    else:
        reason = "SMA 데드크로스"
        sc = -1.0
    strength.append(reason)
    score += sc
    weights['SMA'] = weights.get('SMA', 0) + sc
    weights_detail['SMA'] = (weights['SMA'], reason)

    # ===== RSI =====
    if rsi < rsi_buy_th - rsi_extreme_margin:
        reason = f"RSI 극단적 과매도 ({rsi:.1f})"
        sc = 2
    elif rsi < rsi_buy_th:
        reason = f"RSI 과매도 ({rsi:.1f})"
        sc = 1
    elif rsi > rsi_sell_th + rsi_extreme_margin:
        reason = f"RSI 극단적 과매수 ({rsi:.1f})"
        sc = -2
    elif rsi > rsi_sell_th:
        reason = f"RSI 과매수 ({rsi:.1f})"
        sc = -1
    else:
        reason = f"RSI 중립 ({rsi:.1f})"
        sc = 0
    strength.append(reason)
    score += sc
    weights['RSI'] = weights.get('RSI', 0) + sc
    weights_detail['RSI'] = (weights['RSI'], reason)

    # ===== MACD =====
    macd_diff = macd - macd_signal
    if macd_diff > 0 and macd > 0:
        reason = "MACD 상승(0 위)"
        sc = 1.5
    elif macd_diff > 0:
        reason = "MACD 상승(0 아래)"
        sc = 1.0
    elif macd_diff < 0 and macd < 0:
        reason = "MACD 하락(0 아래)"
        sc = -1.5
    else:
        reason = "MACD 하락(0 위)"
        sc = -1.0
    strength.append(reason)
    score += sc
    weights['MACD'] = weights.get('MACD', 0) + sc
    weights_detail['MACD'] = (weights['MACD'], reason)

    # ===== Bollinger =====
    if close_for_calc < bb_lower:
        reason = "볼린저 하단 돌파"
        sc = 1.0
    elif close_for_calc > bb_upper:
        reason = "볼린저 상단 돌파"
        sc = -1.0
    else:
        reason = "볼린저 밴드 내"
        sc = 0
    strength.append(reason)
    score += sc
    weights['Bollinger'] = weights.get('Bollinger', 0) + sc
    weights_detail['Bollinger'] = (weights['Bollinger'], reason)

    # ===== Ichimoku =====
    cloud_top = max(senkou_a, senkou_b)
    cloud_bot = min(senkou_a, senkou_b)
    sc_total = 0
    if close_for_calc > cloud_top:
        reason = "일목: 구름 상단 돌파"
        sc_total += 1.0
    elif close_for_calc < cloud_bot:
        reason = "일목: 구름 하단"
        sc_total -= 1.0
    else:
        reason = "일목: 구름 내부(혼조)"
    strength.append(reason)

    if df['tenkan_sen'].iloc[-1] > df['kijun_sen'].iloc[-1]:
        sc_total += 0.5
    else:
        sc_total -= 0.5
    if close_for_calc > df['kijun_sen'].iloc[-1]:
        sc_total += 0.5
    if chikou > past_price:
        sc_total += 0.5
    else:
        sc_total -= 0.5

    score += sc_total
    weights['Ichimoku'] = weights.get('Ichimoku', 0) + sc_total
    weights_detail['Ichimoku'] = (weights['Ichimoku'], reason)

    # ===== ADX =====
    if adx > CFG["adx_trend_min"]:
        if plus_di > minus_di:
            reason = "ADX 상승 추세"
            sc = 1.0
        else:
            reason = "ADX 하락 추세"
            sc = -1.0
    else:
        reason = "ADX 약한 추세"
        sc = 0
    strength.append(reason)
    score += sc
    weights['ADX'] = weights.get('ADX', 0) + sc
    weights_detail['ADX'] = (weights['ADX'], reason)

    # ===== CCI =====
    if cci < CFG["cci_ext_os"]:
        reason = "CCI 극단적 과매도"
        sc = 1.5
    elif cci < CFG["cci_os"]:
        reason = "CCI 과매도"
        sc = 1.0
    elif cci > CFG["cci_ext_ob"]:
        reason = "CCI 극단적 과매수"
        sc = -1.5
    elif cci > CFG["cci_ob"]:
        reason = "CCI 과매수"
        sc = -1.0
    else:
        reason = "CCI 중립"
        sc = 0
    strength.append(reason)
    score += sc
    weights['CCI'] = weights.get('CCI', 0) + sc
    weights_detail['CCI'] = (weights['CCI'], reason)

    # ===== EMA Trend =====
    ema50 = df['EMA50'].iloc[-1]
    ema200 = df['EMA200'].iloc[-1]
    if close_for_calc > ema50 > ema200:
        reason = "EMA 추세 ↑ (Close>EMA50>EMA200)"
        sc = 1.5
    elif close_for_calc < ema50 < ema200:
        reason = "EMA 추세 ↓ (Close<EMA50<EMA200)"
        sc = -1.5
    else:
        reason = "EMA 혼조"
        sc = 0.0
    strength.append(reason)
    score += sc
    weights['EMA'] = weights.get('EMA', 0) + sc
    weights_detail['EMA'] = (weights['EMA'], reason)

    # ===== SuperTrend =====
    st_dir = df['SUPERTREND'].iloc[-1]
    if st_dir == 1:
        reason = "SuperTrend 상방"
        sc = 1.0
    else:
        reason = "SuperTrend 하방"
        sc = -1.0
    strength.append(reason)
    score += sc
    weights['SuperTrend'] = weights.get('SuperTrend', 0) + sc
    weights_detail['SuperTrend'] = (weights['SuperTrend'], reason)

    # ===== StochRSI =====
    k_now = _last(df, 'STOCHRSI_K', None)
    d_now = _last(df, 'STOCHRSI_D', None)
    k_prev = _s_iloc(df['STOCHRSI_K'], -2, k_now) if 'STOCHRSI_K' in df else None
    d_prev = _s_iloc(df['STOCHRSI_D'], -2, d_now) if 'STOCHRSI_D' in df else None
    cross_up = (k_prev is not None and d_prev is not None and k_now is not None and d_now is not None and (k_prev <= d_prev) and (k_now > d_now))
    cross_dn = (k_prev is not None and d_prev is not None and k_now is not None and d_now is not None and (k_prev >= d_prev) and (k_now < d_now))

    if cross_up and k_now < 0.2:
        reason = "StochRSI 저점 크로스(매수)"
        sc = 1.0
    elif cross_dn and k_now > 0.8:
        reason = "StochRSI 고점 크로스(매도)"
        sc = -1.0
    else:
        reason = f"StochRSI 중립(K={k_now:.2f},D={d_now:.2f})"
        sc = 0.0
    strength.append(reason)
    score += sc
    weights['StochRSI'] = weights.get('StochRSI', 0) + sc
    weights_detail['StochRSI'] = (weights['StochRSI'], reason)

    # ===== MFI =====
    mfi = df['MFI'].iloc[-1]
    if mfi < 20:
        reason = f"MFI 과매도({mfi:.1f})"
        sc = 0.5
    elif mfi > 80:
        reason = f"MFI 과매수({mfi:.1f})"
        sc = -0.5
    else:
        reason = f"MFI 중립({mfi:.1f})"
        sc = 0.0
    strength.append(reason)
    score += sc
    weights['MFI'] = weights.get('MFI', 0) + sc
    weights_detail['MFI'] = (weights['MFI'], reason)

    # ===== OBV 기울기 =====
    obv_last = _last(df, 'OBV', 0.0)
    obv_prev5 = _s_iloc(df['OBV'], -5, obv_last) if 'OBV' in df else obv_last
    obv_slope = (obv_last - obv_prev5)
    if obv_slope > 0:
        reason = "OBV↑ (수급 우호)"
        sc = 0.5
    else:
        reason = "OBV↓ (수급 약세)"
        sc = -0.5
    strength.append(reason)
    score += sc
    weights['OBV'] = weights.get('OBV', 0) + sc
    weights_detail['OBV'] = (weights['OBV'], reason)

    # 롱/숏 카운트
    agree_long += 1 if sma5 > sma20 else 0
    agree_long += 1 if (macd > macd_signal) else 0
    agree_long += 1 if close_for_calc > cloud_top else 0
    agree_long += 1 if plus_di > minus_di else 0
    agree_long += 1 if rsi < CFG["rsi_oversold"] else 0
    agree_long += 1 if cci < CFG["cci_os"] else 0

    agree_short += 1 if sma5 < sma20 else 0
    agree_short += 1 if (macd < macd_signal) else 0
    agree_short += 1 if close_for_calc < cloud_bot else 0
    agree_short += 1 if minus_di > plus_di else 0
    agree_short += 1 if rsi > CFG["rsi_overbought"] else 0
    agree_short += 1 if cci > CFG["cci_ob"] else 0

    # === Context-aware score adjustment (1d regime/structure) ===
    try:
        _raw_score = float(score)
        _adj_score, _ctx = _adjust_score_with_ctx(symbol, tf, _raw_score)
        score = _adj_score
        if _ctx:
            log(f"[CTX_ADJ] {symbol} {tf} raw={_raw_score:.2f} -> adj={_adj_score:.2f} bias={_ctx.get('ctx_bias'):.2f} regime={_ctx.get('regime')}")
    except Exception as e:
        log(f"[CTX_ADJ_ERR] {symbol} {tf} {e}")

    # 등급 판정
    if rsi < (rsi_buy_th - rsi_extreme_margin) and macd > macd_signal:
        signal = 'STRONG BUY'
    elif rsi > (rsi_sell_th + rsi_extreme_margin) and macd < macd_signal:
        signal = 'STRONG SELL'
    elif score >= CFG["strong_cut"] and agree_long >= 3 and (adx >= adx_trend_min or close_for_calc > cloud_top):
        signal = 'STRONG BUY'
    elif score >= CFG["buy_cut"] and agree_long >= 2:
        signal = 'BUY'
    elif score <= CFG["strong_sell_cut"] and agree_short >= 3 and (adx >= adx_trend_min or close_for_calc < cloud_bot):
        signal = 'STRONG SELL'
    elif score <= CFG["sell_cut"] and agree_short >= 2:
        signal = 'SELL'
    else:
        signal = 'NEUTRAL'

    # ATR 가중치
    atr = df['ATR14'].iloc[-1]
    vol_regime = 'high' if (atr / close_for_calc) > 0.01 else 'low'
    if vol_regime == 'high':
        for ind in ['ADX', 'Ichimoku']:
            if ind in weights:
                bump = 0.2 * np.sign(weights[ind])
                weights[ind] += bump
                score += bump

    # 🔹 가중치 적용 후 weights_detail 값 업데이트
    for ind in weights_detail.keys():
        if ind in weights:
            old_reason = weights_detail[ind][1]  # 기존 이유 유지
            weights_detail[ind] = (weights[ind], old_reason)

    # 최근 신호 중복 방지
    global last_signals
    if 'last_signals' not in globals():
        last_signals = {}
    last_sig, last_score = last_signals.get(tf, (None, None))
    if last_sig == signal and abs(score - last_score) < 0.5:
        return 'NEUTRAL', close_for_calc, rsi, macd, ["최근 동일 신호 감지됨"], score, weights, agree_long, agree_short, weights_detail
    last_signals[tf] = (signal, score)
    

    return signal, price_for_signal, rsi, macd, strength, score, weights, agree_long, agree_short, weights_detail




def save_chart(df, symbol, timeframe):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    # 데이터 체크
    if _len(df) < 2:
        return None

    fig = None
    filename = None
    try:
        # 패널 구성:
        # 0) 가격 + SMA5/20 + EMA50/200 + Bollinger + SuperTrend(라인/밴드) + (OBV는 보조축 라인)
        # 1) RSI
        # 2) MACD(+히스토그램)
        # 3) ADX/+DI/-DI
        # 4) StochRSI(K/D)
        # 5) MFI
        fig, axs = plt.subplots(6, 1, figsize=(14, 20), sharex=True)
        fig.suptitle(f'{symbol} - {timeframe}', fontsize=16)

        # (0) 가격/추세
        ax0 = axs[0]
        ax0.plot(df['timestamp'], df['close'], label='가격', linewidth=1.2, color='black')
        if 'SMA5' in df:  ax0.plot(df['timestamp'], df['SMA5'],  label='SMA5',  linewidth=1.0)
        if 'SMA20' in df: ax0.plot(df['timestamp'], df['SMA20'], label='SMA20', linewidth=1.0)
        if 'EMA50' in df: ax0.plot(df['timestamp'], df['EMA50'], label='EMA50', linewidth=1.0)
        if 'EMA200' in df:ax0.plot(df['timestamp'], df['EMA200'],label='EMA200',linewidth=1.0)

        # Bollinger Band
        if 'BB_UPPER' in df and 'BB_LOWER' in df:
            ax0.fill_between(df['timestamp'], df['BB_UPPER'], df['BB_LOWER'], alpha=0.15, label='Bollinger')

        # SuperTrend line & bands
        if 'SUPERTREND_LINE' in df:
            ax0.plot(df['timestamp'], df['SUPERTREND_LINE'], label='SuperTrend 라인', linewidth=1.0)
        if 'SUPERTREND_UB' in df and 'SUPERTREND_LB' in df:
            ax0.fill_between(df['timestamp'], df['SUPERTREND_UB'], df['SUPERTREND_LB'], alpha=0.10, label='ST 밴드')

        # OBV 보조축(상대적 흐름만 보려는 용도)
        if 'OBV' in df:
            ax0b = ax0.twinx()
            obv_norm = (df['OBV'] - df['OBV'].min()) / max((df['OBV'].max() - df['OBV'].min()), 1e-9)
            ax0b.plot(df['timestamp'], obv_norm, linewidth=0.8, alpha=0.4, label='OBV(정규화)')
            ax0b.set_ylabel('OBV(norm)')
        ax0.set_ylabel('Price')
        ax0.legend(loc='upper left')

        # (1) RSI
        ax1 = axs[1]
        if 'RSI' in df:
            ax1.plot(df['timestamp'], df['RSI'], label='RSI')
            ax1.axhline(70, linestyle='--'); ax1.axhline(30, linestyle='--')
            ax1.legend(loc='upper left')
        ax1.set_ylabel('RSI')

        # (2) MACD + histogram
        ax2 = axs[2]
        if 'MACD' in df and 'MACD_SIGNAL' in df:
            ax2.plot(df['timestamp'], df['MACD'], label='MACD')
            ax2.plot(df['timestamp'], df['MACD_SIGNAL'], label='Signal')
            # 히스토그램 막대 폭(일 단위 float) 계산
            td = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / max((_len(df)-1), 1)
            try:
                barw = td.total_seconds() / 86400.0
            except Exception:
                barw = 0.002  # fallback
            ax2.bar(df['timestamp'], df['MACD'] - df['MACD_SIGNAL'], alpha=0.4, width=barw, label='Hist')
            ax2.legend(loc='upper left')
        ax2.set_ylabel('MACD')

        # (3) ADX & DI
        ax3 = axs[3]
        if 'ADX' in df:      ax3.plot(df['timestamp'], df['ADX'], label='ADX')
        if 'PLUS_DI' in df:  ax3.plot(df['timestamp'], df['PLUS_DI'], label='+DI')
        if 'MINUS_DI' in df: ax3.plot(df['timestamp'], df['MINUS_DI'], label='-DI')
        ax3.legend(loc='upper left')
        ax3.set_ylabel('ADX / DI')

        # (4) StochRSI
        ax4 = axs[4]
        if 'STOCHRSI_K' in df: ax4.plot(df['timestamp'], df['STOCHRSI_K'], label='%K')
        if 'STOCHRSI_D' in df: ax4.plot(df['timestamp'], df['STOCHRSI_D'], label='%D')
        ax4.axhline(0.8, linestyle='--'); ax4.axhline(0.2, linestyle='--')
        ax4.legend(loc='upper left')
        ax4.set_ylabel('StochRSI')

        # (5) MFI
        ax5 = axs[5]
        if 'MFI' in df:
            ax5.plot(df['timestamp'], df['MFI'], label='MFI')
            ax5.axhline(80, linestyle='--'); ax5.axhline(20, linestyle='--')
            ax5.legend(loc='upper left')
        ax5.set_ylabel('MFI')
        ax5.set_xlabel('Time')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        safe_symbol = symbol.replace("/", "").lower()
        filename = f"images/chart_{safe_symbol}_{timeframe}.png"
        plt.savefig(filename)
        return filename

    except Exception as e:
        log(f"❌ 차트 생성 실패 ({symbol} {timeframe}): {e}")
        return None
    finally:
        try:
            if fig is not None:
                plt.close(fig)
        except Exception:
            try:
                plt.close('all')
            except Exception:
                pass



def ichimoku_analysis(df):
    n = _len(df)
    if n < 60:  # 52 롤링 + 26 시프트 여유
        return ["데이터 부족으로 Ichimoku 요약 생략"]

    try:
        last = df.iloc[-1]
    except Exception:
        return ["데이터 부족으로 Ichimoku 요약 생략"]

    analysis = []

    # 전환선 vs 기준선
    try:
        if last['tenkan_sen'] > last['kijun_sen']:
            analysis.append("전환선 > 기준선: **단기 강세** 흐름")
        else:
            analysis.append("전환선 < 기준선: **단기 약세** 흐름")
    except Exception:
        analysis.append("전환선/기준선: 데이터 부족")

    # 현재가 vs 구름
    try:
        if last['close'] > last['senkou_span_a'] and last['close'] > last['senkou_span_b']:
            analysis.append("현재가 > 구름대: **상승장 지속**")
        elif last['close'] < last['senkou_span_a'] and last['close'] < last['senkou_span_b']:
            analysis.append("현재가 < 구름대: **하락장 지속**")
        else:
            analysis.append("현재가 구름대 내부: 혼조세")
    except Exception:
        analysis.append("구름대 비교: 데이터 부족")

    # 후행스팬 비교(26봉 전)
    try:
        if n >= 27 and last['chikou_span'] > df['close'].iloc[-26]:
            analysis.append("후행스팬 > 과거 가격: **강세 지속 신호**")
        else:
            analysis.append("후행스팬 < 과거 가격: **약세 신호**")
    except Exception:
        analysis.append("후행스팬 비교: 데이터 부족")

    return analysis




# ==== 퍼포먼스 스냅샷 빌더 ====
def build_performance_snapshot(
    tf, symbol, display_price, *,
    daily_change_pct=None,      # format_signal_message에서 넘겨줌
    recent_scores=None          # 최근 점수 리스트(예: [2.1, 2.4, ...])
) -> str:
    """
    짧고 실용적인 성과 요약:
      - 가격(USD/KRW)
      - 1일/7일/30일 변동률 (+ 일중 변동률이 있으면 같이)
      - 해당 TF의 누적 수익/승률/총 트레이드
      - 최근 점수 흐름(있으면)
    """
    # 안전 포맷터
    def _pct(v):
        return "-" if v is None else f"{v:+.2f}%"

    # 전일/주간/월간 변동률 계산(일봉 데이터 기준)
    d1 = None
    try:
        d1 = get_ohlcv(symbol, '1d', limit=90)
    except Exception:
        d1 = None

    def _chg_k_days_ago(k):
        try:
            if d1 is None or len(d1) <= (k+1): 
                return None
            prev = float(d1['close'].iloc[-(k+1)])
            curr = float(display_price) if isinstance(display_price, (int, float)) else float(d1['close'].iloc[-1])
            return ((curr - prev) / prev) * 100.0 if prev else None
        except Exception:
            return None

    chg_1d  = _chg_k_days_ago(1)     # 전일 대비
    chg_7d  = _chg_k_days_ago(7)     # 1주
    chg_30d = _chg_k_days_ago(30)    # 1개월

    # 성과 요약(해당 TF 로그 기반)
    perf = None
    try:
        perf = get_latest_performance_summary(symbol, tf)  # {'return','win_rate','total_trades'}
    except Exception:
        perf = None

    # 최근 점수 표시
    score_line = "-"
    if recent_scores and isinstance(recent_scores, (list, tuple)):
        try:
            score_line = " → ".join(f"{float(s):.1f}" for s in recent_scores[-5:])
        except Exception:
            score_line = "-"

    # 본문 구성
    sym = (symbol or "ETH/USDT").split('/')[0].upper()
    tf_tag = tf.upper()
    usd_str = _fmt_usd(display_price) if isinstance(display_price, (int, float)) else "$-"
    krw_str = usd_to_krw(display_price) if isinstance(display_price, (int, float)) else "₩-"

    lines = []
    lines.append("## 📈 **퍼포먼스 스냅샷**")
    lines.append(f"**가격**: {usd_str} / {krw_str}")
    # 일중(daily_change_pct) 있으면 함께 표시
    intra = _pct(daily_change_pct) if isinstance(daily_change_pct, (int, float)) else "-"
    lines.append(f"**변동률**: 1D { _pct(chg_1d) } · 7D { _pct(chg_7d) } · 30D { _pct(chg_30d) } · 일중 {intra}")

    if perf:
        lines.append(f"**성과({sym}-{tf_tag})**: 누적수익 {perf['return']:+.2f}% · 승률 {perf['win_rate']:.1f}% · 트레이드 {perf['total_trades']}회")
    else:
        lines.append(f"**성과({sym}-{tf_tag})**: 데이터 없음")

    lines.append(f"**최근 점수 흐름**: {score_line}")

    return "\n".join(lines)


def save_ichimoku_chart(df, symbol, timeframe):
    import matplotlib
    import matplotlib.pyplot as plt

    if _len(df) < 2:
        return None

    fig = None
    filename = None
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df['timestamp'], df['close'], label='종가', linewidth=1.2)
        if 'tenkan_sen' in df:  ax.plot(df['timestamp'], df['tenkan_sen'], label='전환선', linewidth=1.2)
        if 'kijun_sen' in df:   ax.plot(df['timestamp'], df['kijun_sen'], label='기준선', linewidth=1.2)
        if 'chikou_span' in df: ax.plot(df['timestamp'], df['chikou_span'], label='후행스팬', linewidth=1.0)
        if 'senkou_span_a' in df and 'senkou_span_b' in df:
            ax.plot(df['timestamp'], df['senkou_span_a'], label='선행스팬A', alpha=0.6, linewidth=1.0)
            ax.plot(df['timestamp'], df['senkou_span_b'], label='선행스팬B', alpha=0.6, linewidth=1.0)
            ax.fill_between(df['timestamp'], df['senkou_span_a'], df['senkou_span_b'],
                            where=(df['senkou_span_a'] >= df['senkou_span_b']),
                            alpha=0.25)
            ax.fill_between(df['timestamp'], df['senkou_span_a'], df['senkou_span_b'],
                            where=(df['senkou_span_a'] < df['senkou_span_b']),
                            alpha=0.25)
        ax.set_title(f"Ichimoku Cloud - {symbol} {timeframe}")
        ax.legend(loc='upper left')

        safe_symbol = symbol.replace("/", "").lower()
        filename = f'images/ichimoku_{safe_symbol}_{timeframe}.png'
        plt.tight_layout()
        plt.savefig(filename)
        return filename

    except Exception as e:
        log(f"❌ Ichimoku 차트 생성 실패 ({symbol} {timeframe}): {e}")
        return None
    finally:
        try:
            if fig is not None:
                plt.close(fig)
        except Exception:
            try:
                plt.close('all')
            except Exception:
                pass



# === [NEW] 차트 읽는 법 텍스트 ===--------------------------
def _chart_howto_text(group="A"):
    if group == "A":
        return (
            "① 가격/추세: 검정=가격, EMA50(파랑)/EMA200(빨강), SMA20(회색), BB(옅은 채움)\n"
            "   · 가격>EMA50>EMA200 & SuperTrend 상방 → 상승 추세 신뢰↑\n"
            "   · BB 상단=과열 경계, 하단=반등 후보\n"
            "② Ichimoku: 구름 위=상승/아래=하락, 전환선>기준선=단기 강세"
        )
    if group == "B":
        return (
            "RSI & MACD 읽기\n"
            "· RSI: 30/70 점선, 50축 재진입 방향 주목\n"
            "· MACD: Signal 상향교차 + 0선 위=상승 모멘텀 강화, Hist 0선 상향=추세 강화"
        )
    if group == "C":
        return (
            "ADX/DI & StochRSI\n"
            "· ADX>20=추세장, +DI>-DI면 상승 우위\n"
            "· StochRSI: 0.2↓ 골든=저점/0.8↑ 데드=고점"
        )
    return (
        "MFI(자금흐름)\n"
        "· 80↑ 과매수/20↓ 과매도, 다른 모멘텀과 함께 확인\n"
        "· (선택) OBV↑ 동반시 실제 유입 근거 강화"
    )

# 내부 유틸
def _bar_width_from_time_index(df):
    try:
        td = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / max((len(df)-1), 1)
        return td.total_seconds() / 86400.0  # days
    except Exception:
        return 0.002

# === [NEW] 2·2·2·1 분할 차트 저장 ===
def save_chart_groups(df, symbol, timeframe, outdir="images"):
    """
    4장의 PNG를 생성해 경로 리스트를 반환:
      A: Trend(가격+이동평균+BB+ST) / Ichimoku(요약)
      B: RSI / MACD
      C: ADX&DI / StochRSI
      D: MFI (필요시 OBV 보조축로 확장 가능)
    """
    import matplotlib.pyplot as plt
    import os

    if len(df) < 2:
        return []

    os.makedirs(outdir, exist_ok=True)
    sym = symbol.replace("/", "").lower()
    paths = []

    # ---------- A. Trend ----------
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{symbol} - {timeframe} · Trend", fontsize=14)

    ax = axs[0]
    ax.plot(df['timestamp'], df['close'], color='black', linewidth=1.9, label="가격")

    if 'SMA20' in df:   ax.plot(df['timestamp'], df['SMA20'],  color='#888888', linewidth=1.0, label='SMA20')
    if 'EMA50' in df:   ax.plot(df['timestamp'], df['EMA50'],  color='#1f77b4', linewidth=1.8, label='EMA50')
    if 'EMA200' in df:  ax.plot(df['timestamp'], df['EMA200'], color='#d62728', linewidth=2.0, label='EMA200')

    if 'BB_UPPER' in df and 'BB_LOWER' in df:
        ax.fill_between(df['timestamp'], df['BB_UPPER'], df['BB_LOWER'], alpha=0.12, color='#1f77b4', label='Bollinger')

    if 'SUPERTREND_LINE' in df:
        ax.plot(df['timestamp'], df['SUPERTREND_LINE'], color='#444444', linewidth=1.2, label='SuperTrend')

    ax.set_ylabel("Price")
    ax.legend(loc='upper left')

    # Ichimoku 요약 패널
    ax = axs[1]
    ax.plot(df['timestamp'], df['close'], color='black', linewidth=1.2, label='종가')
    if 'tenkan_sen' in df:  ax.plot(df['timestamp'], df['tenkan_sen'],  linewidth=1.2, label='전환선')
    if 'kijun_sen' in df:   ax.plot(df['timestamp'], df['kijun_sen'],   linewidth=1.2, label='기준선')
    if 'chikou_span' in df: ax.plot(df['timestamp'], df['chikou_span'], linewidth=1.0, label='후행스팬')
    if 'senkou_span_a' in df and 'senkou_span_b' in df:
        ax.plot(df['timestamp'], df['senkou_span_a'], alpha=0.6, linewidth=1.0, label='선행스팬A')
        ax.plot(df['timestamp'], df['senkou_span_b'], alpha=0.6, linewidth=1.0, label='선행스팬B')
        ax.fill_between(df['timestamp'], df['senkou_span_a'], df['senkou_span_b'],
                        where=(df['senkou_span_a'] >= df['senkou_span_b']), alpha=0.20)
    ax.set_ylabel("Ichimoku")
    ax.legend(loc='upper left')

    fig.text(0.01, 0.01, _chart_howto_text("A"),
             ha='left', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#999', alpha=0.92))
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    pA = os.path.join(outdir, f"chart_{sym}_{timeframe}_A_trend.png")
    fig.savefig(pA, dpi=140); plt.close(fig); paths.append(pA)

    # ---------- B. Momentum ----------
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{symbol} - {timeframe} · Momentum", fontsize=14)

    # RSI
    ax = axs[0]
    if 'RSI' in df:
        ax.plot(df['timestamp'], df['RSI'], linewidth=1.2, label='RSI')
        ax.axhline(70, linestyle='--', color='#888888'); ax.axhline(30, linestyle='--', color='#888888')
        ax.set_ylabel('RSI'); ax.legend(loc='upper left')

    # MACD
    ax = axs[1]
    if 'MACD' in df and 'MACD_SIGNAL' in df:
        ax.plot(df['timestamp'], df['MACD'],         linewidth=1.6, color='#1f77b4', label='MACD')
        ax.plot(df['timestamp'], df['MACD_SIGNAL'],  linewidth=1.2, color='#ff7f0e', label='Signal')
        ax.bar(df['timestamp'], df['MACD']-df['MACD_SIGNAL'], width=_bar_width_from_time_index(df), alpha=0.35, label='Hist')
        ax.set_ylabel('MACD'); ax.legend(loc='upper left')

    fig.text(0.01, 0.01, _chart_howto_text("B"),
             ha='left', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#999', alpha=0.92))
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    pB = os.path.join(outdir, f"chart_{sym}_{timeframe}_B_momentum.png")
    fig.savefig(pB, dpi=140); plt.close(fig); paths.append(pB)

    # ---------- C. Strength & Oscillator ----------
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{symbol} - {timeframe} · Strength", fontsize=14)

    # ADX / DI
    ax = axs[0]
    if 'ADX' in df:      ax.plot(df['timestamp'], df['ADX'],       linewidth=1.4, color='#1f77b4', label='ADX')
    if 'PLUS_DI' in df:  ax.plot(df['timestamp'], df['PLUS_DI'],   linewidth=1.0, color='#2ca02c', label='+DI')
    if 'MINUS_DI' in df: ax.plot(df['timestamp'], df['MINUS_DI'],  linewidth=1.0, color='#ff7f0e', label='-DI')
    ax.axhline(20, linestyle=':', color='#aaaaaa')
    ax.set_ylabel('ADX / DI'); ax.legend(loc='upper left')

    # StochRSI
    ax = axs[1]
    if 'STOCHRSI_K' in df: ax.plot(df['timestamp'], df['STOCHRSI_K'], linewidth=1.2, label='%K')
    if 'STOCHRSI_D' in df: ax.plot(df['timestamp'], df['STOCHRSI_D'], linewidth=1.0, label='%D')
    ax.axhline(0.8, linestyle='--', color='#888888'); ax.axhline(0.2, linestyle='--', color='#888888')
    ax.set_ylabel('StochRSI'); ax.legend(loc='upper left')

    fig.text(0.01, 0.01, _chart_howto_text("C"),
             ha='left', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#999', alpha=0.92))
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    pC = os.path.join(outdir, f"chart_{sym}_{timeframe}_C_strength.png")
    fig.savefig(pC, dpi=140); plt.close(fig); paths.append(pC)

    # ---------- D. Money Flow ----------
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"{symbol} - {timeframe} · Money Flow", fontsize=14)
    if 'MFI' in df:
        ax.plot(df['timestamp'], df['MFI'], linewidth=1.2, label='MFI')
        ax.axhline(80, linestyle='--', color='#888888'); ax.axhline(20, linestyle='--', color='#888888')
        ax.set_ylabel('MFI'); ax.legend(loc='upper left')
    ax.set_xlabel('Time')
    fig.text(0.01, 0.02, _chart_howto_text("D"),
             ha='left', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#999', alpha=0.92))
    plt.tight_layout(rect=[0, 0.09, 1, 0.97])
    pD = os.path.join(outdir, f"chart_{sym}_{timeframe}_D_flow.png")
    fig.savefig(pD, dpi=140); plt.close(fig); paths.append(pD)

    return paths
#-----------------------------------------------------------------------------------


def format_signal_message(
    tf, signal, price, pnl, strength, df,
    entry_time=None, entry_price=None, score=None,
    weights=None, weights_detail=None,  # 🔹 새 매개변수 추가
    prev_score_value=None,
    agree_long=None,
    agree_short=None,
    rr_value=None,
    sma_dist_pct=None,
    kijun_dist_pct=None,
    atr_pct=None,
    symbol="ETH/USDT",
    daily_change_pct=None,
    score_history=None,
    recent_scores=None,
    live_price=None,
    show_risk: bool = False
):
    tf_str = {'15m': '15분봉', '1h': '1시간봉', '4h': '4시간봉', '1d': '일봉'}[tf]
    now_str = datetime.now().strftime("%m월 %d일 %H:%M")
    display_price = live_price if isinstance(live_price, (int, float)) else price
    # 🔒 데이터 길이 가드
    n = _len(df)
    if n == 0:
        now_str = datetime.now().strftime("%m월 %d일 %H:%M")
        symbol_short = (symbol or "ETH/USDT").split('/')[0].upper()
        tf_compact = tf.upper()
        usdkrw_short = get_usdkrw_rate()
        price_krw = (display_price * usdkrw_short) if isinstance(usdkrw_short, (int,float)) and usdkrw_short>0 and isinstance(display_price,(int,float)) else None
        short_msg = f"{symbol_short}-{tf_compact} " + (f"${display_price:,.2f}" if isinstance(display_price,(int,float)) else "$-")
        if price_krw:
            short_msg += f"/₩{price_krw:,.0f}"
        short_msg += f" {signal} {now_str.split()[-1]}"

        main_msg = f"## [{tf}] 데이터 없음\n**🕒 일시:** {now_str}\n지표 산출을 위한 캔들이 부족합니다."
        summary_msg = "📝 애널리스트 코멘트: 데이터 부족으로 생략"
        return main_msg, summary_msg, short_msg
    
    # ✅ 등급 분류
    if score is not None:
        if score >= CFG["strong_cut"]:
            grade = "🔥 STRONG BUY (강한 매수)"
        elif score >= CFG["buy_cut"]:
            grade = "🟢 BUY (약한 매수)"
        elif score <= CFG["strong_sell_cut"]:
            grade = "💀 STRONG SELL (강한 매도)"
        elif score <= CFG["sell_cut"]:
            grade = "🔴 SELL (약한 매도)"
        else:
            grade = "⚪ NEUTRAL (중립)"
    else:
        grade = "❓ UNKNOWN"

    # ✅ 기본 메시지 시작
    header_sig = "BUY" if (score is not None and score >= CFG["buy_cut"]) else ("SELL" if (score is not None and score <= CFG["sell_cut"]) else "NEUTRAL")
    main_msg = f"## [{tf_str}] {header_sig} \n"
    main_msg += f"** 일시:** {now_str}\n"
   

    # 🇰🇷 원화 환산가 (USDKRW 실시간) — None 가드
    usdkrw = get_usdkrw_rate()
    if isinstance(display_price, (int, float)):
        krw_txt = ""
        if isinstance(usdkrw, (int, float)) and usdkrw > 0:
            krw_txt = f" / {_fmt_krw(display_price * usdkrw)} (USDKRW {usdkrw:,.2f})"
        main_msg += f" **현재가:** **{_fmt_usd(display_price)}{krw_txt}**\n"
    else:
        main_msg += f" **현재가:** **{_fmt_usd(display_price)}**\n"
    if daily_change_pct is not None:
        main_msg += f"**일봉 변동률:** **{daily_change_pct:+.2f}%**\n"
    if entry_price and entry_time:
        main_msg += f"**진입 시점:** {entry_time} ({_fmt_usd(entry_price)})\n"
    if pnl is not None:
        main_msg += f"**수익률:** {pnl:.2f}%\n"

    if isinstance(prev_score_value, (int, float)) and isinstance(score, (int, float)):
        delta = score - prev_score_value
        direction = "증가 ▲" if delta > 0 else "감소 🔽" if delta < 0 else "변화 없음"
        main_msg += f"\n🔁 이전 점수 대비: {prev_score_value:.1f} → {score:.1f} ({delta:+.1f}, {direction})\n"


    # entry_price가 없더라도 현재가/종가로 폴백하여 표시
    basis_price = None
    if isinstance(entry_price, (int, float)) and entry_price > 0:
        basis_price = float(entry_price)
    elif (str(signal).startswith("BUY") or str(signal).startswith("SELL")) and isinstance(price, (int, float)):
        basis_price = float(price)  # 폴백: 현재가
    else:
        last_close = _last(df, 'close', None)
        if isinstance(last_close, (int, float)) and last_close:
            basis_price = float(last_close)  # 최후 폴백: 종가


    # [ANCHOR: risk_section_guard_begin]
    risk_msg = ""
    if show_risk:
        risk_msg += "\n### 📌 손절·익절·트레일링"

        if basis_price is not None:
            sig_is_buy = str(signal).startswith("BUY")

            _cfg = globals()
            hs_on  = (_cfg.get('USE_HARD_STOP', {}) or {}).get(tf, True)
            hs_pct = (_cfg.get('HARD_STOP_PCT', {}) or {}).get(tf, 3.0)

        # TP 설정: 전역
        _tp_map = _cfg.get('take_profit_pct', {}) or {}
        tp_pct_local = _tp_map.get(tf, 2.0)
        tp = basis_price * (1 + tp_pct_local / 100) if sig_is_buy else basis_price * (1 - tp_pct_local / 100)

        # 퍼센트 트레일링: 전역 + USE_TRAILING
        _ts_map = _cfg.get('trailing_stop_pct', {}) or {}
        ts_pct = _ts_map.get(tf, 0.0)
        use_trail = (_cfg.get('USE_TRAILING', {}) or {}).get(tf, True) and ts_pct > 0

        # 하드 스탑(4h/1d만 ON)
        if hs_on and hs_pct and hs_pct > 0:
            sl = basis_price * (1 - hs_pct / 100) if sig_is_buy else basis_price * (1 + hs_pct / 100)
            risk_msg += f"\n\n- **하드 스탑**: ${sl:.2f} ({hs_pct}%) — {tf} 활성화\n"
        else:
            risk_msg += "\n\n- **하드 스탑**: 사용 안 함 (트레일링/MA 스탑 사용)\n"

        # MA 스탑 표시
        ma_cfg = _cfg.get('MA_STOP_CFG', {})
        rule = (ma_cfg.get('tf_rules') or {}).get(tf)

        # TF별 버퍼 우선 적용
        buf = 0.0
        if rule:
            _, _, *rest = rule
            buf = (rest[0] if rest else ma_cfg.get('buffer_pct', 0.0))


        if ma_cfg.get('enabled') and rule:
            ma_type, period, *_ = rule
            ma_col = f"{ma_type.upper()}{period}"
            ma_val = None
            if ma_col in df.columns:
                try:
                    v = df[ma_col].iloc[-1]
                    ma_val = float(v) if v == v else None  # NaN guard
                except Exception:
                    ma_val = None

            confirm_txt = ", 종가 기준" if ma_cfg.get('confirm') == 'close' else ", 저/고가 터치 기준"
            buf_txt = f", 버퍼 {buf:.1f}%" if buf else ""

            if ma_val is not None:
                # 📌 현재가 대비 % 차이 계산
                if isinstance(price, (int, float)) and price > 0:
                    diff_pct = ((price - ma_val) / price) * 100
                    direction = "위" if price >= ma_val else "아래"
                    diff_txt = f"가격 기준 {diff_pct:+.2f}% ({direction})"
                else:
                    diff_txt = ""
                risk_msg += f"- **MA 스탑**: {ma_col}=**${ma_val:.2f}**({diff_txt}{confirm_txt}{buf_txt})\n"
            else:
                risk_msg += f"**MA 스탑**: {ma_col}({confirm_txt}{buf_txt})\n"


        # 익절 표시
        risk_msg += f"- **익절가**: ${tp:.2f} 현재 분봉기준({tp_pct_local}%)\n"



        # ----------------- 실행 체크리스트(쉬운 표현 + 설명 포함) -----------------
        risk_msg += "### 🎯 체크리스트\n"

        # 기준 가격(now) 확보: price → 종가 폴백
        now_price = None
        if isinstance(price, (int, float)) and price:
            now_price = float(price)
        elif _len(df) > 0 and 'close' in df:
            try:
                now_price = float(df['close'].iloc[-1])
            except Exception:
                now_price = None

        if now_price:
            _cfg = globals()
            sig_is_buy = str(signal).startswith("BUY")

            # (옵션) 여러 시간대 합의
            try:
                    if agree_long is not None and agree_short is not None:
                        risk_msg += f"- 여러 시간대 분석 결과: 매수 **{agree_long}** / 매도 **{agree_short}** — 같은 방향 표가 많을수록 신뢰도 ↑\n"
            except Exception:
                pass
            
            # ===== 계산 결과 변수들(액션 힌트에 재사용) =====
            rr_value = None          # 손익비
            risk_pct = None          # 손실 한도(%)
            sma_dist_pct = None      # 평균선까지 거리(%)
            kijun_dist_pct = None    # 일목 기준선까지 거리(%)
            atr_pct = None           # 변동성(%)

            # 1) 손익비 스냅샷(지금 진입 가정) — 하드 스탑 우선, 없으면 MA 스탑 기준
            hs_on  = (_cfg.get('USE_HARD_STOP', {}) or {}).get(tf, False)
            hs_pct = (_cfg.get('HARD_STOP_PCT', {}) or {}).get(tf, 0.0)
            tp_pct_local = (_cfg.get('take_profit_pct', {}) or {}).get(tf, 0.0)

            # 보수적 리스크 바닥값(전역 설정에 있으면 그 값 우선)
            MIN_RISK_FLOOR = (_cfg.get('MIN_RISK_FLOOR', {}) or {'15m':0.25,'1h':0.50,'4h':0.75,'1d':1.00})
            risk_candidates = []

            if hs_on and hs_pct > 0:
                # 하드 스탑 설정이 있으면 그 퍼센트를 최소 리스크 후보에 포함
                risk_candidates.append(float(hs_pct))
            else:
                # === MA 스탑 기반 리스크 추정 ===
                ma_cfg = _cfg.get('MA_STOP_CFG', {})
                rule = (ma_cfg.get('tf_rules') or {}).get(tf)
                if ma_cfg.get('enabled') and rule:
                    ma_type, period, *rest = rule
                    buf = (rest[0] if rest else ma_cfg.get('buffer_pct', 0.0))  # ← TF별 버퍼 우선
                    ma_col = f"{ma_type.upper()}{period}"
                    if ma_col in df.columns and pd.notna(df[ma_col].iloc[-1]):
                        ma_val = float(df[ma_col].iloc[-1])
                        thr = ma_val * (1 - buf/100.0) if sig_is_buy else ma_val * (1 + buf/100.0)
                        raw_risk_pct = abs(now_price - thr) / now_price * 100.0
                        risk_candidates.append(raw_risk_pct)


                        # (b) MA 선 자체까지 거리(버퍼 제거)
                        ma_gap_pct = abs(now_price - ma_val) / now_price * 100.0
                        risk_candidates.append(ma_gap_pct)

            # (c) 시간대별 최소 리스크 바닥값 적용
            risk_floor = float(MIN_RISK_FLOOR.get(tf, 0.50))
            risk_pct = max([x for x in risk_candidates if x is not None] + [risk_floor])

            # 손익비 계산 및 출력
            if risk_pct is not None and tp_pct_local:
                rr = tp_pct_local / max(risk_pct, 1e-9)
                rr_value = rr
                rr_hint = "유리(1.5배 이상)" if rr >= 1.5 else ("보통(1.0~1.5)" if rr >= 1.0 else "불리(1.0 미만)")

                # 과대평가 경고(리스크가 바닥값에 걸리거나 손익비가 과도하게 큰 경우)
                warn = ""
                if risk_pct <= risk_floor + 1e-9 or rr >= 10:
                    warn = " — ※ 평균선에 매우 근접: 손익비가 과대평가될 수 있음"

                risk_msg += f"- 손익비(지금 들어갈 경우): **{rr:.2f}배** (손실 한도 {risk_pct:.2f}%, 이익 목표 {tp_pct_local:.2f}%) — **{rr_hint}**{warn}\n"

            # 2) 중요 레벨까지 거리(%) — 평균선 / 일목 기준선 / 20봉 고저 / 변동성
            prox_lines = []

            # 평균선(이 TF의 스탑 기준선)
            try:
                ma_cfg = _cfg.get('MA_STOP_CFG', {})
                rule = (ma_cfg.get('tf_rules') or {}).get(tf)
                if ma_cfg.get('enabled') and rule:
                    ma_type, period, *_ = rule
                    ma_col = f"{ma_type.upper()}{period}"
                    if ma_col in df.columns and pd.notna(df[ma_col].iloc[-1]):
                        ma_val = float(df[ma_col].iloc[-1])
                        ma_dist = (now_price - ma_val) / now_price * 100.0
                        sma_dist_pct = ma_dist
                        direction = "위" if ma_dist >= 0 else "아래"
                        bias = "상승 흐름 유지" if ma_dist >= 0 else "평균선 아래(이탈 주의)"
                        prox_lines.append(f"- 평균선({ma_col})까지: **{ma_dist:+.2f}%** {direction} — {bias}")
            except Exception:
                pass

            # 일목 기준선(kijun)
            try:
                if 'kijun_sen' in df and pd.notna(df['kijun_sen'].iloc[-1]):
                    kijun = float(df['kijun_sen'].iloc[-1])
                    kijun_dist = (now_price - kijun) / now_price * 100.0
                    kijun_dist_pct = kijun_dist
                    direction = "위" if kijun_dist >= 0 else "아래"
                    bias = "단기 상승 쪽" if kijun_dist >= 0 else "단기 하락 쪽"
                    near_txt = " · 기준선 매우 근접(±0.5%)=되돌림 주의" if abs(kijun_dist) < 0.5 else ""
                    prox_lines.append(f"- 일목 기준선까지: **{kijun_dist:+.2f}%** {direction} — {bias}{near_txt}")
            except Exception:
                pass

            # 최근 20개 봉 고점/저점까지
            try:
                if 'high' in df and 'low' in df and _len(df) >= 20:
                    lookback = 20
                    swing_high = float(df['high'].rolling(lookback).max().iloc[-1])
                    swing_low  = float(df['low'].rolling(lookback).min().iloc[-1])
                    to_break_hi = (swing_high - now_price) / now_price * 100.0
                    to_break_lo = (now_price - swing_low)  / now_price * 100.0
                    note_bits = []
                    if to_break_hi <= 1.0: note_bits.append("상단 1% 이내=돌파 관찰")
                    if to_break_lo <= 1.0: note_bits.append("하단 1% 이내=방어 준비")
                    note = f" — {' · '.join(note_bits)}" if note_bits else ""
                    prox_lines.append(f"- 최근 20개 봉 최고가까지: **{to_break_hi:.2f}%** / 최저가까지: **{to_break_lo:.2f}%**{note}")
            except Exception:
                pass

            # 변동성(ATR, 최근 14봉 평균)
            try:
                if 'ATR14' in df and pd.notna(df['ATR14'].iloc[-1]) and now_price:
                    atr_pct = float(df['ATR14'].iloc[-1]) / now_price * 100.0
                    if atr_pct < 2.0: atr_note = "낮음(흔들림 적음, 돌파는 둔할 수 있음)"
                    elif atr_pct < 3.5: atr_note = "보통"
                    elif atr_pct < 5.0: atr_note = "높음(물량 축소 권장)"
                    else: atr_note = "매우 높음(급변 위험)"
                    prox_lines.append(f"- 변동성(최근 14봉 평균): **{atr_pct:.2f}%** — {atr_note}")
            except Exception:
                pass

                if prox_lines:
                    risk_msg += "\n".join(prox_lines) + "\n"
            

            # 🎯 실행 체크리스트 하단 액션 힌트 (쉬운 표현)
            rr_text = None
            if rr_value is not None:
                if rr_value >= 1.5:
                    rr_text = "손익비 유리(≥1.5배) — 진입 우선 검토"
                elif rr_value >= 1.0:
                    rr_text = "손익비 보통(1.0~1.5배) — 규모 축소/분할 접근"
                else:
                    rr_text = "손익비 불리(<1.0배) — 보류 권장"

            # 평균선/기준선 근접(±1%)은 추세 유지/돌파 관찰 신호
            dist_bits = []
            if sma_dist_pct is not None and abs(sma_dist_pct) <= 1.0:
                dist_bits.append("평균선 근접(±1%)")
            if kijun_dist_pct is not None and abs(kijun_dist_pct) <= 1.0:
                dist_bits.append("일목 기준선 근접(±1%)")
            dist_text = " / ".join(dist_bits) + " — 추세 유지·돌파 여부 확인" if dist_bits else None

            # 변동성 수준에 따른 행동 힌트
            vol_text = None
            if atr_pct is not None:
                if atr_pct < 1.0:
                    vol_text = "변동성 매우 낮음 — 수익 제한/돌파 실패 가능, 손절 짧게"
                elif atr_pct < 3.0:
                    vol_text = "변동성 낮음 — 비교적 안정적, 추세 추종 유리"
                elif atr_pct < 5.0:
                    vol_text = "변동성 높음 — 흔들림 큼, 진입 규모 축소"
                else:
                    vol_text = "변동성 매우 높음 — 급변 위험, 관망 또는 소액"

            # 종합 액션 힌트 출력(항목별 개별 줄)
            hints = [h for h in (rr_text, dist_text, vol_text) if h]
            if hints:
                risk_msg += "\n➡️ **액션 힌트**\n" + "\n".join(f"- {h}" for h in hints) + "\n"

        else:
            risk_msg += "- 가격 데이터 부족으로 체크리스트를 만들 수 없습니다.\n"
        # ------------------------------------------------------------

    if show_risk and risk_msg:
        main_msg += risk_msg
    # [ANCHOR: risk_section_guard_end]

    # ✅ 점수 및 등급
    main_msg += "\n### **📊 점수 기반 판단**\n"
    if score is not None:
        main_msg += f"- 최종 종합 지표 점수: **{score:.1f}**\n"
        main_msg += f"- 판단 등급: **{grade}**\n"

        # 매수 매도 동의 투표
    if agree_long is not None and agree_short is not None:
        main_msg += f"- 매수 매도 동의 투표 (**상승**/**하락**): **{agree_long}** / **{agree_short}**\n"

    # 최근 N개 점수 표시
    if recent_scores:
        seq = " → ".join(f"{s:.1f}" for s in recent_scores)
        main_msg += f"- 점수기록(최근 {len(recent_scores)}건): {seq}\n"

    
    # ✅ 지표별 기여도 (점수 + 이유)
    TOP_N = 3
    top_items = []
    if weights_detail and isinstance(weights_detail, dict):
        # (지표명, 점수, 이유)로 변환
        items = []
        for ind, val in weights_detail.items():
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                sc, rsn = val
            else:
                sc, rsn = (val if isinstance(val, (int,float)) else 0.0), "-"
            items.append((ind, float(sc), str(rsn)))
        # 절대값 큰 순으로 상위 N개
        items.sort(key=lambda x: abs(x[1]), reverse=True)
        top_items = items[:TOP_N]
    elif weights and isinstance(weights, dict):
        items = [(k, float(v), "-") for k, v in weights.items()]
        items.sort(key=lambda x: abs(x[1]), reverse=True)
        top_items = items[:TOP_N]


    # === ⚙️ 지표 시너지 인사이트 (위쪽 섹션에 배치해도 OK)
    try:
        syn_lines = []
        if _len(df) > 1:
            last = df.iloc[-1]

            # 안전 추출 (NameError 방지)
            close_val = float(last['close'])              if 'close'        in df else None
            ema50     = float(last['EMA50'])              if 'EMA50'        in df else None
            ema200    = float(last['EMA200'])             if 'EMA200'       in df else None
            bb_up     = float(last['BB_UPPER'])           if 'BB_UPPER'     in df else None
            bb_lo     = float(last['BB_LOWER'])           if 'BB_LOWER'     in df else None
            st_dir    = float(last['SUPERTREND'])         if 'SUPERTREND'   in df else None
            mfi_val   = float(last['MFI'])                if 'MFI'          in df else None
            cci_val   = float(last['CCI'])                if 'CCI'          in df else None
            macd_val  = float(last['MACD'])               if 'MACD'         in df else None
            macd_sig  = float(last['MACD_SIGNAL'])        if 'MACD_SIGNAL'  in df else None
            adx_val   = float(last['ADX'])                if 'ADX'          in df else None
            rsi_val   = float(last['RSI'])                if 'RSI'          in df else None
            kijun_val = float(last['kijun_sen'])          if 'kijun_sen'    in df else None

            cloud_top_loc = cloud_bot_loc = None
            if 'senkou_span_a' in df and 'senkou_span_b' in df:
                try:
                    cloud_top_loc = float(max(last['senkou_span_a'], last['senkou_span_b']))
                    cloud_bot_loc = float(min(last['senkou_span_a'], last['senkou_span_b']))
                except Exception:
                    pass

            atr_pct_val = None
            if 'ATR14' in df and close_val:
                try:
                    atr_pct_val = float(last['ATR14']) / float(close_val) * 100.0
                except Exception:
                    atr_pct_val = None

            # OBV 기울기
            obv_slope = None
            if 'OBV' in df and _len(df) >= 5:
                obv_last  = _last(df, 'OBV', None)
                obv_prev5 = _s_iloc(df['OBV'], -5, obv_last)
                if obv_last is not None and obv_prev5 is not None:
                    obv_slope = obv_last - obv_prev5

            # 시너지 계산
            syn = _synergy_insights(
                df,
                adx=adx_val, plus_di=_last(df,'PLUS_DI',None), minus_di=_last(df,'MINUS_DI',None),
                rsi=rsi_val, macd=macd_val, macd_signal=macd_sig, st_dir=st_dir,
                close=close_val, ema50=ema50, ema200=ema200, kijun=kijun_val,
                cloud_top=cloud_top_loc, cloud_bot=cloud_bot_loc, bb_up=bb_up, bb_lo=bb_lo,
                obv_slope=( (_last(df,'OBV',None) - _s_iloc(df['OBV'],-5,_last(df,'OBV',None))) if ('OBV' in df and _len(df)>=5) else None ),
                mfi=mfi_val, cci=cci_val, atr_pct=atr_pct_val, max_items=5
            ) or []

            syn_lines = [f"- {s}" for s in syn] if syn else ["- 현재 조합에서 두드러진 시너지/충돌 신호 없음"]

        # 제목 굵게 + 줄바꿈 출력
        main_msg += "\n### **🧾 애널리스트 인사이트**\n" + "\n".join(syn_lines if syn_lines else ["- 현재 조합에서 두드러진 시너지/충돌 신호 없음"]) + "\n"

    except Exception as e:
        main_msg += f"\n### **🧾 애널리스트 인사이트**\n- 계산 중 오류: {e}\n"

    # ✅ Ichimoku 분석
    ichimoku_result = ichimoku_analysis(df)
    main_msg += "### ☁️ Ichimoku 분석 요약\n"
    main_msg += '\n'.join(["- " + line for line in ichimoku_result])


    # 2000자 제한 여유 절단 (메인만)
    MAX_DISCORD_MSG_LEN = 1900
    if len(main_msg) > MAX_DISCORD_MSG_LEN:
        main_msg = main_msg[:MAX_DISCORD_MSG_LEN] + "\n...(이하 생략)"
  
    # 애널리스트 해석
    try:
        last = df.iloc[-1]

        cloud_top  = float(max(last.get('senkou_span_a', np.nan), last.get('senkou_span_b', np.nan)))
        cloud_bot  = float(min(last.get('senkou_span_a', np.nan), last.get('senkou_span_b', np.nan)))
        kijun      = float(last.get('kijun_sen', np.nan))
        last_close = float(last.get('close', np.nan))

        adx     = float(last.get('ADX', np.nan)) if 'ADX' in df else np.nan
        rsi     = float(last.get('RSI', np.nan)) if 'RSI' in df else np.nan
        atr_pct = float(last.get('ATR14', np.nan) / last_close * 100) if ('ATR14' in df and last_close) else np.nan

        if last_close > cloud_top:
            trend = "상승장(구름 위)"
        elif last_close < cloud_bot:
            trend = "하락장(구름 아래)"
        else:
            trend = "혼조(구름 내부)"


        if score is not None:
            if score >= CFG["strong_cut"]:
                bias = "강한 매수 우세"
            elif score >= CFG["buy_cut"]:
                bias = "약한 매수 우세"
            elif score <= CFG["strong_sell_cut"]:
                bias = "강한 매도 우세"
            elif score <= CFG["sell_cut"]:
                bias = "약한 매도 우세"
            else:
                bias = "혼조"
        else:
            bias = "불명"

        vol = "높음" if (not np.isnan(atr_pct) and atr_pct >= 1.2) else ("보통" if not np.isnan(atr_pct) else "N/A")

        top_desc = "-"
        if weights_detail and isinstance(weights_detail, dict):
            top_inds = sorted(weights_detail.items(), key=lambda x: abs(x[1][0] if isinstance(x[1], (list, tuple)) else x[1]), reverse=True)[:3]
            tops = []
            for ind, val in top_inds:
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    score_val, reason = val
                else:
                    score_val, reason = val, "-"
                tops.append(f"{ind}({score_val:+.1f}, {reason})")
            top_desc = ", ".join(tops) if tops else "-"

        lookback = 20
        swing_high = float(df['high'].rolling(lookback).max().iloc[-1]) if 'high' in df else np.nan
        swing_low  = float(df['low'].rolling(lookback).min().iloc[-1]) if 'low' in df else np.nan
        bb_up      = float(last.get('BB_UPPER', np.nan)) if 'BB_UPPER' in df else np.nan
        bb_lo      = float(last.get('BB_LOWER', np.nan)) if 'BB_LOWER' in df else np.nan

        if str(signal).startswith("BUY"):
            t1, t2 = (bb_up, swing_high)
            inv1, inv2 = (kijun, cloud_bot)
            checklist = ["종가가 기준선(kijun) 위", "전환선↑기준선", "후행스팬이 26봉 전 종가 위", "ADX>20 & RSI>50"]
        elif str(signal).startswith("SELL"):
            t1, t2 = (bb_lo, swing_low)
            inv1, inv2 = (kijun, cloud_top)
            checklist = ["종가가 기준선(kijun) 아래", "전환선↓기준선", "후행스팬이 26봉 전 종가 아래", "ADX>20 & RSI<50"]
        else:
            t1, t2 = (bb_up, swing_high)
            inv1, inv2 = (kijun, cloud_bot)
            checklist = ["구름 이탈 여부", "기준선 상·하방 복귀", "ADX 20 돌파", "RSI 50 축 상·하 이탈"]

         # ===== 시간프레임별 '강한 지표'를 동적으로 뽑아 표시 =====
        # 후보군(새 지표 포함)
        TF_CANDIDATES = {
            '15m': ['SMA','EMA','Ichimoku','MACD','RSI','StochRSI','Bollinger','SuperTrend','OBV','MFI','ADX','CCI'],
            '1h' : ['SMA','EMA','Ichimoku','MACD','RSI','StochRSI','Bollinger','SuperTrend','OBV','MFI','ADX','CCI'],
            '4h' : ['SMA','EMA','Ichimoku','MACD','RSI','StochRSI','Bollinger','SuperTrend','OBV','MFI','ADX','CCI'],
            '1d' : ['SMA','EMA','Ichimoku','MACD','RSI','StochRSI','Bollinger','SuperTrend','OBV','MFI','ADX','CCI'],
        }

        # 각 지표의 '이론적 최대 기여치' (현재 스코어링 로직 기준으로 추정)
        MAX_SCORES = {
            'SMA': 1.5,
            'RSI': 2.0,          # 극단/과매수·과매도까지 고려
            'MACD': 1.5,
            'Bollinger': 1.0,
            'Ichimoku': 2.5,     # 구름/전환·기준/치코 조합
            'ADX': 1.0,          # +/-1.0로 반영
            'CCI': 1.5,
            'EMA': 1.5,
            'SuperTrend': 1.0,
            'StochRSI': 1.0,
            'MFI': 0.5,
            'OBV': 0.5,
        }

        # 현재 TF에서 후보군 중 실제 점수(weights)가 있는 지표만 뽑고, 절대값 기준 Top-3
        cand = [i for i in TF_CANDIDATES.get(tf, []) if i in (weights or {})]
        top_inds = sorted(cand, key=lambda i: abs(weights.get(i, 0.0)), reverse=True)[:3]

        # 메타라인(애널리스트 코멘트 하단 요약용)
        if weights and top_inds:
            meta_line = ", ".join(f"{ind}({weights.get(ind,0.0):+.1f})".replace("+ ", "+").replace("- ", "-") for ind in top_inds)
        else:
            meta_line = "-"

        # 종합 지표 기여도
        main_msg += "\n\n📌 **종합 지표 기여도**:\n"

        items_all = []
        if isinstance(weights, dict) and weights:
            for ind, sc in weights.items():
                sc = float(sc)
                det = weights_detail.get(ind, "")
                if isinstance(det, (list, tuple)) and len(det) >= 2:
                    rsn = str(det[1])
                else:
                    rsn = str(det)
                items_all.append((ind, sc, rsn))

        # 0점 제거 후 정렬
        ZERO_EPS = 1e-9
        items_nz = [t for t in items_all if abs(t[1]) > ZERO_EPS]
        items_nz.sort(key=lambda x: (abs(x[1]), x[1]), reverse=True)

        # +상위3, -하위3
        pos = [(i,s,r) for i,s,r in items_nz if s > 0][:3]
        neg = [(i,s,r) for i,s,r in items_nz if s < 0][:3]

        def _line(i,s,r): return f"- {i}: {s:+.1f} ({r})\n".replace("+ ", "+").replace("- ", "-")

        shown = 0
        for t in pos: main_msg += _line(*t); shown += 1
        for t in neg: main_msg += _line(*t); shown += 1

        # 기타 합침
        others = [t for t in items_nz if t not in pos + neg]
        if others:
            etc = ", ".join(f"{i}({s:+.1f})".replace("+ ", "+").replace("- ", "-") for i,s,_ in others)
            main_msg += f"- 기타: {etc}\n"
        if shown == 0:
            main_msg += "- 유의미한 지표 기여가 없습니다.\n"



        # ===== 퍼포먼스 스냅샷 생성 =====
        try:
            summary_msg = build_performance_snapshot(
                tf=tf,
                symbol=symbol,
                display_price=display_price,
                daily_change_pct=daily_change_pct,
                recent_scores=recent_scores
            )
        except Exception as e:
            summary_msg = f"📈 퍼포먼스 스냅샷 생성 중 오류: {e}"


        # ✅ 누적 성과 요약(예외 가드)
        try:
            path = log_path(symbol, tf)
            if os.path.exists(path):
                hist = pd.read_csv(path)
                if 'timeframe' in hist.columns:
                    hist = hist[hist['timeframe'] == tf]
                total = len(hist)
                wins = ((hist['pnl'] > 0).fillna(False)).sum() if 'pnl' in hist.columns else 0
                winrate = (wins / total * 100) if total else 0.0
                cumret = hist['pnl'].fillna(0).sum() if 'pnl' in hist.columns else 0.0
            else:
                total, winrate, cumret = 0, 0.0, 0.0
        except Exception:
            total, winrate, cumret = 0, 0.0, 0.0

        summary_msg += (
            f"\n\n **누적 성과 요약**\n"
            f"- 누적 수익률: {cumret:+.2f}%\n"
            f"- 승률: {winrate:.1f}%\n"
            f"- 총 트레이드: {total}회"
        )

    except Exception as e:
            # 큰 해석 블록에서 오류 나더라도 메시지가 끊기지 않게 안전 폴백
            try:
                summary_msg = f"⚠️ 해석 생성 중 오류: {e}"
            except Exception:
                summary_msg = "⚠️ 해석 생성 중 알 수 없는 오류"


    # === 📱 모바일 푸시 전용 짧은 메시지 ===
    symbol_short = (symbol or "ETH/USDT").split('/')[0].upper()
    tf_compact = tf.upper()
    time_only = datetime.now().strftime("%H:%M")
    daily_part = f"{daily_change_pct:+.1f}%(일변)" if isinstance(daily_change_pct, (int, float)) else ""

    # 🔹 같은 TF 직전봉 대비 변화율(안전 가드)
    try:
        tf_change_pct = None
        if len(df) >= 2:
            prev_close = float(df['close'].iloc[-2])
            curr_close = float(df['close'].iloc[-1])
            if prev_close:
                tf_change_pct = (curr_close - prev_close) / prev_close * 100
    except Exception:
        tf_change_pct = None
    tf_part = f"{tf_change_pct:+.2f}%({tf_compact})" if isinstance(tf_change_pct, (int, float)) else ""

    # 🔹 환율·원화 — None 가드
    usdkrw_short = get_usdkrw_rate()
    if isinstance(usdkrw_short, (int, float)) and usdkrw_short > 0 and isinstance(display_price, (int, float)):
        price_krw = display_price * usdkrw_short
        krw_str = f"₩{price_krw:,.0f}"
    else:
        krw_str = "₩-"

    score_str = f"{score:.1f}" if isinstance(score, (int, float)) else "-"

    # 최종 콤팩트 포맷
    # 예: ETH-1H $4,628.76/₩6,251,000 +0.8%(일변) +0.25%(1H) 3.9 BUY 12:28
    parts = [
        f"{symbol_short}-{tf_compact}",
        f"${display_price:,.2f}/{krw_str}",
        daily_part if daily_part else None,
        tf_part if tf_part else None,
        score_str,
        header_sig,
        time_only
    ]

    short_msg = " ".join([p for p in parts if p])

    # 🔚 반드시 반환!
    return main_msg, summary_msg, short_msg



def log_to_csv(symbol, tf, signal, price, rsi, macd,
               pnl=None, entry_price=None, entry_time=None,
               score=None, reasons=None, weights=None):
    # [PATCH-③] 종료 성격 신호면 TF 점유 해제 (페이퍼)
    try:
        if str(signal).upper() in ("MA STOP","EXIT","CLOSE","SL","TP","STOP","TAKE_PROFIT","STOP_LOSS"):
            if tf in PAPER_POS_TF:
                PAPER_POS_TF.pop(tf, None)
                _save_json(PAPER_POS_TF_FILE, PAPER_POS_TF)
            k = f"{symbol}|{tf}"
            if k in PAPER_POS:
                PAPER_POS.pop(k, None)
                _save_json(PAPER_POS_FILE, PAPER_POS)
    except Exception:
        pass
    if os.getenv("SANITIZE_LOG_PRICE","0") == "1":
        price = sanitize_price_for_tf(symbol, tf, price)



    symbol_clean = symbol.replace("/", "")  # ETH/USDT → ETHUSDT
    filename = f'logs/signals_{symbol_clean}_{tf}.csv'

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pnl_str = f"{pnl:.2f}%" if pnl is not None else ""
    entry_price_str = f"{entry_price:.2f}" if entry_price is not None else ""
    entry_time_str = entry_time if entry_time else ""
    score_str = score if score is not None else ""
    
    if isinstance(reasons, (list, tuple)):
        reasons_str = '"' + " | ".join(reasons) + '"'
    else:
        reasons_str = ""

    weights_str = " | ".join([f"{k}:{v:+.2f}" for k, v in weights.items()]) \
                  if weights and isinstance(weights, dict) else ""

    row = pd.DataFrame([[
        now, symbol_clean, tf, signal, price, rsi, macd,
        entry_price_str, entry_time_str, pnl_str, score_str,
        reasons_str, weights_str
    ]], columns=[
        "datetime", "symbol", "timeframe", "signal", "price", "rsi", "macd",
        "entry_price", "entry_time", "pnl", "score", "reasons", "weights"
    ])

    if os.path.exists(filename):
        row.to_csv(filename, mode='a', header=False, index=False)
    else:
        row.to_csv(filename, mode='w', header=True, index=False)


def plot_score_history(symbol, tf):
    fp = log_path(symbol, tf)
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp)
    df = df[df['timeframe'] == tf].copy()
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df[df['score'].notnull()]
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.sort_values('datetime')


    if df.empty:
        return None  # 데이터가 없으면 None 반환

    plt.figure(figsize=(12, 4))
    plt.plot(df['datetime'], df['score'], label='Score', color='blue')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f"Score History - {tf}")
    plt.xlabel("시간")
    plt.ylabel("점수")
    plt.grid(True)
    plt.tight_layout()
    filename = f"logs/score_history_{_symtag(symbol)}_{tf}.png"
    plt.savefig(filename)
    plt.close()
    return filename

def generate_performance_stats(tf, symbol='ETH/USDT'):
    """
    심볼별 로그 파일을 읽어 해당 타임프레임 성과 이미지를 만들어 반환.
    logs/signals_{sym}.csv 가 없으면 None.
    """
    import os, pandas as pd, matplotlib.pyplot as plt
    symtag = _symtag(symbol)
    path = log_path(symbol, tf)
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if 'timeframe' not in df.columns:
        return None
    df = df[df['timeframe'] == tf].copy()
    if df.empty:
        return None

    # 숫자 컬럼 캐스팅
    for c in ['price','rsi','macd','score','pnl']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 간단 집계
    total = len(df)
    wins = ((df['pnl'] > 0).fillna(False)).sum() if 'pnl' in df.columns else 0
    winrate = (wins/total*100) if total else 0.0
    cumret = df['pnl'].fillna(0).sum() if 'pnl' in df.columns else 0.0

    # 누적 수익률/점수 추이 그래프
    out = f"logs/perf_{symtag}_{tf}.png"
    plt.figure(figsize=(8,4))
    if 'pnl' in df.columns:
        df['cum'] = df['pnl'].fillna(0).cumsum()
        df['cum'].plot()
        plt.title(f"{symbol} {tf} 누적 수익 (총 {total}회, 승률 {winrate:.1f}%)")
        plt.xlabel("trade #"); plt.ylabel("cum PnL")
    else:
        df['score'].plot()
        plt.title(f"{symbol} {tf} 점수 추이 (총 {total}회)")
        plt.xlabel("signal #"); plt.ylabel("score")
    plt.tight_layout(); plt.savefig(out); plt.close()
    return out

def get_latest_performance_summary(symbol, tf):
    symtag = symbol.replace("/", "").lower()
    fp = f"logs/signals_{symtag}_{tf}.csv"
    if not os.path.exists(fp):
        return {'return': 0, 'win_rate': 0, 'total_trades': 0}

    df = pd.read_csv(fp)
    df = df[df['timeframe'] == tf]
    df = df[df['pnl'] != '']
    if df.empty:
        return {'return': 0, 'win_rate': 0, 'total_trades': 0}

    df['pnl'] = pd.to_numeric(df['pnl'].astype(str).str.replace('%', ''), errors='coerce')

    cumulative_return = df['pnl'].sum()
    total_trades = len(df)
    if total_trades == 0:
        return None
    win_rate = (df['pnl'] > 0).sum() / total_trades * 100

    total_trades = len(df)

    return {
        'return': round(cumulative_return, 2),
        'win_rate': round(win_rate, 2),
        'total_trades': total_trades
    }


# === 주문 엔진 =============================================
def create_exchange():
    """
    Spot/Futures 겸용 ccxt 인스턴스 생성.
    - 비ASCII(한글/이모지)가 포함된 키는 자동 무시(공개 API만 사용)
    """
    try:
        cls = getattr(ccxt, EXCHANGE_ID)
    except AttributeError:
        log(f"[INIT] unsupported exchange: {EXCHANGE_ID}")
        return None

    api_key = (os.getenv("BINANCE_API_KEY") or os.getenv("API_KEY") or "").strip()
    secret  = (os.getenv("BINANCE_SECRET")  or os.getenv("API_SECRET") or "").strip()

    # 키에 비ASCII 문자가 있으면 무효화
    def _is_ascii(s): 
        try:
            s.encode("ascii")
            return True
        except Exception:
            return False

    if not _is_ascii(api_key) or not _is_ascii(secret):
        if api_key or secret:
            log("[INIT] non-ASCII found in API key/secret → ignore keys and use public endpoints only")
        api_key, secret = "", ""

    opts = {
        'enableRateLimit': True,
        'options': {'adjustForTimeDifference': True}
    }
    if api_key and secret:
        opts['apiKey'] = api_key
        opts['secret'] = secret

    # Spot/Futures 옵션
    if EXCHANGE_ID == "binance":
        opts.setdefault('options', {})['defaultType'] = 'spot'
    elif EXCHANGE_ID == "binanceusdm":
        opts.setdefault('options', {})['defaultType'] = 'future'

    ex = cls(opts)
    try:
        if SANDBOX:
            ex.set_sandbox_mode(True)
    except Exception:
        pass
    try:
        ex.load_markets()
    except Exception as e:
        log(f"[INIT] load_markets warn: {e}")
    return ex



async def _fetch_balance_safe(ex):
    return await asyncio.to_thread(ex.fetch_balance)

async def _market_buy(ex, symbol, amount):
    return await asyncio.to_thread(ex.create_market_buy_order, symbol, amount)

async def _market_sell(ex, symbol, amount):
    return await asyncio.to_thread(ex.create_market_sell_order, symbol, amount)

def _amount_to_precision(ex, symbol, amount):
    try:
        return float(ex.amount_to_precision(symbol, amount))
    except Exception:
        return float(amount)

def _price_to_precision(ex, symbol, price):
    try:
        return float(ex.price_to_precision(symbol, price))
    except Exception:
        return float(price)

def _min_notional_ok(ex, symbol, price, amount):
    try:
        m = ex.market(symbol)
        notional = price * amount
        min_cost = 0.0
        # Binance의 경우 'limits'에 최소 주문 금액 정보가 들어있을 수 있음
        if 'limits' in m and 'cost' in m['limits'] and m['limits']['cost'].get('min'):
            min_cost = float(m['limits']['cost']['min'])
        min_cost = max(min_cost, MIN_NOTIONAL)
        return notional >= min_cost
    except Exception:
        # 정보가 없으면 보수적으로 MIN_NOTIONAL 사용
        return (price * amount) >= MIN_NOTIONAL



async def handle_trigger(symbol, tf, trigger_mode, signal, display_price, c_ts, entry_map):
    key = (symbol, tf)
    state = TRIGGER_STATE.get(key, 'FLAT')
    if trigger_mode == 'close':
        await maybe_execute_trade(symbol, tf, signal, last_price=display_price, candle_ts=c_ts)
        if signal not in ('BUY', 'SELL'):
            TRIGGER_STATE[key] = 'FLAT'
        return

    if trigger_mode == 'intrabar':
        await maybe_execute_trade(symbol, tf, signal, last_price=display_price, candle_ts=c_ts)
        return

    if trigger_mode == 'intrabar_confirm':
        if state == 'FLAT':
            if signal in ('BUY', 'SELL'):
                TRIGGER_STATE[key] = 'ARMED'
                ARMED_SIGNAL[key] = signal
                ARMED_TS[key] = c_ts
                await maybe_execute_trade(symbol, tf, signal, last_price=display_price, candle_ts=c_ts)
        elif state == 'ARMED':
            if c_ts > ARMED_TS.get(key, 0):
                if signal == ARMED_SIGNAL.get(key):
                    TRIGGER_STATE[key] = 'CONFIRMED'
                else:
                    opp = 'SELL' if ARMED_SIGNAL.get(key) == 'BUY' else 'BUY'
                    log(f"[REVERT] {symbol} {tf}: intrabar trigger reverted")
                    await maybe_execute_trade(symbol, tf, opp, last_price=display_price, candle_ts=c_ts)
                    entry_map[key] = None
                    TRIGGER_STATE[key] = 'FLAT'
                    ARMED_SIGNAL.pop(key, None)
                    ARMED_TS.pop(key, None)
        elif state == 'CONFIRMED':
            side_c = "LONG" if signal == "BUY" else "SHORT"
            snap2 = await get_price_snapshot(symbol)
            bar1m = _fetch_recent_bar_1m(symbol)
            _ex_guard = FUT_EXCHANGE or PUB_FUT_EXCHANGE
            exec_price = _exec_price_model(_ex_guard, symbol, tf, side_c, "entry", snap2, bar1m)
            if exec_price is None:
                log(f"[FILL_MODEL] intrabar_confirm missing price {symbol} {tf}")
                return
            await maybe_execute_trade(symbol, tf, signal, last_price=exec_price, candle_ts=c_ts)
            if signal not in ('BUY', 'SELL'):
                TRIGGER_STATE[key] = 'FLAT'
                ARMED_SIGNAL.pop(key, None)
                ARMED_TS.pop(key, None)
async def maybe_execute_trade(symbol, tf, signal, last_price, candle_ts=None):
    # [ANCHOR: ENTRY_CORE_V3_BEGIN]
    log(f"[ENTRY_V3] {symbol} {tf} state=pre")
    if candle_ts is None:
        log(f"⏭ {symbol} {tf}: skip reason=DATA")
        return
    candle_ts = int(candle_ts)
    if idem_hit(symbol, tf, candle_ts):
        log(f"⏭ {symbol} {tf}: skip (already executed this candle)")
        log(f"⏭ {symbol} {tf}: skip reason=IDEMP")
        return
    # --- normalize strong/weak signals ---
    _BUY_SET = {"BUY", "STRONG BUY", "WEAK BUY"}
    _SELL_SET = {"SELL", "STRONG SELL", "WEAK SELL"}
    exec_signal = "BUY" if signal in _BUY_SET else ("SELL" if signal in _SELL_SET else None)
    if exec_signal is None:
        log(f"⏭ {symbol} {tf}: skip (signal={signal})")
        log(f"⏭ {symbol} {tf}: skip reason=NEUTRAL")
        return

    # --- Hedge-side policy (paper path as futures) ---
    try:
        if not _hedge_side_allowed(symbol, tf, exec_signal):
            log(f"[PAPER] skip {symbol} {tf} {exec_signal}: hedge side policy")
            return
    except Exception as e:
        log(f"[PAPER] hedge policy warn {symbol} {tf}: {e}")

    side = ("LONG" if exec_signal=="BUY" else "SHORT")

    # [ADD] Kill-switch guard
    if signal in ("BUY","SELL") and _panic_active():
        log(f"[PANIC] skip entry {symbol} {tf} due to panic flag")
        return

    # [ADD] Daily limit guard (entry 차단)
    if signal in ("BUY","SELL"):
        ok, reason = _daily_limits_ok(capital_get())
        if not ok:
            log(f"[DAILY-LIMIT] skip entry {symbol} {tf}: {reason} -> action={DAILY_LIMIT_ACTION}")
            if DAILY_LIMIT_ACTION == "pause":
                await cmd_pause_all()
            elif DAILY_LIMIT_ACTION == "panic":
                _panic_on()
                if PANIC_FORCE_CLOSE:
                    await close_all_positions(reason="PANIC")
            return
    # [ANCHOR: REENTRY_GUARD_BEGIN]  << ADD >>
    try:
        key = (symbol, tf, "LONG" if exec_signal=="BUY" else "SHORT")
        node = _REENTRY_MEM.get(str(key), {})
        now = time.time()

        side_block = float(node.get("side_block_until") or 0)
        re_block_sec = int(os.getenv("REENTRY_SIDE_BLOCK_SEC","0") or 0)
        if re_block_sec > 0 and now < side_block:
            if int(os.getenv("REENTRY_DEBUG","0") or 0):
                log(f"[REENTRY] skip same-side cooldown until {side_block:.0f} ({(side_block-now):.0f}s)")
            return

        min_away = float(os.getenv("REENTRY_MIN_PRICE_AWAY_PCT","0") or 0.0)
        if min_away > 0 and node.get("last_exit_px"):
            away = abs(float(last_price) - float(node["last_exit_px"])) / float(node["last_exit_px"]) * 100.0
            if away < min_away:
                if int(os.getenv("REENTRY_DEBUG","0") or 0):
                    log(f"[REENTRY] skip: price away {away:.2f}% < {min_away:.2f}% (last_exit {node['last_exit_px']})")
                return

        min_d = float(os.getenv("REENTRY_MIN_SCORE_DELTA","0") or 0.0)
        last_score = node.get("last_entry_score")
        cur_score = EXEC_STATE.get(('score', symbol, tf))
        if min_d > 0 and last_score is not None and cur_score is not None:
            delta = float(cur_score) - float(last_score)
            if delta < min_d:
                if int(os.getenv("REENTRY_DEBUG","0") or 0):
                    log(f"[REENTRY] skip: score delta {delta:.3f} < {min_d:.3f}")
                return

        if int(node.get("loss_streak") or 0) >= int(os.getenv("REENTRY_BLOCK_AFTER_N_LOSSES","0") or 0):
            until = float(node.get("loss_block_until") or 0)
            if now < until:
                if int(os.getenv("REENTRY_DEBUG","0") or 0):
                    log(f"[REENTRY] skip: loss-block until {until:.0f}")
                return
    except Exception as e:
        log(f"[REENTRY] guard warn: {e}")
    # [ANCHOR: REENTRY_GUARD_END]

    # [ANCHOR: PROTECTIVE_CHECK_BEFORE_ENTRY]
    key = _pp_key(symbol, tf, side)
    pos = (PAPER_POS or {}).get(key)
    if pos:
        entry = float(pos.get("entry_price") or pos.get("entry") or 0)
        clamped, bar1m = await safe_price_hint(symbol)
        if not _outlier_guard(clamped, bar1m):
            tp_price = pos.get("tp_price"); sl_price = pos.get("sl_price")
            tr_eff = pos.get("eff_tr_pct") if (pos.get("eff_tr_pct") is not None) else pos.get("tr_pct")
            lev = float(pos.get("lev") or 1.0)
            op_ts = float(pos.get("opened_ts") or 0)/1000.0
            ok_exit, reason, trig_px, dbg = _eval_exit(symbol, tf, side, entry, clamped, tp_price, sl_price, tr_eff, (symbol, tf), lev, op_ts)
            if ok_exit:

                exec_px = await _choose_exec_price(symbol, tf, reason, side, float(trig_px), bar1m)

                info = await _paper_close(symbol, tf, exec_px, reason, side=side) if TRADE_MODE=="paper" else None
                if info:

                    await _notify_trade_exit(symbol, tf, side=info["side"], entry_price=info["entry_price"], exit_price=exec_px, reason=(reason or "TP/SL"), mode="paper", pnl_pct=info.get("pnl_pct"), qty=info.get("qty"), pnl_usdt=info.get("net_usdt"))
                elif TRADE_MODE!="paper":
                    await futures_close_all(symbol, tf, exit_price=exec_px, reason=reason)
                # 엔트리 대신 보호청산 했으므로 반환

                return
        log(f"⏭ {symbol} {tf}: open pos exists → skip new entry")
        log(f"⏭ {symbol} {tf}: skip reason=OCCUPIED")
        return

    # ① 라우팅 검사 (먼저)
    if not _route_allows(symbol, tf):
        log(f"⏭ {symbol} {tf}: skip reason=ROUTE")
        return

    # ② 게이트키퍼

    cand = {"symbol": symbol, "dir": exec_signal, "score": EXEC_STATE.get(('score', symbol, tf))}
    allowed = gatekeeper_offer(tf, candle_ts * 1000, cand)

    if not allowed:
        log(f"⏸ {symbol} {tf}: pending gatekeeper (waiting/loser)")
        log(f"⏭ {symbol} {tf}: skip reason=GATEKEEPER")
        return


    if _env_on("DEBUG") or _env_on("FILL_MODEL_DEBUG"):

        try:
            log(f"[GK] mode={TRIGGER_MODE} px_src(entry)={os.getenv('ENTRY_EXEC_PRICE_SOURCE','chosen')} px_src(exit)={os.getenv('EXIT_PRICE_SOURCE','mark')} clamp={os.getenv('BAR_BOUND_CLAMP','1')}")
        except Exception:
            pass

    # --- TF occupancy check (paper vs futures) ---
    occ = FUT_POS_TF.get(tf) if TRADE_MODE == "futures" else PAPER_POS_TF.get(tf)
    # Safety: also consider cross-cache occupancy (in case of stale state)
    occ = occ or PAPER_POS_TF.get(tf) or FUT_POS_TF.get(tf)
    has_real = False
    if tf not in IGNORE_OCCUPANCY_TFS and occ:
        try:
            has_real = _has_open_position(occ, tf, TRADE_MODE)
        except Exception:
            has_real = False

        # Disallow different symbol on the same TF when ALLOW_BOTH_PER_TF=0
        if (not ALLOW_BOTH_PER_TF) and (str(occ) != str(symbol)):
            log(f"⏭ {symbol} {tf}: skip reason=OCCUPIED(other={occ})")
            return

        # Strict: if any real open pos exists on this TF → skip re-entry
        if POS_TF_STRICT and has_real:
            log(f"⏭ {symbol} {tf}: skip reason=OCCUPIED")
            return

        # Autorepair: clear stale occupancy if no real pos remains
        if POS_TF_AUTOREPAIR and not has_real:
            try:
                if TRADE_MODE == "futures":
                    if FUT_POS_TF.get(tf):
                        FUT_POS_TF.pop(tf, None); _save_json(OPEN_TF_FILE, FUT_POS_TF)
                else:
                    if PAPER_POS_TF.get(tf):
                        PAPER_POS_TF.pop(tf, None); _save_json(PAPER_POS_TF_FILE, PAPER_POS_TF)
                log(f"[OCCUPANCY] cleared stale tf={tf} (was {occ})")
            except Exception:
                pass
    # NOTE: Do NOT pre-commit TF occupancy here anymore.

    # [ANCHOR: AVOID_OVERWRITE_OPEN_POS]  (REPLACED)
    key_side = _pp_key(symbol, tf, side)
    existing_paper = (PAPER_POS or {}).get(key_side)
    has_paper = existing_paper is not None
    fut_qty, fut_side = await _fut_get_open_qty_side(symbol)
    has_futures = fut_qty > 0 and (fut_side == side)

    if (has_paper or has_futures) and SCALE_ENABLE:
        # NOTE: use EXEC_STATE stored score
        cur_score = EXEC_STATE.get(('score', symbol, tf))
        last_score = None
        same_side = None
        legs = 1
        lev_used = int(_req_leverage(symbol, tf))
        up_thr = float(SCALE_UP_DELTA.get(tf, 0.6))
        dn_thr = float(SCALE_DN_DELTA.get(tf, 0.8))
        step_pct = float(SCALE_STEP_PCT.get(tf, 0.25))
        red_pct  = float(SCALE_REDUCE_PCT.get(tf, 0.20))
        tf_base_cap = _margin_for_tf(tf)
        max_cap = tf_base_cap
        did_scale = False
        base_notional = float((existing_paper or {}).get("plan_total_notional") or (FUT_POS.get(symbol, {}).get("plan_total_notional") if has_futures else 0.0) or (tf_base_cap * lev_used))

        side = "LONG" if exec_signal == "BUY" else "SHORT"
        try:
            _pb, _ctx = _playbook_adjust_risk(symbol, tf, side, None, None, None, lev_used, None)
        except Exception as e:
            _pb = None
            log(f"[PB_ERR] {symbol} {tf} {e}")

        # === Playbook scaling overrides (multipliers & threshold shifts) ===
        try:
            if PLAYBOOK_SCALE_OVERRIDE and '_pb' in locals() and _pb:
                sc_step_mul   = float(_pb.get("scale_step_mul", 1.0))
                sc_reduce_mul = float(_pb.get("scale_reduce_mul", 1.0))
                sc_legs_add   = int(_pb.get("scale_legs_add", 0))
                up_shift      = float(_pb.get("scale_up_shift", 0.0))
                down_shift    = float(_pb.get("scale_down_shift", 0.0))
                step_pct = max(0.0, step_pct * sc_step_mul)
                red_pct  = max(0.0, red_pct * sc_reduce_mul)
                SCALE_MAX_LEGS_EFF = max(0, SCALE_MAX_LEGS + sc_legs_add)
                up_thr = max(0.0, up_thr + up_shift)
                dn_thr = max(0.0, dn_thr + down_shift)
            else:
                SCALE_MAX_LEGS_EFF = SCALE_MAX_LEGS
        except Exception as e:
            log(f"[PB_SCALE_ERR] {symbol} {tf} {e}")
            SCALE_MAX_LEGS_EFF = SCALE_MAX_LEGS
        # === Bracket selection for this symbol/tf/side ===
        try:
            BRACKETS_WS = _select_brackets_for(symbol, side, max_legs_eff=SCALE_MAX_LEGS_EFF)
        except Exception as e:
            log(f"[BRKT_ERR] {symbol} {tf} {e}")
            BRACKETS_WS = _parse_brackets(SCALE_BRACKETS_DEFAULT, SCALE_MAX_LEGS_EFF)
        log(f"[PB_CAP] {symbol} {tf} alloc_cap={_pb.get('alloc_abs_cap') if _pb else 0} lev_cap={_pb.get('lev_cap') if _pb else 0}")
        log(f"[PB_SCALE] {symbol} {tf} step×{_pb.get('scale_step_mul') if _pb else 1} reduce×{_pb.get('scale_reduce_mul') if _pb else 1} legs+{_pb.get('scale_legs_add') if _pb else 0} upΔ{_pb.get('scale_up_shift') if _pb else 0} downΔ{_pb.get('scale_down_shift') if _pb else 0}")

        # PAPER branch state
        if has_paper:
            same_side = (existing_paper.get("side") == ("LONG" if exec_signal=="BUY" else "SHORT"))
            legs = int(existing_paper.get("legs") or 1)
            last_score = float(existing_paper.get("last_score") or 0.0)

        # FUTURES branch state
        if has_futures and not has_paper:
            # Evaluate side equality by futures side
            want_side = "LONG" if exec_signal=="BUY" else "SHORT"
            same_side = (fut_side == want_side)
            # you may store last_score per (symbol, tf) in a FUT_STATE map if desired
            last_score = float(EXEC_STATE.get(("last_score", symbol, tf)) or 0.0)

        if cur_score is None:
            idem_mark(symbol, tf, candle_ts)
            return  # cannot decide deltas without score

        # SCALE-IN (same side + improving)
        if same_side and (cur_score - last_score) >= up_thr and legs < int(SCALE_MAX_LEGS_EFF):
            pos_ref = existing_paper if has_paper else FUT_POS.get(symbol, {})
            brk_idx = int(legs)
            targets = _plan_bracket_targets(base_notional, BRACKETS_WS)
            new_leg_notional = max(0.0, targets[brk_idx] - sum(l.get("notional",0.0) for l in pos_ref.get("legs", [])[:brk_idx+1]))
            if new_leg_notional <= 0:
                new_leg_notional = base_notional * step_pct  # safe fallback
            add_base = new_leg_notional / max(lev_used, 1e-9)
            used_base = float((pos_ref or {}).get("used_base_margin") or 0.0)
            add_base = max(0.0, min(add_base, max_cap - used_base))
            notional_add = add_base * lev_used
            if notional_add >= SCALE_MIN_ADD_NOTIONAL and add_base > 0.0:
                if TRADE_MODE == "paper":
                    add_eff = add_base
                    add_qty = (add_eff * lev_used) / float(last_price)
                    _ex_guard = FUT_EXCHANGE or PUB_FUT_EXCHANGE
                    if ENFORCE_MARKET_RULES and _ex_guard:
                        add_qty = _fut_amount_to_precision(_ex_guard, symbol, add_qty)
                        if not _fut_min_notional_ok(_ex_guard, symbol, float(last_price), add_qty):
                            if SCALE_LOG:
                                logging.info(f"[PAPER_SCALE_SKIP] below min_notional {symbol} tf={tf} add_qty={add_qty}")
                            add_eff = 0.0
                            add_qty = 0.0
                    old_qty = float(existing_paper.get("qty") or 0.0)
                    old_entry = float(existing_paper.get("entry_price") or last_price)
                    new_qty = old_qty + add_qty
                    new_entry = (old_qty*old_entry + add_qty*float(last_price)) / max(new_qty,1e-9)
                    existing_paper.update({
                        "qty": new_qty,
                        "entry_price": new_entry,
                        "eff_margin": float(existing_paper.get("eff_margin") or 0.0) + add_eff,
                        "used_base_margin": used_base + add_base,
                        "legs": legs + 1,
                        "last_score": float(cur_score),
                        "last_update_ms": int(time.time()*1000),
                    })

                    try:
                        existing_paper.setdefault("legs", [])
                        existing_paper["legs"].append({"notional": float(new_leg_notional), "price": last_price, "ts": time.time()})
                    except Exception:
                        pass

                    avg = float(existing_paper["entry_price"])
                    tp_pct = float(existing_paper.get("tp_pct", 0.0))
                    sl_pct = float(existing_paper.get("sl_pct", 0.0))
                    tr_pct = float(existing_paper.get("tr_pct", 0.0))
                    PAPER_POS[key] = existing_paper
                    try:
                        _paper_ensure_tp_sl_trailing(symbol, tf, existing_paper.get("side"), avg, tp_pct, sl_pct, tr_pct, existing_paper.get("lev"))
                    except Exception as e:
                        log(f"[PAPER] ensure tp/sl/tr warn {symbol} {tf}: {e}")
                    _save_json(PAPER_POS_FILE, PAPER_POS)
                    did_scale = True
                else:
                    add_qty = await _fut_scale_in(symbol, float(last_price), notional_add, "LONG" if exec_signal=="BUY" else "SHORT")
                    if add_qty > 0:
                        did_scale = True

                        fp = FUT_POS.get(symbol, {})
                        old_qty = float(fp.get("qty", 0.0))
                        old_entry = float(fp.get("entry", last_price))
                        new_qty = old_qty + add_qty
                        new_entry = (old_qty*old_entry + add_qty*float(last_price)) / max(new_qty,1e-9)
                        fp.update({"qty": new_qty, "entry": new_entry})
                        tp_pct_fp = float(fp.get("tp_pct", 0.0))
                        sl_pct_fp = float(fp.get("sl_pct", 0.0))
                        if fp.get("side") == "LONG":
                            fp["tp_price"] = (new_entry*(1+tp_pct_fp/100.0)) if tp_pct_fp>0 else None
                            fp["sl_price"] = (new_entry*(1-sl_pct_fp/100.0)) if sl_pct_fp>0 else None
                        else:
                            fp["tp_price"] = (new_entry*(1-tp_pct_fp/100.0)) if tp_pct_fp>0 else None
                            fp["sl_price"] = (new_entry*(1+sl_pct_fp/100.0)) if sl_pct_fp>0 else None
                        FUT_POS[symbol] = fp
                        _save_json(OPEN_POS_FILE, FUT_POS)

                        await _fut_rearm_brackets(symbol, tf, float(last_price), "LONG" if exec_signal=="BUY" else "SHORT")

                        try:
                            if TRADE_MODE=='futures' and CSV_SCALE_EVENTS:
                                kind = "SCALE_ADD"
                                _csv_log_scale_event(symbol, tf, kind, side, float(add_qty if 'add_qty' in locals() else 0.0), float(last_price), "SCALE_ADD")
                        except Exception:
                            pass

            if did_scale and SCALE_LOG:
                log(f"🔼 scale-in {symbol} {tf}: +{add_base:.2f} base (lev×{lev_used}) at {last_price:.2f} (Δscore={cur_score-last_score:.2f})")

        # SCALE-OUT (same side + weakening)
        elif same_side and (last_score - cur_score) >= dn_thr:
            pos_ref = existing_paper if has_paper else FUT_POS.get(symbol, {})
            current_notional = (float(existing_paper.get("qty") or 0.0) * float(last_price)) if has_paper else (fut_qty * float(last_price))
            try:
                targets = _plan_bracket_targets(base_notional, BRACKETS_WS)
                legs = list(pos_ref.get("legs", []))
                reduce_size = 0.0
                for i in range(len(legs)-1, -1, -1):
                    cur = float(legs[i].get("notional", 0.0))
                    tgt = float(targets[i] if i < len(targets) else 0.0)
                    excess = max(0.0, cur - tgt)
                    if excess <= 0:
                        continue
                    take = min(excess, current_notional * red_pct - reduce_size)
                    reduce_size += take
                    if reduce_size >= current_notional * red_pct:
                        break
                if reduce_size <= 0:
                    reduce_size = current_notional * red_pct
            except Exception:
                reduce_size = current_notional * red_pct
            if TRADE_MODE == "paper":
                red_qty = reduce_size / float(last_price)
                info = await _paper_reduce(symbol, tf, side, red_qty, float(last_price)) if red_qty>0 else None
                if info: did_scale = True
            else:
                red_qty = reduce_size / float(last_price)
                closed = await _fut_reduce(symbol, red_qty, "LONG" if exec_signal=="BUY" else "SHORT") if red_qty>0 else 0.0
                if closed > 0:
                    did_scale = True

                    fp = FUT_POS.get(symbol, {})
                    fp_qty = max(0.0, float(fp.get("qty", 0.0)) - closed)
                    fp["qty"] = fp_qty
                    FUT_POS[symbol] = fp
                    _save_json(OPEN_POS_FILE, FUT_POS)

                    await _fut_rearm_brackets(symbol, tf, float(last_price), "LONG" if exec_signal=="BUY" else "SHORT")


                    try:
                        if TRADE_MODE=='futures' and CSV_SCALE_EVENTS:
                            kind = "SCALE_REDUCE"
                            _csv_log_scale_event(symbol, tf, kind, side, float(closed if 'closed' in locals() else 0.0), float(last_price), "SCALE_REDUCE")
                    except Exception:
                        pass


            try:
                pos = PAPER_POS.get(_pp_key(symbol, tf, side)) if TRADE_MODE=='paper' else None
                if isinstance(pos, dict):
                    need = float(reduce_size)
                    legs = list(pos.get("legs", []))
                    for i in range(len(legs)-1, -1, -1):
                        take = min(need, float(legs[i].get("notional",0.0)))
                        legs[i]["notional"] = max(0.0, float(legs[i].get("notional",0.0)) - take)
                        need -= take
                        if need <= 1e-9: break
                    pos["legs"] = [l for l in legs if l.get("notional",0.0) > 0]
            except Exception:
                pass

            if did_scale and SCALE_LOG:
                log(f"🔽 scale-out {symbol} {tf}: -{red_pct*100:.1f}% qty at {last_price:.2f} (Δscore={last_score-cur_score:.2f})")

        # === Periodic rebalance across brackets (paper: execute; futures: log plan) ===
        try:
            pos = PAPER_POS.get(_pp_key(symbol, tf, side)) if TRADE_MODE=='paper' else FUT_POS.get(symbol)
            if isinstance(pos, dict) and SCALE_REALLOCATE_BRACKETS:
                prev_ctx = pos.get("last_ctx"); new_ctx = CTX_STATE.get(symbol) or _compute_context(symbol)
                last_ts  = float(pos.get("last_realloc_ts") or 0.0)
                if _should_realloc(prev_ctx, new_ctx, last_ts, side):
                    base_total = float(pos.get("plan_total_notional") or base_notional)
                    ws = BRACKETS_WS
                    targets = _plan_bracket_targets(base_total, ws)
                    legs = list(pos.get("legs", []))
                    while len(legs) < len(targets):
                        legs.append({"notional": 0.0, "price": last_price, "ts": time.time()})
                    deltas = [float(targets[i]) - float(legs[i].get("notional",0.0)) for i in range(len(targets))]
                    plan = [(i, d) for i,d in enumerate(deltas) if abs(d) >= SCALE_REALLOC_MIN_USDT]
                    if plan:
                        log(f"[BRKT_REALLOC] {symbol} {tf} side={side} plan={plan} targets={targets}")
                        if TRADE_MODE=='paper':
                            for i,d in plan:
                                legs[i]["notional"] = max(0.0, legs[i].get("notional",0.0) + d)
                                legs[i]["ts"] = time.time(); legs[i]["price"] = last_price
                            pos["legs"] = [l for l in legs if l.get("notional",0.0) > 0]
                        else:

                            # Futures live execution: issue reduceOnly/add market orders per plan
                            if REALLOC_FUTURES_EXECUTE:
                                for i, d_usdt in plan:
                                    note = f"BRKT_REALLOC:{_scale_note_label(i, d_usdt) if '_scale_note_label' in globals() else 'leg'+str(i)}"
                                    await _futures_exec_delta(symbol, tf, side, float(d_usdt), float(last_price), note)
                            else:
                                log(f"[BRKT_REALLOC_SKIP] {symbol} {tf} exec=off plan={plan}")

                        pos["last_ctx"] = new_ctx
                        pos["last_realloc_ts"] = time.time()
                    else:
                        pos["last_ctx"] = new_ctx
        except Exception as e:
            log(f"[BRKT_REALLOC_ERR] {symbol} {tf} {e}")

        # update last_score memory
        try: EXEC_STATE[("last_score", symbol, tf)] = float(cur_score)
        except: pass

        # notify & finalize
        if did_scale:
            try:
                cid = _get_trade_channel_id(symbol, tf); ch = client.get_channel(cid) if cid else None
                if ch:
                    action = "ADD" if (cur_score >= last_score) else "REDUCE"
                    await ch.send(f"🧪 {action} 〔{symbol} · {tf}〕 • price: {_fmt_usd(last_price)} • lev×{lev_used}")
            except: pass
            try:
                act = "SCALE_IN" if (cur_score >= last_score) else "SCALE_OUT"
                _log_scale_csv(symbol, tf, act, qty=(add_qty if act=="SCALE_IN" else (red_qty if TRADE_MODE=="paper" else closed)), price=float(last_price))
            except: pass
            idem_mark(symbol, tf, candle_ts)
            return
        else:
            # nothing to scale — keep open pos untouched
            idem_mark(symbol, tf, candle_ts)
            return

    # Prefer live futures exchange for market limits; fallback to public futures markets on paper.
    _ex_guard = FUT_EXCHANGE or PUB_FUT_EXCHANGE
    alloc = _preview_allocation_and_qty(
        symbol=symbol,
        tf=tf,
        signal=exec_signal,
        price=float(last_price),
        ex=_ex_guard
    )
    base_margin = alloc["base_margin"]
    eff_margin  = alloc["eff_margin"]
    lev_used    = alloc["lev_used"]
    qty         = alloc["qty"]
    tp_pct      = alloc.get("tp_pct")
    sl_pct      = alloc.get("sl_pct")
    tr_pct      = alloc.get("tr_pct")
    lev         = alloc.get("lev_used")

# [PATCH NEG/CCA GATE BEGIN — maybe_execute_trade]
    try:

        # ── MTF 구조 게이트 적용 ──────────────────────────────────────────────
        try:
            mtf_dec = _mtf_struct_guard(symbol, tf, exec_signal)
            if mtf_dec.get("action") == "BLOCK":
                log(f"[MTF-GATE] {symbol} {tf} {exec_signal} → WAIT: {mtf_dec.get('reason')}")
                return
        except Exception as _e:
            log(f"[MTF-GATE-WARN] {symbol} {tf} {type(_e).__name__}: {_e}")
        # ─────────────────────────────────────────────────────────────────────

        # 토글/임계치 (env는 사용자가 설정)
        coh_on      = str(cfg_get("COHERENCE_MODE", "on")).lower() in ("1","on","true","yes")
        if coh_on:
            ner_min   = float(cfg_get("NER_MIN",   "0.25"))
            ner_tgt   = float(cfg_get("NER_TARGET","0.35"))
            plr_max   = float(cfg_get("PLR_MAX",   "0.80"))
            scout_pct = float(cfg_get("SCOUT_ALLOC_PCT", "0.15"))
            edge_dlt  = float(cfg_get("EDGE_SWITCH_DELTA", "0.12"))

            side_str   = "LONG" if exec_signal == "BUY" else "SHORT"
            price_ref  = float(last_price)
            L, S       = _symbol_exposure(symbol, price_ref)

            # 후보 notional (eff_margin*lev 또는 qty*price)
            cand_notional = (float(eff_margin or 0.0) * float(lev_used or 0.0))
            if (cand_notional <= 0.0) and qty and price_ref > 0:
                cand_notional = float(qty) * price_ref

            # CCA: 점수우위 검증 (반대사이드가 강하면 스카웃)
            new_score = abs(float(EXEC_STATE.get(('score', symbol, tf)) or 0.0))
            opp_best  = _best_opposite_score(symbol, side_str)
            cca_weaken = (opp_best > 0.0) and ((new_score - opp_best) < edge_dlt)

            # NEG: NER/PLR 예측
            ner_next, plr_next = _ner_plr(L, S, side_str, cand_notional)

            scale_factor = 1.0
            reason_tags  = []

            # PLR 상한 위반 → 스카웃
            if plr_next > plr_max:
                scale_factor = min(scale_factor, scout_pct); reason_tags.append(f"PLR>{plr_max:.2f}")

            # lighter side 진입 시 NER>=ner_min 유지 위한 x 상한 적용
            heavier = "LONG" if L >= S else "SHORT"
            if side_str != heavier:
                x_max = _x_max_for_ner_min(L, S, side_str, ner_min)
                if x_max <= 0:
                    scale_factor = min(scale_factor, scout_pct); reason_tags.append("NER_CAP0")
                elif cand_notional > x_max:
                    scale_factor = min(scale_factor, max(0.0, x_max / max(cand_notional, 1e-9)))
                    reason_tags.append("NER_SCALE")

            # 점수우위 부족 시 스카웃
            if cca_weaken:
                scale_factor = min(scale_factor, scout_pct); reason_tags.append("EDGE")


            # 상위TF 근접으로 스카웃만 허용된 경우
            try:
                mtf_dec = _mtf_struct_guard(symbol, tf, exec_signal)
                if mtf_dec.get("action") == "SCOUT":
                    scale_factor = min(scale_factor, scout_pct)
                    reason_tags.append("MTF_NEAR")
            except Exception:
                pass


            # 스케일/스카웃 반영
            if scale_factor < 0.999:
                eff_margin = float(eff_margin) * float(scale_factor)
                qty        = (float(eff_margin) * float(lev_used or 1.0)) / max(price_ref, 1e-9)

                EXEC_STATE[("coh_tags", symbol, tf)] = ",".join(reason_tags)

                log(f"[COHERENCE] {symbol} {tf} {side_str} scale×{scale_factor:.2f} tags={','.join(reason_tags)}")
    except Exception as e:
        log(f"[COHERENCE_WARN] {symbol} {tf} {e}")
# [PATCH NEG/CCA GATE END — maybe_execute_trade]

    # [PATCH SAT APPLY BEGIN — maybe_execute_trade]
    try:
        side_str  = "LONG" if exec_signal == "BUY" else "SHORT"
        entry_ref = float(last_price)
        sat = _style_sl_tp(symbol, tf, side_str, entry_ref, entry_ref)
        if sat.get("sl_price") is not None:
            sl_price = sat["sl_price"]
            sl_pct = abs((entry_ref - sat["sl_price"]) / entry_ref * 100.0)
        if sat.get("tp_price") is not None:
            tp_price = sat["tp_price"]
            tp_pct = abs((sat["tp_price"] - entry_ref) / entry_ref * 100.0)
        tr_pct = sat.get("trail_pct", 0.0)
        EXEC_STATE[("style", symbol, tf)]  = sat.get("style")
        EXEC_STATE[("regime", symbol, tf)] = sat.get("regime")
        EXEC_STATE[("sl_mode", symbol, tf)] = sat.get("mode")
        EXEC_STATE[("rr", symbol, tf)] = sat.get("rr")
        EXEC_STATE[("atr_mult", symbol, tf)] = sat.get("atr_mult")
        log(f"[STYLE] {symbol} {tf} {side_str} style={sat.get('style')} regime={sat.get('regime')} mode={sat.get('mode')} rr={sat.get('rr')}")
    except Exception as e:
        log(f"[STYLE_WARN] {symbol} {tf} {e}")
    # [PATCH SAT APPLY END — maybe_execute_trade]

    _pb_label   = alloc.get("pb_label")
    _pb_w       = alloc.get("pb_w")
    _pb_alloc_mul = alloc.get("pb_alloc_mul")


    side = "LONG" if exec_signal == "BUY" else "SHORT"

    snap = await get_price_snapshot(symbol)
    bar1m = _fetch_recent_bar_1m(symbol)
    exec_price = _exec_price_model(_ex_guard, symbol, tf, side, "entry", snap, bar1m)
    if exec_price is None:
        log(f"[FILL_MODEL] missing ref price for entry {symbol} {tf}")
        return


    _ex_guard = _ex_guard if '_ex_guard' in locals() else (FUT_EXCHANGE or PUB_FUT_EXCHANGE)
    if ENFORCE_MARKET_RULES and _ex_guard:
        # futures-aware rounding + min_notional gate
        qty = _fut_amount_to_precision(_ex_guard, symbol, qty)
        if not _fut_min_notional_ok(_ex_guard, symbol, exec_price, qty):
            logging.warning(f"[PAPER_RULES] below min_notional: {symbol} {tf} qty={qty}")
            return
        exec_price = _price_to_precision(_ex_guard, symbol, float(exec_price))


    if _env_on("DEBUG") or _env_on("FILL_MODEL_DEBUG"):

        try:
            log(f"[FILL_MODEL] ENTRY {symbol} {tf} side={side} ref={os.getenv('ENTRY_EXEC_PRICE_SOURCE','chosen')} slp={_resolve_slippage_pct(symbol, tf, 'entry')}")
        except Exception:
            pass
    if _POS_LOCK:
        async with _POS_LOCK:
            PAPER_POS[key] = {
                "side": side,
                "entry": float(exec_price),
                "entry_price": float(exec_price),
                "qty": qty,
                "eff_margin": eff_margin,
                "lev": lev_used,
                "ts_ms": int(time.time()*1000),
                "high": float(exec_price),
                "low": float(exec_price),

            }
            slip = _req_slippage_pct(symbol, tf)
            eff_tp_pct, eff_sl_pct, _, _ = _eff_risk_pcts(tp_pct, sl_pct, tr_pct, lev_used)
            try:
                tp_price, sl_price, tr_pct_eff = _paper_ensure_tp_sl_trailing(
                    symbol, tf, side,
                    entry_price=float(exec_price),
                    tp_pct=(tp_pct if (tp_pct is not None) else None),
                    sl_pct=(sl_pct if (sl_pct is not None) else None),
                    tr_pct=(tr_pct if (tr_pct is not None) else None),
                    lev=float(lev_used or 1.0)
                )
            except Exception as e:
                log(f"[PAPER] ensure tp/sl/tr warn {symbol} {tf}: {e}")
                tr_pct_eff = None
            PAPER_POS[key].update({
                "tp_pct": tp_pct, "sl_pct": sl_pct, "tr_pct": tr_pct,
                "lev": float(lev_used or 1.0),
                "eff_tp_pct": eff_tp_pct, "eff_sl_pct": eff_sl_pct, "eff_tr_pct": tr_pct_eff,
                "risk_mode": RISK_INTERPRET_MODE,
                "slippage_pct": slip
            })
            _save_json(PAPER_POS_FILE, PAPER_POS)
            try:
                # now that the position is persisted, mark TF occupancy
                PAPER_POS_TF[tf] = symbol
                _save_json(PAPER_POS_TF_FILE, PAPER_POS_TF)
            except Exception:
                pass
    else:
        PAPER_POS[key] = {
            "side": side,
            "entry": float(exec_price),
            "entry_price": float(exec_price),
            "qty": qty,
            "eff_margin": eff_margin,
            "lev": lev_used,
            "ts_ms": int(time.time()*1000),
            "high": float(exec_price),
            "low": float(exec_price),

        }
        slip = _req_slippage_pct(symbol, tf)
        eff_tp_pct, eff_sl_pct, _, _ = _eff_risk_pcts(tp_pct, sl_pct, tr_pct, lev_used)
        try:
            tp_price, sl_price, tr_pct_eff = _paper_ensure_tp_sl_trailing(
                symbol, tf, side,
                entry_price=float(exec_price),

                tp_pct=(tp_pct if (tp_pct is not None) else None),
                sl_pct=(sl_pct if (sl_pct is not None) else None),
                tr_pct=(tr_pct if (tr_pct is not None) else None),
                lev=float(lev_used or 1.0)
            )
        except Exception as e:
            log(f"[PAPER] ensure tp/sl/tr warn {symbol} {tf}: {e}")
            tr_pct_eff = None
        PAPER_POS[key].update({
            "tp_pct": tp_pct, "sl_pct": sl_pct, "tr_pct": tr_pct,
            "lev": float(lev_used or 1.0),
            "eff_tp_pct": eff_tp_pct, "eff_sl_pct": eff_sl_pct, "eff_tr_pct": tr_pct_eff,
            "risk_mode": RISK_INTERPRET_MODE,
            "slippage_pct": slip
        })
        _save_json(PAPER_POS_FILE, PAPER_POS)
        try:
            # now that the position is persisted, mark TF occupancy
            PAPER_POS_TF[tf] = symbol
            _save_json(PAPER_POS_TF_FILE, PAPER_POS_TF)
        except Exception:
            pass
    # also write OPEN row for paper mode
    extra = ",".join([
          f"mode={'paper' if TRADE_MODE=='paper' else 'futures'}",
          f"lev={float(lev_used or 1.0):.2f}", f"risk_mode={RISK_INTERPRET_MODE}",
          f"tp_pct={(tp_pct if tp_pct is not None else '')}",
          f"sl_pct={(sl_pct if sl_pct is not None else '')}",
          f"tr_pct={(tr_pct if tr_pct is not None else '')}",
          f"eff_tp_pct={(eff_tp_pct if eff_tp_pct is not None else '')}",
          f"eff_sl_pct={(eff_sl_pct if eff_sl_pct is not None else '')}",
          f"eff_tr_pct={(tr_pct_eff if tr_pct_eff is not None else '')}",
          f"tp_price={(tp_price if tp_price else '')}", f"sl_price={(sl_price if sl_price else '')}",
          f"pb_label={_pb_label if '_pb_label' in locals() else ''}",
          f"pb_alloc_mul={_pb_alloc_mul if '_pb_alloc_mul' in locals() else ''}"
      ])
    if PAPER_CSV_OPEN_LOG:
        _log_trade_csv(symbol, tf, "OPEN", side, qty, exec_price, extra=extra)
# [ANCHOR: REENTRY_ON_ENTRY_SUCCESS]  << ADD WHERE ORDER SUCCEEDS >>
    try:
        key = (symbol, tf, side)
        node = _REENTRY_MEM.get(str(key), {})
        ttl = time.time() + int(os.getenv("REENTRY_SIDE_BLOCK_SEC","0") or 0)
        node["side_block_until"] = ttl
        node["last_entry_score"] = float(EXEC_STATE.get(('score', symbol, tf)) or node.get("last_entry_score") or 0.0)
        _REENTRY_MEM[str(key)] = node
        reentry_state_save()
    except Exception as e:
        log(f"[REENTRY] on_entry warn: {e}")

# [ANCHOR: POSITION_OPEN_HOOK]
    # --- Bracket legs state on open ---
    try:
        pos_obj = PAPER_POS.get(_pp_key(symbol, tf, side)) if TRADE_MODE=='paper' else FUT_POS.get(symbol)
        if isinstance(pos_obj, dict):
            # persist legs array and last-realloc metadata
            pos_obj.setdefault("legs", [])  # list of {"notional":..., "price":..., "ts":...}
            pos_obj.setdefault("plan_total_notional", float(notional_used if 'notional_used' in locals() else qty*float(exec_price)))
            pos_obj.setdefault("last_ctx", CTX_STATE.get(symbol))
            pos_obj.setdefault("last_realloc_ts", 0.0)
    except Exception:
        pass
    # initialize trailing baseline at entry (per (symbol, tf))
    try:
        k2 = (symbol, tf)
        entry_price = float(exec_price)
        if str(side).upper() == "LONG":
            highest_price[k2] = entry_price
            lowest_price.pop(k2, None)
        else:
            lowest_price[k2] = entry_price
            highest_price.pop(k2, None)
        trail_peak_roe[k2] = 0.0
    except Exception:
        pass
    previous_signal[(symbol, tf)] = exec_signal
    entry_data[(symbol, tf)] = (float(exec_price), datetime.now().strftime("%m월 %d일 %H:%M"))

    if TRADE_MODE == "paper" and PAPER_STRICT_NONZERO and (not base_margin or not eff_margin or not qty):
        logging.warning("[PAPER_WARN] zero allocation on paper entry: check PART A")

    await _notify_trade_entry(
        symbol, tf, exec_signal,
        mode="paper", price=float(exec_price),
        qty=qty,
        base_margin=base_margin, eff_margin=eff_margin,
        lev_used=lev_used,
        score=EXEC_STATE.get(('score', symbol, tf)),
        pb_label=_pb_label, pb_w=_pb_w, pb_alloc_mul=_pb_alloc_mul
    )

    # [ANCHOR: IDEMP_MARK_BEFORE_RETURN]
    idem_mark(symbol, tf, candle_ts)
    # [ANCHOR: ENTRY_CORE_V3_END]

# 모듈 로드 시점에 한 번 생성 (라이브 모드에서만 의미 있음)
try:
    GLOBAL_EXCHANGE = create_exchange() if (AUTO_TRADE and TRADE_MODE == "spot") else None
except Exception as _e:
    log(f"[INIT] exchange init fail: {_e}")
    GLOBAL_EXCHANGE = None

# === 총자본·배분 설정 ===
TOTAL_CAPITAL_USDT = float(os.getenv("TOTAL_CAPITAL_USDT", "2000"))

# [ANCHOR: CAPITAL_MGR_BEGIN]
from typing import Optional

# env 읽기 (RUNTIME_CFG 오버레이 지원)
def _cfgf(key, default):
    try:
        return float(cfg_get(key, str(default)))
    except Exception:
        return float(default)

def _cfg(key, default):
    v = cfg_get(key, None)
    return (v if v is not None else default)

# 총자본 소스/초기값
CAPITAL_SOURCE = _cfg("CAPITAL_SOURCE", "paper").lower()  # paper | exchange (현 버전은 paper만 사용)
CAPITAL_BASE   = _cfgf("CAPITAL_BASE", _cfg("TOTAL_CAPITAL_USDT", "2000"))
CAPITAL_INCLUDE_UPNL = int(_cfg("CAPITAL_INCLUDE_UPNL", "0") or "0")  # 표시용(배분에는 별도 가중 루틴 사용)

# 알림 꼬리표
ALERT_SHOW_CAPITAL = (str(_cfg("ALERT_SHOW_CAPITAL", "1")) == "1")
PLANNER_ID         = str(_cfg("PLANNER_ID", "") or "").strip()

# UPNL 가중 배분 옵션
ALLOC_USE_UPNL       = int(_cfg("ALLOC_USE_UPNL", "1") or "1")
ALLOC_UPNL_MODE      = str(_cfg("ALLOC_UPNL_MODE", "ASYM")).upper()   # NET | ASYM | POS_ONLY | NEG_ONLY
ALLOC_UPNL_W_POS     = _cfgf("ALLOC_UPNL_W_POS", 0.50)
ALLOC_UPNL_W_NEG     = _cfgf("ALLOC_UPNL_W_NEG", 1.25)
ALLOC_UPNL_EMA_ALPHA = _cfgf("ALLOC_UPNL_EMA_ALPHA", 0.30)
ALLOC_UPNL_CLAMP_PCT = _cfgf("ALLOC_UPNL_CLAMP_PCT", 20.0)
ALLOC_DEBUG          = int(_cfg("ALLOC_DEBUG", "1") or "1")

# 런타임 총자본(페이퍼)
_CAPITAL_RT: Optional[float] = float(CAPITAL_BASE)
_UPNL_EMA_VAL: Optional[float] = None

def capital_get() -> float:
    """현재 총자본(페이퍼). 실거래는 추후 확장."""
    return float(_CAPITAL_RT or 0.0)


def capital_apply_realized_pnl(delta_usd: float, fees_usd: float = 0.0):
    """실현손익을 총자본에 반영"""
    global _CAPITAL_RT
    try:
        _CAPITAL_RT = float(( _CAPITAL_RT or 0.0 ) + float(delta_usd) - float(fees_usd))
    except Exception:
        pass

# [ANCHOR: CAPITAL_PERSIST_BLOCK]  << ADD NEW >>
import json, csv, os, datetime as _dt

CAPITAL_PERSIST       = int(os.getenv("CAPITAL_PERSIST","1") or 1)
CAPITAL_STATE_PATH    = os.getenv("CAPITAL_STATE_PATH","./data/capital_state.json")
# BACKWARD-COMPAT: allow legacy CAPITAL_CSV_PATH as alias
_CAPITAL_CSV_PATH_ALIAS = (os.getenv("CAPITAL_CSV_PATH","") or "").strip()
CAPITAL_LEDGER_CSV    = (os.getenv("CAPITAL_LEDGER_CSV") or _CAPITAL_CSV_PATH_ALIAS or "./data/capital_ledger.csv")
CAPITAL_LEDGER_ENABLE = int(os.getenv("CAPITAL_LEDGER_ENABLE","1") or 1)
CAPITAL_LEDGER_APPEND_HEADERS = int(os.getenv("CAPITAL_LEDGER_APPEND_HEADERS","1") or 1)

CAP_RESET_ALLOW = int(os.getenv("CAP_RESET_ALLOW","1") or 1)
CAP_RESET_MIN   = float(os.getenv("CAP_RESET_MIN","0") or 0.0)
CAP_RESET_LOG   = int(os.getenv("CAP_RESET_LOG","1") or 1)

def _ensure_parent_dir(path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass

async def capital_save_state():
    """현재 총자본을 json 상태 파일에 저장"""
    try:
        if not CAPITAL_PERSIST:
            return
        _ensure_parent_dir(CAPITAL_STATE_PATH)
        if _STATE_LOCK:
            async with _STATE_LOCK:
                with open(CAPITAL_STATE_PATH, "w", encoding="utf-8") as f:
                    json.dump({"capital": capital_get(), "ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00","Z")}, f)
        else:
            with open(CAPITAL_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump({"capital": capital_get(), "ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00","Z")}, f)
    except Exception as e:
        log(f"[CAPITAL] save_state warn: {e}")

async def capital_load_state():
    """재시작 시 마지막 자본 복원"""
    global _CAPITAL_RT
    if not CAPITAL_PERSIST:
        return
    try:
        if os.path.isfile(CAPITAL_STATE_PATH):
            if _STATE_LOCK:
                async with _STATE_LOCK:
                    with open(CAPITAL_STATE_PATH, "r", encoding="utf-8") as f:
                        obj = json.load(f)
            else:
                with open(CAPITAL_STATE_PATH, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            val = float(obj.get("capital", CAPITAL_BASE))
            _CAPITAL_RT = val
            log(f"[CAPITAL] restored: {val:,.2f} from {CAPITAL_STATE_PATH}")
    except Exception as e:
        log(f"[CAPITAL] load_state warn: {e}")

async def _csv_append(path: str, headers: list[str], row: dict):
    """헤더 보장 + append"""
    try:
        _ensure_parent_dir(path)
        write_header = CAPITAL_LEDGER_APPEND_HEADERS and (not os.path.isfile(path) or os.path.getsize(path)==0)
        if _STATE_LOCK:
            async with _STATE_LOCK:
                with open(path, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=headers)
                    if write_header:
                        w.writeheader()
                    w.writerow(row)
        else:
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=headers)
                if write_header:
                    w.writeheader()
                w.writerow(row)
    except Exception as e:
        log(f"[CSV] append warn: {path}: {e}")

async def capital_ledger_write(event: str, **kw):
    """
    자본 원장에 한 줄 기록. event 예: CLOSE, RESET, DEPOSIT, WITHDRAW
    kw: symbol, tf, side, reason, entry_price, exit_price, qty, gross_usdt, fees_usdt, net_usdt, capital_after
    """
    if not CAPITAL_LEDGER_ENABLE:
        return
    try:
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        row = {
            "ts_utc": now,
            "event": event,
            "capital_after": f"{capital_get():.8f}",
        }
        for k,v in kw.items():
            row[k] = v
        headers = ["ts_utc","event","symbol","tf","side","reason",
                   "entry_price","exit_price","qty","gross_usdt","fees_usdt","net_usdt","capital_after"]
        await _csv_append(CAPITAL_LEDGER_CSV, headers, row)
    except Exception as e:
        log(f"[CAPITAL] ledger warn: {e}")
# [ANCHOR: CAPITAL_PERSIST_BLOCK_END]

# [ANCHOR: REENTRY_STATE_BEGIN]  << ADD NEW >>
import time, pathlib, json

REENTRY_STATE_PATH = "./data/reentry_state.json"
_REENTRY_MEM = {}  # {(symbol, tf, side): {"last_exit_px":..., "last_entry_score":..., "side_block_until": ts, "loss_streak": int, "loss_block_until": ts}}

def reentry_state_load():
    global _REENTRY_MEM
    try:
        with open(REENTRY_STATE_PATH, "r", encoding="utf-8") as f:
            _REENTRY_MEM = json.load(f)
    except Exception:
        _REENTRY_MEM = {}

def reentry_state_save():
    try:
        pathlib.Path(REENTRY_STATE_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(REENTRY_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(_REENTRY_MEM, f)
    except Exception as e:
        log(f"[REENTRY] save warn: {e}")

reentry_state_load()
# [ANCHOR: REENTRY_STATE_END]

# [ANCHOR: RESILIENT_FETCHERS_BEGIN]  << ADD NEW >>
import asyncio as _asyncio

PRICE_FETCH_MAX_RETRY = int(os.getenv("PRICE_FETCH_MAX_RETRY","3") or 3)
PRICE_FETCH_BACKOFF_MS = int(os.getenv("PRICE_FETCH_BACKOFF_MS","200") or 200)
PRICE_FALLBACK_ORDER = [x.strip() for x in os.getenv("PRICE_FALLBACK_ORDER","mark,last,mid,index").split(",")]
MARK_CLAMP_TO_LAST = int(os.getenv("MARK_CLAMP_TO_LAST","1") or 1)

async def _sleep_ms(ms:int):
    await _asyncio.sleep(max(ms,0)/1000.0)

async def _fetch_with_retry(fn, *args, **kwargs):
    """코루틴 fn을 재시도/백오프와 함께 호출"""
    err = None
    for i in range(max(1, PRICE_FETCH_MAX_RETRY)):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            err = e
            await _sleep_ms(PRICE_FETCH_BACKOFF_MS * (i+1))
    raise err

async def safe_price_hint(symbol:str):
    """
    스냅샷 후보 우선순위 → 값 선택 → (필요 시) 1m 캔들로 클램프 + 이상치 가드
    """

    snap = (await _fetch_with_retry(get_price_snapshot, symbol)) or {}

    # ✅ None 가드 + 옵션 폴백
    if not isinstance(snap, dict) or not snap:

        if os.getenv("PRICE_FALLBACK_ON_NONE", "1") == "1":
            try:
                df = get_ohlcv(symbol, "1m", limit=1)
                last = float(df["close"].iloc[-1]) if hasattr(df, "iloc") and len(df) else 0.0
            except Exception:
                last = 0.0

            return _sanitize_exit_price(symbol, last)
        return _sanitize_exit_price(symbol, 0.0)

    # 트리거 소스 우선(존재하면 cand 고정)
    pref = (TRIGGER_PRICE_SOURCE or "").strip().lower()
    if pref and (pref in snap) and (snap.get(pref) is not None):
        cand = float(snap.get(pref))
    else:
        cand = None

    # 후보 가격 선택
    if cand is None:
        for k in PRICE_FALLBACK_ORDER:
            v = snap.get(k)

            if v is not None:
                cand = float(v); break

    # mark 직접사용 제한 → last 있으면 last로 클램프

    if MARK_CLAMP_TO_LAST and (cand is not None) and ("mark" in PRICE_FALLBACK_ORDER) and (snap.get("mark") == cand):
        last = snap.get("last")
        if last is not None:

            cand = float(last)

    bar_async = await _fetch_recent_bar_1m_async(symbol)
    clamped, bar = _sanitize_exit_price_with_bar(symbol, float(cand or 0.0), bar_async)

    # 이상치면 1회 재조회(✅ None 가드)
    if _outlier_guard(clamped, bar):

        snap2 = (await _fetch_with_retry(get_price_snapshot, symbol)) or {}
        cand2 = float(snap2.get("last") or snap2.get("mid") or cand or 0.0)
        bar_async = await _fetch_recent_bar_1m_async(symbol)
        clamped, bar = _sanitize_exit_price_with_bar(symbol, cand2, bar_async)

    return clamped, bar
# [ANCHOR: RESILIENT_FETCHERS_END]

# [ANCHOR: EXEC_PRICE_MODEL_BEGIN]  << ADD NEW >>
def _resolve_slippage_pct(symbol: str, tf: str, when: str) -> float:
    """
    Resolve slippage pct priority:
      SLIPPAGE_BY_TF > SLIPPAGE_PCT_ENTRY/EXIT > SLIPPAGE_PCT (global) > 0.0
    """
    try:
        by_tf = os.getenv("SLIPPAGE_BY_TF", "")
        if by_tf:
            for kv in by_tf.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    if k.strip() == tf:
                        return float(v.strip())
    except Exception:
        pass
    try:
        if when == "entry":
            v = os.getenv("SLIPPAGE_PCT_ENTRY")
            if v:
                return float(v)
        elif when == "exit":
            v = os.getenv("SLIPPAGE_PCT_EXIT")
            if v:
                return float(v)
    except Exception:
        pass
    try:
        v = os.getenv("SLIPPAGE_PCT")
        return float(v) if v else 0.0
    except Exception:
        return 0.0

def _bar_clamp(px: float, bar: dict|None) -> float:
    """Clamp price within current 1m bar bounds if BAR_BOUND_CLAMP=1."""
    if os.getenv("BAR_BOUND_CLAMP", "1") != "1":
        return px
    try:
        lo = float(bar.get("low")) if bar and bar.get("low") is not None else None
        hi = float(bar.get("high")) if bar and bar.get("high") is not None else None
        if lo is not None and px < lo:
            return lo
        if hi is not None and px > hi:
            return hi
    except Exception:
        pass
    return px

def _pick_ref_price(symbol: str, when: str, snap: dict) -> float|None:
    """
    Choose a reference price from snapshot for execution.
    Entry: ENTRY_EXEC_PRICE_SOURCE (default 'chosen').
    Exit:  EXIT_PRICE_SOURCE      (default 'mark').
    """
    src = os.getenv("ENTRY_EXEC_PRICE_SOURCE", "chosen") if when == "entry" else os.getenv("EXIT_PRICE_SOURCE", "mark")
    return (
        snap.get(src)
        or snap.get("chosen")
        or snap.get("mid")
        or snap.get("last")
        or snap.get("mark")
        or snap.get("index")
    )

def _exec_price_model(ex, symbol: str, tf: str, side: str, when: str, snap: dict, bar: dict|None, ref_override: float|None = None) -> float|None:
    """
    Compute the execution price:
      1) pick reference (or use ref_override),
      2) apply worst-case slippage by side (+ for BUY, - for SELL),
      3) clamp to 1m bar if enabled,
      4) precision-round via exchange if available.
    """
    sideU = (side or "").upper()
    base = float(ref_override) if (ref_override is not None) else _pick_ref_price(symbol, when, snap)
    if base is None:
        return None
    slp = _resolve_slippage_pct(symbol, tf, when)
    if sideU in ("LONG", "BUY"):
        px = base * (1.0 + slp / 100.0)
    else:
        px = base * (1.0 - slp / 100.0)
    px = _bar_clamp(px, bar)
    try:
        if ex:
            px = _price_to_precision(ex, symbol, px)
    except Exception:
        pass
    return float(px)
# [ANCHOR: EXEC_PRICE_MODEL_END]

# [ANCHOR: TRAIL_GUARDS_BEGIN]
def _tf_map_get(env_key:str, tf:str, default:float) -> float:
    raw = os.getenv(env_key, "")
    try:
        items = dict([p.split(":") for p in raw.split(",") if ":" in p])
        val = float(items.get(tf, default))
    except Exception:
        val = float(default)
    return val

def _roe_pct(side:str, entry:float, last:float, lev:float) -> float:
    chg = (last-entry)/entry*100.0
    mult = 1.0 if side.upper()=="LONG" else -1.0
    return chg*mult*float(lev or 1.0)

def _arm_allowed(side:str, tf:str, entry:float, last:float, lev:float, opened_ts:float) -> bool:
    arm_delta = _tf_map_get("TRAIL_ARM_DELTA_MIN_PCT_BY_TF", tf, 0.5)
    if abs((last-entry)/entry*100.0) < arm_delta:
        return False
    if os.getenv("TRAIL_ONLY_AFTER_BREAK_EVEN","1")=="1":
        if _roe_pct(side, entry, last, lev) <= 0.0:
            return False
    need_profit = _tf_map_get("TRAIL_MIN_PROFIT_PCT_BY_TF", tf, 0.0)
    if need_profit > 0 and _roe_pct(side, entry, last, lev) < need_profit:
        return False
    hold_need = _tf_map_get("TRAIL_MIN_HOLD_SEC_BY_TF", tf, 0.0)
    if hold_need > 0:
        now = _dt.datetime.now(_dt.timezone.utc).timestamp()
        if (now - float(opened_ts or now)) < hold_need:
            return False
    return True

def _trail_priority_secondary() -> bool:
    return (os.getenv("TRAIL_PRIORITY","secondary") or "secondary").lower() == "secondary"
# [ANCHOR: TRAIL_GUARDS_END]


# [ANCHOR: DASH_ADV_HELPERS_BEGIN]
import json as _json, pathlib as _pathlib, math as _math

DASHBOARD_MAE_MFE = int(os.getenv("DASHBOARD_MAE_MFE","1") or 1)
DASHBOARD_RISK_BAR = int(os.getenv("DASHBOARD_RISK_BAR","1") or 1)
RISK_WARN_NEAR_SL_PCT = float(os.getenv("RISK_WARN_NEAR_SL_PCT","0.5") or 0.5)
RISK_BAR_WIDTH = int(os.getenv("RISK_BAR_WIDTH","12") or 12)
POS_STATS_STATE_PATH = os.getenv("POS_STATS_STATE_PATH","./data/pos_stats.json")

DASH_STATE_PATH = os.getenv("DASH_STATE_PATH", "./data/dashboard_state.json")

DASHBOARD_FUNDING = int(os.getenv("DASHBOARD_FUNDING","1") or 1)
FUNDING_COUNTDOWN_ONLY = int(os.getenv("FUNDING_COUNTDOWN_ONLY","1") or 1)
FUNDING_EXCHANGE_HINT = os.getenv("FUNDING_EXCHANGE_HINT","")

# 포지션별 극값/MAE/MFE 저장소 (세션 지속)
_POS_STATS = None

def _dash_state_load():
    try:
        with open(DASH_STATE_PATH, "r", encoding="utf-8") as f:
            s = _json.load(f)
            if isinstance(s, dict) and s.get("msg_id") and s.get("ch_id"):
                _DASHBOARD_STATE.update(s)
    except Exception:
        pass

def _dash_state_save():
    try:
        _pathlib.Path(DASH_STATE_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(DASH_STATE_PATH, "w", encoding="utf-8") as f:
            _json.dump({"msg_id": _DASHBOARD_STATE.get("msg_id", 0),
                        "ch_id":  _DASHBOARD_STATE.get("ch_id", 0)}, f)
    except Exception as e:
        log(f"[DASH] state save warn: {e}")

def _pos_stats_key(symbol:str, tf:str, side:str, entry:float, qty:float) -> str:
    return f"{symbol}|{tf}|{side}|{entry:.8f}|{qty:.8f}"

def _pos_stats_load():
    """
    POS_STATS_STATE_PATH에서 MAE/MFE 상태를 로드.
    파일이 없거나, 손상되었거나, null/비-dict이면 {}로 정규화.
    """
    global _POS_STATS
    if isinstance(_POS_STATS, dict):
        return _POS_STATS

    data = None
    try:
        with open(POS_STATS_STATE_PATH, "r", encoding="utf-8") as f:
            data = _json.load(f)  # null이면 None
    except Exception:
        data = {}

    if not isinstance(data, dict):
        log("[DASH] pos_stats file not dict -> reset to {}")
        data = {}
        if os.getenv("POS_STATS_AUTOFIX", "1") == "1":
            try:
                _pathlib.Path(POS_STATS_STATE_PATH).parent.mkdir(parents=True, exist_ok=True)
                with open(POS_STATS_STATE_PATH, "w", encoding="utf-8") as f:
                    _json.dump(data, f, ensure_ascii=False)
            except Exception as e:
                log(f"[DASH] pos_stats autofix warn: {e}")

    _POS_STATS = data
    return _POS_STATS

def _pos_stats_save():
    """
    _POS_STATS가 dict일 때만 저장. 그 외는 {}로 저장하여 null 재발 방지.
    """
    global _POS_STATS
    data = _POS_STATS if isinstance(_POS_STATS, dict) else {}
    try:
        _pathlib.Path(POS_STATS_STATE_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(POS_STATS_STATE_PATH, "w", encoding="utf-8") as f:
            _json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        log(f"[DASH] pos_stats save warn: {e}")

def _update_mae_mfe(symbol:str, tf:str, side:str, entry:float, last:float, qty:float, lev:float):
    """
    MAE/MFE (% on margin) 추정: 진입 이후 최저/최고 가격을 추적해 ROE%로 환산.
    """
    if not DASHBOARD_MAE_MFE: return (0.0, 0.0)
    st = _pos_stats_load()
    # ← 로더가 정규화하지만, 재발 감시용 최소 단언(개발 중 추적)
    assert isinstance(st, dict), f"POS_STATS must be dict, got {type(st).__name__}"
    k = _pos_stats_key(symbol, tf, side, entry, qty)
    node = st.get(k) or {"lo": entry, "hi": entry}
    node["lo"] = min(node["lo"], last)
    node["hi"] = max(node["hi"], last)
    st[k] = node

    # 변동률(%) × 방향 × 레버리지
    def _roe(p:float):
        chg_pct = (p - entry)/entry*100.0
        mult = (1.0 if side.upper()=="LONG" else -1.0)
        return chg_pct*mult*float(lev or 1.0)

    mae = min(_roe(node["lo"]), _roe(node["hi"]))  # 최악값(손실쪽)
    mfe = max(_roe(node["lo"]), _roe(node["hi"]))  # 최고값(이익쪽)
    return mae, mfe

def _risk_bar_pct(dist_to_sl_pct: float, dist_to_tp_pct: float) -> str:
    """
    거리 기반 텍스트 바: [■■■▢▢▢] 형태.
    dist_to_* 는 '현재가로부터 남은 거리(%)' (양수).
    """
    if not DASHBOARD_RISK_BAR: return ""
    span = (dist_to_sl_pct + dist_to_tp_pct)
    if span <= 0:
        return ""
    # SL쪽 가까울수록 왼쪽 채움이 적게 보이게 (SL=0이면 거의 빈 칸)
    filled = int(max(0, min(RISK_BAR_WIDTH, (dist_to_sl_pct/span)*RISK_BAR_WIDTH)))
    empty = RISK_BAR_WIDTH - filled
    return "〔" + ("■"*filled) + ("▢"*empty) + "〕"

def _warn_icon(dist_to_sl_pct: float) -> str:
    try:
        return " ⚠️" if dist_to_sl_pct <= RISK_WARN_NEAR_SL_PCT else ""
    except Exception:
        return ""

async def _funding_hint(symbol:str) -> str:
    """
    펀딩 카운트다운/추정. 거래소 API 유무에 따라 best-effort.
    반환 예: 'FUND 05:12' or 'FUND 05:12 / +0.0100%'
    """
    if not DASHBOARD_FUNDING:
        return ""
    try:
        # 프로젝트에 이미 연결된 클라이언트/헬퍼가 있으면 사용.
        # 여기서는 의존성 피하려고 best-effort 가짜 헬퍼를 사용하거나, 없으면 빈 문자열.
        # ex) next_funding_ts, predicted_rate = await fetch_funding_info(symbol, hint=FUNDING_EXCHANGE_HINT)
        next_funding_ts = None
        predicted_rate = None
        # 없으면 N/A
        if not next_funding_ts:
            return ""
        # 카운트다운
        now = datetime.now(timezone.utc).timestamp()
        sec = max(0, int(next_funding_ts - now))
        hh, mm = sec//3600, (sec%3600)//60
        if FUNDING_COUNTDOWN_ONLY or predicted_rate is None:
            return f" FUND {hh:02d}:{mm:02d}"
        else:
            return f" FUND {hh:02d}:{mm:02d} / {predicted_rate:+.4f}%"
    except Exception:
        return ""
# [ANCHOR: DASH_ADV_HELPERS_END]


# [ANCHOR: DASH_UPNL_HELPERS_BEGIN]
from typing import List, Dict, Tuple

def _pnl_usdt(side: str, entry: float, last: float, qty: float) -> float:
    # 선물/페이퍼: 달러 손익 = (last - entry) * qty * (롱=+1, 숏=-1)
    mult = 1.0 if str(side).upper() == "LONG" else -1.0
    return (last - entry) * qty * mult

def _pnl_pct_on_margin(side: str, entry: float, last: float, lev: float) -> float:
    # 퍼센트(마진 기준): 가격변동률(%) × 방향 × 레버리지
    mult = 1.0 if str(side).upper() == "LONG" else -1.0
    chg_pct = (last - entry) / entry * 100.0
    return chg_pct * mult * float(lev or 1.0)

async def gather_positions_upnl() -> Tuple[List[Dict], Dict]:
    """
    열린 포지션을 순회하며 1분봉 가드가 적용된 가격으로 UPNL/ROE를 계산, 합계/정렬 정보까지 반환
    returns (rows, totals)
      rows: [{symbol, tf, side, qty, entry, last, lev, upnl_usdt, upnl_pct_on_margin,
              notional, mae_pct, mfe_pct, dist_sl_pct, dist_tp_pct, riskbar, warn, fund}]

      totals: {upnl_usdt_sum, upnl_pct_on_equity}
    """
    rows: List[Dict] = []
    upnl_sum = 0.0
    upnl_sum_net = 0.0
    fees_exit_est_sum = 0.0
    fees_entry_sum = 0.0
    # 포지션 소스: 페이퍼/실거래 공용 요약 유틸 사용 (프로젝트 내 존재). 없다면 PAPER_POS를 직접 순회.
    for pos in get_open_positions_iter():

        if os.getenv("DASH_TRACE","0")=="1":
            assert isinstance(pos, dict), f"gather() pos type={type(pos).__name__}"

        symbol = pos["symbol"]; tf = pos["tf"]
        entry  = float(pos.get("entry_price") or pos.get("entry") or 0.0)
        lev    = float(pos.get("lev") or 1.0)
        side   = pos.get("side","").upper()
        qty    = float(pos.get("qty") or 0.0)

        # ✅ 실시간가 힌트(1m 가드/폴백 포함)
        last, bar1m = await safe_price_hint(symbol)

        upnl = _pnl_usdt(side, entry, last, qty)
        roe_pct = _pnl_pct_on_margin(side, entry, last, lev)
        notional = last * qty

        r = {
            "symbol": symbol,
            "tf": tf,
            "side": side,
            "qty": qty,
            "entry": entry,
            "last": last,
            "lev": lev,
            "upnl_usdt": upnl,
            "upnl_pct_on_margin": roe_pct,
            "notional": notional,
        }

        # 표시용 마진모드 (요청값 기준; 실선물은 fetch로 동기화되므로 거의 일치)
        try:
            _mm, _ = _req_margin_mode(symbol, tf)
            r["margin_mode"] = _mm
        except Exception:
            r["margin_mode"] = os.getenv("FUT_MARGIN","ISOLATED")

        if pos.get("eff_margin") is not None:
            try:
                r["eff_margin"] = float(pos.get("eff_margin"))
            except Exception:
                pass

        fees = _estimate_fees_usdt(symbol, qty, entry, last)
        r["fee_entry_usdt"] = fees["entry_fee"]
        r["fee_exit_est_usdt"] = fees["exit_fee_est"]

        mode = (os.getenv("UPNL_INCLUDE_FEES", "exit_only") or "exit_only").lower()
        upnl_net = r["upnl_usdt"]
        if mode in ("both", "all"):
            upnl_net = upnl_net - fees["exit_fee_est"] - fees["entry_fee"]
        elif mode in ("exit", "exit_only"):
            upnl_net = upnl_net - fees["exit_fee_est"]
        r["upnl_usdt_net"] = upnl_net

        upnl_sum += upnl
        upnl_sum_net += upnl_net
        fees_exit_est_sum += fees["exit_fee_est"]
        fees_entry_sum += fees["entry_fee"]

        # === margin used (real capital) ===
        try:
            notional = abs(r.get("qty", 0.0)) * float(r.get("entry", 0.0))
            lev_now = float(r.get("lev", 0.0) or 1.0)
            margin_used = float(r.get("eff_margin") or (notional / max(1.0, lev_now)))
            r["margin_used_usdt"] = margin_used
        except Exception:
            r["margin_used_usdt"] = 0.0

        rows.append(r)


        # === MAE/MFE 업데이트 ===
        mae_pct, mfe_pct = _update_mae_mfe(symbol, tf, side, entry, last, qty, lev)

        # === SL/TP까지 거리(%) 추정 ===
        dist_sl_pct, dist_tp_pct = None, None
        slp = pos.get("sl") or pos.get("sl_price")
        tpp = pos.get("tp") or pos.get("tp_price")
        if slp and tpp:
            # 현재가 기준 남은 거리(%), 음수면 이미 지나침
            dist_sl_pct = abs((last - float(slp))/last*100.0)
            dist_tp_pct = abs((float(tpp) - last)/last*100.0)
        riskbar = _risk_bar_pct(dist_sl_pct or 0.0, dist_tp_pct or 0.0)
        warn = _warn_icon(dist_sl_pct or 999.0)

        # === 펀딩 힌트 ===
        fund = await _funding_hint(symbol)

        r.update({
            "mae_pct": mae_pct,
            "mfe_pct": mfe_pct,
            "dist_sl_pct": dist_sl_pct,
            "dist_tp_pct": dist_tp_pct,
            "riskbar": riskbar,
            "warn": warn,
            "fund": fund,
        })


    # 정렬
    mode = (os.getenv("DASHBOARD_SORT","by_notional") or "by_notional").lower()
    if mode == "by_upnl":
        rows.sort(key=lambda r: r["upnl_usdt"], reverse=True)
    elif mode == "by_symbol":
        rows.sort(key=lambda r: (r["symbol"], r["tf"]))
    else:
        rows.sort(key=lambda r: r["notional"], reverse=True)

    # 합계 퍼센트는 현재 Equity 대비(실현 총자본 기준)로 계산
    eq_base = float(capital_get() or 1.0)
    totals = {
        "upnl_usdt_sum": upnl_sum,
        "upnl_usdt_sum_net": upnl_sum_net,
        "fees_exit_est_sum": fees_exit_est_sum,
        "fees_entry_sum": fees_entry_sum,
        "upnl_pct_on_equity": (upnl_sum / eq_base * 100.0),
    }

    _pos_stats_save()

    return rows, totals
# [ANCHOR: DASH_UPNL_HELPERS_END]

# [ANCHOR: CONFIG_DUMP_HELPERS]  << ADD NEW (TOP-LEVEL FUNCS) >>
CONFIG_DUMP_MODE = os.getenv("CONFIG_DUMP_MODE","chunk").lower()  # chunk | file
CONFIG_CHUNK_LEN = int(os.getenv("CONFIG_CHUNK_LEN","1800") or 1800)
CONFIG_FILENAME  = os.getenv("CONFIG_FILENAME","config_dump.txt")

def _build_config_dump_text() -> str:
    # 필요한 항목만 골라 가독성 있게 출력
    lines = []
    lines.append("[RUNTIME]")
    lines.append(f"TRADE_MODE={TRADE_MODE}, EXCHANGE_ID={EXCHANGE_ID}, HEDGE_MODE={HEDGE_MODE}")
    lines.append(f"EXIT_EVAL_MODE={os.getenv('EXIT_EVAL_MODE')}, EXIT_FILL_MODE={os.getenv('EXIT_FILL_MODE')}, OUTLIER_MAX_1M={os.getenv('OUTLIER_MAX_1M')}")
    lines.append(f"CAPITAL_BASE={CAPITAL_BASE}, CAPITAL={capital_get():.2f}, PERSIST={CAPITAL_PERSIST}")
    lines.append(f"ALLOC_BY_TF={os.getenv('ALLOC_BY_TF')}")
    lines.append(f"LEVERAGE_BY_SYMBOL={os.getenv('LEVERAGE_BY_SYMBOL')}")
    lines.append(f"SLIPPAGE_PCT={os.getenv('SLIPPAGE_PCT')}, SLIPPAGE_BY_SYMBOL={os.getenv('SLIPPAGE_BY_SYMBOL')}")
    lines.append(f"TP/SL/TR BY SYMBOL={os.getenv('TP_PCT_BY_SYMBOL')} | {os.getenv('SL_PCT_BY_SYMBOL')} | {os.getenv('TRAIL_PCT_BY_SYMBOL')}")
    lines.append("")
    return "\n".join(lines)

async def _send_long_text_or_file(ch, text: str, fname: str):
    if CONFIG_DUMP_MODE == "file" or len(text) > 1900 and CONFIG_DUMP_MODE != "chunk":
        try:
            _ensure_parent_dir(f"./data/{fname}")
        except Exception:
            pass
        path = f"./data/{fname}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        await ch.send(file=discord.File(path, filename=fname))
        return
    if len(text) <= CONFIG_CHUNK_LEN:
        await ch.send(f"```{text}```")
        return
    i = 0
    n = len(text)
    while i < n:
        chunk = text[i:i+CONFIG_CHUNK_LEN]
        await ch.send(f"```{chunk}```")
        i += CONFIG_CHUNK_LEN


# === Discord chunked sender (2,000자 제한 대응) ================================
def _split_text_chunks(text: str, limit: int = 1900) -> list[str]:
    """문단/헤더 우선으로 자르고, 넘치면 줄단위로 보수 분할."""
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= limit:
        return [text]
    paras = re.split(r"(\n{2,})", text)  # 문단 경계 유지
    chunks, cur = [], ""
    for seg in paras:
        if not seg:
            continue
        if len(cur) + len(seg) <= limit:
            cur += seg
        else:
            if cur:
                chunks.append(cur)
            if len(seg) <= limit:
                cur = seg
            else:
                # 매우 긴 문단 → 줄 단위 분해
                lines = seg.splitlines(True)
                cur = ""
                for ln in lines:
                    if len(cur) + len(ln) > limit:
                        if cur:
                            chunks.append(cur)
                        cur = ln
                    else:
                        cur += ln
    if cur:
        chunks.append(cur)
    return chunks


async def _discord_send_chunked(channel, text: str, *, files=None, silent: bool = True, header_prefix: str = "Part"):
    """
    새 메시지 전송 전용. 첫 파트에만 파일 첨부.
    반환: [message_id, ...]
    """
    try:
        parts = _split_text_chunks(text, 1900)
        ids = []
        total = len(parts)
        for i, body in enumerate(parts, 1):
            head = (f"[{header_prefix} {i}/{total}]\n" if total > 1 else "")
            if i == 1 and files:
                m = await channel.send(content=head + body, files=files, silent=silent)
            else:
                m = await channel.send(content=head + body, silent=silent)
            ids.append(m.id)
        return ids
    except Exception as e:
        # 50035 등 폴백: 하드 컷
        if ("50035" in str(e)) or ("Must be 2000" in str(e)):
            try:
                hard = text[:1900]
                m = await channel.send(content=hard, silent=silent)
                return [m.id]
            except Exception:
                pass
        raise


async def _discord_edit_chunked(msg, channel, text: str, *, tag_key: str = "DASH", header_prefix: str = "Part"):
    """
    기존 '첫 메시지'를 edit하고 나머지 파트는 삭제 후 재전송.
    파트 ID는 CTX_STATE['_chunks'][tag_key]에 저장.
    """
    parts = _split_text_chunks(text, 1900)
    total = len(parts)
    first = f"[{header_prefix} 1/{total}]\n{parts[0]}" if total > 1 else parts[0]
    try:
        await msg.edit(content=first)
    except Exception as e:
        if ("50035" in str(e)) or ("Must be 2000" in str(e)):
            # 더 강하게 잘라 다시 시도
            await msg.edit(content=(first[:1900]))
        else:
            raise

    # 이전 꼬리 파트 삭제
    try:
        chunks_map = CTX_STATE.setdefault("_chunks", {})
        old = chunks_map.get(tag_key, [])
        for mid in old[1:]:
            try:
                m = await channel.fetch_message(int(mid))
                await m.delete()
            except Exception:
                pass
    except Exception:
        pass

    # 새 꼬리 파트 전송
    new_ids = [int(msg.id)]
    for i, body in enumerate(parts[1:], 2):
        head = f"[{header_prefix} {i}/{total}]\n"
        m = await channel.send(content=head + body, silent=True)
        new_ids.append(int(m.id))
    CTX_STATE.setdefault("_chunks", {})[tag_key] = new_ids
    return new_ids
# ==============================================================================


def _ema_update(x: float, alpha: float) -> float:
    global _UPNL_EMA_VAL
    if alpha <= 0:
        _UPNL_EMA_VAL = x
        return x
    if _UPNL_EMA_VAL is None:
        _UPNL_EMA_VAL = x
        return x
    _UPNL_EMA_VAL = alpha*x + (1-alpha)*_UPNL_EMA_VAL
    return _UPNL_EMA_VAL

def _calc_upnl_now() -> float:
    """
    매우 간단한 미실현손익 합산(페이퍼 포지션만).
    """
    try:
        tot = 0.0
        for key, pos in (PAPER_POS or {}).items():
            side = str(pos.get("side","" )).upper()
            qty  = float(pos.get("qty") or 0.0)
            entry= float(pos.get("entry_price") or 0.0)
            sym  = str(pos.get("symbol") or "").upper() or key.split("|")[0]
            if qty <= 0 or entry <= 0:
                continue
            lp = get_last_price(sym) or entry
            delta = (float(lp) - entry) * (1 if side=="LONG" else -1)
            tot += qty * delta
        return float(tot)
    except Exception:
        return 0.0

def _upnl_weighted_component(upnl_raw: float) -> float:
    up_s = _ema_update(float(upnl_raw or 0.0), float(ALLOC_UPNL_EMA_ALPHA))
    pos = max(up_s, 0.0)
    neg = min(up_s, 0.0)
    if ALLOC_UPNL_MODE == "POS_ONLY":
        contrib = pos * ALLOC_UPNL_W_POS
    elif ALLOC_UPNL_MODE == "NEG_ONLY":
        contrib = neg * ALLOC_UPNL_W_NEG
    elif ALLOC_UPNL_MODE == "ASYM":
        contrib = pos * ALLOC_UPNL_W_POS + neg * ALLOC_UPNL_W_NEG
    else:
        contrib = up_s * ALLOC_UPNL_W_POS  # NET: 동일 가중

    # base 대비 기여 한도 클램프
    base = capital_get()
    lim  = abs(base) * (ALLOC_UPNL_CLAMP_PCT/100.0)
    return max(min(contrib, lim), -lim)

def planning_capital_for_allocation() -> tuple[float, float, float]:
    """
    배분용 계획자본 계산
    returns: (base_cap, upnl_contrib, plan_cap)
    """
    base = capital_get()
    if not ALLOC_USE_UPNL:
        return float(base), 0.0, float(base)
    upnl = _calc_upnl_now()
    contrib = _upnl_weighted_component(upnl)
    plan = max(float(base) + float(contrib), 0.0)
    return float(base), float(contrib), float(plan)
# [ANCHOR: CAPITAL_MGR_END]


ALLOC_BY_TF_RAW    = os.getenv("ALLOC_BY_TF", "")   # 예: "15m:0.10,1h:0.15,4h:0.25,1d:0.40"
RESERVE_PCT        = float(os.getenv("RESERVE_PCT", "0.10"))

import re as _re
def _parse_pct_map(raw):
    m = {}
    if not raw:
        return m
    for part in _re.split(r"[;,]\s*", raw.strip()):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        try:
            m[k.strip()] = float(v.strip())
        except:
            pass
    return m

ALLOC_TF = _parse_pct_map(ALLOC_BY_TF_RAW)

def _margin_for_tf(tf):
    # ALLOC_TF에 정의가 없으면 기존 FUT_MGN_USDT 사용(하위호환)
    pct = ALLOC_TF.get(tf)
    if pct is None:
        return FUT_MGN_USDT
    try:

        base_cap, upnl_contrib, plan_cap = planning_capital_for_allocation()
        return max(0.0, float(plan_cap) * float(pct))

    except Exception:
        return FUT_MGN_USDT


# (NEW) unified (symbol, tf) key helper
def _key2(symbol: str, tf: str):
    return (str(symbol), str(tf))


# [ANCHOR: SCALE_CFG_BEGIN]
SCALE_ENABLE = (cfg_get("SCALE_ENABLE", "1") == "1")
def _parse_pct_map2(raw: str, default=0.0):
    d = {}
    if not raw: return d
    for part in re.split(r"[;,]\s*", raw.strip()):
        if ":" not in part: continue
        k, v = part.split(":", 1)
        try: d[k.strip()] = float(v.strip())
        except: d[k.strip()] = default
    return d

SCALE_MAX_LEGS = int(cfg_get("SCALE_MAX_LEGS", "3"))
SCALE_UP_DELTA = _parse_pct_map2(cfg_get("SCALE_UP_SCORE_DELTA", "15m:0.5,1h:0.6,4h:0.6,1d:0.7"))
SCALE_DN_DELTA = _parse_pct_map2(cfg_get("SCALE_DOWN_SCORE_DELTA", "15m:0.6,1h:0.7,4h:0.8,1d:1.0"))
SCALE_STEP_PCT = _parse_pct_map2(cfg_get("SCALE_STEP_PCT", "15m:0.25,1h:0.25,4h:0.25,1d:0.25"))
SCALE_REDUCE_PCT = _parse_pct_map2(cfg_get("SCALE_REDUCE_PCT", "15m:0.20,1h:0.20,4h:0.20,1d:0.20"))
SCALE_MIN_ADD_NOTIONAL = float(cfg_get("SCALE_MIN_ADD_NOTIONAL_USDT", "15"))
# ==== Rebalancing Brackets (regime-aware scaling allocation) ====
# Enable fine-grained bracket reallocation (supersedes old boolean if set)
SCALE_REALLOCATE_BRACKETS = (cfg_get("SCALE_REALLOCATE_BRACKETS", "1") == "1")
# Bracket specs: "legs|w1,w2,w3" (weights sum doesn't need to be 1; we normalize)
SCALE_BRACKETS_DEFAULT = cfg_get("SCALE_BRACKETS_DEFAULT", "3|0.50,0.30,0.20")
SCALE_BRACKETS_ALIGN   = cfg_get("SCALE_BRACKETS_ALIGN",   "4|0.45,0.25,0.20,0.10")
SCALE_BRACKETS_CONTRA  = cfg_get("SCALE_BRACKETS_CONTRA",  "2|0.60,0.40")
SCALE_BRACKETS_RANGE   = cfg_get("SCALE_BRACKETS_RANGE",   "3|0.40,0.35,0.25")
# Reallocation triggers (any true → consider rebalance)
SCALE_REALLOC_ON_ALIGN_CHANGE = (cfg_get("SCALE_REALLOC_ON_ALIGN_CHANGE", "1") == "1")
SCALE_REALLOC_ON_BIAS_STEP    = (cfg_get("SCALE_REALLOC_ON_BIAS_STEP",    "1") == "1")
SCALE_REALLOC_BIAS_STEPS      = cfg_get("SCALE_REALLOC_BIAS_STEPS",       "0.33,0.66")
# Cooldown to avoid thrashing
SCALE_REALLOC_COOLDOWN_SEC    = int(float(cfg_get("SCALE_REALLOC_COOLDOWN_SEC", "600")))
# Minimum per-bracket notional (USDT) to actually rebalance; below → skip/noise
SCALE_REALLOC_MIN_USDT        = float(cfg_get("SCALE_REALLOC_MIN_USDT", "10"))
# ==== Futures Rebalance Execution (reduceOnly / market) ====
REALLOC_FUTURES_EXECUTE   = (cfg_get("REALLOC_FUTURES_EXECUTE", "1") == "1")
REALLOC_ORDER_TYPE        = cfg_get("REALLOC_ORDER_TYPE", "market").lower()   # market only (default)
REALLOC_MIN_QTY           = float(cfg_get("REALLOC_MIN_QTY", "0"))            # 0 = off
REALLOC_MAX_RETRIES       = int(float(cfg_get("REALLOC_MAX_RETRIES", "2")))
REALLOC_RETRY_SLEEP_SEC   = float(cfg_get("REALLOC_RETRY_SLEEP_SEC", "0.5"))
CSV_SCALE_EVENTS          = (cfg_get("CSV_SCALE_EVENTS", "1") == "1")
SCALE_LOG = (cfg_get("SCALE_LOG", "1") == "1")
# [ANCHOR: SCALE_CFG_END]

# === Signal strength & MTF bias ===
from collections import defaultdict

# 마지막 신호 저장: (symbol, tf) -> {'dir': 'BUY'/'SELL', 'score': float|None, 'ts': ISO}
SIG_STATE = {}

_TF_ORDER = ['15m', '1h', '4h', '1d']
def _higher_tfs(tf):
    try:
        i = _TF_ORDER.index(tf)
        return _TF_ORDER[i+1:]
    except Exception:
        return ['1h','4h','1d']  # fallback

def _parse_kv_map(raw, to_float=False, upper_key=True):
    m = {}
    if not raw:
        return m
    import re as _re
    for part in _re.split(r"[;,]\s*", raw.strip()):
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = (k.strip().upper() if upper_key else k.strip())
        v = v.strip()
        m[k] = float(v) if to_float else v
    return m

_STRENGTH_W = _parse_kv_map(cfg_get("STRENGTH_WEIGHTS", ""), to_float=True)
if not _STRENGTH_W:
    _STRENGTH_W = {
        'STRONG_BUY':0.80, 'BUY':0.55, 'WEAK_BUY':0.30,
        'STRONG_SELL':0.80, 'SELL':0.55, 'WEAK_SELL':0.30
    }

_BUCKET_RAW = cfg_get("STRENGTH_BUCKETS", "80:STRONG,60:BASE,0:WEAK")
# 내림차순 정렬된 [(th, label)] 리스트
_STRENGTH_BUCKETS = sorted(
    [(int(k), v.upper()) for k, v in _parse_kv_map(_BUCKET_RAW, to_float=False, upper_key=False).items()],
    key=lambda x: -x[0]
)

_MTF_F = _parse_kv_map(cfg_get("MTF_FACTORS", "ALL_ALIGN:1.00,MAJ_ALIGN:1.25,SOME_ALIGN:1.10,NO_ALIGN:0.85,MAJ_OPPOSE:0.60,ALL_OPPOSE:0.40"), to_float=True)
_FULL_ON_ALL = (cfg_get("FULL_ALLOC_ON_ALL_ALIGN", "1") == "1")
_DEBUG_ALLOC = (cfg_get("DEBUG_ALLOC_LOG", "0") == "1")

# ==== Regime / Structure Context (1d 기반) ====
REGIME_ENABLE       = (cfg_get("REGIME_ENABLE", "1") == "1")
REGIME_TF           = cfg_get("REGIME_TF", "1d")
REGIME_LOOKBACK     = int(float(cfg_get("REGIME_LOOKBACK", "180")))
REGIME_TREND_R2_MIN = float(cfg_get("REGIME_TREND_R2_MIN", "0.30"))
REGIME_ADX_MIN      = float(cfg_get("REGIME_ADX_MIN", "20"))
STRUCT_ZIGZAG_PCT   = float(cfg_get("STRUCT_ZIGZAG_PCT", "3.0"))
# --- SCE (A/B) params ---
STRUCT_ENABLE          = (cfg_get("STRUCT_ENABLE", "1") == "1")
STRUCT_ATR_LEN         = int(float(cfg_get("STRUCT_ATR_LEN", "14")))
STRUCT_NEAR_THR_ATR    = float(cfg_get("STRUCT_NEAR_THR_ATR", "0.8"))
STRUCT_MAX_LEVELS      = int(float(cfg_get("STRUCT_MAX_LEVELS", "6")))
# (B 단계에서 사용)
STRUCT_BREAK_CLOSE_ATR = float(cfg_get("STRUCT_BREAK_CLOSE_ATR", "0.2"))

# --- SCE (E) MTF gate/bias ---
MTF_STRUCT_BIAS        = (cfg_get("MTF_STRUCT_BIAS", "on").lower() in ("1","on","true","yes"))
MTF_STRUCT_MAP_STR     = cfg_get("MTF_STRUCT_MAP", "15m:1h,4h;1h:4h,1d;4h:1d")
MTF_NEAR_THR_ATR       = float(cfg_get("MTF_NEAR_THR_ATR", "1.0"))
MTF_SCOUT_ONLY_NEAR    = (cfg_get("MTF_SCOUT_ONLY_NEAR", "1") == "1")
MTF_BLOCK_NEAR         = (cfg_get("MTF_BLOCK_NEAR", "0") == "1")  # true면 근접 시 진입 보류
MTF_ALERT_ENABLE       = (cfg_get("MTF_ALERT_ENABLE", "1") == "1")
MTF_ALERT_COOLDOWN_SEC = int(float(cfg_get("MTF_ALERT_COOLDOWN_SEC", "1800")))
MTF_ALERT_PREWARN_ATR  = float(cfg_get("MTF_ALERT_PREWARN_ATR", "0.6"))  # 사전경고 임계(ATR배수)

CHANNEL_BANDS_STD   = float(cfg_get("CHANNEL_BANDS_STD", "1.5"))
AVWAP_ANCHORS       = cfg_get("AVWAP_ANCHORS", "SWING_HI,SWING_LO,LAST_BREAKOUT")
CTX_ALPHA           = float(cfg_get("CTX_ALPHA", "0.35"))
CTX_BETA            = float(cfg_get("CTX_BETA", "0.50"))
REGIME_PLAYBOOK     = (cfg_get("REGIME_PLAYBOOK", "1") == "1")
ALERT_CTX_LINES     = (cfg_get("ALERT_CTX_LINES", "1") == "1")
CTX_TTL_SEC         = int(float(cfg_get("CTX_TTL_SEC", "300")))

# ==== Regime Playbook (risk & sizing auto-tuning) ====
PLAYBOOK_ENABLE         = (cfg_get("PLAYBOOK_ENABLE", "1") == "1")
# Base multipliers when signal is ALIGNED with regime direction (LONG vs TREND_UP, SHORT vs TREND_DOWN)
PB_ALIGN_TP_MUL         = float(cfg_get("PB_ALIGN_TP_MUL", "1.25"))
PB_ALIGN_SL_MUL         = float(cfg_get("PB_ALIGN_SL_MUL", "1.15"))  # wider stop in trend
PB_ALIGN_TR_MUL         = float(cfg_get("PB_ALIGN_TR_MUL", "1.20"))  # looser trail in trend
PB_ALIGN_ALLOC_MUL      = float(cfg_get("PB_ALIGN_ALLOC_MUL", "1.25"))
PB_ALIGN_LEV_CAP        = float(cfg_get("PB_ALIGN_LEV_CAP", "0"))    # 0=disabled (no extra cap)
# Base multipliers when signal is CONTRA to regime direction (counter-trend)
PB_CONTRA_TP_MUL        = float(cfg_get("PB_CONTRA_TP_MUL", "0.75"))
PB_CONTRA_SL_MUL        = float(cfg_get("PB_CONTRA_SL_MUL", "0.70")) # tighter stop counter-trend
PB_CONTRA_TR_MUL        = float(cfg_get("PB_CONTRA_TR_MUL", "0.60")) # tighter trail counter-trend
PB_CONTRA_ALLOC_MUL     = float(cfg_get("PB_CONTRA_ALLOC_MUL", "0.60"))
PB_CONTRA_LEV_CAP       = float(cfg_get("PB_CONTRA_LEV_CAP", "5"))
# RANGE regime multipliers (mean-reversion friendly defaults)
PB_RANGE_TP_MUL         = float(cfg_get("PB_RANGE_TP_MUL", "0.80"))
PB_RANGE_SL_MUL         = float(cfg_get("PB_RANGE_SL_MUL", "0.60"))
PB_RANGE_TR_MUL         = float(cfg_get("PB_RANGE_TR_MUL", "0.60"))
PB_RANGE_ALLOC_MUL      = float(cfg_get("PB_RANGE_ALLOC_MUL", "0.80"))
PB_RANGE_LEV_CAP        = float(cfg_get("PB_RANGE_LEV_CAP", "5"))
# How strongly context bias (|CTX_BIAS| in [0,1]) scales multipliers toward the regime profile
PB_INTENSITY            = float(cfg_get("PB_INTENSITY", "1.00"))  # 0..1 (1 = full effect)


# ==== Playbook hard limits & scaling overrides ====
PLAYBOOK_HARD_LIMITS     = (cfg_get("PLAYBOOK_HARD_LIMITS", "1") == "1")
# Absolute allocation caps per trade (USDT notionals); 0=disabled
PB_ALIGN_ALLOC_ABS_CAP   = float(cfg_get("PB_ALIGN_ALLOC_ABS_CAP", "0"))
PB_CONTRA_ALLOC_ABS_CAP  = float(cfg_get("PB_CONTRA_ALLOC_ABS_CAP", "0"))
PB_RANGE_ALLOC_ABS_CAP   = float(cfg_get("PB_RANGE_ALLOC_ABS_CAP", "0"))
# Regime-specific max leverage caps; 0=disabled (falls back to prior PB_*_LEV_CAP)
PB_ALIGN_MAX_LEV         = float(cfg_get("PB_ALIGN_MAX_LEV", "0"))
PB_CONTRA_MAX_LEV        = float(cfg_get("PB_CONTRA_MAX_LEV", "0"))
PB_RANGE_MAX_LEV         = float(cfg_get("PB_RANGE_MAX_LEV", "0"))
# Scaling overrides (multipliers shift existing SCALE_* by TF)
PLAYBOOK_SCALE_OVERRIDE  = (cfg_get("PLAYBOOK_SCALE_OVERRIDE", "1") == "1")
PB_ALIGN_SCALE_STEP_MUL  = float(cfg_get("PB_ALIGN_SCALE_STEP_MUL", "1.20"))
PB_CONTRA_SCALE_STEP_MUL = float(cfg_get("PB_CONTRA_SCALE_STEP_MUL", "0.70"))
PB_RANGE_SCALE_STEP_MUL  = float(cfg_get("PB_RANGE_SCALE_STEP_MUL", "0.85"))
PB_ALIGN_SCALE_REDUCE_MUL  = float(cfg_get("PB_ALIGN_SCALE_REDUCE_MUL", "1.10"))
PB_CONTRA_SCALE_REDUCE_MUL = float(cfg_get("PB_CONTRA_SCALE_REDUCE_MUL", "1.30"))
PB_RANGE_SCALE_REDUCE_MUL  = float(cfg_get("PB_RANGE_SCALE_REDUCE_MUL", "1.10"))
# Max legs adders (e.g., +1 in trend, -1 in contra; negatives are clamped to 0)
PB_ALIGN_SCALE_MAX_LEGS_ADD  = int(float(cfg_get("PB_ALIGN_SCALE_MAX_LEGS_ADD", "1")))
PB_CONTRA_SCALE_MAX_LEGS_ADD = int(float(cfg_get("PB_CONTRA_SCALE_MAX_LEGS_ADD", "-1")))
PB_RANGE_SCALE_MAX_LEGS_ADD  = int(float(cfg_get("PB_RANGE_SCALE_MAX_LEGS_ADD", "0")))
# Score-delta threshold shifts for scale up/down (applied to per-TF thresholds)
PB_ALIGN_SCALE_UP_DELTA_SHIFT   = float(cfg_get("PB_ALIGN_SCALE_UP_DELTA_SHIFT", "-0.05"))
PB_CONTRA_SCALE_UP_DELTA_SHIFT  = float(cfg_get("PB_CONTRA_SCALE_UP_DELTA_SHIFT", "0.10"))
PB_RANGE_SCALE_UP_DELTA_SHIFT   = float(cfg_get("PB_RANGE_SCALE_UP_DELTA_SHIFT", "0.00"))
PB_ALIGN_SCALE_DOWN_DELTA_SHIFT  = float(cfg_get("PB_ALIGN_SCALE_DOWN_DELTA_SHIFT", "0.00"))
PB_CONTRA_SCALE_DOWN_DELTA_SHIFT = float(cfg_get("PB_CONTRA_SCALE_DOWN_DELTA_SHIFT", "-0.05"))
PB_RANGE_SCALE_DOWN_DELTA_SHIFT  = float(cfg_get("PB_RANGE_SCALE_DOWN_DELTA_SHIFT", "0.00"))


def _bucketize(score: float|None):
    if score is None:
        return 'BASE'
    try:
        s = float(score)
    except Exception:
        return 'BASE'
    for th, label in _STRENGTH_BUCKETS:
        if s >= th:
            return label  # STRONG / BASE / WEAK
    return 'BASE'

def _strength_label(direction: str, score: float|None):
    # ex) dir='BUY', score=85 -> 'STRONG_BUY'; score=50 -> 'WEAK_BUY'
    base = (direction or 'BUY').upper()
    bucket = _bucketize(score)
    if bucket == 'STRONG':
        return f"STRONG_{base}"
    elif bucket == 'WEAK':
        return f"WEAK_{base}"
    return base  # BASE

def _strength_factor(direction: str, score: float|None) -> float:
    lab = _strength_label(direction, score)
    return float(_STRENGTH_W.get(lab, _STRENGTH_W.get(direction.upper(), 0.55)))

def _record_signal(symbol, tf, direction, score=None):
    try:
        SIG_STATE[(symbol, tf)] = {'dir': direction, 'score': score, 'ts': datetime.now().isoformat()}
    except Exception:
        pass

def _mtf_factor(symbol, tf, direction) -> tuple[float, bool]:
    """상위 TF 동의/반대 정도에 따라 계수와 '전부 일치' 여부 반환"""
    htfs = _higher_tfs(tf)
    if not htfs:
        return 1.0, False
    agree = oppose = 0
    for htf in htfs:
        rec = SIG_STATE.get((symbol, htf))
        if not rec or not rec.get('dir'):
            continue
        if rec['dir'] == direction:
            agree += 1
        else:
            oppose += 1
    all_cnt = agree + oppose
    all_agree = (all_cnt > 0 and agree == all_cnt)
    all_opp   = (all_cnt > 0 and oppose == all_cnt)
    if all_agree:
        return float(_MTF_F.get('ALL_ALIGN', 1.0)), True
    if all_opp:
        return float(_MTF_F.get('ALL_OPPOSE', 0.4)), False
    if agree >= 2:
        return float(_MTF_F.get('MAJ_ALIGN', 1.25)), False
    if agree == 1:
        return float(_MTF_F.get('SOME_ALIGN', 1.10)), False
    if agree == 0 and oppose == 0:
        return 1.0, False
    return float(_MTF_F.get('NO_ALIGN', 0.85)), False

# === TF 단일점유 시, 더 좋은 후보 선택 ===
PEER_SET = {"BTC/USDT", "ETH/USDT"}  # 같은 TF에서 경쟁시키는 심볼 집합

def _last_sig(symbol: str, tf: str):
    """최근 기록된 신호/점수/시각. 없으면 None 반환."""
    try:
        srec = SIG_STATE.get((symbol, tf)) or {}
        sig = srec.get("signal")
        score = srec.get("score")
        ts = srec.get("ts")
        return sig, score, ts
    except Exception:
        return None, None, None

def _signal_priority(symbol: str, tf: str, signal: str) -> float:
    """
    후보 우선순위 점수(높을수록 우수): sf * mf + (score/100)*w
    - sf: 강도 계수(0~1)
    - mf: 상위TF 계수(0.4~1.25쯤), 최종 사용비율은 min(1.0, sf*mf)로 캡
    - score: 0~100 가정 (없으면 50)
    """
    try:
        score = EXEC_STATE.get(('score', symbol, tf))
    except Exception:
        score = None
    if score is None:
        # 최근 SIG_STATE에 있을 수도 있음
        _, score, _ = _last_sig(symbol, tf)
    if score is None:
        score = 50.0

    sf = _strength_factor(signal, score) or 0.0
    mf, _all = _mtf_factor(symbol, tf, signal)
    if mf is None:
        mf = 1.0

    w = float(os.getenv("PICK_W_SCORE", "0.50"))
    pri = float(sf) * float(mf) + (float(score) / 100.0) * w
    return float(pri)

def _best_symbol_for_tf(tf: str, signal: str) -> str | None:
    """동일 TF에서 signal 방향 같은 후보 중 우선순위가 가장 높은 심볼을 고름."""
    cands = []
    for sym in PEER_SET:
        sig, _sc, _ts = _last_sig(sym, tf)
        if sig is None:
            continue
        if sig.upper() != signal.upper():
            continue
        cands.append(sym)
    if not cands:
        return None
    # 우선순위 계산
    best = max(cands, key=lambda s: _signal_priority(s, tf, signal))
    return best

def _is_best_candidate(symbol: str, tf: str, signal: str) -> bool:
    """현재 symbol이 해당 TF에서 가장 우수 후보인지 판정."""
    if os.getenv("PICK_BEST_PER_TF", "1") != "1":
        return True
    # 경쟁 풀에 현재 symbol이 없으면 True
    if symbol not in PEER_SET:
        return True
    best = _best_symbol_for_tf(tf, signal)
    return (best is None) or (best == symbol)


def _mtf_alignment_text(symbol: str, tf: str, direction: str):
    """
    예: ('ETH/USDT','15m','BUY') -> ("1h: BUY, 4h: BUY, 1d: SELL · 합의 2/3", 2, 1)
    상위 TF에 기록된 최근 방향(SIG_STATE)을 요약해 텍스트/집계 반환
    """
    htfs = _higher_tfs(tf)
    if not htfs:
        return "상위TF 없음", 0, 0

    parts = []
    agree = oppose = 0
    seen = 0
    for htf in htfs:
        rec = SIG_STATE.get((symbol, htf))
        if rec is None:
            parts.append(f"{htf}: -")
        else:
            d = rec.get('dir')
            if d:
                parts.append(f"{htf}: {d}")
                seen += 1
                if d == direction:
                    agree += 1
                else:
                    oppose += 1
            else:
                parts.append(f"{htf}: Neutral")
                seen += 1

    if seen == 0:
        tail = "데이터 없음"
    else:
        tail = f"합의 {agree}/{seen}"
    return (", ".join(parts) + f" · {tail}", agree, oppose)


def _qty_from_margin_eff2(ex, symbol, price, margin, tf=None):
    # 레버리지 상한 클램프 포함 + 심볼별 TF 오버라이드 반영
    req_lev = int(_req_leverage(symbol, tf))                         # ← 변경
    limits  = _market_limits(ex, symbol)
    eff_lev = int(_clamp(req_lev, 1, int(limits.get('max_lev') or 125)))
    notional = float(margin) * eff_lev
    if notional <= 0 or price <= 0:
        return 0.0
    return notional / float(price)

# === Allocation & qty preview (for notify) ===
def _preview_allocation_and_qty(symbol: str, tf: str, signal: str, price: float, ex=None):
    """
    알림에 넣을 '배분/계수/레버리지/수량' 미리 계산.
    ex 없으면(페이퍼) 거래소 한도 클램프는 생략하고 TF 레버리지를 그대로 사용.
    return dict(base_margin, eff_margin, sf, mf, all_align, lev_used, qty)
    """
    base_margin = _margin_for_tf(tf)

    score = None
    try:
        score = EXEC_STATE.get(('score', symbol, tf))
    except Exception:
        pass

    sf = _strength_factor(signal, score)
    mf, all_align = _mtf_factor(symbol, tf, signal)
    frac = min(1.0, sf * mf)
    if all_align and _FULL_ON_ALL:
        frac = 1.0
    side = "LONG" if signal == "BUY" else "SHORT"
    tp_pct = _req_tp_pct(symbol, tf, (take_profit_pct or {}))
    sl_pct = _req_sl_pct(symbol, tf, (HARD_STOP_PCT or {}))
    tr_pct = _req_trail_pct(symbol, tf, (trailing_stop_pct or {}))
    req_lev = int(_req_leverage(symbol, tf))
    lev_used = req_lev
    if ex:
        try:
            limits = _market_limits(ex, symbol)
            lev_used = int(_clamp(req_lev, 1, int(limits.get('max_lev') or 125)))
        except Exception:
            pass
    try:
        _pb, _ = _playbook_adjust_risk(symbol, tf, side, tp_pct, sl_pct, tr_pct, lev_used, None)
        tp_pct = _pb.get("tp", tp_pct)
        sl_pct = _pb.get("sl", sl_pct)
        tr_pct = _pb.get("tr", tr_pct)
        _pb_label = _pb.get("label", "PB_OFF")
        _pb_w = _pb.get("eff_w", 0.0)
        _pb_alloc_mul = float(_pb.get("alloc_mul", 1.0))
        _pb_lev_cap = float(_pb.get("lev_cap", 0.0))

        _pb_cap = float(_pb.get("alloc_abs_cap", 0.0))
    except Exception:
        _pb_label = "PB_ERR"; _pb_w = 0.0; _pb_alloc_mul = 1.0; _pb_lev_cap = 0.0; _pb_cap = 0.0
    eff_margin = base_margin * frac * _pb_alloc_mul
    if eff_margin > base_margin:
        eff_margin = base_margin
    try:
        if _pb_cap > 0:
            eff_margin = min(eff_margin, _pb_cap)
    except Exception:
        pass
    try:
        if _pb_lev_cap > 0 and float(lev_used or 1.0) > _pb_lev_cap:
            lev_used = int(_pb_lev_cap)
    except Exception:
        pass


    # 수량(미리보기)
    qty = 0.0
    try:
        if price and eff_margin and lev_used:
            qty = (float(eff_margin) * float(lev_used)) / max(float(price), 1e-9)
            if ex:
                # precision/stepSize rounding applied when exchange provided
                qty = _fut_amount_to_precision(ex, symbol, qty)
    except Exception:
        qty = 0.0

    return {
        'base_margin': float(base_margin),
        'eff_margin': float(eff_margin),
        'sf': float(sf),
        'mf': float(mf),
        'all_align': bool(all_align),
        'lev_used': int(lev_used),
        'qty': float(qty),
        'tp_pct': tp_pct,
        'sl_pct': sl_pct,
        'tr_pct': tr_pct,
        'pb_label': _pb_label,
        'pb_w': _pb_w,
        'pb_alloc_mul': _pb_alloc_mul
    }


# === 라우팅(ETH/BTC) & 동시 TF 제한 ===
ROUTE_BY_TF_RAW   = os.getenv("ROUTE_BY_TF", "")  # 예: "15m:ETH,1h:BTC,4h:AUTO,1d:AUTO"
ALLOW_BOTH_PER_TF = os.getenv("ALLOW_BOTH_PER_TF", "0") == "1"
IGNORE_OCCUPANCY_TFS = set([x.strip() for x in os.getenv("IGNORE_OCCUPANCY_TFS","" ).split(",") if x.strip()])

DEBOUNCE_SEC   = int(os.getenv('DEBOUNCE_SEC', '10'))
COOLDOWN_SEC   = int(os.getenv('COOLDOWN_SEC', '30'))
MIN_HOLD_SEC   = int(os.getenv('MIN_HOLD_SEC', '0'))
HYSTERESIS_PCT = float(os.getenv('HYSTERESIS_PCT', '0.05'))

def trigger_mode_for(tf: str) -> str:
    return os.getenv(f'TRIGGER_MODE_TF_{tf.upper()}', os.getenv('TRIGGER_MODE', 'close')).lower()

def _parse_route_map(raw):
    m = {}
    if not raw:
        return m
    for part in _re.split(r"[;,]\s*", raw.strip()):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        m[k.strip()] = v.strip().upper()
    return m

ROUTE_TF = _parse_route_map(ROUTE_BY_TF_RAW)

def _route_allows(symbol, tf):
    # [ANCHOR: allow_daily_tf_toggle]
    if tf == '1d' and os.getenv("ALLOW_DAILY_TF","1") == "1":
        return True
    rule = (ROUTE_TF.get(tf) or "AUTO").upper()
    if rule in ("AUTO", "BOTH", "ALL"):
        return True
    if rule == "ETH":
        return symbol.startswith("ETH/")
    if rule == "BTC":
        return symbol.startswith("BTC/")
    return True


# === JSON helpers & runtime state (단일 정의) ===
import os, json

os.makedirs("logs", exist_ok=True)

OPEN_POS_FILE = "logs/futures_positions.json"      # 심볼 보유 추적
OPEN_TF_FILE  = "logs/futures_positions_tf.json"   # TF별 점유 심볼 추적

def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"[WARN] json save fail {path}: {e}")

# === [ANCHOR: IDEMP_UTILS] 아이템포턴스(중복진입 방지) 유틸 ===
IDEMP_FILE = "logs/idempotence.json"
_IDEMP = _load_json(IDEMP_FILE, {})  # dict: key -> 1

def _idem_key(symbol: str, tf: str, candle_ts: int) -> str:
    return f"{symbol}|{tf}|{int(candle_ts)}"

def idem_hit(symbol: str, tf: str, candle_ts: int) -> bool:
    try:
        return _IDEMP.get(_idem_key(symbol, tf, candle_ts), 0) == 1
    except Exception:
        return False

def idem_mark(symbol: str, tf: str, candle_ts: int):
    try:
        _IDEMP[_idem_key(symbol, tf, candle_ts)] = 1
        _save_json(IDEMP_FILE, _IDEMP)
    except Exception:
        pass

def idem_clear_symbol_tf(symbol: str, tf: str):
    """Remove all idempotence marks for (symbol, tf) regardless of candle_ts."""
    try:
        prefix = f"{symbol}|{tf}|"
        keys = [k for k in list(_IDEMP.keys()) if k.startswith(prefix)]
        for k in keys: _IDEMP.pop(k, None)
        _save_json(IDEMP_FILE, _IDEMP)
    except Exception: pass

def idem_clear_all():
    try:
        _IDEMP.clear()
        _save_json(IDEMP_FILE, _IDEMP)
    except Exception: pass

PAPER_POS_TF_FILE = "logs/paper_positions_tf.json"
PAPER_POS_TF = _load_json(PAPER_POS_TF_FILE, {})   # key: tf -> symbol (paper 전용)

PAPER_POS_FILE = "logs/paper_positions.json"
PAPER_POS = _load_json(PAPER_POS_FILE, {})   # key: f"{symbol}|{tf}" -> {side, entry, opened_ts, high, low}

FUT_POS    = _load_json(OPEN_POS_FILE, {})         # symbol -> {'side','qty','entry'}
FUT_POS_TF = _load_json(OPEN_TF_FILE, {})          # tf -> "BTC/USDT" 또는 "ETH/USDT"

# [PATCH NEG/CCA HELPERS BEGIN]
def _symbol_exposure(symbol: str, ref_price: float) -> tuple[float, float]:
    """
    현재 심볼의 총 Long/Short 노치오날 합계를 계산 (paper+futures).
    반환: (L_notional, S_notional)
    """
    L = S = 0.0
    try:
        # PAPER
        for k, pos in (PAPER_POS or {}).items():
            if not isinstance(k, str) or not k.startswith(f"{symbol}|"):
                continue
            side = str(pos.get("side") or "").upper()
            qty  = float(pos.get("qty") or 0.0)
            lev  = float(pos.get("lev") or 0.0)
            eff  = float(pos.get("eff_margin") or 0.0)
            notional = (qty * ref_price) if (qty and ref_price) else (eff * lev)
            if side == "LONG": L += max(0.0, notional)
            elif side == "SHORT": S += max(0.0, notional)
    except Exception:
        pass
    try:
        # FUTURES (단일 심볼 키)
        fp = (FUT_POS or {}).get(symbol) or {}
        fside = str(fp.get("side") or "").upper()
        fqty  = abs(float(fp.get("qty") or 0.0))
        fnot  = fqty * ref_price if ref_price else 0.0
        if fside == "LONG": L += max(0.0, fnot)
        elif fside == "SHORT": S += max(0.0, fnot)
    except Exception:
        pass
    return float(L), float(S)

def _ner_plr(L: float, S: float, side: str, x: float) -> tuple[float, float]:
    """
    현재 L,S에 후보 포지션 notional(x)을 더했을 때의 (NER, PLR) 계산.
    NER = |L'-S'|/(L'+S'),  PLR = min/max
    """
    Ln = L + (x if side == "LONG" else 0.0)
    Sn = S + (x if side == "SHORT" else 0.0)
    tot = Ln + Sn
    if tot <= 0:
        return 1.0, 0.0
    net = abs(Ln - Sn)
    ner = net / tot
    mx  = max(Ln, Sn)
    mn  = min(Ln, Sn)
    plr = (mn / mx) if mx > 0 else 0.0
    return float(ner), float(plr)

def _x_max_for_ner_min(L: float, S: float, side: str, t: float) -> float:
    """
    '가벼운 쪽(lighter side)'으로 진입할 때 NER>=t를 유지하기 위한 최대 notional x 상한.
    무제한/해당없음이면 큰 값 반환(=제한 없음), 불가능하면 0 반환.
    """
    t = float(t)
    if t >= 1.0:
        return 0.0
    # heavier 판단
    heavier = "LONG" if L >= S else "SHORT"
    if side == heavier:
        return float("inf")  # 무제한 (추가해도 NER 상승/유지)
    # lighter로 진입 → x 상한 공식
    if side == "LONG" and S >= L:
        # NER = (S - L - x)/(S + L + x) >= t  → x <= ((1-t)S - (1+t)L)/(1+t)
        num = (1.0 - t) * S - (1.0 + t) * L
        den = (1.0 + t)
        return max(0.0, num / den)
    if side == "SHORT" and L >= S:
        # NER = (L - S - x)/(L + S + x) >= t  → x <= ((1-t)L - (1+t)S)/(1+t)
        num = (1.0 - t) * L - (1.0 + t) * S
        den = (1.0 + t)
        return max(0.0, num / den)
    return 0.0

def _best_opposite_score(symbol: str, side: str) -> float:
    """
    열린 '반대 사이드' 포지션들의 score(절대값) 중 최댓값 반환.
    EXEC_STATE[('score', symbol, tf)] 사용.
    """
    best = 0.0
    try:
        # PAPER
        for k, pos in (PAPER_POS or {}).items():
            if not isinstance(k, str) or not k.startswith(f"{symbol}|"):
                continue
            tf   = k.split("|", 1)[1]
            pside = str(pos.get("side") or "").upper()
            if (pside == "LONG" and side == "SHORT") or (pside == "SHORT" and side == "LONG"):
                sc = abs(float(EXEC_STATE.get(('score', symbol, tf)) or 0.0))
                if sc > best: best = sc
    except Exception:
        pass
    try:
        # FUTURES (열린 TF는 FUT_POS_TF에서 역추적)
        fp = (FUT_POS or {}).get(symbol) or {}
        fside = str(fp.get("side") or "").upper()
        if (fside == "LONG" and side == "SHORT") or (fside == "SHORT" and side == "LONG"):
            for tf, sym in (FUT_POS_TF or {}).items():
                if sym == symbol:
                    sc = abs(float(EXEC_STATE.get(('score', symbol, tf)) or 0.0))
                    if sc > best: best = sc
    except Exception:
        pass
    return float(best)
# [PATCH NEG/CCA HELPERS END]


# [PATCH SAT HELPERS BEGIN]
def _ind(symbol:str, tf:str, key:str, default:float=0.0) -> float:
    """지표 캐시 EXEC_STATE에서 안전하게 값 로드."""
    try:
        return float(EXEC_STATE.get((key, symbol, tf)) or default)
    except Exception:
        return float(default)

def _detect_regime(symbol:str, tf:str) -> str:
    """
    단순 레짐 분류:
      - trend: EMA 정렬+ADX↑
      - meanrev: RSI z-score/BB%b 과매수·과매도
      - vol_hi: ATR% 상위 퍼센타일
      - neutral: 그 외
    """
    adx  = _ind(symbol, tf, 'adx', 0.0)
    ema_fast = _ind(symbol, tf, 'ema_fast', 0.0)
    ema_slow = _ind(symbol, tf, 'ema_slow', 0.0)
    rsiz = _ind(symbol, tf, 'rsi_z', 0.0)
    bbp  = _ind(symbol, tf, 'bb_percent', 0.5)
    atrp = _ind(symbol, tf, 'atr_pct', 0.0)  # ATR/price * 100
    adx_thr  = float(cfg_get("REGIME_ADX_THR", "20"))
    rsiz_thr = float(cfg_get("REGIME_RSIZ_THR", "1.0"))
    bb_hi    = float(cfg_get("REGIME_BBP_HI", "0.90"))
    bb_lo    = float(cfg_get("REGIME_BBP_LO", "0.10"))
    atr_hi   = float(cfg_get("REGIME_ATRP_HI", "1.2"))
    trend_ok = (ema_fast > ema_slow and adx >= adx_thr) or (ema_fast < ema_slow and adx >= adx_thr)
    mean_ok  = abs(rsiz) >= rsiz_thr or (bbp >= bb_hi or bbp <= bb_lo)
    vol_ok   = atrp >= atr_hi
    if vol_ok and not trend_ok:
        return "vol_hi"
    if trend_ok and not mean_ok:
        return "trend"
    if mean_ok and not trend_ok:
        return "meanrev"
    return "neutral"

def _select_style(symbol:str, tf:str, fallback:str="intraday") -> str:
    """스타일 자동선택: STYLE_AUTO_ENABLE=on 이면 레짐에 따라 덮어쓰기."""
    base_map = str(cfg_get("STYLE_BY_TF", "15m:scalp,1h:intraday,4h:swing,1d:position"))
    style = fallback
    for pair in base_map.split(","):
        if ":" in pair:
            k,v = pair.split(":",1)
            if k.strip()==tf:
                style = v.strip(); break
    if str(cfg_get("STYLE_AUTO_ENABLE","1")).lower() in ("1","on","true","yes"):
        regime = _detect_regime(symbol, tf)
        rules  = str(cfg_get("REGIME_RULES", "trend:swing|position;meanrev:intraday|scalp;vol_hi:scalp;neutral:intraday"))
        for r in rules.split(";"):
            if ":" not in r: continue
            rk, rv = r.split(":",1)
            if rk.strip()==regime:
                style = rv.split("|")[0].strip()
                break
    return style

def _style_params(style:str) -> dict:
    """스타일별 파라미터 읽기."""
    atr_len = int(float(cfg_get("ATR_LEN","14")))
    atr_mult_str = str(cfg_get("ATR_MULT","scalp:0.8,intraday:1.2,swing:2.0,position:2.5"))
    rr_str       = str(cfg_get("RR_BY_STYLE","scalp:1.5,intraday:2.0,swing:3.0,position:3.5"))
    trail_str    = str(cfg_get("TP_TRAIL_PCT","scalp:0.25,intraday:0.5,swing:0.8,position:1.0"))
    slpct_str    = str(cfg_get("STYLE_SL_PCT","scalp:0.35,intraday:0.7,swing:1.2,position:1.8"))
    tppct_str    = str(cfg_get("STYLE_TP_PCT","scalp:0.55,intraday:1.4,swing:3.6,position:6.0"))
    def pick(smap, key, default):
        for p in smap.split(","):
            if ":" in p:
                k,v = p.split(":",1)
                if k.strip()==key:
                    return float(v)
        return float(default)
    return {
        "atr_len": atr_len,
        "atr_mult": pick(atr_mult_str, style, 1.0),
        "rr": pick(rr_str, style, 2.0),
        "trail_pct": pick(trail_str, style, 0.5),
        "sl_pct": pick(slpct_str, style, 0.7),
        "tp_pct": pick(tppct_str, style, 1.4),
        "buffer_atr": float(cfg_get("STRUCT_BUFFER_ATR","0.3")),
        "mode_order": [m.strip() for m in str(cfg_get("SL_MODE_ORDER","structure,atr,percent")).split(",") if m.strip()],
    }

def _style_sl_tp(symbol:str, tf:str, side:str, entry_price:float, ref_price:float) -> dict:
    """
    스타일·레짐 기반 SL/TP/Trail 계산.
    반환: dict(sl_price, tp_price, trail_pct, style, regime, mode, rr, atr_mult)
    """
    style  = _select_style(symbol, tf)
    regime = _detect_regime(symbol, tf)
    P  = float(entry_price or ref_price or 0.0)
    if P <= 0:
        return {"style":style,"regime":regime,"mode":"none","rr":0,"atr_mult":0,"trail_pct":0}
    prc = float(ref_price or entry_price or P)
    prm = _style_params(style)
    atr = _ind(symbol, tf, f"atr{int(prm['atr_len'])}", 0.0)
    mode_used = None
    sl = tp = None
    for mode in prm["mode_order"]:
        mode = mode.lower()
        if mode=="structure":
            swing_hi = _ind(symbol, tf, "swing_high", 0.0)
            swing_lo = _ind(symbol, tf, "swing_low", 0.0)
            if swing_hi>0 and swing_lo>0 and atr>0:
                if side.upper()=="LONG":
                    sl = max(0.0, swing_lo - (atr * prm["buffer_atr"]))
                    tp = P + (P - sl) * prm["rr"]
                else:
                    sl = min(1e18, swing_hi + (atr * prm["buffer_atr"]))
                    tp = P - (sl - P) * prm["rr"]
                mode_used = "structure"
                break
        elif mode=="atr" and atr>0:
            sl_dist = atr * prm["atr_mult"]
            if side.upper()=="LONG":
                sl = max(0.0, P - sl_dist)
                tp = P + sl_dist * prm["rr"]
            else:
                sl = min(1e18, P + sl_dist)
                tp = P - sl_dist * prm["rr"]
            mode_used = "atr"
            break
        elif mode=="percent":
            sl_pct = prm["sl_pct"] / 100.0
            tp_pct = prm["tp_pct"] / 100.0
            if side.upper()=="LONG":
                sl = max(0.0, P * (1.0 - sl_pct))
                tp = P * (1.0 + tp_pct)
            else:
                sl = P * (1.0 + sl_pct)
                tp = P * (1.0 - tp_pct)
            mode_used = "percent"
            break
    trail_pct = prm["trail_pct"]
    return {
        "sl_price": float(sl) if sl else None,
        "tp_price": float(tp) if tp else None,
        "trail_pct": float(trail_pct),
        "style": style,
        "regime": regime,
        "mode": mode_used or "none",
        "rr": float(prm["rr"]),
        "atr_mult": float(prm["atr_mult"]),
    }
# [PATCH SAT HELPERS END]

# repair hi/lo baselines on boot (applies to all TFs)
try:
    for key, pos in (PAPER_POS or {}).items():
        try:
            sym, tf, side = key.split("|", 2)
        except Exception:
            continue
        side = str(side).upper()
        entry = float(pos.get("entry_price") or 0.0)
        k2 = (sym, tf)
        if side == "LONG":
            highest_price[k2] = max(float(highest_price.get(k2, 0.0)), entry)
        elif side == "SHORT":
            lowest_price[k2]  = min(float(lowest_price.get(k2, 1e30)), entry)
except Exception: pass


def _has_open_position(symbol: str, tf: str, mode: str) -> bool:
    if mode == "paper":
        return any(PAPER_POS.get(_pp_key(symbol, tf, s)) for s in ("LONG","SHORT"))
    pos = FUT_POS.get(symbol)
    try:
        return bool(pos and abs(float(pos.get("qty", 0))) > 0)
    except Exception:
        return False


async def _paper_close(symbol: str, tf: str, exit_price: float, exit_reason: str | None = None, side: str | None = None):
    async def _do_close():
        # Resolve side if not given: if only one exists → use it; if both → prefer previous_signal or LONG
        _cands = [s for s in ("LONG","SHORT") if PAPER_POS.get(_pp_key(symbol, tf, s))]
        use_side = (side or (_cands[0] if len(_cands)==1 else None) or
                    ("LONG" if "BUY" == (previous_signal.get((symbol,tf)) or "").upper() else
                     "SHORT" if "SELL" == (previous_signal.get((symbol,tf)) or "").upper() else "LONG"))
        key = _pp_key(symbol, tf, use_side)
        pos = PAPER_POS.get(key)
        if not pos: return None
        eff_side = (pos.get("side") or use_side or "").upper()

        entry = float(pos.get("entry_price") or pos.get("entry") or 0.0)
        qty   = float(pos.get("qty") or pos.get("quantity") or 0.0)
        pnl_pct = None
        try:
            if entry > 0 and exit_price > 0:
                gross = ((exit_price - entry) / entry) * 100.0 if eff_side == "LONG" else ((entry - exit_price) / entry) * 100.0
                pnl_pct = gross
        except Exception:
            pnl_pct = None

        PAPER_POS.pop(key, None)
        # Free TF occupancy only if no side remains for this (symbol, tf)
        try:

            other = "SHORT" if eff_side=="LONG" else "LONG"

            still_open = PAPER_POS.get(_pp_key(symbol, tf, other))
            if not still_open and PAPER_POS_TF.get(tf) == symbol:
                PAPER_POS_TF.pop(tf, None)
                _save_json(PAPER_POS_TF_FILE, PAPER_POS_TF)
        except Exception:
            pass
        _save_json(PAPER_POS_FILE, PAPER_POS)

        # === 실현손익(USDT) 및 수수료 추정 ===
        qty = float(pos.get("qty") or 0.0)
        # [ANCHOR: PAPER_FEES_FUNDING_BEGIN]
        ex = FUT_EXCHANGE if FUT_EXCHANGE else PUB_FUT_EXCHANGE


        side_up = 1 if str(eff_side).upper() == "LONG" else -1
        gross_usdt = (float(exit_price) - float(entry)) * float(qty) * side_up

        # --- funding estimation (optional) ---
        opened_ms = None
        try:
            opened_ms = int(pos.get("ts_ms") or pos.get("opened_ts") or 0)
        except Exception:
            opened_ms = None
        closed_ms = int(time.time() * 1000)

        funding_fee = 0.0
        if ESTIMATE_FUNDING_IN_PNL:
            try:
                avg_px = ((float(entry) + float(exit_price)) / 2.0) if (entry and exit_price) else 0.0
                notional = float(avg_px) * float(qty)
                funding_fee = await _estimate_funding_fee(ex, symbol, notional, opened_ms, closed_ms)
            except Exception:
                funding_fee = 0.0

        # --- fees (dynamic when USE_DYNAMIC_FEE=1) ---
        if INCLUDE_FEES_IN_PNL:
            entry_bps = _fee_bps("MARKET", ex=ex, symbol=symbol)
            exit_bps  = _fee_bps("MARKET", ex=ex, symbol=symbol)
            fee_entry = _fee_usdt(entry, float(qty), entry_bps)
            fee_exit  = _fee_usdt(exit_price, float(qty), exit_bps)
        else:
            fee_entry = 0.0
            fee_exit  = 0.0


        fees_usdt = float(fee_entry + fee_exit)
        if ESTIMATE_FUNDING_IN_PNL:
            fees_usdt += float(funding_fee)


        net_usdt = float(gross_usdt) - float(fees_usdt)
        # [ANCHOR: PAPER_FEES_FUNDING_END]


        before_cap = capital_get()
        try:
            capital_apply_realized_pnl(gross_usdt, fees_usdt)
        except Exception as _e:
            log(f"[CAPITAL] apply pnl warn: {symbol} {tf}: {_e}")
        after_cap = capital_get()
        delta_cap = after_cap - before_cap
        delta_pct = (delta_cap / before_cap * 100.0) if before_cap > 0 else 0.0

        # CSV: paper CLOSE
        try:
            if PAPER_CSV_CLOSE_LOG:

                lev = float((pos or {}).get("lev") or 1.0)
                pnl_on_margin = (pnl_pct*lev) if (pnl_pct is not None) else None
                extra = ",".join([
                    "mode=paper", f"lev={lev:.2f}",
                    f"pnl_pct_price={(pnl_pct if pnl_pct is not None else 0):.4f}",
                    f"pnl_pct_on_margin={(pnl_on_margin if pnl_on_margin is not None else 0):.4f}",
                    f"reason={exit_reason}"
                ])
                _log_trade_csv(symbol, tf, "CLOSE", eff_side, float((pos or {}).get('qty',0.0)), float(exit_price), extra=extra)

        except Exception as e:
            log(f"[CSV_CLOSE_WARN] paper {symbol} {tf}: {e}")

        # [ANCHOR: PAPER_CLOSE_AND_NOTIFY_LEDGER]
        try:
            await capital_ledger_write(
                "CLOSE",
                symbol=symbol, tf=tf, side=eff_side, reason=(exit_reason or ""),
                entry_price=f"{entry:.8f}", exit_price=f"{exit_price:.8f}",
                qty=f"{qty:.8f}",
                gross_usdt=f"{gross_usdt:.8f}", fees_usdt=f"{fees_usdt:.8f}", net_usdt=f"{net_usdt:.8f}",
                capital_after=f"{after_cap:.8f}",
            )
            await capital_save_state()
        except Exception as _e:
            log(f"[CAPITAL] on_close ledger/save warn: {_e}")
        realized_pnl_usdt = net_usdt
        # [ANCHOR: REENTRY_ON_CLOSE]  << ADD NEAR CLOSE LEDGER/SAVE >>
        try:
            key = (symbol, tf, eff_side)
            node = _REENTRY_MEM.get(str(key), {})
            node["last_exit_px"] = float(exit_price)
            node["last_entry_score"] = float(entry_score) if "entry_score" in locals() else float(node.get("last_entry_score") or 0.0)
            if realized_pnl_usdt < 0:
                streak = int(node.get("loss_streak") or 0) + 1
                node["loss_streak"] = streak
                if streak >= int(os.getenv("REENTRY_BLOCK_AFTER_N_LOSSES","0") or 0):
                    ttl = time.time() + int(os.getenv("REENTRY_BLOCK_TTL_SEC","0") or 0)
                    node["loss_block_until"] = ttl
            else:
                node["loss_streak"] = 0
                node["loss_block_until"] = 0
            _REENTRY_MEM[str(key)] = node
            reentry_state_save()
        except Exception as e:
            log(f"[REENTRY] on_close warn: {e}")
        # IDEMP: allow re-entry after manual/forced close
        try: idem_clear_symbol_tf(symbol, tf)
        except Exception: pass

        return {"side": eff_side, "entry_price": entry, "pnl_pct": pnl_pct, "qty": qty, "net_usdt": net_usdt}

    if _POS_LOCK:
        async with _POS_LOCK:
            return await _do_close()
    else:
        return await _do_close()


# [ANCHOR: PAPER_PARTIAL_CLOSE_BEGIN]
async def _paper_reduce(symbol: str, tf: str, side: str, reduce_qty: float, exit_price: float):
    async def _do_reduce():
        key = _pp_key(symbol, tf, side)
        pos = PAPER_POS.get(key)
        if not pos or reduce_qty <= 0: return None
        side = (pos.get("side") or side or "").upper()
        qty_old = float(pos.get("qty",0.0))
        if qty_old <= 0: return None
        reduce_qty = min(reduce_qty, qty_old)
        _ex_guard = FUT_EXCHANGE or PUB_FUT_EXCHANGE
        if ENFORCE_MARKET_RULES and _ex_guard:
            reduce_qty = _fut_amount_to_precision(_ex_guard, symbol, reduce_qty)
            reduce_qty = min(reduce_qty, qty_old)
            if reduce_qty <= 0:
                return None
        entry = float(pos.get("entry_price") or pos.get("entry") or 0.0)
        pnl_usdt = (exit_price - entry) * reduce_qty if side=="LONG" else (entry - exit_price) * reduce_qty
        qty_new = qty_old - reduce_qty
        if qty_new <= 0: return await _paper_close(symbol, tf, exit_price, side=side)
        eff_margin_old = float(pos.get("eff_margin") or 0.0)
        eff_margin_new = eff_margin_old * (qty_new/qty_old)
        pos["qty"] = qty_new
        pos["eff_margin"] = eff_margin_new
        pos["last_update_ms"] = int(time.time()*1000)
        PAPER_POS[key] = pos
        _save_json(PAPER_POS_FILE, PAPER_POS)
        return {"pnl": pnl_usdt, "qty_closed": reduce_qty, "qty_left": qty_new}

    if _POS_LOCK:
        async with _POS_LOCK:
            return await _do_reduce()
    else:
        return await _do_reduce()

# Back-compat shim (used by older call sites)
async def _paper_reduce_compat(symbol: str, tf: str, reduce_qty: float, exit_price: float):
    # Try LONG then SHORT; if both exist, prefer the one with matching direction by PnL sign.
    for side in ("LONG", "SHORT"):
        if PAPER_POS.get(_pp_key(symbol, tf, side)):
            return await _paper_reduce(symbol, tf, side, reduce_qty, exit_price)
    return None
# [ANCHOR: PAPER_PARTIAL_CLOSE_END]

# [ANCHOR: HYDRATE_FROM_DISK_BEGIN]
def _hydrate_from_disk():
    try:
        # ==== Helpers ====
        def _pp_key(sym: str, tf: str, side: str) -> str:
            return f"{sym}|{tf}|{side.upper()}"

        globals()["_pp_key"] = _pp_key  # export helper

        # ==== Load & migrate existing PAPER_POS ====
        global PAPER_POS, PAPER_POS_TF
        migrated = {}
        for k, v in (PAPER_POS or {}).items():
            try:
                parts = k.split("|")
                if len(parts) == 2:
                    sym, tf = parts
                    side = str(v.get("side") or "").upper() or "LONG"
                    newk = _pp_key(sym, tf, side)
                else:
                    sym, tf, side = parts[0], parts[1], parts[2].upper()
                    newk = _pp_key(sym, tf, side)
                migrated[newk] = v
            except Exception:
                continue
        PAPER_POS = migrated

        # ==== Rebuild TF occupancy from migrated keys ====
        PAPER_POS_TF = PAPER_POS_TF or {}
        seen_tfs = {}
        for k in PAPER_POS.keys():
            try:
                sym, tf, _ = k.split("|", 2)
                # Occupy only if empty
                if not PAPER_POS_TF.get(tf):
                    PAPER_POS_TF[tf] = sym
                seen_tfs[tf] = True
            except Exception:
                continue
        # Prune TFs that have no matching paper positions
        for tf in list(PAPER_POS_TF.keys()):
            if tf not in seen_tfs:
                PAPER_POS_TF.pop(tf, None)

        _save_json(PAPER_POS_FILE, PAPER_POS)
        _save_json(PAPER_POS_TF_FILE, PAPER_POS_TF)
    except Exception as e:
        log(f"[HYDRATE] warn: {e}")
# [ANCHOR: HYDRATE_FROM_DISK_END]

# === [ADD] Safety / Daily limits / Kill-switch (singleton) ===
from datetime import datetime, timezone, timedelta
import json, os, pathlib, math

KILL_SWITCH_ENABLE   = os.getenv("KILL_SWITCH_ENABLE","1")=="1"
PANIC_FLAG_PATH      = os.getenv("PANIC_FLAG_PATH","./data/panic.flag")
PANIC_CANCEL_OPEN    = os.getenv("PANIC_CANCEL_OPEN_ORDERS","1")=="1"
PANIC_FORCE_CLOSE    = os.getenv("PANIC_FORCE_CLOSE","0")=="1"

DAILY_MAX_LOSS_USDT  = float(os.getenv("DAILY_MAX_LOSS_USDT","0") or 0)
DAILY_MAX_LOSS_PCT   = float(os.getenv("DAILY_MAX_LOSS_PCT","0") or 0)
DAILY_MAX_CLOSES     = int(os.getenv("DAILY_MAX_CLOSES","0") or 0)
DAILY_LIMIT_ACTION   = (os.getenv("DAILY_LIMIT_ACTION","pause") or "pause").lower()
DAILY_RESET_HOUR_KST = int(os.getenv("DAILY_RESET_HOUR_KST","9") or 9)

ENFORCE_MARKET_RULES = os.getenv("ENFORCE_MARKET_RULES","1")=="1"
MIN_NOTIONAL         = float(os.getenv("MIN_NOTIONAL","5") or 5)
TICK_ENFORCE         = os.getenv("TICK_ENFORCE","1")=="1"

TRADE_CSV_PATH       = os.getenv("TRADE_CSV_PATH","./logs/trades.csv")
CAPITAL_CSV_PATH     = os.getenv("CAPITAL_CSV_PATH","./logs/capital_ledger.csv")

DASHBOARD_ENABLE     = os.getenv("DASHBOARD_ENABLE","1")=="1"
DASHBOARD_CHANNEL_ID = int(os.getenv("DASHBOARD_CHANNEL_ID","0") or 0)
DASHBOARD_UPDATE_SEC = int(os.getenv("DASHBOARD_UPDATE_SEC","10") or 10)
PRESENCE_ENABLE      = os.getenv("PRESENCE_ENABLE","1")=="1"

def _kst_now():
    return datetime.now(timezone.utc) + timedelta(hours=9)

def _daily_key_kst(dt=None):
    if dt is None: dt=_kst_now()
    return dt.strftime("%Y-%m-%d")

def _panic_active():
    return KILL_SWITCH_ENABLE and os.path.exists(PANIC_FLAG_PATH)

def _panic_on():
    if not KILL_SWITCH_ENABLE: return
    pathlib.Path(PANIC_FLAG_PATH).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(PANIC_FLAG_PATH).write_text("1")

def _panic_off():
    try: os.remove(PANIC_FLAG_PATH)
    except FileNotFoundError: pass

_DAILY_STATE_PATH = "./data/daily_limits.json"

def _daily_state_load():
    try:
        with open(_DAILY_STATE_PATH,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _daily_state_save(st):
    pathlib.Path(_DAILY_STATE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(_DAILY_STATE_PATH,"w",encoding="utf-8") as f:
        json.dump(st,f,ensure_ascii=False,indent=2)

def _daily_reset_if_needed(capital_now: float):
    st = _daily_state_load()
    key = _daily_key_kst()
    want_hour = DAILY_RESET_HOUR_KST
    last = st.get("key","")
    last_hour = st.get("hour",want_hour)
    if key != last or last_hour != want_hour:
        st = {"key": key, "hour": want_hour, "realized_usdt": 0.0, "realized_pct": 0.0,
              "closes": 0, "base_capital": capital_now}
        _daily_state_save(st)
    return st

def _daily_register_close(pnl_usdt: float, capital_after: float):
    st = _daily_state_load()
    if not st: st = _daily_reset_if_needed(capital_after)
    st["realized_usdt"] = float(st.get("realized_usdt",0.0)) + float(pnl_usdt)
    base = float(st.get("base_capital", capital_after) or 1.0)
    st["realized_pct"]  = (st["realized_usdt"] / base) * 100.0
    st["closes"]        = int(st.get("closes",0)) + 1
    _daily_state_save(st)
    return st

def _daily_limits_ok(capital_now: float) -> (bool, str):
    st = _daily_reset_if_needed(capital_now)
    loss_usdt = float(st.get("realized_usdt",0.0))
    loss_pct  = float(st.get("realized_pct",0.0))
    closes    = int(st.get("closes",0))
    if DAILY_MAX_LOSS_USDT>0 and (-loss_usdt) >= DAILY_MAX_LOSS_USDT:
        return False, f"daily loss(usdt) reached: {loss_usdt:.2f}"
    if DAILY_MAX_LOSS_PCT>0 and (-loss_pct) >= DAILY_MAX_LOSS_PCT:
        return False, f"daily loss(pct) reached: {loss_pct:.2f}%"
    if DAILY_MAX_CLOSES>0 and closes >= DAILY_MAX_CLOSES:
        return False, f"daily max closes reached: {closes}"
    return True, ""

# === Margin Switch Queue: 포지션/오더 때문에 실패한 마진 전환을 예약 ===
MARGIN_Q_FILE = "logs/margin_switch_queue.json"
MARGIN_Q = _load_json(MARGIN_Q_FILE, {})   # symbol -> {"target":"ISOLATED"/"CROSSED", "ts": "...", "retries": 0, "last_error": ""}

def _enqueue_margin_switch(symbol: str, target: str, why: str = ""):
    tgt = _normalize_margin(target)
    now = datetime.now().isoformat(timespec="seconds")
    rec = MARGIN_Q.get(symbol) or {}
    rec.update({"target": tgt, "ts": now, "retries": int(rec.get("retries", 0)), "last_error": (rec.get("last_error") or why)})
    MARGIN_Q[symbol] = rec
    _save_json(MARGIN_Q_FILE, MARGIN_Q)
    log(f"[FUT] queued margin switch {symbol} -> {tgt} ({why})")

async def _cancel_all_orders(ex, symbol: str):
    try:
        opens = await _post(ex.fetch_open_orders, symbol)
        for o in opens or []:
            try:
                await _post(ex.cancel_order, o.get('id'), symbol)
            except Exception as e:
                log(f"[FUT] cancel warn {symbol} id={o.get('id')}: {e}")
    except Exception as e:
        log(f"[FUT] fetch_open_orders warn {symbol}: {e}")

async def cancel_all_open_orders():
    ex = FUT_EXCHANGE
    if not ex:
        return
    try:
        symbols = set(FUT_POS.keys()) | {"BTC/USDT", "ETH/USDT"}
        for sym in symbols:
            await _cancel_all_orders(ex, sym)
    except Exception as e:
        log(f"[FUT] cancel_all_open_orders warn: {e}")

async def close_all_positions(reason="PANIC"):
    n = 0
    for key, pos in list(PAPER_POS.items()):
        try:
            sym, tf, side = key.split("|",2)
        except Exception:
            continue
        fallback = float(pos.get("entry_price",0.0))
        await _paper_close(sym, tf, get_last_price(sym, fallback), reason, side=side)
        n += 1
    for tfk, sym in list(FUT_POS_TF.items()):
        await futures_close_all(sym, tfk, reason=reason)
        n += 1
    if CLEAR_IDEMP_ON_CLOSEALL:
        try: idem_clear_all()
        except Exception: pass
    return n

async def _has_open_pos_or_orders(ex, symbol: str) -> bool:
    try:
        qty, side, _ = await _fetch_pos_qty(ex, symbol)
        if abs(float(qty or 0)) > 0:
            return True
    except Exception:
        return True
    try:
        opens = await _post(ex.fetch_open_orders, symbol)
        return bool(opens)
    except Exception:
        return True

async def _apply_margin_switch_if_possible(ex, symbol: str):
    rec = MARGIN_Q.get(symbol)
    if not rec:
        return False
    target = _normalize_margin(rec.get("target") or "")
    # 스팸 방지: 최근 시도 후 10초 이내면 skip
    try:
        last_ts = rec.get("ts")
        if last_ts:
            dt = datetime.fromisoformat(last_ts)
            if (datetime.now() - dt).total_seconds() < 10:
                return False
    except Exception:
        pass

    # 포지션/오더 없도록 보장
    if await _has_open_pos_or_orders(ex, symbol):
        return False

    # 혹시 남은 오더 제거 시도
    await _cancel_all_orders(ex, symbol)

    # 전환 시도
    try:
        m = ex.market(symbol); sym_id = m.get('id') or symbol.replace('/','')
        if hasattr(ex, 'fapiPrivate_post_margintype'):
            await _post(ex.fapiPrivate_post_margintype, {'symbol': sym_id, 'marginType': target})
        elif hasattr(ex, 'set_margin_mode'):
            await _post(ex.set_margin_mode, target, symbol)
        # 성공 → 큐 제거
        MARGIN_Q.pop(symbol, None)
        _save_json(MARGIN_Q_FILE, MARGIN_Q)
        log(f"[FUT] margin switched OK {symbol} -> {target}")
        return True
    except Exception as e:
        # 실패 → 재시도 정보 갱신
        rec["ts"] = datetime.now().isoformat(timespec="seconds")
        rec["retries"] = int(rec.get("retries", 0)) + 1
        rec["last_error"] = str(e)
        MARGIN_Q[symbol] = rec
        _save_json(MARGIN_Q_FILE, MARGIN_Q)
        log(f"[FUT] margin switch retry queued {symbol} -> {target}: {e}")
        return False

async def _apply_all_pending_margin_switches(ex):
    # 심볼 단위 일괄 처리 (루프 입구에서 가끔 호출)
    if not MARGIN_Q:
        return
    for sym in list(MARGIN_Q.keys()):
        try:
            await _apply_margin_switch_if_possible(ex, sym)
        except Exception as e:
            log(f"[FUT] margin queue process warn {sym}: {e}")


# [ANCHOR: FUT_MARGIN_LEV_FETCH_BEGIN]  << ADD NEW >>
async def _fetch_current_margin_leverage(ex, symbol: str):
    """
    Return (mode, lev) where:
      mode ∈ {'ISOLATED','CROSSED'} (None if unknown)
      lev  ∈ int or None
    Tries Binance 'positionRisk' first; falls back to fetch_positions().
    """
    try:
        m = ex.market(symbol)
        sym_id = m.get('id') or symbol.replace('/', '')
    except Exception:
        sym_id = symbol.replace('/', '')

    mode = None
    lev  = None

    # 1) Binance positionRisk (preferred)
    try:
        if hasattr(ex, 'fapiPrivate_get_positionrisk'):
            rows = await _post(ex.fapiPrivate_get_positionrisk, {'symbol': sym_id})
            row = rows[0] if isinstance(rows, list) and rows else rows
            if row:
                iso = row.get('isolated')
                try:
                    if isinstance(iso, str):
                        iso = iso.lower() in ('true', '1', 'yes')
                    mode = 'ISOLATED' if iso else 'CROSSED'
                except Exception:
                    pass
                try:
                    lev = int(float(row.get('leverage') or 0))
                except Exception:
                    pass
    except Exception as e:
        log(f"[FUT] margin/lev fetch warn {symbol}: {e}")

    # 2) Fallback: fetch_positions
    if mode is None or lev is None:
        try:
            pos = await _post(ex.fetch_positions, [symbol])
            if isinstance(pos, list) and pos:
                p = pos[0] if isinstance(pos[0], dict) else {}
                info = p.get('info', {}) if isinstance(p, dict) else {}
                if lev is None:
                    try:
                        lev = int(float(p.get('leverage') or info.get('leverage') or 0))
                    except Exception:
                        pass
                if mode is None:
                    iso = info.get('isolated') if isinstance(info, dict) else None
                    if iso is not None:
                        if isinstance(iso, str):
                            iso = iso.lower() in ('true','1','yes')
                        mode = 'ISOLATED' if iso else 'CROSSED'
        except Exception:
            pass

    return (mode, lev)
# [ANCHOR: FUT_MARGIN_LEV_FETCH_END]

# [ANCHOR: FUT_PRECHECK_BEGIN]  << ADD NEW >>
async def _check_and_sync_symbol_settings(ex, symbol: str, tf: str) -> bool:
    """
    Verify current margin/leverage equals requested; fix if possible; enqueue & attempt immediate switch if margin mismatch persists.
    Returns True if settings are aligned; otherwise False (the caller should skip ordering).
    """
    try:
        wanted_margin, _ = _req_margin_mode(symbol, tf)
    except Exception:
        wanted_margin = 'ISOLATED'

    try:
        req_lev = int(_req_leverage(symbol, tf))
        lim = _market_limits(ex, symbol)
        req_lev = int(_clamp(req_lev, 1, int(lim.get('max_lev', 125))))
    except Exception:
        req_lev = 1

    cur_mode, cur_lev = await _fetch_current_margin_leverage(ex, symbol)
    if cur_mode == wanted_margin and (cur_lev is None or int(cur_lev) == int(req_lev)):
        return True

    # Try to apply settings normally
    try:
        await _ensure_symbol_settings(ex, symbol, tf)
    except Exception as e:
        log(f"[FUT] precheck ensure warn {symbol}: {e}")

    cur_mode2, cur_lev2 = await _fetch_current_margin_leverage(ex, symbol)
    if cur_mode2 == wanted_margin and (cur_lev2 is not None and int(cur_lev2) == int(req_lev)):
        return True

    # If margin still mismatched, enqueue & attempt immediate switch
    try:
        if cur_mode2 != wanted_margin:
            _enqueue_margin_switch(symbol, wanted_margin, why="pre-order mismatch")
            await _apply_margin_switch_if_possible(ex, symbol)
            cur_mode3, cur_lev3 = await _fetch_current_margin_leverage(ex, symbol)
            if cur_mode3 == wanted_margin and (cur_lev3 is not None and int(cur_lev3) == int(req_lev)):
                return True
    except Exception as e:
        log(f"[FUT] precheck queue/apply warn {symbol}: {e}")

    return False
# [ANCHOR: FUT_PRECHECK_END]

# ==========================
#   USDT-M Futures Engine
# ==========================
AUTO_TRADE   = os.getenv("AUTO_TRADE", "0") == "1"
TRADE_MODE   = os.getenv("TRADE_MODE", "paper")   # 'paper' | 'spot' | 'futures'
EXCHANGE_ID  = os.getenv("EXCHANGE_ID", "binanceusdm")
SANDBOX      = os.getenv("SANDBOX", "1") == "1"

FUT_MGN_USDT = float(os.getenv("FUT_MGN_USDT", "10"))    # 1회 진입 증거금(USDT)
FUT_LEVERAGE = int(os.getenv("FUT_LEVERAGE", os.getenv("LEVERAGE", "3")))
FUT_MARGIN   = os.getenv("FUT_MARGIN", os.getenv("MARGIN_TYPE", "ISOLATED")).upper()  # ISOLATED|CROSS
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.25"))  # 허용 슬리피지(%)

# TF별 TP/SL 퍼센트는 기존 설정을 그대로 사용:
#  - take_profit_pct[tf]
#  - HARD_STOP_PCT[tf]

FUT_EXCHANGE = None
FUT_ORDERS = {}      # (symbol, tf) -> {'tp': order_id, 'sl': order_id}
FUT_POS_TF = _load_json(OPEN_TF_FILE, {})  # key: tf -> symbol
os.makedirs("logs", exist_ok=True)


def _ppct(p, q):  # % 차이
    try:
        return abs((p - q) / q) * 100.0
    except Exception:
        return 999.0

def _qty_from_margin(price, tf=None):
    # (TF별 증거금 * TF별 레버리지) / 현재가 → 수량
    lev = TF_LEVERAGE.get(tf, FUT_LEVERAGE)
    margin = _margin_for_tf(tf)           # ← 총자본 배분 반영
    notional = margin * lev
    if notional <= 0 or price <= 0:
        return 0.0
    return notional / float(price)

def _mk_ex():
    """
    Futures 모드일 때만 ccxt 인스턴스를 만들고,
    API 키에 비ASCII 문자가 섞여 있으면 선물 엔진을 비활성화해
    'latin-1' 인코딩 예외를 원천 차단합니다.
    """
    if not AUTO_TRADE or TRADE_MODE != "futures":
        return None
    try:
        cls = getattr(ccxt, EXCHANGE_ID)
    except AttributeError:
        log(f"[FUT] Unsupported exchange: {EXCHANGE_ID}")
        return None

    api_key = (os.getenv("BINANCE_API_KEY") or os.getenv("API_KEY") or "").strip()
    secret  = (os.getenv("BINANCE_SECRET")  or os.getenv("API_SECRET") or "").strip()

    if not api_key or not secret:
        log("[FUT] No API keys. Futures disabled.")
        return None

    def _is_ascii(s: str) -> bool:
        try:
            s.encode('ascii')
            return True
        except Exception:
            return False

    # 🚫 한글/이모지 등 비ASCII가 단 1자라도 있으면 헤더 인코딩에서 'latin-1' 에러 발생
    if (not _is_ascii(api_key)) or (not _is_ascii(secret)):
        log("❌ [FUT] API 키에 비ASCII 문자가 포함되어 선물 엔진을 비활성화합니다. "
            "key.env의 BINANCE_API_KEY/BINANCE_SECRET에 실제 영문/숫자 키만 넣어주세요.")
        return None

    ex = cls({
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    try:
        if SANDBOX:
            ex.set_sandbox_mode(True)
        ex.load_markets()
    except Exception as e:
        log(f"[FUT] init error: {e}")
    return ex

# [ANCHOR: FUT_PUB_EX_BEGIN]  << ADD NEW >>
PUB_FUT_EXCHANGE = None

def _mk_pub_fut_ex():
    """
    Build a public futures exchange instance (no API keys) so that
    paper mode can still read futures mark/index prices.
    """
    try:
        ex_id = EXCHANGE_ID
        # Map common spot id -> its futures counterpart (Binance)
        if ex_id == "binance":
            ex_id = "binanceusdm"
        cls = getattr(ccxt, ex_id)
    except Exception as e:
        log(f"[FUT_PUB] Unsupported exchange id for public futures: {EXCHANGE_ID} ({e})")
        return None
    try:
        ex = cls({
            'enableRateLimit': True,
            'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
        })
        # Optional: honor SANDBOX if explicitly desired
        try:
            if SANDBOX:
                ex.set_sandbox_mode(True)
        except Exception:
            pass
        try:
            ex.load_markets()
        except Exception as e:
            log(f"[FUT_PUB] load_markets warn: {e}")
        return ex
    except Exception as e:
        log(f"[FUT_PUB] init failed: {e}")
        return None
# [ANCHOR: FUT_PUB_EX_END]


async def _post(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)

async def _ensure_symbol_settings(ex, symbol, tf=None):
    """
    TF 규칙에 따라 마진/레버리지를 적용.
    - 마진 전환 실패(포지션/오더 존재 등)는 큐에 등록하여 청산 직후 재시도.
    """
    try:
        m = ex.market(symbol)
        sym_id = m.get('id') or symbol.replace('/', '')

        # ---- margin type (overrides with queue on fail) ----
        try:
            wanted_margin, src = _req_margin_mode(symbol, tf)
            if _MARGIN_DEBUG:
                log(f"[CONF] margin request {symbol} {tf}: {wanted_margin} ({src})")

            if hasattr(ex, 'fapiPrivate_post_margintype'):
                await _post(ex.fapiPrivate_post_margintype, {'symbol': sym_id, 'marginType': wanted_margin})
            elif hasattr(ex, 'set_margin_mode'):
                await _post(ex.set_margin_mode, wanted_margin, symbol)
        except Exception as e:
            emsg = str(e).lower()
            if ('margin' in emsg) and ('open' in emsg or 'position' in emsg or 'order' in emsg or 'cannot' in emsg):
                # 포지션/오더로 전환 불가 → 큐에 등록하여 청산 후 자동 재시도
                _enqueue_margin_switch(symbol, wanted_margin, why=str(e))
                if _MARGIN_DEBUG:
                    log(f"[CONF] queued margin switch {symbol} {tf} -> {wanted_margin} ({src}): {e}")
            else:
                log(f"[FUT] margin set warn {symbol}: {e}")


        # ---- leverage (clamp to exchange max) ----
        try:
            req = int(_req_leverage(symbol, tf))                     # ← 변경: TF별 심볼 오버라이드 사용
            limits = (m.get('limits') or {}).get('leverage') or {}
            mx = int(float(limits.get('max') or 125))
            eff = int(_clamp(req, 1, mx))
            if hasattr(ex, 'fapiPrivate_post_leverage'):
                await _post(ex.fapiPrivate_post_leverage, {'symbol': sym_id, 'leverage': eff})
            elif hasattr(ex, 'set_leverage'):
                await _post(ex.set_leverage, eff, symbol)
        except Exception as e:
            log(f"[FUT] leverage set warn {symbol}: {e}")

    except Exception as e:
        log(f"[FUT] symbol setting failed {symbol}: {e}")




async def _ensure_account_settings(ex):
    # 듀얼(헤지) 모드 on/off
    try:
        if hasattr(ex, 'fapiPrivate_post_positionside_dual'):
            await _post(
                ex.fapiPrivate_post_positionside_dual,
                {'dualSidePosition': 'true' if HEDGE_MODE else 'false'}
            )
        elif hasattr(ex, 'set_position_mode'):
            await _post(ex.set_position_mode, HEDGE_MODE)
    except Exception as e:
        log(f'[FUT] position mode warn: {e}')


async def _fetch_pos_qty(ex, symbol):
    # 순포지션 수량(+ long, - short)
    try:
        poss = await _post(ex.fetch_positions, [symbol])
        for p in poss or []:
            if (p.get('symbol') or p.get('info',{}).get('symbol')) == symbol:
                amt = float(p.get('contracts') or p.get('positionAmt') or 0)
                side = 'LONG' if amt > 0 else ('SHORT' if amt < 0 else None)
                return amt, side, float(p.get('entryPrice') or p.get('info',{}).get('entryPrice') or 0)
    except Exception:
        pass
    return 0.0, None, 0.0

async def _market(ex, symbol, side, amount, reduceOnly=False):
    params = {'reduceOnly': True} if reduceOnly else {}
    # client order id (모든 시장가 주문에 부여)
    try:
        import uuid, time
        if getattr(ex, 'id', '') in ('binanceusdm', 'binance'):
            prefix = 'cls' if reduceOnly else 'mkt'
            params['newClientOrderId'] = f"sb-{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
    except Exception:
        pass
    return await _post(ex.create_order, symbol, 'market', side, amount, None, params)

async def _stop_market(ex, symbol, side, stop_price, closePosition=True, positionSide=None, params=None):
    params = dict(params or {})
    params.update({'stopPrice': float(stop_price), 'closePosition': bool(closePosition)})
    if str(cfg_get("STOP_TRIGGER_PRICE","mark")).lower() in ("mark","mark_price","markprice"):
        params["workingType"] = "MARK_PRICE"
    if HEDGE_MODE:
        params["positionSide"] = ("LONG" if side.upper()=="LONG" else "SHORT")
    params["reduceOnly"] = True
    try:
        import uuid, time
        if getattr(ex, 'id', '') in ('binanceusdm', 'binance'):
            params['newClientOrderId'] = f"sb-sl-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
    except Exception:
        pass
    return await _post(ex.create_order, symbol, 'STOP_MARKET', side, None, None, params)

async def _tp_market(ex, symbol, side, stop_price, closePosition=True, positionSide=None, params=None):
    params = dict(params or {})
    params.update({'stopPrice': float(stop_price), 'closePosition': bool(closePosition)})
    if str(cfg_get("STOP_TRIGGER_PRICE","mark")).lower() in ("mark","mark_price","markprice"):
        params["workingType"] = "MARK_PRICE"
    if HEDGE_MODE:
        params["positionSide"] = ("LONG" if side.upper()=="LONG" else "SHORT")
    params["reduceOnly"] = True
    try:
        import uuid, time
        if getattr(ex, 'id', '') in ('binanceusdm', 'binance'):
            params['newClientOrderId'] = f"sb-tp-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
    except Exception:
        pass
    return await _post(ex.create_order, symbol, 'TAKE_PROFIT_MARKET', side, None, None, params)

# [ANCHOR: FUT_SCALE_HELPERS_BEGIN]
async def _fut_get_open_qty_side(symbol:str):
    """Return (qty, side_str) from current futures position map."""
    pos = FUT_POS.get(symbol) or {}
    qty = float(pos.get("qty",0))
    side = "LONG" if qty>0 else ("SHORT" if qty<0 else "")
    return abs(qty), side

async def _fut_scale_in(symbol:str, price:float, notional_usdt:float, side:str):
    """Add to winner via market order; returns executed qty (approx)."""
    if notional_usdt <= 0: return 0.0
    lev = int(_req_leverage(symbol, tf=None))  # or per-TF leverage if available
    qty = (notional_usdt) / max(price, 1e-9)
    ex = FUT_EXCHANGE
    if not ex: return 0.0
    side_ccxt = "buy" if side=="LONG" else "sell"
    try:
        await ex.create_order(symbol, type="market", side=side_ccxt, amount=qty, params={"reduceOnly": False})
        return qty
    except Exception as e:
        logging.error(f"[FUT_SCALE_IN_ERR] {symbol} {side} notional={notional_usdt}: {e}")
        return 0.0

async def _fut_reduce(symbol:str, reduce_qty:float, side:str):
    """Partial reduce via reduceOnly market order; returns closed qty (approx)."""
    if reduce_qty <= 0: return 0.0
    ex = FUT_EXCHANGE
    if not ex: return 0.0
    side_ccxt = "sell" if side=="LONG" else "buy"
    try:
        await ex.create_order(symbol, type="market", side=side_ccxt, amount=reduce_qty, params={"reduceOnly": True})
        return reduce_qty
    except Exception as e:
        logging.error(f"[FUT_REDUCE_ERR] {symbol} {side} qty={reduce_qty}: {e}")
        return 0.0

async def _cancel_symbol_conditional_orders(symbol:str):
    ex = FUT_EXCHANGE
    if not ex: return
    try:
        await _cancel_all_orders(ex, symbol)
    except Exception as e:
        log(f"[FUT_CANCEL_WARN] {symbol}: {e}")

async def _fut_rearm_brackets(symbol:str, tf:str, last_price:float, side:str):
    """After scaling, cancel old TP/SL/Trail (if any) and re-arm with updated size/avg."""
    if not SCALE_REALLOCATE_BRACKETS: return
    # [PATCH PROTECT RE-ARM GUARD BEGIN]
    try:
        key = ("rearm_ts", symbol, tf)
        min_sec_map = str(cfg_get("PROTECT_REARM_MIN_SEC","15m:60,1h:180,4h:600,1d:1800"))
        min_sec = 60
        for p in min_sec_map.split(","):
            if ":" in p:
                k,v = p.split(":",1)
                if k.strip()==tf:
                    min_sec = int(float(v)); break
        last_ts = int(EXEC_STATE.get(key) or 0)
        now_ts  = int(time.time())
        if last_ts and (now_ts - last_ts) < min_sec:
            log(f"[PROTECT] skip re-arm tf={tf} remain={min_sec-(now_ts-last_ts)}s")
            return
        EXEC_STATE[key] = now_ts
    except Exception:
        pass
    # [PATCH PROTECT RE-ARM GUARD END]
    try:
        await _cancel_symbol_conditional_orders(symbol)
    except Exception as e:
        logging.warning(f"[FUT_REARM_CANCEL_WARN] {symbol}: {e}")
    try:
        pos = FUT_POS.get(symbol) or {}
        await _place_protect_orders(
            FUT_EXCHANGE, symbol, tf, side, float(last_price),
            tp_pct=pos.get("tp_pct"), sl_pct=pos.get("sl_pct"), tr_pct=pos.get("tr_pct")
        )
    except Exception as e:
        logging.error(f"[FUT_REARM_ERR] {symbol}: {e}")
# [ANCHOR: FUT_SCALE_HELPERS_END]

async def _ensure_tp_sl_trailing(symbol: str, tf: str, price: float, side: str):
    pos = FUT_POS.get(symbol) or {}
    tp_pct = pos.get("tp_pct")
    sl_pct = pos.get("sl_pct")
    tr_pct = pos.get("tr_pct")
    lev    = pos.get("lev")
    eff_tp_pct, eff_sl_pct, eff_tr_pct, _ = _eff_risk_pcts(tp_pct, sl_pct, tr_pct, lev)
    if str(side).upper() == "LONG":
        tp_px = price*(1+(eff_tp_pct or 0)/100) if eff_tp_pct else None
        sl_px = price*(1-(eff_sl_pct or 0)/100) if eff_sl_pct else None
        ps = 'LONG'; tp_side = sl_side = 'sell'
    else:
        tp_px = price*(1-(eff_tp_pct or 0)/100) if eff_tp_pct else None
        sl_px = price*(1+(eff_sl_pct or 0)/100) if eff_sl_pct else None
        ps = 'SHORT'; tp_side = sl_side = 'buy'
    tr_p = eff_tr_pct
    try:
        await _cancel_symbol_conditional_orders(symbol)
    except Exception:
        pass
    try:
        if tp_px:
            await _tp_market(FUT_EXCHANGE, symbol, tp_side, tp_px, positionSide=ps)
        if sl_px:
            await _stop_market(FUT_EXCHANGE, symbol, sl_side, sl_px, positionSide=ps)
    except Exception as e:
        logging.warning(f"[_ensure_tp_sl_trailing_warn] {symbol} {tf}: {e}")
    return {"tp_price": tp_px, "sl_price": sl_px, "tr_pct": tr_p}

def _log_trade_csv(symbol, tf, action, side, qty, price, extra=None):
    path = "logs/futures_trades.csv"
    os.makedirs("logs", exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join([
            now, symbol, tf, action, side or "", f"{qty:.6f}", f"{float(price):.4f}", (extra or "")
        ]) + "\n")

def _log_scale_csv(symbol, tf, action, qty, price, extra=""):
    # action in {"SCALE_IN","SCALE_OUT"}
    try:
        _log_trade_csv(symbol, tf, action, action, qty, price, extra=extra)
    except Exception as e:
        logging.warning(f"[CSV_SCALE_WARN] {symbol} {tf} {action}: {e}")

async def _estimate_funding_fee(ex, symbol, notional, opened_ms: int|None, closed_ms: int|None) -> float:
    """
    funding_fee ≈ notional * sum(funding_rate_i)
    - 우선순위: fetchFundingRateHistory → fetchFundingRate → 추정불가(0)
    - 오류/미지원 시 0.0
    """
    try:
        if not (ESTIMATE_FUNDING_IN_PNL and ex and symbol and notional > 0 and opened_ms and closed_ms and opened_ms < closed_ms):
            return 0.0

        # 펀딩 타임스탬프 경계(통상 8h). 경계를 하나도 안 지났다면 0 처리.
        eight_h = 8*60*60*1000
        # opened 이후 첫 8시간 경계
        first_cut = ((opened_ms // eight_h) + 1) * eight_h
        if first_cut > closed_ms:
            return 0.0

        rates_sum = 0.0

        # 1) 이력 지원
        fn_hist = getattr(ex, "fetchFundingRateHistory", None) or getattr(ex, "fetch_funding_rate_history", None)
        if callable(fn_hist):
            # 적당히 넉넉한 범위로 요청
            since = opened_ms - eight_h
            limit = 200
            rows = await _post(fn_hist, symbol, since, limit)
            for r in rows or []:
                ts = int(r.get("timestamp") or r.get("fundingTime") or (r.get("info") or {}).get("fundingTime") or 0)
                if not ts or ts < opened_ms or ts > closed_ms:
                    continue
                rate = r.get("fundingRate")
                if rate is None:
                    rate = (r.get("info") or {}).get("fundingRate")
                try:
                    rates_sum += float(rate)
                except Exception:
                    pass
            return float(notional) * float(rates_sum)

        # 2) 현재 레이트만 제공 → 경계 통과 횟수로 보정
        fn_cur = getattr(ex, "fetchFundingRate", None) or getattr(ex, "fetch_funding_rate", None)
        if callable(fn_cur):
            cur = await _post(fn_cur, symbol)
            rate = cur.get("fundingRate") if isinstance(cur, dict) else None
            rate = float(rate) if rate is not None else 0.0
            crossings = max(1, (closed_ms - first_cut) // eight_h + 1)
            return float(notional) * float(rate) * float(crossings)

        return 0.0
    except Exception:
        return 0.0


def _pnl_close(ex, symbol, side, qty, entry_price, exit_price,
               entry_order_type="MARKET", exit_order_type="MARKET",
               funding_fee_usdt: float = 0.0) -> float:
    """
    USDT-M 기준:
      gross = (exit-entry)*qty (LONG) / (entry-exit)*qty (SHORT)
      net   = gross - entry_fee - exit_fee - funding_fee
    """
    qty = float(qty); ep = float(entry_price); xp = float(exit_price)
    if side == 'LONG':
        gross = (xp - ep) * qty
    elif side == 'SHORT':
        gross = (ep - xp) * qty
    else:
        gross = 0.0

    if not INCLUDE_FEES_IN_PNL:
        return gross

    fee_entry_bps = _fee_bps(entry_order_type, ex=ex, symbol=symbol)
    fee_exit_bps  = _fee_bps(exit_order_type,  ex=ex, symbol=symbol)
    fee_entry = _fee_usdt(ep, qty, fee_entry_bps)  # 보통 진입은 테이커(시장가)
    fee_exit  = _fee_usdt(xp, qty, fee_exit_bps)   # TP/SL 시장가 테이커
    funding   = float(funding_fee_usdt or 0.0)
    return gross - fee_entry - fee_exit - funding


async def _log_pnl(ex, symbol, tf, close_reason, side, qty, entry_price, exit_price,
                   opened_ms: int|None = None, closed_ms: int|None = None) -> float:
    """
    futures_pnl.csv 9번째 칸은 'net PnL(USDT)'로 기록 (리포트 합산 대상)
    뒤에 확장 정보 컬럼(총손익, 수수료합, 진입/청산 수수료, 펀딩비)을 추가 기록
    """
    qty = float(qty); ep = float(entry_price); xp = float(exit_price)

    # 펀딩 추정 (노치오날 = 평균가격 * 수량 으로 근사)
    notional = ((ep + xp) / 2.0) * qty if (ep > 0 and xp > 0) else 0.0
    funding_fee = 0.0
    try:
        funding_fee = await _estimate_funding_fee(ex, symbol, notional, opened_ms, closed_ms)
    except Exception:
        funding_fee = 0.0

    net = _pnl_close(ex, symbol, side, qty, ep, xp,
                     entry_order_type="MARKET", exit_order_type="MARKET",
                     funding_fee_usdt=funding_fee)

    # 투명성: 부가 정보도 덧붙여 둠
    gross = (xp - ep) * qty if side == 'LONG' else (ep - xp) * qty
    fee_e = _fee_usdt(ep, qty, _fee_bps("MARKET", ex=ex, symbol=symbol))
    fee_x = _fee_usdt(xp, qty, _fee_bps("MARKET", ex=ex, symbol=symbol))
    fee_t = fee_e + fee_x

    path = "logs/futures_pnl.csv"
    os.makedirs("logs", exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join([
            now, symbol, tf, str(close_reason or ""), str(side or ""), f"{qty:.6f}",
            f"{ep:.4f}", f"{xp:.4f}", f"{net:.4f}",     # 9번째: net PnL
            f"{gross:.4f}", f"{fee_t:.4f}", f"{fee_e:.4f}", f"{fee_x:.4f}", f"{float(funding_fee):.4f}"
        ]) + "\n")
    return net



async def _place_protect_orders(ex, symbol, tf, side, entry_price, tp_pct=None, sl_pct=None, tr_pct=None):
    """
    TP/SL/Trailing — 심볼×TF 오버라이드 반영.
    듀얼(헤지) 모드면 positionSide 명시.
    """
    # 이미 상단에서 환경변수들을 파싱해서 만든 소문자 dict들을 사용
    tp_map = (take_profit_pct or {})
    sl_map = (HARD_STOP_PCT or {})
    tr_map = (trailing_stop_pct or {})  # <-- 대문자 금지

    if tp_pct is None:
        tp_pct = _req_tp_pct(symbol, tf, tp_map)
    if sl_pct is None:
        sl_pct = _req_sl_pct(symbol, tf, sl_map)
    if tr_pct is None:
        tr_pct = _req_trail_pct(symbol, tf, tr_map)

    tp_price = sl_price = None
    tp_order = sl_order = None

    if side == 'LONG':
        if tp_pct > 0: tp_price = entry_price * (1 + tp_pct/100.0)
        if sl_pct > 0: sl_price = entry_price * (1 - sl_pct/100.0)
        if tp_price: tp_order = await _tp_market(ex, symbol, 'sell', tp_price, positionSide='LONG')
        if sl_price: sl_order = await _stop_market(ex, symbol, 'sell', sl_price, positionSide='LONG')
    else:  # SHORT
        if tp_pct > 0: tp_price = entry_price * (1 - tp_pct/100.0)
        if sl_pct > 0: sl_price = entry_price * (1 + sl_pct/100.0)
        if tp_price: tp_order = await _tp_market(ex, symbol, 'buy',  tp_price, positionSide='SHORT')
        if sl_price: sl_order = await _stop_market(ex, symbol, 'buy', sl_price, positionSide='SHORT')

    # trailing은 거래소별 옵션 차이가 커서 여기선 pct만 리턴하거나, 별도 함수에서 처리 권장
    return {"tp": tp_order, "sl": sl_order, "tp_price": tp_price, "sl_price": sl_price}

    # (선택) 트레일링 스탑은 별도 구현 위치가 있으면 그쪽에도 tr_pct를 반영
    # ex) trailing worker 또는 진입 루틴의 추적 최고/최저값 업데이트 로직에서 폭(tr_pct)을 사용

    FUT_ORDERS[(symbol, tf)] = {
        'tp': (tp_order or {}).get('id') if isinstance(tp_order, dict) else None,
        'sl': (sl_order or {}).get('id') if isinstance(sl_order, dict) else None,
        'tp_price': float(tp_price) if tp_price else None,
        'sl_price': float(sl_price) if sl_price else None,
        'tp_pct': float(tp_pct), 'sl_pct': float(sl_pct), 'tr_pct': float(tr_pct),
    }


# === trade entry notify helpers ===
def _get_trade_channel_id(symbol: str, tf: str) -> int|None:
    sym = symbol.split('/')[0].upper()  # ETH, BTC
    key = f"TRADE_CH_{sym}_{tf.upper().replace('M','M').replace('H','H').replace('D','D')}"
    val = os.getenv(key) or os.getenv("TRADE_CHANNEL_ID")
    try:
        cid = int(str(val).strip())
        return cid if cid > 0 else None
    except Exception:
        return None

# === formatting helpers ===
def _fmt_usd(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)

def _fmt_qty(q):
    try:
        # 과도하게 긴 소수 방지 (유효자리 6)
        return f"{float(q):.6g}"
    except Exception:
        return str(q)

def _fmt_pct(frac):
    try:
        return f"{float(frac)*100:.2f}%"
    except Exception:
        return "-"

# (구) _fmt_aloc_line는 더이상 사용하지 않음 → 알림에서 바로 포맷팅

# [ANCHOR: NOTIFY_ENTRY_BEGIN]
def _format_entry_message(symbol:str, tf:str, side:str, mode:str, price:float, lev:float,
                          qty:float, notional:float, tp:float, sl:float, trail_pct:float,
                          strength_label:str, strength_mult:float, equity_now:float,
                          strength_score:float=None, eff_margin_used:float=None,
                          tp_pct:float=None, sl_pct:float=None):
    """
    Entry alert formatter (KO/EN). Shows REAL capital used (margin), optional notional,
    TP/SL with +/-%, and strength score if given.
    """
    ko = (os.getenv("DASH_LOCALE","ko") == "ko")
    show_notional = (os.getenv("ALERT_SHOW_NOTIONAL","0") == "1")
    fees = _estimate_fees_usdt(symbol, qty, price, price)
    # real capital used
    margin_used = float(eff_margin_used) if eff_margin_used is not None else (abs(qty)*float(price) / max(1.0, float(lev or 1.0)))

    # helper: percent labels
    def _pct_label(x, sign="+"):
        try:
            return f"{float(x):.2f}%"
        except Exception:
            return "n/a"

    if ko:
        head_emoji = "🟢" if side.upper().startswith("LONG") or "BUY" in side.upper() else "🔴"
        side_kr = "LONG" if side.upper().startswith("LONG") or "BUY" in side.upper() else "SHORT"
        score_seg = (f" / 점수 {float(strength_score):.2f}" if strength_score is not None else "")
        line1 = f"{head_emoji} 진입 | {symbol} · {tf} · {side_kr} ×{lev:g}"
        line2 = f"• 가격/수량: ${price:,.2f} / {qty:.4f}"
        if show_notional:
            line2 += f"  (노치오날 ${notional:,.2f})"
        line3 = f"• 실자본 사용: ${margin_used:,.2f} / 현재자본 ${equity_now:,.2f}"
        line4 = f"• 강도: {strength_label} (×{strength_mult:.2f}){score_seg}"
        tp_txt = f"+{_pct_label(tp_pct)}" if tp_pct is not None else ""
        sl_txt = f"-{_pct_label(sl_pct)}" if sl_pct is not None else ""
        line5 = f"• 리스크: TP ${tp:,.2f} {tp_txt} / SL ${sl:,.2f} {sl_txt} / 트레일 {float(trail_pct):.2f}%"
        try:
            st  = EXEC_STATE.get(("style", symbol, tf)) or "-"
            rg  = EXEC_STATE.get(("regime", symbol, tf)) or "-"
            sm  = EXEC_STATE.get(("sl_mode", symbol, tf)) or "-"
            rr  = EXEC_STATE.get(("rr", symbol, tf)) or 0
            am  = EXEC_STATE.get(("atr_mult", symbol, tf)) or 0
            wkt = ("MARK" if str(cfg_get("STOP_TRIGGER_PRICE","mark")).lower() in ("mark","mark_price","markprice") else "LAST")
            coh = EXEC_STATE.get(("coh_tags", symbol, tf)) or ""
            line_style = f"• Style: {st} / Regime: {rg} / SLmode: {sm} / RR {rr} / ATR×{am} / trigger {wkt} {('· '+coh) if coh else ''}"
        except Exception:
            line_style = None
        line6 = f"• 수수료(진입/추정청산): -${fees['entry_fee']:.2f} / -${fees['exit_fee_est']:.2f}"
        lines = [line1, line2, line3, line4, line5]
        if line_style:
            lines.append(line_style)
        lines.append(line6)
        return "\n".join(lines)
    else:
        score_seg = (f" / score {float(strength_score):.2f}" if strength_score is not None else "")
        line1 = f"🟢 ENTRY | {symbol} · {tf} · {side} ×{lev:g}"
        line2 = f"• Price/Qty: ${price:,.2f} / {qty:.4f}"
        if show_notional:
            line2 += f"  (Notional ${notional:,.2f})"
        line3 = f"• Real capital used: ${margin_used:,.2f} / Equity ${equity_now:,.2f}"
        line4 = f"• Strength: {strength_label} (×{strength_mult:.2f}){score_seg}"
        tp_txt = f"+{_pct_label(tp_pct)}" if tp_pct is not None else ""
        sl_txt = f"-{_pct_label(sl_pct)}" if sl_pct is not None else ""
        line5 = f"• Risk: TP ${tp:,.2f} {tp_txt} / SL ${sl:,.2f} {sl_txt} / Trail {float(trail_pct):.2f}%"
        try:
            st  = EXEC_STATE.get(("style", symbol, tf)) or "-"
            rg  = EXEC_STATE.get(("regime", symbol, tf)) or "-"
            sm  = EXEC_STATE.get(("sl_mode", symbol, tf)) or "-"
            rr  = EXEC_STATE.get(("rr", symbol, tf)) or 0
            am  = EXEC_STATE.get(("atr_mult", symbol, tf)) or 0
            wkt = ("MARK" if str(cfg_get("STOP_TRIGGER_PRICE","mark")).lower() in ("mark","mark_price","markprice") else "LAST")
            coh = EXEC_STATE.get(("coh_tags", symbol, tf)) or ""
            line_style = f"• Style: {st} / Regime: {rg} / SLmode: {sm} / RR {rr} / ATR×{am} / trigger {wkt} {('· '+coh) if coh else ''}"
        except Exception:
            line_style = None
        line6 = f"• Fees (entry/est. close): -${fees['entry_fee']:.2f} / -${fees['exit_fee_est']:.2f}"
        lines = [line1, line2, line3, line4, line5]
        if line_style:
            lines.append(line_style)
        lines.append(line6)
        return "\n".join(lines)


async def _notify_trade_entry(symbol: str, tf: str, signal: str, *,
                              mode: str,              # 'futures' or 'spot' or 'paper'
                              price: float, qty: float|None,
                              base_margin: float|None=None, eff_margin: float|None=None,
                              lev_used: int|None=None,
                              score: float|None=None,
                              pb_label: str|None=None, pb_w: float=0.0, pb_alloc_mul: float=1.0):
    """
    진입 알림: 모드/가격/레버리지/강도/상위TF/배분(총자본→TF배분→강도×MTF→최종)/수량·노치오날
    """
    try:
        cid = _get_trade_channel_id(symbol, tf)
        if not cid:
            log(f"[SKIP] trade notify {symbol} {tf}: no channel id")
            return
        ch = client.get_channel(cid)
        if not ch:
            log(f"[SKIP] trade notify {symbol} {tf}: channel not found {cid}")
            return

        if mode == "paper" and PAPER_STRICT_NONZERO and ((not base_margin) or (not eff_margin) or (not qty)):
            logging.warning("[PAPER_WARN] zero allocation on paper entry: check PART A")

        # 강도/MTF 요약
        sf = mf = None
        all_align = False
        strength_label = None
        try:
            strength_label = _strength_label(signal, score)
            sf = _strength_factor(signal, score)                     # 예: 0.55
            mf, all_align = _mtf_factor(symbol, tf, signal)          # 예: 1.25
            align_text, _, _ = _mtf_alignment_text(symbol, tf, signal)
        except Exception:
            align_text = "-"

        # [ANCHOR: ENTRY_ALLOC_CALC]  << REPLACE BLOCK >>

        use_frac  = None
        notional  = None
        if eff_margin and base_margin:
            use_frac = float(eff_margin) / float(base_margin)
        if eff_margin and lev_used:
            try:
                notional = float(eff_margin) * int(lev_used)
            except Exception:
                notional = None

        tp_price = sl_price = None
        tr_pct_eff = 0.0
        try:
            tpv = _req_tp_pct(symbol, tf, (take_profit_pct or {}))
            slv = _req_sl_pct(symbol, tf, (HARD_STOP_PCT or {}))
            trv = _req_trail_pct(symbol, tf, (trailing_stop_pct or {}))
            eff_tp_pct, eff_sl_pct, eff_tr_pct, _src = _eff_risk_pcts(tpv, slv, trv, lev_used)
            if signal == "BUY":
                tp_price = price * (1 + (eff_tp_pct or 0) / 100) if eff_tp_pct else None
                sl_price = price * (1 - (eff_sl_pct or 0) / 100) if eff_sl_pct else None
            else:
                tp_price = price * (1 - (eff_tp_pct or 0) / 100) if eff_tp_pct else None
                sl_price = price * (1 + (eff_sl_pct or 0) / 100) if eff_sl_pct else None
            tr_pct_eff = eff_tr_pct if eff_tr_pct is not None else 0.0
        except Exception:
            pass

        eff_margin_used = float(eff_margin) if eff_margin is not None else None
        msg = _format_entry_message(
            symbol,
            tf,
            signal,
            mode,
            float(price),
            float(lev_used or 1),
            float(qty or 0.0),
            float(notional or 0.0),
            float(tp_price or 0.0),
            float(sl_price or 0.0),
            float(tr_pct_eff or 0.0),
            strength_label or "",
            float(sf or 0.0),
            capital_get(),
            strength_score=score,
            eff_margin_used=eff_margin_used,
            tp_pct=eff_tp_pct if 'eff_tp_pct' in locals() else None,
            sl_pct=eff_sl_pct if 'eff_sl_pct' in locals() else None,
        )
        await ch.send(msg)
    except Exception as e:
        log(f"[NOTIFY] trade entry warn {symbol} {tf}: {e}")

# === trade exit notify helper ===
# [ANCHOR: EXIT_NOTIFY_HELPER]
async def _notify_trade_exit(symbol: str, tf: str, *,
                             side: str,          # 'LONG'|'SHORT'|'SPOT'
                             entry_price: float,
                             exit_price: float,
                             reason: str,
                             mode: str,          # 'futures'|'spot'|'paper'
                             pnl_pct: float | None = None,
                             qty: float | None = None,

                             status: str | None = None,
                             pnl_usdt: float | None = None):

    try:
        cid = _get_trade_channel_id(symbol, tf)
        if not cid:
            return
        ch = client.get_channel(cid)
        if not ch:
            return

        pnl_pct_val = pnl_pct
        if pnl_pct_val is None and isinstance(entry_price, (int, float)) and entry_price > 0:
            try:
                mult = 1.0
                if side.upper() == 'LONG':
                    mult = 1.0
                elif side.upper() == 'SHORT':
                    mult = -1.0
                else:
                    mult = 1.0 if float(exit_price) >= float(entry_price) else -1.0
                pnl_pct_val = ((float(exit_price)/float(entry_price))-1.0) * 100.0 * mult
            except Exception:
                pnl_pct_val = None
        realized_pnl_usdt = float(pnl_usdt or 0.0)
        realized_pnl_pct = float(pnl_pct_val if pnl_pct_val is not None else 0.0)
        st = _daily_register_close(pnl_usdt=realized_pnl_usdt, capital_after=capital_get())
        try:
            pathlib.Path(os.path.dirname(TRADE_CSV_PATH)).mkdir(parents=True, exist_ok=True)
            new_file = not os.path.exists(TRADE_CSV_PATH)
            with open(TRADE_CSV_PATH,"a",encoding="utf-8",newline="") as f:
                import csv
                w = csv.writer(f)
                if new_file:
                    w.writerow(["ymd","ts","symbol","tf","side","reason",
                                "entry","exit","pnl_usdt","pnl_pct",
                                "fee_est","slip_bp",
                                "upnl_before","upnl_after",
                                "equity_before","equity_after",
                                "mae_pct","mfe_pct"])
                w.writerow([
                    _daily_key_kst(), _kst_now().isoformat(),
                    symbol, tf, side, reason,
                    round(entry_price,6), round(exit_price,6),
                    round(realized_pnl_usdt,4), round(realized_pnl_pct,4),
                    round(fee_est_usdt,4) if 'fee_est_usdt' in locals() else 0.0,
                    round(slip_bps,2) if 'slip_bps' in locals() else 0.0,
                    round(upnl_before,4) if 'upnl_before' in locals() else 0.0,
                    round(upnl_after,4)  if 'upnl_after'  in locals() else 0.0,
                    round(equity_before,4) if 'equity_before' in locals() else capital_get(),
                    round(equity_after,4)  if 'equity_after'  in locals() else capital_get(),
                    round(mae_pct,4) if 'mae_pct' in locals() else 0.0,
                    round(mfe_pct,4) if 'mfe_pct' in locals() else 0.0
                ])
        except Exception as e:
            log(f"[CSV] trade log warn: {e}")
        is_gain = (pnl_pct_val is not None and pnl_pct_val >= 0)
        emoji = "🟢" if is_gain else "🔴"
        label = "익절" if is_gain else "손절"
        title = f"{emoji} {label} ({side}) 〔{symbol} · {tf}〕"
        lines = [
            f"• 모드: {('🧪 페이퍼' if mode=='paper' else ('선물' if mode=='futures' else '현물'))}",
            f"• 진입가/청산가: ${entry_price:,.2f} → ${exit_price:,.2f}",
            f"• 사유: {reason}",
        ]
        if pnl_pct_val is not None:
            lines.append(f"• 손익률: {pnl_pct_val:.2f}%")
        if status:
            lines.append(f"• 상태: {status}")

        # [ANCHOR: PAPER_CLOSE_AND_NOTIFY]

        if ALERT_SHOW_CAPITAL:
            after_cap = capital_get()
            tail = (f" [Planner: {PLANNER_ID}]" if PLANNER_ID else "")
            if pnl_usdt is not None:
                before_cap = after_cap - float(pnl_usdt)
                delta_pct = (float(pnl_usdt) / before_cap * 100.0) if before_cap > 0 else 0.0
                lines.append(f"• 총자본(종결후): {_fmt_usd(after_cap)} | 변화: {_fmt_usd(pnl_usdt)} ({delta_pct:+.2f}%){tail}")
            else:
                lines.append(f"• 총자본(종결후): {_fmt_usd(after_cap)}{tail}")


        # [ANCHOR: EXIT_NOTIFY_TAIL]
        if ALERT_SHOW_CAPITAL and PLANNER_ID and all("Planner:" not in s for s in lines):
            lines.append(f"• Planner: {PLANNER_ID}")

        await ch.send("\n".join([title] + lines))
    except Exception as e:
        log(f"[NOTIFY] trade exit warn {symbol} {tf}: {e}")

    # [ANCHOR: SET_COOLDOWN_ON_EXIT]
    try:
        if ENABLE_COOLDOWN:
            import time
            LAST_EXIT_TS[tf] = time.time()
            COOLDOWN_UNTIL[tf] = LAST_EXIT_TS[tf] + float(POST_EXIT_COOLDOWN_SEC.get(tf, 0.0))
            log(f"⏳ cooldown set: {tf} until {COOLDOWN_UNTIL.get(tf, 0):.0f}")
    except Exception:
        pass

    # [ANCHOR: POSITION_CLOSE_HOOK]
    if AFTER_CLOSE_PAUSE:
        PAUSE_UNTIL[(symbol, tf)] = 2**62
        log(f"⏸ post-close paused {symbol} {tf}")


async def futures_close_all(symbol, tf, exit_price=None, reason="CLOSE") -> bool:
    ex = FUT_EXCHANGE
    if not (AUTO_TRADE and TRADE_MODE == "futures" and ex):
        return False
    ok = False
    pos = FUT_POS.get(symbol) or {}
    try:
        qty, side, entry = await _fetch_pos_qty(ex, symbol)
        if not side or abs(qty) <= 0:
            return False
        close_side = 'sell' if side == 'LONG' else 'buy'
        await _market(ex, symbol, close_side, abs(qty), reduceOnly=True)
        import time  # 이미 상단에 있으면 생략
        opened_ms = None
        try:
            opened_ms = int((FUT_POS.get(symbol) or {}).get("opened_ts") or 0)
        except Exception:
            opened_ms = None
        closed_ms = int(time.time()*1000)

        await _log_pnl(
            ex, symbol, tf, reason, side, abs(qty),
            float(entry), float(exit_price or entry),
            opened_ms=opened_ms, closed_ms=closed_ms
        )
        # CSV: futures CLOSE
        try:
            if FUTURES_CSV_CLOSE_LOG:

                lev = float(pos.get("lev") or 1.0) if isinstance(pos, dict) else 1.0
                extra = ",".join(["mode=futures", f"lev={lev:.2f}", f"reason={reason}"])

                _log_trade_csv(symbol, tf, "CLOSE", side, abs(qty), float(exit_price or entry), extra=extra)
        except Exception as e:
            log(f"[CSV_CLOSE_WARN] futures {symbol} {tf}: {e}")

        # Ledger: futures CLOSE (paper와 동일 스키마)
        try:
            ent = float(entry or 0.0)
            exi = float(exit_price or ent)
            qty_abs = float(abs(qty))
            # gross_usdt: 방향성 반영 (LONG: exi-ent, SHORT: ent-exi) × 수량
            gross_usdt = (exi - ent) * qty_abs if side.upper() == "LONG" else (ent - exi) * qty_abs

            # fees_usdt: maker/taker 동적 소스 포함(이미 _fee_bps/INCLUDE_FEES_IN_PNL 경로 존재 가정)
            if INCLUDE_FEES_IN_PNL:
                try:
                    entry_bps = _fee_bps("MARKET", ex=ex, symbol=symbol)
                    exit_bps  = _fee_bps("MARKET",  ex=ex, symbol=symbol)
                    fees_usdt = qty_abs * ent * (entry_bps/10000.0) + qty_abs * exi * (exit_bps/10000.0)
                except Exception:
                    fees_usdt = max(0.0, float(locals().get("fee_est_usdt", 0.0)))  # 기존 추정치가 있으면 사용
            else:
                fees_usdt = 0.0

            net_usdt  = gross_usdt - fees_usdt
            after_cap = capital_get() + float(net_usdt)  # futures 모드에선 상태 반영은 안 하고 원장에만 'after' 필드로 기록

            await capital_ledger_write(
                "CLOSE",
                symbol=symbol, tf=tf, side=str(side or "").upper(), reason=str(reason or "CLOSE"),
                entry_price=f"{ent:.8f}", exit_price=f"{exi:.8f}",
                qty=f"{qty_abs:.8f}", gross_usdt=f"{gross_usdt:.8f}",
                fees_usdt=f"{fees_usdt:.8f}", net_usdt=f"{net_usdt:.8f}",
                capital_after=f"{after_cap:.8f}",
            )
        except Exception as le:
            log(f"[CAPITAL] futures ledger warn {symbol} {tf}: {le}")

        # --- B-2: 선물 청산 알림(공통 헬퍼 호출) ---
        try:
            ent = float(entry or 0.0)
            exi = float(exit_price or ent or 0.0)
            if ent > 0 and exi > 0:
                gross_pct = ((exi - ent) / ent) * 100.0 if side == 'LONG' else ((ent - exi) / ent) * 100.0
                entry_bps = _fee_bps("MARKET", ex=ex, symbol=symbol) if INCLUDE_FEES_IN_PNL else 0.0
                exit_bps  = _fee_bps("MARKET", ex=ex, symbol=symbol) if INCLUDE_FEES_IN_PNL else 0.0
                fee_pct   = (entry_bps + exit_bps) / 100.0
                pnl_pct   = gross_pct - fee_pct

            else:
                pnl_pct = None
            await _notify_trade_exit(
                symbol, tf,
                side=side, entry_price=ent, exit_price=exi,
                reason=str(reason or "CLOSE"),
                mode="futures",
                pnl_pct=pnl_pct,
            )

        except Exception as ne:
            log(f"[NOTIFY] fut exit warn {symbol} {tf}: {ne}")

        ok = True
    except Exception as e:
        log(f"[FUT] close_all error {symbol} {tf}: {e}")
        ok = False
    finally:
        # 열린 주문 정리 + 마진 전환 재시도(있다면) + 상태 초기화
        try:
            await _cancel_all_orders(ex, symbol)
        except Exception as e:
            log(f"[FUT] cancel after close warn {symbol}: {e}")
        try:
            await _apply_margin_switch_if_possible(ex, symbol)
        except Exception as e:
            log(f"[FUT] margin apply after close warn {symbol}: {e}")
        FUT_POS.pop(symbol, None); _save_json(OPEN_POS_FILE, FUT_POS)
        if FUT_POS_TF.get(tf) == symbol:
            FUT_POS_TF.pop(tf, None); _save_json(OPEN_TF_FILE, FUT_POS_TF)
        # IDEMP: allow re-entry after manual/forced close
        try: idem_clear_symbol_tf(symbol, tf)
        except Exception: pass
    return ok

async def futures_close_symbol_tf(symbol, tf, reason="MANUAL"):
    return await futures_close_all(symbol, tf, reason=reason)

async def _auto_close_and_notify_eth(
    channel, tf, symbol_eth, action, reason,
    entry_price, curr_price, exit_price,
    rsi, macd, entry_time, score
):
    """
    ETH 자동 종료: 선물 청산(성공/실패 라벨) → 알림 전송 → CSV/상태 정리
    """
    # reason이 비었으면 action으로 대체
    action_reason = reason or action
    key2 = _key2(symbol_eth, tf)

    if not _has_open_position(symbol_eth, tf, TRADE_MODE):
        if EXIT_DEBUG:
            logging.info(f"[EXIT_DEBUG] skip exit: no open position for {symbol_eth} {tf}")
        return

    if TRADE_MODE == "paper":
        info = await _paper_close(symbol_eth, tf, float(exit_price), action_reason)
        if info:
            try:
                await _notify_trade_exit(
                    symbol_eth, tf,
                    side=info["side"],
                    entry_price=info["entry_price"],
                    exit_price=float(exit_price),
                    reason=str(action_reason),
                    mode="paper",
                    pnl_pct=info.get("pnl_pct"),
                    qty=info.get("qty"),
                    pnl_usdt=info.get("net_usdt"),

                )
            except Exception as e:
                log(f"[NOTIFY] paper exit warn {symbol_eth} {tf}: {e}")
        return

    # 표시용 신호: 진입/청산 가격으로 추정(스팟 기준)
    display_signal = "BUY"
    if entry_price is not None and exit_price is not None:
        display_signal = "BUY" if float(exit_price) >= float(entry_price) else "SELL"

    pnl = None


    # [ANCHOR: EXIT_NOTIFY_FIX_BEGIN]

    ep = float(entry_price or 0.0)

    # 선물 청산 먼저
    executed = await futures_close_all(symbol_eth, tf, exit_price=exit_price, reason=action_reason)
    status_text = "✅ 선물 청산" if executed else "🧪 시뮬레이션/미실행"
    is_futures = executed

    # 알림 (공통 헬퍼 사용)
    try:
        # 방향(페이퍼에서도 LONG/SHORT 표기 위해 추정)
        side_guess = 'LONG'
        try:
            ds = str(display_signal).upper()
            if 'SELL' in ds:
                side_guess = 'SHORT'
        except Exception:
            pass

        await _notify_trade_exit(
            symbol_eth, tf,
            side=side_guess,
            entry_price=ep,
            exit_price=float(exit_price),
            reason=str(action_reason),
            mode=("futures" if is_futures else "paper"),
            pnl_pct=(float(pnl) if pnl is not None else None),
            status=status_text

        )
    except Exception as e:
        log(f"[NOTIFY] paper/fut exit (ETH) warn {symbol_eth} {tf}: {e}")
    # [ANCHOR: EXIT_NOTIFY_FIX_END]


    # CSV/상태 정리 (ETH는 접미사 없이 공통 변수 사용)
    exit_price = sanitize_price_for_tf(symbol_eth, tf, exit_price)
    log_to_csv(
        symbol_eth, tf, action, exit_price,
        rsi, macd, pnl,
        entry_price=ep, entry_time=entry_time,
        score=score, reasons=[action_reason]
    )

    previous_signal[key2] = action
    previous_score[tf]  = None
    entry_data[key2]      = None
    highest_price[key2]   = None
    lowest_price[key2]    = None
    trail_peak_roe[key2]  = None



async def _auto_close_and_notify_btc(
    channel, tf, symbol, action, reason,
    entry_price, curr_price, exit_price,
    rsi, macd, entry_time, score
):
    """
    BTC 자동 종료: 선물 청산 → 알림 전송 → CSV/상태 정리
    """
    action_reason = (reason or action or "").strip()
    if not _has_open_position(symbol, tf, TRADE_MODE):
        if EXIT_DEBUG:
            logging.info(f"[EXIT_DEBUG] skip exit: no open position for {symbol} {tf}")
        return
    ep = float(entry_price or 0.0)
    cp = float(curr_price or 0.0)
    xp = float(exit_price or cp or 0.0)

    if TRADE_MODE == "paper":
        info = await _paper_close(symbol, tf, xp, action_reason)
        if info:
            try:
                await _notify_trade_exit(
                    symbol, tf,
                    side=info["side"],
                    entry_price=info["entry_price"],
                    exit_price=xp,
                    reason=action_reason,
                    mode="paper",
                    pnl_pct=info.get("pnl_pct"),
                    qty=info.get("qty"),
                    pnl_usdt=info.get("net_usdt"),

                )
            except Exception as e:
                log(f"[NOTIFY] paper exit warn {symbol} {tf}: {e}")
        return

    # 선물 청산 시도
    status = "🧪 시뮬레이션/미실행"
    try:
        executed = await futures_close_all(symbol, tf, exit_price=xp, reason=action_reason)
        status = "✅ 선물 청산" if executed else "🧪 시뮬레이션/미실행"
    except Exception as e:
        log(f"[NOTIFY] paper/fut exit (BTC) warn {symbol} {tf}: {e}")

    # (표시용) 대략 PnL% 계산 — 수수료 반영 옵션
    pnl_pct = None
    try:
        if ep > 0 and xp > 0:
            # 직전 포지션 방향을 모르면 BUY 기준으로 계산해도 무방(알림용)
            long_like = True
            gross = ((xp - ep) / ep) * 100.0 if long_like else ((ep - xp) / ep) * 100.0
            if INCLUDE_FEES_IN_PNL:
                fee_bps = _fee_bps("MARKET", ex=FUT_EXCHANGE, symbol=symbol) * 2  # 진입+청산
                gross -= (fee_bps / 100.0)
            pnl_pct = gross
    except Exception:
        pnl_pct = None

    # 알림(공통 헬퍼)
    try:
        key2 = (symbol, tf)
        await _notify_trade_exit(
            symbol, tf,
            side=previous_signal.get(key2, ""),  # 있으면 사용
            entry_price=ep, exit_price=xp,
            reason=action_reason, mode="futures",
            pnl_pct=pnl_pct
        )
    except Exception as ne:
        log(f"[NOTIFY] btc exit send warn {symbol} {tf}: {ne}")

    # CSV/상태 정리
    xp = sanitize_price_for_tf(symbol, tf, xp)
    log_to_csv(
        symbol, tf, action, xp,
        rsi, macd, None,
        entry_price=ep, entry_time=entry_time,
        score=score, reasons=[action_reason]
    )
    previous_signal[key2] = action
    previous_score_btc[tf]  = None
    entry_data[key2]      = None
    highest_price.pop(key2, None)
    lowest_price.pop(key2, None)
    trail_peak_roe.pop(key2, None)




async def maybe_execute_futures_trade(symbol, tf, signal, signal_price, candle_ts):
    """ BUY→롱 오픈 / SELL→숏 오픈. 반대 신호면 청산 후 반대방향 진입.
        동일 캔들이나 중복 재시도는 idem key로 방지.
    """
    # 현재 신호가 찍힌 가격(없으면 마지막 호가)
    last = float(signal_price) if signal_price is not None else float(fetch_live_price(symbol) or 0.0)

    if not (AUTO_TRADE and TRADE_MODE == "futures"):
        return

    exec_signal = _normalize_exec_signal(signal)
    if exec_signal not in ("BUY", "SELL"):
        return
    
    # --- TF 후보 선정: 같은 TF에서 더 우수한 심볼만 허용 ---
    if not ALLOW_BOTH_PER_TF:
        # 아직 해당 TF에 열린 포지션이 없다면, 후보비교로 더 좋은 쪽만 통과
        if not FUT_POS_TF.get(tf) and not PAPER_POS_TF.get(tf):
            if not _is_best_candidate(symbol, tf, exec_signal):
                log(f"[FUT] skip {symbol} {tf} {exec_signal}: better candidate exists")
                return

    ex = FUT_EXCHANGE
    if not ex:
        return
    
    # --- 헤지(듀얼) 모드 신호 정책: LONG_ONLY/SHORT_ONLY/BOTH ---
    try:
        if not _hedge_side_allowed(symbol, tf, exec_signal):
            log(f"[FUT] skip {symbol} {tf} {exec_signal}: hedge side policy")
            return
    except Exception as e:
        log(f"[FUT] hedge policy warn {symbol} {tf}: {e}")

        # 진입 전에 한번 전체 큐 처리 (포지션 없는 심볼은 즉시 전환)
    try:
        await _apply_all_pending_margin_switches(ex)
    except Exception as e:
        log(f"[FUT] margin queue sweep warn: {e}")

    # 라우팅 가드
    if not _route_allows(symbol, tf):
        return

    # 같은 TF에서 동시(ETH/BTC) 포지션 금지(옵션)
    if not ALLOW_BOTH_PER_TF:
        other = FUT_POS_TF.get(tf)
        if other and other != symbol:
            return

    await _ensure_account_settings(ex)            # 듀얼 모드 등
    await _ensure_symbol_settings(ex, symbol, tf) # TF별 레버리지/마진

    # [ANCHOR: PREORDER_MARGIN_PRECHECK_BEGIN]  << ADD NEW >>
    try:
        if os.getenv("MARGIN_PRECHECK_ENFORCE", "1") == "1":
            ok = await _check_and_sync_symbol_settings(ex, symbol, tf)
            if not ok:
                log(f"[FUT] skip {symbol} {tf}: margin/leverage not ready")
                return
    except Exception as e:
        log(f"[FUT] precheck warn {symbol} {tf}: {e}")
    # [ANCHOR: PREORDER_MARGIN_PRECHECK_END]

    # --- 슬리피지 가드(심볼×TF 오버라이드 반영) ---
    limit_pct = _req_slippage_pct(symbol, tf)  # ex) BTC 4h=0.4, ETH 4h=0.9
    cur = float(last)
    sig = float(signal_price or last)
    if sig > 0:
        diff_pct = abs(cur - sig) / sig * 100.0
        if diff_pct > float(limit_pct):
            log(f"[FUT] skip {symbol} {tf} {exec_signal}: slippage {diff_pct:.2f}% > {limit_pct:.2f}%")
            return


    # === 강도×MTF 바이어스 기반 최종 증거금 계산 ===
    # 1) 상태기록(상위 TF 바이어스용)
    local_score = None
    try:
        # 분석 파트에서 score를 구해 넘겨주는 흐름이라면, 여기서 대입
        # 없으면 None으로 두면 버킷 'BASE' 처리
        local_score = EXEC_STATE.get(('score', symbol, tf))
    except Exception:
        pass
    _record_signal(symbol, tf, exec_signal, local_score)

    # 2) 기본 증거금(총자본 × TF배분)
    base_margin = _margin_for_tf(tf)  # capital_get() × ALLOC_TF[tf] or fallback(FUT_MGN_USDT)

    # 3) 강도 가중
    sf = _strength_factor(exec_signal, local_score)

    # 4) 상위 TF 바이어스
    mf, all_align = _mtf_factor(symbol, tf, exec_signal)

    # 5) 최종 증거금 비율
    frac = min(1.0, sf * mf)
    if all_align and _FULL_ON_ALL:
        frac = 1.0

    side = "LONG" if exec_signal == "BUY" else "SHORT"
    tp_pct = _req_tp_pct(symbol, tf, (take_profit_pct or {}))
    sl_pct = _req_sl_pct(symbol, tf, (HARD_STOP_PCT or {}))
    tr_pct = _req_trail_pct(symbol, tf, (trailing_stop_pct or {}))
    lev = _req_leverage(symbol, tf)
    # === Playbook: adjust risk % and allocation by regime context ===
    try:
        _pb, _ctx_used = _playbook_adjust_risk(symbol, tf, side, tp_pct, sl_pct, tr_pct, lev, None)
        tp_pct = _pb.get("tp", tp_pct)
        sl_pct = _pb.get("sl", sl_pct)
        tr_pct = _pb.get("tr", tr_pct)
        _pb_label = _pb.get("label", "PB_OFF")
        _pb_w     = _pb.get("eff_w", 0.0)
        _pb_alloc_mul = float(_pb.get("alloc_mul", 1.0))
        _pb_lev_cap   = float(_pb.get("lev_cap", 0.0))

        _pb_cap       = float(_pb.get("alloc_abs_cap", 0.0))
    except Exception as e:
        log(f"[PB_ERR] {symbol} {tf} {e}")
        _pb_label = "PB_ERR"; _pb_w = 0.0; _pb_alloc_mul = 1.0; _pb_lev_cap = 0.0; _pb_cap = 0.0
    log(f"[PB] {symbol} {tf} side={side} label={_pb_label} w={_pb_w:.2f} tp={tp_pct} sl={sl_pct} tr={tr_pct} alloc×{_pb_alloc_mul:.2f} lev_cap={_pb_lev_cap}")
    log(f"[PB_CAP] {symbol} {tf} alloc_cap={_pb.get('alloc_abs_cap') if '_pb' in locals() and _pb else 0} lev_cap={_pb.get('lev_cap') if '_pb' in locals() and _pb else 0}")
    log(f"[PB_SCALE] {symbol} {tf} step×{_pb.get('scale_step_mul') if '_pb' in locals() and _pb else 1} reduce×{_pb.get('scale_reduce_mul') if '_pb' in locals() and _pb else 1} legs+{_pb.get('scale_legs_add') if '_pb' in locals() and _pb else 0} upΔ{_pb.get('scale_up_shift') if '_pb' in locals() and _pb else 0} downΔ{_pb.get('scale_down_shift') if '_pb' in locals() and _pb else 0}")


    eff_margin = base_margin * frac * (_pb_alloc_mul if '_pb_alloc_mul' in locals() else 1.0)
    if eff_margin > base_margin:
        eff_margin = base_margin

    try:
        if _pb_cap > 0:
            eff_margin = min(eff_margin, _pb_cap)
    except Exception:
        pass

    # Playbook/max leverage cap (0 = no cap)
    try:
        _pb_lev_cap2 = float(_pb.get("lev_cap", 0.0)) if '_pb' in locals() and _pb else 0.0
        if _pb_lev_cap2 > 0 and float(lev or 1.0) > _pb_lev_cap2:
            lev = _pb_lev_cap2
    except Exception:
        pass


    # 디버그 로그(옵션)
    if _DEBUG_ALLOC:
        await channel.send(
            f"⚙️ 배분 내역 {symbol} {tf}\n"
            f"• 기본: ${base_margin:.2f}\n"
            f"• 강도계수: ×{sf:.2f}\n"
            f"• MTF계수: ×{mf:.2f} (all_align={all_align})\n"
            f"• 최종 증거금: ${eff_margin:.2f}"
        )
        log(f"[ALLOC-DEBUG] {symbol} {tf} {exec_signal} req_lev={req_lev} limits={limits} -> qty≈{qty:.6f}")

# [PATCH NEG/CCA GATE BEGIN — maybe_execute_futures_trade]
    try:
        coh_on = str(cfg_get("COHERENCE_MODE", "on")).lower() in ("1","on","true","yes")
        if coh_on:
            ner_min   = float(cfg_get("NER_MIN",   "0.25"))
            ner_tgt   = float(cfg_get("NER_TARGET","0.35"))
            plr_max   = float(cfg_get("PLR_MAX",   "0.80"))
            scout_pct = float(cfg_get("SCOUT_ALLOC_PCT", "0.15"))
            edge_dlt  = float(cfg_get("EDGE_SWITCH_DELTA", "0.12"))

            side_str  = "LONG" if exec_signal == "BUY" else "SHORT"
            price_ref = float(last)
            L, S      = _symbol_exposure(symbol, price_ref)

            # 후보 notional (eff_margin*lev)
            cand_notional = float(eff_margin or 0.0) * float(lev or 0.0)

            # CCA: 점수우위
            new_score = abs(float(EXEC_STATE.get(('score', symbol, tf)) or 0.0))
            opp_best  = _best_opposite_score(symbol, side_str)
            cca_weaken = (opp_best > 0.0) and ((new_score - opp_best) < edge_dlt)

            # NEG: 예측치
            ner_next, plr_next = _ner_plr(L, S, side_str, cand_notional)

            scale_factor = 1.0
            reason_tags  = []

            if plr_next > plr_max:
                scale_factor = min(scale_factor, scout_pct); reason_tags.append(f"PLR>{plr_max:.2f}")

            heavier = "LONG" if L >= S else "SHORT"
            if side_str != heavier:
                x_max = _x_max_for_ner_min(L, S, side_str, ner_min)
                if x_max <= 0:
                    scale_factor = min(scale_factor, scout_pct); reason_tags.append("NER_CAP0")
                elif cand_notional > x_max:
                    scale_factor = min(scale_factor, max(0.0, x_max / max(cand_notional, 1e-9)))
                    reason_tags.append("NER_SCALE")

            if cca_weaken:
                scale_factor = min(scale_factor, scout_pct); reason_tags.append("EDGE")


            # 상위TF 근접으로 스카웃만 허용된 경우
            try:
                mtf_dec = _mtf_struct_guard(symbol, tf, exec_signal)
                if mtf_dec.get("action") == "SCOUT":
                    scale_factor = min(scale_factor, scout_pct)
                    reason_tags.append("MTF_NEAR")
            except Exception:
                pass

            if scale_factor < 0.999:
                eff_margin = float(eff_margin) * float(scale_factor)
                EXEC_STATE[("coh_tags", symbol, tf)] = ",".join(reason_tags)

                log(f"[COHERENCE] FUT {symbol} {tf} {side_str} scale×{scale_factor:.2f} tags={','.join(reason_tags)}")
    except Exception as e:
        log(f"[COHERENCE_WARN] FUT {symbol} {tf} {e}")
# [PATCH NEG/CCA GATE END — maybe_execute_futures_trade]

    # [PATCH SAT APPLY BEGIN — maybe_execute_futures_trade]
    try:
        side_str  = "LONG" if exec_signal == "BUY" else "SHORT"
        entry_ref = float(last)
        sat = _style_sl_tp(symbol, tf, side_str, entry_ref, entry_ref)
        if sat.get("sl_price") is not None:
            sl_price = sat["sl_price"]
            sl_pct = abs((entry_ref - sat["sl_price"]) / entry_ref * 100.0)
        if sat.get("tp_price") is not None:
            tp_price = sat["tp_price"]
            tp_pct = abs((sat["tp_price"] - entry_ref) / entry_ref * 100.0)
        tr_pct = sat.get("trail_pct", 0.0)
        EXEC_STATE[("style", symbol, tf)]  = sat.get("style")
        EXEC_STATE[("regime", symbol, tf)] = sat.get("regime")
        EXEC_STATE[("sl_mode", symbol, tf)] = sat.get("mode")
        EXEC_STATE[("rr", symbol, tf)] = sat.get("rr")
        EXEC_STATE[("atr_mult", symbol, tf)] = sat.get("atr_mult")
        log(f"[STYLE] FUT {symbol} {tf} {side_str} style={sat.get('style')} regime={sat.get('regime')} mode={sat.get('mode')} rr={sat.get('rr')}")
    except Exception as e:
        log(f"[STYLE_WARN] FUT {symbol} {tf} {e}")
    # [PATCH SAT APPLY END — maybe_execute_futures_trade]

    # 6) 수량 계산(레버리지 상한 반영) → 정밀도/최소노치오날 체크
    qty_raw = _qty_from_margin_eff2(ex, symbol, last, eff_margin, tf)
    qty     = _ensure_fut_qty(ex, symbol, last, qty_raw)
    if qty <= 0:
        log(f"[FUT] skip (qty/nominal too small) {symbol} {tf} at {last}")
        return


    # 현재 포지션 확인
    pos_qty, pos_side, pos_entry = await _fetch_pos_qty(ex, symbol)

    # 반대면 청산
    if pos_side and ((exec_signal == "BUY" and pos_side == "SHORT") or (exec_signal == "SELL" and pos_side == "LONG")):
        await futures_close_all(symbol, tf, exit_price=last, reason="REVERSE")

    # 진입
    try:
        if exec_signal == "BUY":
            ord_ = await _market(ex, symbol, 'buy', qty, reduceOnly=False)
            # LONG
            FUT_POS[symbol] = {'side': 'LONG', 'qty': float(qty), 'entry': float(last), 'opened_ts': int(time.time()*1000)}
            side = 'LONG'
            FUT_POS_TF[tf] = symbol
            _save_json(OPEN_TF_FILE, FUT_POS_TF)
        else:
            ord_ = await _market(ex, symbol, 'sell', qty, reduceOnly=False)
            # SHORT
            FUT_POS[symbol] = {'side': 'SHORT','qty': float(qty), 'entry': float(last), 'opened_ts': int(time.time()*1000)}
            side = 'SHORT'
            FUT_POS_TF[tf] = symbol
            _save_json(OPEN_TF_FILE, FUT_POS_TF)

        eff_tp_pct, eff_sl_pct, eff_tr_pct, _src = _eff_risk_pcts(tp_pct, sl_pct, tr_pct, lev)
        if side == 'LONG':
            tp_price = (float(last) * (1 + (eff_tp_pct or 0)/100)) if eff_tp_pct else None
            sl_price = (float(last) * (1 - (eff_sl_pct or 0)/100)) if eff_sl_pct else None
        else:
            tp_price = (float(last) * (1 - (eff_tp_pct or 0)/100)) if eff_tp_pct else None
            sl_price = (float(last) * (1 + (eff_sl_pct or 0)/100)) if eff_sl_pct else None
        tr_pct_eff = eff_tr_pct

        extra = ",".join([
            f"id={(ord_.get('id') if isinstance(ord_, dict) else '')}",
            f"mode=futures",
            f"lev={float(lev or 1.0):.2f}", f"risk_mode={RISK_INTERPRET_MODE}",
            f"tp_pct={(tp_pct if tp_pct is not None else '')}",
            f"sl_pct={(sl_pct if sl_pct is not None else '')}",
            f"tr_pct={(tr_pct if tr_pct is not None else '')}",
            f"eff_tp_pct={(eff_tp_pct if eff_tp_pct is not None else '')}",
            f"eff_sl_pct={(eff_sl_pct if eff_sl_pct is not None else '')}",
            f"eff_tr_pct={(tr_pct_eff if tr_pct_eff is not None else '')}",
            f"tp_price={(tp_price if tp_price else '')}", f"sl_price={(sl_price if sl_price else '')}",
            f"pb_label={_pb_label if '_pb_label' in locals() else ''}",
            f"pb_alloc_mul={_pb_alloc_mul if '_pb_alloc_mul' in locals() else ''}"
        ])
        _log_trade_csv(symbol, tf, "OPEN", side, qty, last, extra=extra)

        # (NEW) persist risk to FUT_POS as well
        try:
            FUT_POS[symbol] = {
                **(FUT_POS.get(symbol) or {}),
                "tp_pct": tp_pct, "sl_pct": sl_pct, "tr_pct": tr_pct,
                "tp_price": tp_price, "sl_price": sl_price,
                "lev": float(lev or 1.0),
                "eff_tp_pct": eff_tp_pct, "eff_sl_pct": eff_sl_pct, "eff_tr_pct": tr_pct_eff,
                "risk_mode": RISK_INTERPRET_MODE
            }
            _save_json(OPEN_POS_FILE, FUT_POS)
        except Exception as e:
            logging.warning(f"[FUT_POS_RISK_SAVE_WARN] {symbol} {tf}: {e}")


        # [ANCHOR: POSITION_OPEN_HOOK]
        # --- Bracket legs state on open ---
        try:
            pos_obj = PAPER_POS.get(_pp_key(symbol, tf, side)) if TRADE_MODE=='paper' else FUT_POS.get(symbol)
            if isinstance(pos_obj, dict):
                pos_obj.setdefault("legs", [])
                pos_obj.setdefault("plan_total_notional", float(notional_used if 'notional_used' in locals() else qty*float(last)))
                pos_obj.setdefault("last_ctx", CTX_STATE.get(symbol))
                pos_obj.setdefault("last_realloc_ts", 0.0)
        except Exception:
            pass
        # initialize trailing baseline at entry (per (symbol, tf))
        try:
            k2 = (symbol, tf)
            entry_price = float(last)
            if str(side).upper() == 'LONG':
                highest_price[k2] = entry_price
                lowest_price.pop(k2, None)
            else:
                lowest_price[k2] = entry_price
                highest_price.pop(k2, None)
            trail_peak_roe[k2] = 0.0
        except Exception:
            pass
        previous_signal[(symbol, tf)] = 'BUY' if side == 'LONG' else 'SELL'
        entry_data[(symbol, tf)] = (float(last), datetime.now().strftime("%m월 %d일 %H:%M"))

        # 보호 주문(TP/SL) 동시 등록
        await _place_protect_orders(ex, symbol, tf, side, float(last), tp_pct=tp_pct, sl_pct=sl_pct, tr_pct=tr_pct)

        # (체결 후) 디스코드 알림
        try:
            await _notify_trade_entry(
                symbol, tf, exec_signal, mode="futures",
                price=float(last), qty=float(qty),
                base_margin=float(base_margin), eff_margin=float(eff_margin),
                lev_used=int(lev),
                score=EXEC_STATE.get(('score', symbol, tf)),
                pb_label=_pb_label, pb_w=_pb_w, pb_alloc_mul=_pb_alloc_mul
            )
            # 🔒 같은 캔들 재진입 방지 플래그
            if candle_ts is not None:
                ENTERED_CANDLE[(symbol, tf)] = int(candle_ts)

        except Exception as e:
            log(f"[NOTIFY] futures entry warn {symbol} {tf}: {e}")

    except Exception as e:
        log(f"[FUT] order failed {symbol} {tf} {signal}: {e}")



# 시작 시 거래소 준비
try:
    FUT_EXCHANGE = _mk_ex()
except Exception as e:
    log(f"[FUT] exchange init fail: {e}")
    FUT_EXCHANGE = None

# [ANCHOR: FUT_PUB_BOOT_BEGIN]  << ADD NEW >>
try:
    PUB_FUT_EXCHANGE = _mk_pub_fut_ex()
except Exception as e:
    log(f"[FUT_PUB] init fail: {e}")
    PUB_FUT_EXCHANGE = None
# [ANCHOR: FUT_PUB_BOOT_END]


_DASHBOARD_STATE = {"msg_id": 0, "ch_id": 0}
_DASH_TASK_RUNNING = False
_dash_state_load()


async def _dash_channel(client):
    ch_id = DASHBOARD_CHANNEL_ID or int(os.getenv("PNL_REPORT_CHANNEL_ID","0") or 0)
    if not ch_id: return None
    try: return await client.fetch_channel(ch_id)
    except Exception: return None

async def _dash_get_or_create_message(client):
    ch = await _dash_channel(client)
    if not ch:
        return None

    try:
        mid = int(_DASHBOARD_STATE.get("msg_id") or 0)
        if mid and _DASHBOARD_STATE.get("ch_id") == ch.id:

            return ch.get_partial_message(mid)  # no history fetch required
    except Exception:
        _DASHBOARD_STATE["msg_id"] = 0
        _dash_state_save()

    # create once
    m = await ch.send("📊 initializing dashboard…")
    _DASHBOARD_STATE["msg_id"] = m.id
    _DASHBOARD_STATE["ch_id"] = ch.id
    _dash_state_save()
    log(f"[DASH] created dashboard msg id={m.id} ch={ch.id}")

    return ch.get_partial_message(m.id)

def get_open_positions_iter():
    """Yield unified open position dicts from paper/futures stores."""
    out = []
    paper = PAPER_POS or {}
    fut   = FUT_POS or {}
    if not paper and not fut:
        # 디스크/거래소 하이드레이션이 늦는 경우를 대비한 1회 폴백
        try:
            _hydrate_from_disk()

            paper = PAPER_POS or {}
            fut   = FUT_POS or {}
        except Exception:
            pass

    try:
        for key, pos in paper.items():
            try:
                sym, tf, side = key.split("|", 2)
                out.append({
                    "symbol": sym,
                    "tf": tf,
                    "side": str(side).upper(),
                    "qty": float(pos.get("qty") or 0.0),
                    "entry_price": float(pos.get("entry_price") or pos.get("entry") or 0.0),
                    "lev": float(pos.get("lev") or 1.0),
                })
            except Exception:
                continue
        for sym, pos in fut.items():
            try:
                out.append({
                    "symbol": sym,
                    "tf": pos.get("tf", "FUT"),
                    "side": str(pos.get("side", "")).upper(),
                    "qty": float(pos.get("qty") or 0.0),
                    "entry_price": float(pos.get("entry") or pos.get("entry_price") or 0.0),
                    "lev": float(pos.get("lev") or 1.0),
                })
            except Exception:
                continue
    except Exception:
        pass
    return out

def _cooldown_remain_sec(symbol: str, tf: str) -> int:
    try:
        last = (STRUCT_ALERT_STATE or {}).get((symbol, tf, "ALERT_TEXT"), {}).get("ts", 0)
        if last <= 0:
            return 0
        remain = (MTF_ALERT_COOLDOWN_SEC * 1000 - max(0, _now_ms() - last)) // 1000
        return max(0, int(remain))
    except Exception:
        return 0


def _struct_shortline(symbol: str, tf: str) -> str:
    """
    • 최근접 저항/지지 거리(ATR배수) + 구조 사유 1~2개 요약
    """
    try:

        # 캐시 우선
        rows = _load_ohlcv(symbol, tf, limit=240)
        df = _sce_build_df_from_ohlcv(rows) if rows else None
        if df is None or len(df) < 60:
            return f"{symbol} {tf}: 데이터 부족"
        ent = _struct_cache_get(symbol, tf, _df_last_ts(df))
        if ent and ent.get("ctx"):
            ctx = ent["ctx"]
        else:
            ctx = build_struct_context_basic(df, tf)
            _struct_cache_put(symbol, tf, _df_last_ts(df), ctx, ent.get("img") if ent else None)

        near = ctx.get("nearest") or {}
        res, sup = near.get("res"), near.get("sup")
        bits = []
        if res: bits.append(f"저항 {res[2]:.2f}×ATR")
        if sup: bits.append(f"지지 {sup[2]:.2f}×ATR")

        # 구조 사유(추세선/채널/컨플루언스) 2개만
        rsn = []
        for (txt, sc, key) in ctx.get("reasons", []):
            if key.startswith(("TREND","CHAN","STRUCT_CONFLUENCE","STRUCT_GAP")):
                rsn.append(txt)
            if len(rsn) >= 2:
                break
        if rsn:
            bits.append(" / ".join(rsn))
        return f"{symbol.split('/')[0]}-{tf}: " + (" · ".join(bits) if bits else "구조 정보 없음")
    except Exception as e:
        return f"{symbol} {tf}: 구조 요약 실패({type(e).__name__})"


async def _dash_struct_block() -> list[str]:
    """
    대시보드 상단 고정 블록:
    ◼ 구조 컨텍스트 / ◼ MTF 게이트 / ◼ 알림 쿨다운
    """
    out = []
    try:
        symbols = ("ETH/USDT","BTC/USDT")
        tfs     = ("1h",)  # 요구사항: 1h 기준 요약 (필요시 15m/4h/1d 추가 가능)
        # 1) 구조 컨텍스트
        out.append("◼ 구조 컨텍스트")
        for s in symbols:
            for tf in tfs:
                out.append(" - " + _struct_shortline(s, tf))
        # 2) MTF 게이트 (1h→상위 TF)
        out.append("◼ MTF 게이트")
        for s in symbols:
            tf = "1h"
            try:
                buy = _mtf_struct_guard(s, tf, "BUY"); sell = _mtf_struct_guard(s, tf, "SELL")
                out.append(f" - {s.split('/')[0]}-{tf}: BUY={buy.get('action','?')} / SELL={sell.get('action','?')}  ({buy.get('reason') or sell.get('reason') or '—'})")
            except Exception:
                out.append(f" - {s.split('/')[0]}-{tf}: 게이트 계산 실패")
        # 3) 알림 쿨다운
        out.append("◼ 알림 쿨다운")
        for s in symbols:
            tf = "1h"
            sec = _cooldown_remain_sec(s, tf)
            out.append(f" - {s.split('/')[0]}-{tf}: 남은 {sec}s")
    except Exception as e:
        out.append(f"(구조 요약 생성 실패: {type(e).__name__})")
    return out



# === SCE text render for analysis messages ====================================
def _render_struct_context_text(symbol: str, tf: str, df=None, ctx=None) -> str:
    """
    분석 알림 본문 공통 섹션:
    - 최근접 저항/지지: 값 / 거리(ATR배수)
    - 추세선: up/down 근접/돌파
    - 회귀/피보 채널: 상/하단 접근/이탈
    - 컨플루언스/협곡
    """
    try:

        # --- 폴백/최소행수 파라미터 ---
        MIN_ROWS = int(os.getenv("SCE_MIN_ROWS", "60"))
        LIMIT    = int(os.getenv("SCE_FETCH_LIMIT", "400"))
        rows = None
        # 1) 입력 df 우선 사용, 없으면 로더
        if df is None:
            rows = _load_ohlcv(symbol, tf, limit=LIMIT)
            df2  = _sce_build_df_from_ohlcv(rows) if rows else None
        else:
            df2 = df
        # 2) 폴백: 로더 부족 시 메인 분석 df를 강제 재사용
        if df2 is None or len(df2) < MIN_ROWS:
            try:
                _rows_fb = _load_ohlcv(symbol, tf, limit=LIMIT)
                _df_fb   = _sce_build_df_from_ohlcv(_rows_fb) if _rows_fb else None
                if _df_fb is not None and len(_df_fb) >= MIN_ROWS:
                    df2 = _df_fb
            except Exception as e:
                log(f"[SCE_FALLBACK_WARN] {symbol} {tf} reload fail: {e}")
        # 3) 최종 부족 시 안내 + 길이 로깅
        if df2 is None or len(df2) < MIN_ROWS:
            log(f"[SCE_SHORT] {symbol} {tf} rows={0 if df2 is None else len(df2)} < {MIN_ROWS}")
            return "◼ 구조 컨텍스트\n- 데이터 부족"
        if ctx is None:
            ctx = build_struct_context_basic(df2, tf)


        lines = ["◼ 구조 컨텍스트"]
        near = (ctx.get("nearest") or {})
        res, sup = near.get("res"), near.get("sup")
        if res or sup:
            rtxt = f"{res[1]:.2f} ({res[2]:.2f}×ATR)" if res else "-"
            stxt = f"{sup[1]:.2f} ({sup[2]:.2f}×ATR)" if sup else "-"
            lines.append(f"- 최근접 저항/지지: {rtxt} / {stxt}")
        else:
            lines.append("- 최근접 저항/지지: -")

        # 이유 요약: TREND, CHAN, CONFLUENCE/GAP 카테고리별 최대 1~2줄
        cats = {"TREND": [], "CHAN": [], "CONF": [], "GAP": []}
        for (text, sc, key) in ctx.get("reasons", []):
            if key.startswith("TREND"):
                cats["TREND"].append(text)
            elif key.startswith("CHAN"):
                cats["CHAN"].append(text)
            elif key.startswith("STRUCT_CONFLUENCE"):
                cats["CONF"].append(text)
            elif key.startswith("STRUCT_GAP"):
                cats["GAP"].append(text)

        if cats["TREND"]:
            lines.append(f"- 추세선: {cats['TREND'][0]}")
        if cats["CHAN"]:
            lines.append(f"- 채널: {cats['CHAN'][0]}")
        if cats["CONF"]:
            lines.append(f"- 컨플루언스: {cats['CONF'][0]}")
        if cats["GAP"]:
            lines.append(f"- 협곡: {cats['GAP'][0]}")

        return "\n".join(lines)
    except Exception as e:
        return f"◼ 구조 컨텍스트\n- 생성 실패: {type(e).__name__}"

# ==============================================================================

def _render_struct_legend(ctx: dict, tf: str) -> str:
    if os.getenv("STRUCT_LEGEND_ENABLE", "1") != "1":
        return ""
    lines = [
        "",
        "🔎 **구조 해석 가이드**",
        "• 수평레벨: 가격↔레벨 거리(ATR배수) 작을수록 반대포지션 위험↑",
        "• 추세선: 선 아래 종가마감=하향 유지, 상향선 재진입=스카웃",
        "• 회귀채널: 상단=롱 익절/숏 관심, 하단=숏 익절/분할매수 관심",
        "• 피보채널: 0.382/0.618/1.0 평행선 접촉 시 반응/돌파 체크",
        "• 컨플루언스: 다중 레벨이 ATR×ε 내 겹치면 신뢰도↑",
    ]
    return "\n".join(lines)




# === Structure overlay renderer (matplotlib) ==================================
def render_struct_overlay(symbol: str, tf: str, df, struct_info,
                          save_dir: str = "./charts", width: int = 1600, height: int = 900) -> str | None:
    """
    캔들 + 수평 레벨 + 추세선(최근 피벗 2점) + 회귀 채널 + 피보 채널을 그려 저장.
    반환: 파일 경로 (실패 시 None)
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        if df is None or len(df) < 60 or struct_info is None:
            return None

        # 좌표 준비
        x = np.arange(len(df))
        o,h,l,c = df['open'].values, df['high'].values, df['low'].values, df['close'].values

        # Figure
        dpi = 100
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_title(f"{symbol} • {tf} • Structure Overlay", loc="left")

        # 캔들 (간단 구현)
        barw = 0.6
        for i in range(len(df)):
            color = "#2ca02c" if c[i] >= o[i] else "#d62728"
            ax.vlines(i, l[i], h[i], linewidth=1, color=color, alpha=0.8)
            rb = Rectangle((i - barw/2, min(o[i], c[i])),
                           barw, abs(c[i]-o[i]),
                           facecolor=color, edgecolor=color, alpha=0.6)
            ax.add_patch(rb)

        last_x = len(df)-1
        price  = float(c[-1])
        atr    = struct_info.get("atr", 0.0) or 0.0

        # 수평 레벨
        ne = struct_info.get("nearest") or {}
        res = ne.get("res"); sup = ne.get("sup")
        for (t, v) in struct_info.get("levels", [])[:10]:
            lw = 1.2; ls = "--"; col = "#888888"
            if t == "ATH": col = "#c21807"; lw = 1.8
            if t == "ATL": col = "#1565c0"; lw = 1.8
            if res and v == float(res[1]): lw = 2.4; col = "#c21807"; ls="-"
            if sup and v == float(sup[1]): lw = 2.4; col = "#1565c0"; ls="-"
            ax.hlines(v, 0, last_x, colors=col, linestyles=ls, linewidth=lw, alpha=0.9)

        # 추세선
        try:
            tls = _sce_best_trendlines(df)
            for dirn in ("up","down"):
                tl = tls.get(dirn)
                if not tl: continue
                y1 = _sce_value_on_line(tl, 0)
                y2 = _sce_value_on_line(tl, last_x)
                ls = "--"; col = "#2e7d32" if dirn=="up" else "#b71c1c"
                ax.plot([0,last_x],[y1,y2], ls=ls, lw=1.8, color=col, alpha=0.9, label=f"{dirn} TL")
        except Exception:
            pass

        # 회귀 채널
        try:
            reg = _sce_linreg_channel(df)
            if reg:
                ax.plot([0,last_x],[reg["upper"]]*2, lw=1.4, color="#6a1b9a", ls="-", alpha=0.9, label="Reg ↑")
                ax.plot([0,last_x],[reg["lower"]]*2, lw=1.4, color="#6a1b9a", ls="--", alpha=0.9, label="Reg ↓")
        except Exception:
            pass

        # 피보 채널
        try:
            fib = _sce_fib_channel(df)
            if fib:
                ups, downs = fib["ups"], fib["downs"]
                for v in ups:
                    ax.plot([0,last_x],[v]*2, lw=1.0, color="#f39c12", ls="-", alpha=0.8)
                for v in downs:
                    ax.plot([0,last_x],[v]*2, lw=1.0, color="#f39c12", ls="--", alpha=0.8)
        except Exception:
            pass

        ax.set_xlim(-1, last_x+1)
        ypad = (np.nanmax(h[-120:]) - np.nanmin(l[-120:])) * 0.05 if len(df)>120 else (np.nanmax(h) - np.nanmin(l)) * 0.05
        ax.set_ylim(np.nanmin(l[-120:]) - ypad, np.nanmax(h[-120:]) + ypad)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

        out = os.path.join(save_dir, f"struct_{symbol.replace('/','-')}_{tf}_{int(time.time())}.png")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        return out
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        log(f"[STRUCT_OVERLAY_ERR] {symbol} {tf} {type(e).__name__}: {e}")
        return None
# ==============================================================================



async def _dash_render_text():

    st = _daily_state_load() or {}  # ← Nonesafe
    cap_realized = capital_get()
    rows, totals = await gather_positions_upnl()  # ← async 버전만 사용
    if os.getenv("DASH_TRACE","0")=="1":
        assert isinstance(st, dict), f"DASH st type={type(st).__name__}"
        assert isinstance(totals, dict), f"DASH totals type={type(totals).__name__}"
        assert isinstance(rows, list), f"DASH rows type={type(rows).__name__}"
        bad = [(i, type(r).__name__) for i, r in enumerate(rows) if not isinstance(r, dict)]
        if bad:
            raise TypeError(f"DASH rows bad entries: {bad[:5]}")


    eq_mode = (os.getenv("DASHBOARD_EQUITY_MODE", "live") or "live").lower()
    if eq_mode == "live":
        eq_now = float(cap_realized) + float(
            (totals or {}).get(
                "upnl_usdt_sum_net", (totals or {}).get("upnl_usdt_sum", 0.0)
            )
        )
    else:
        eq_now = float(cap_realized)


    lines = []
    ko = os.getenv("DASH_LOCALE", "ko").lower().startswith("ko")
    show_fees = (os.getenv("DASH_SHOW_FEES", "1") == "1")
    eq_mode = (os.getenv("DASHBOARD_EQUITY_MODE", "live") or "live").lower()

    # 헤더
    if ko:
        lines.append(f"자산: **${eq_now:,.2f}**" + (" **(live)**" if eq_mode == "live" else " **(realized)**"))
        lines.append(
            f"일일 손익: {float(st.get('realized_usdt',0.0)):+.2f} USDT "
            f"({float(st.get('realized_pct',0.0)):+.2f}%) | 청산건수={int(st.get('closes',0))}"
        )
    else:
        lines.append(f"Equity: ${eq_now:,.2f}" + (" (live)" if eq_mode == "live" else " (realized)"))
        lines.append(
            f"Day PnL: {float(st.get('realized_usdt',0.0)):+.2f} USDT ({float(st.get('realized_pct',0.0)):+.2f}%) | closes={int(st.get('closes',0))}"
        )

    # [ANCHOR: DASH_RENDER_BEGIN]

    ko = os.getenv("DASH_LOCALE","ko").lower().startswith("ko")
    if ko:
        lev_mode = (os.getenv("DASH_LEV_HEADER_MODE", "applied") or "applied").lower()
        if lev_mode == "applied" and rows:
            _bkt = {}
            for r in rows:
                sym = str(r["symbol"]).split("/")[0]
                tfv = r["tf"]
                lv = int(float(r.get("lev", 0) or 0))
                _bkt.setdefault(sym, []).append(f"{tfv}={lv}")
            lev_line = "; ".join(f"{k} " + ", ".join(sorted(set(vs))) for k, vs in _bkt.items())
        else:
            lev_line = os.getenv("TF_LEVERAGE", "")
        lines.append(f"모드:{os.getenv('TRADE_MODE')} / 레버리지:{lev_line} / 슬리피지:{os.getenv('SLIPPAGE_PCT')}%")
    else:
        lev_mode = (os.getenv("DASH_LEV_HEADER_MODE", "applied") or "applied").lower()
        if lev_mode == "applied" and rows:
            _bkt = {}
            for r in rows:
                sym = str(r["symbol"]).split("/")[0]
                tfv = r["tf"]
                lv = int(float(r.get("lev", 0) or 0))
                _bkt.setdefault(sym, []).append(f"{tfv}={lv}")
            lev_line = "; ".join(f"{k} " + ", ".join(sorted(set(vs))) for k, vs in _bkt.items())
        else:
            lev_line = os.getenv("TF_LEVERAGE", "")
        lines.append(f"mode:{os.getenv('TRADE_MODE')} / lev:{lev_line} / slippage:{os.getenv('SLIPPAGE_PCT')}%")

    # ◼ SCE/MTF 고정 섹션 추가
    try:
        blk = await _dash_struct_block()
        if blk:
            lines += blk + [""]
    except Exception as _e:
        log(f"[DASH_STRUCT_BLOCK_WARN] {_e}")

    # 합계 UPNL(수수료 옵션 포함)
    if os.getenv("DASHBOARD_SHOW_TOTAL_UPNL", "1") == "1":
        up_sum = float(totals.get("upnl_usdt_sum", 0.0))
        up_net = float(totals.get("upnl_usdt_sum_net", up_sum))
        pct_on_eq = float(totals.get("upnl_pct_on_equity", 0.0))
        if ko:
            lines.append(f"미실현 손익: **{up_net:+.2f} USDT** (**{pct_on_eq:+.2f}%** of equity)")
            if os.getenv("DASH_SHOW_FEES","1") == "1":
                fe = float(totals.get("fees_entry_sum",0.0))
                fx = float(totals.get("fees_exit_est_sum",0.0))
                lines.append(f"수수료(누적/예상청산): -{fe:.2f} / -{fx:.2f} USDT")
        else:
            lines.append(f"Open UPNL: {up_net:+.2f} USDT ({pct_on_eq:+.2f}% of equity)")
            if show_fees:
                fe = float(totals.get("fees_entry_sum", 0.0))
                fx = float(totals.get("fees_exit_est_sum", 0.0))
                lines.append(f"Fees (paid/est. close): -{fe:.2f} / -{fx:.2f} USDT")

    # 섹션 제목
    count = len(rows)
    lines.append("— 포지션 —" if ko else "— open positions —")
    # 포지션 목록
    for r in rows:
        # 불릿+줄바꿈 포맷 (ko/en)
        upnl_net = r.get("upnl_usdt_net", r.get("upnl_usdt", 0.0))
        mae = r.get("mae_pct", 0.0); mfe = r.get("mfe_pct", 0.0)
        fe  = float(r.get("fee_entry_usdt",0.0)); fx = float(r.get("fee_exit_est_usdt",0.0))
        margin_used = float(r.get("margin_used_usdt", 0.0))
        show_notional = (os.getenv("DASH_SHOW_NOTIONAL","0") == "1")
        notional = abs(r.get("qty",0.0))*float(r.get("entry",0.0))
        # 리스크 항목(있을 때만)
        tp_price = r.get("tp_price"); sl_price = r.get("sl_price")
        tp_pct = r.get("eff_tp_pct", r.get("tp_pct")); sl_pct = r.get("eff_sl_pct", r.get("sl_pct"))

        if ko:
            head = f"• **{r['symbol']} · {r['tf']} · {r['side']} ×{r['lev']:g} ({r.get('margin_mode','ISOLATED')})**"
            line_price = f"  • 가격/수량: ${float(r['entry']):,.2f} → ${float(r['last']):,.2f} / {r['qty']:.4f}"
            if show_notional:
                line_price += f"  (노치오날 ${notional:,.2f})"
            line_cap   = f"  • 실자본 사용: **${margin_used:,.2f}**"
            line_pnl   = (
                f"  • 손익: **{r['upnl_pct_on_margin']:+.2f}% / {upnl_net:+,.2f} USDT**"
                f"   ·   MAE {mae:+.2f}% · MFE {mfe:+.2f}%"
            )
            line_fee   = (f"  • 수수료: 진입 -${fe:.2f} / 청산(추정) -${fx:.2f}") if show_fees else ""
            line_risk  = ""
            if (tp_price is not None) and (sl_price is not None) and (tp_pct is not None) and (sl_pct is not None):
                line_risk = f"  • 리스크: TP ${float(tp_price):,.2f} (+{float(tp_pct):.2f}%) / SL ${float(sl_price):,.2f} (-{float(sl_pct):.2f}%)"
            chunk = "\n".join([s for s in [head, line_price, line_cap, line_pnl, line_fee, line_risk] if s])
            lines.append(chunk)
        else:
            head = f"• {r['symbol']} · {r['tf']} · {r['side']} ×{r['lev']:g}"
            line_price = f"  • Price/Qty: ${float(r['entry']):,.2f} → ${float(r['last']):,.2f} / {r['qty']:.4f}"
            if show_notional:
                line_price += f"  (Notional ${notional:,.2f})"
            line_cap   = f"  • Margin used: ${margin_used:,.2f}"
            line_pnl   = f"  • PnL: {r['upnl_pct_on_margin']:+.2f}% / {upnl_net:+,.2f} USDT   ·   MAE {mae:+.2f}% · MFE {mfe:+.2f}%"
            line_fee   = (f"  • Fees: entry -${fe:.2f} / close(est) -${fx:.2f}") if show_fees else ""
            line_risk  = ""
            if (tp_price is not None) and (sl_price is not None) and (tp_pct is not None) and (sl_pct is not None):
                line_risk = f"  • Risk: TP ${float(tp_price):,.2f} (+{float(tp_pct):.2f}%) / SL ${float(sl_price):,.2f} (-{float(sl_pct):.2f}%)"
            chunk = "\n".join([s for s in [head, line_price, line_cap, line_pnl, line_fee, line_risk] if s])
            lines.append(chunk)

    return "\n".join(lines), st, eq_now, totals


async def _dash_loop(client):
    if not DASHBOARD_ENABLE: return
    while True:
        try:
            if os.getenv("DASH_TRACE","0")=="1":
                log("[DASH:TRACE] enter loop")
            msg = await _dash_get_or_create_message(client)
            if os.getenv("DASH_TRACE","0")=="1":
                log(f"[DASH:TRACE] have_msg={bool(msg)}")
            txt, st, eq_now, totals = await _dash_render_text()
            if os.getenv("DASH_TRACE","0")=="1":
                log(f"[DASH:TRACE] render_ok types st={type(st).__name__}, totals={type(totals).__name__}")
            if msg:
                try:
                    # 기존 단일 edit → 멀티파트 edit
                    await _discord_edit_chunked(msg, msg.channel, txt, tag_key="DASH", header_prefix="Dashboard")
                except Exception as e:

                    if "Unknown Message" in str(e) or "Not Found" in str(e):
                        _DASHBOARD_STATE["msg_id"] = 0
                        _dash_state_save()
                        log("[DASH] dashboard message missing – will recreate")
                    else:
                        log(f"[DASH] edit warn: {e}")

            if PRESENCE_ENABLE:

                if os.getenv("DASH_TRACE","0")=="1":
                    log("[DASH:TRACE] before presence")

                eq  = float(eq_now)
                day = float((st or {}).get("realized_usdt", 0.0))
                ou  = float((totals or {}).get("upnl_usdt_sum", 0.0))
                await client.change_presence(
                    activity=discord.Activity(
                        type=discord.ActivityType.watching,
                        name=f"Eq ${eq:,.0f} | Day {day:+.0f} | Open {ou:+.0f}"
                    )
                )

        except Exception as e:
            if os.getenv("DASH_TRACE","0")=="1":
                import traceback
                tb = traceback.format_exc()
                log(f"[DASH][TRACE] stack:\n{tb}")
            log(f"[DASH] warn: {e}")
            await asyncio.sleep(max(3, DASHBOARD_UPDATE_SEC))
            continue
        await asyncio.sleep(max(3, DASHBOARD_UPDATE_SEC))


async def _sync_open_state_on_ready():
    # 페이퍼: 파일 로드로 충분 (이미 상단에서 로드됨)
    # 선물: 거래소 포지션 동기화
    try:
        ex = FUT_EXCHANGE
        if ex:
            for sym in ("BTC/USDT","ETH/USDT"):
                qty, side, entry = await _fetch_pos_qty(ex, sym)
                if side and abs(qty) > 0:
                    FUT_POS[sym] = {"side": side, "qty": float(qty), "entry": float(entry), "opened_ts": int(time.time()*1000)}
                    FUT_POS_TF.setdefault("15m", None)
            _save_json(OPEN_POS_FILE, FUT_POS)
            _save_json(OPEN_TF_FILE, FUT_POS_TF)
    except Exception as e:
        log(f"[SYNC] warn: {e}")

# === Hedge mode & TF-level overrides ===
HEDGE_MODE   = os.getenv("HEDGE_MODE", "1") == "1"

import re

TF_LEVERAGE = _parse_tf_map(os.getenv("TF_LEVERAGE", os.getenv("LEVERAGE_BY_TF", "")), int)   # 예: {'15m':7,'1h':5,...}
TF_MARGIN   = _parse_tf_map(os.getenv("TF_MARGIN", os.getenv("MARGIN_BY_TF", "")), lambda x: x.upper())                  # 예: {'15m':'ISOLATED','4h':'CROSS',...}

# === Per-symbol per-TF margin-mode overrides ===
import re as _re

def _parse_float_by_symbol(raw: str):
    """
    예: 'BTC:15m=0.5,4h=0.4;ETH:4h=0.9'
    -> {'BTC': {'15m':0.5,'4h':0.4}, 'ETH': {'4h':0.9}}
    """
    out = {}
    if not raw:
        return out
    for block in _re.split(r"[;|]\s*", raw.strip()):
        if not block or ":" not in block:
            continue
        sym, tail = block.split(":", 1)
        sym = sym.strip().upper()
        out.setdefault(sym, {})
        for pair in _re.split(r"[,\s]+", tail.strip()):
            if not pair or "=" not in pair:
                continue
            tf, v = pair.split("=", 1)
            try:
                out[sym][tf.strip()] = float(v)
            except Exception:
                pass
    return out

def _parse_side_policy(raw: str):
    """
    예: 'BTC:4h=LONG_ONLY;ETH:4h=BOTH' -> {'BTC': {'4h':'LONG_ONLY'}, 'ETH': {'4h':'BOTH'}}
    """
    out = {}
    if not raw:
        return out
    for block in _re.split(r"[;|]\s*", raw.strip()):
        if not block or ":" not in block:
            continue
        sym, tail = block.split(":", 1)
        sym = sym.strip().upper()
        out.setdefault(sym, {})
        for pair in _re.split(r"[,\s]+", tail.strip()):
            if not pair or "=" not in pair:
                continue
            tf, v = pair.split("=", 1)
            out[sym][tf.strip()] = (v or "").strip().upper()
    return out

# ENV 로드
_SLIP_BY_SYMBOL   = _parse_float_by_symbol(cfg_get("SLIPPAGE_BY_SYMBOL", ""))
_TP_BY_SYMBOL     = _parse_float_by_symbol(cfg_get("TP_PCT_BY_SYMBOL", ""))
_SL_BY_SYMBOL     = _parse_float_by_symbol(cfg_get("SL_PCT_BY_SYMBOL", ""))
_TRAIL_BY_SYMBOL  = _parse_float_by_symbol(cfg_get("TRAIL_PCT_BY_SYMBOL", ""))
_SIDE_POL_BY_SYM  = _parse_side_policy(cfg_get("HEDGE_SIDE_POLICY", ""))

def _req_float_map(sym_map: dict, tf_map: dict, tf: str, default: float|None):
    """
    우선순위: 심볼×TF(overrides) > TF 전역 맵 > 전역 기본(default)
    tf_map 예: TAKE_PROFIT_PCT / HARD_STOP_PCT / TRAILING_STOP_PCT (dict)
    """
    if default is None:
        default = 0.0
    return float(tf_map.get(tf, default))

def _req_slippage_pct(symbol: str, tf: str) -> float:
    base = symbol.split("/")[0].upper()
    if base in _SLIP_BY_SYMBOL and tf in _SLIP_BY_SYMBOL[base]:
        return float(_SLIP_BY_SYMBOL[base][tf])
    return float(os.getenv("SLIPPAGE_PCT", "0.7"))  # 전역 기본

def _req_tp_pct(symbol: str, tf: str, tf_map: dict) -> float:
    base = symbol.split("/")[0].upper()
    if base in _TP_BY_SYMBOL and tf in _TP_BY_SYMBOL[base]:
        return float(_TP_BY_SYMBOL[base][tf])
    return _req_float_map(_TP_BY_SYMBOL.get(base, {}), tf_map, tf, 0.0)

def _req_sl_pct(symbol: str, tf: str, tf_map: dict) -> float:
    base = symbol.split("/")[0].upper()
    if base in _SL_BY_SYMBOL and tf in _SL_BY_SYMBOL[base]:
        return float(_SL_BY_SYMBOL[base][tf])
    return _req_float_map(_SL_BY_SYMBOL.get(base, {}), tf_map, tf, 0.0)

def _req_trail_pct(symbol: str, tf: str, tf_map: dict) -> float:
    base = symbol.split("/")[0].upper()
    if base in _TRAIL_BY_SYMBOL and tf in _TRAIL_BY_SYMBOL[base]:
        return float(_TRAIL_BY_SYMBOL[base][tf])
    return _req_float_map(_TRAIL_BY_SYMBOL.get(base, {}), tf_map, tf, 0.0)

# [ANCHOR: CFG_RESOLUTION_BEGIN]
def eff_leverage(symbol: str, tf: str) -> int:
    """Resolve effective leverage for symbol×tf."""
    try:
        return int(_req_leverage(symbol, tf))
    except Exception:
        try:
            return int(TF_LEVERAGE.get(tf, 1))
        except Exception:
            return 1

def eff_margin(symbol: str, tf: str) -> str:
    """Resolve effective margin mode for symbol×tf."""
    try:
        mode, _ = _req_margin_mode(symbol, tf)
        return mode
    except Exception:
        return FUT_MARGIN

def eff_slippage(symbol: str, tf: str) -> float:
    """Resolve effective slippage percent for symbol×tf."""
    try:
        return float(_req_slippage_pct(symbol, tf))
    except Exception:
        return float(os.getenv("SLIPPAGE_PCT", "0.0"))

def eff_tp_pct(symbol: str, tf: str) -> float:
    """Resolve effective TP% for symbol×tf."""
    try:
        return float(_req_tp_pct(symbol, tf, (take_profit_pct or {})))
    except Exception:
        return 0.0

def eff_sl_pct(symbol: str, tf: str) -> float:
    """Resolve effective SL% for symbol×tf."""
    try:
        return float(_req_sl_pct(symbol, tf, (HARD_STOP_PCT or {})))
    except Exception:
        return 0.0

def eff_trail_pct(symbol: str, tf: str) -> float:
    """Resolve effective trailing-stop % for symbol×tf."""
    try:
        return float(_req_trail_pct(symbol, tf, (trailing_stop_pct or {})))
    except Exception:
        return 0.0
# [ANCHOR: CFG_RESOLUTION_END]

# [ANCHOR: PAPER_PROTECT_ORDERS_BEGIN]
def _paper_ensure_tp_sl_trailing(symbol: str, tf: str, side: str,
                                 entry_price: float, tp_pct: float | None,
                                 sl_pct: float | None, tr_pct: float | None,
                                 lev: float | None) -> tuple[float | None, float | None, float | None]:
    """
    Compute and persist paper tp/sl and effective trailing pct using the same math
    futures uses (eff on price, leverage-aware when RISK_INTERPRET_MODE=MARGIN_RETURN).
    Returns (tp_price, sl_price, eff_tr_pct).
    """
    eff_tp_pct, eff_sl_pct, eff_tr_pct, _ = _eff_risk_pcts(tp_pct, sl_pct, tr_pct, lev)
    tp_price = None
    sl_price = None

    if eff_tp_pct and eff_tp_pct > 0:
        if str(side).upper() == "LONG":
            tp_price = float(entry_price) * (1.0 + eff_tp_pct / 100.0)
        else:
            tp_price = float(entry_price) * (1.0 - eff_tp_pct / 100.0)

    if eff_sl_pct and eff_sl_pct > 0:
        if str(side).upper() == "LONG":
            sl_price = float(entry_price) * (1.0 - eff_sl_pct / 100.0)
        else:
            sl_price = float(entry_price) * (1.0 + eff_sl_pct / 100.0)

    k = _pp_key(symbol, tf, side)
    pos = PAPER_POS.get(k) or {}
    pos["tp_price"] = tp_price
    pos["sl_price"] = sl_price
    pos["eff_tr_pct"] = eff_tr_pct
    if os.getenv("PAPER_EXIT_REDUCEONLY", "1") == "1":
        pos["reduce_only"] = True
    PAPER_POS[k] = pos
    _save_json(PAPER_POS_FILE, PAPER_POS)
    return tp_price, sl_price, eff_tr_pct
# [ANCHOR: PAPER_PROTECT_ORDERS_END]

# === Trailing helpers (apply to all TFs) ===

TRAIL_ARM_DELTA_MIN_PCT = float(cfg_get("TRAIL_ARM_DELTA_MIN_PCT", "0.0"))
TRAIL_ARM_DELTA_MIN_PCT_BY_TF = cfg_get("TRAIL_ARM_DELTA_MIN_PCT_BY_TF", "")


def _arm_min_for_tf(tf: str) -> float:
    try:
        if not TRAIL_ARM_DELTA_MIN_PCT_BY_TF:
            return TRAIL_ARM_DELTA_MIN_PCT
        m = {kv.split(":")[0].strip(): float(kv.split(":")[1])
             for kv in TRAIL_ARM_DELTA_MIN_PCT_BY_TF.split(",")
             if ":" in kv and "=" not in kv}
        return float(m.get(tf, TRAIL_ARM_DELTA_MIN_PCT))
    except Exception:
        return TRAIL_ARM_DELTA_MIN_PCT

def _trail_hp(entry: float, hp: float | None) -> float:
    # long baseline must never be below entry
    return max(float(entry), float(hp or 0.0))

def _trail_lp(entry: float, lp: float | None) -> float:
    # short baseline must never be above entry
    return min(float(entry), float(lp or 1e30))

def _compute_trail(side: str, entry: float, tr_pct: float,
                   hp: float | None, lp: float | None, tf: str):

    """Return (trail_price or None, armed(bool), base(float))"""
    ts = float(tr_pct)/100.0 if tr_pct is not None else 0.0

    if ts <= 0.0:
        return (None, False, None)
    arm_min = _arm_min_for_tf(tf)/100.0
    if str(side).upper() == "LONG":
        base = _trail_hp(entry, hp)
        armed = base >= float(entry)*(1.0 + arm_min)
        return ((base*(1.0 - ts)) if armed else None, armed, base)
    else:
        base = _trail_lp(entry, lp)
        armed = base <= float(entry)*(1.0 - arm_min)
        return ((base*(1.0 + ts)) if armed else None, armed, base)


def _eff_risk_pcts(tp_pct, sl_pct, tr_pct, lev):
    """Return (eff_tp_pct, eff_sl_pct, eff_tr_pct, mode) where effective % are on price-basis."""
    try: l = float(lev or 1.0)
    except Exception: l = 1.0
    mode = RISK_INTERPRET_MODE
    if mode == "MARGIN_RETURN":
        e_tp = (float(tp_pct) / l) if (tp_pct is not None) else None
        e_sl = (float(sl_pct) / l) if (sl_pct is not None) else None
        e_tr = (float(tr_pct) / l) if (tr_pct is not None and APPLY_LEV_TO_TRAIL) else (float(tr_pct) if tr_pct is not None else None)
        return (e_tp, e_sl, e_tr, "MARGIN_RETURN")
    # default PRICE_PCT
    return (float(tp_pct) if tp_pct is not None else None,
            float(sl_pct) if sl_pct is not None else None,
            float(tr_pct) if tr_pct is not None else None,
            "PRICE_PCT")


# === Unified exit evaluation on 1m (for ALL TFs) ===
def _eval_exit(symbol: str, tf: str, side: str,
               entry_price: float, last_price_hint: float,
               tp_price: float|None, sl_price: float|None,
               tr_pct: float|None, key2: tuple,
               lev: float = 1.0, opened_ts: float | None = None):
    """

    Returns: (should_exit: bool, reason: str, trigger_price: float, dbg: str)
    - Uses 1m bar and EXIT_EVAL_MODE (TOUCH | CLOSE)
    - Price is sanitized/clamped; outlier-guarded
    - Trailing is secondary to TP/SL and obeys arm guards

    """
    clamped, bar = _sanitize_exit_price(symbol, last_price_hint)
    if _outlier_guard(clamped, bar):
        return (False, "OUTLIER_SKIP", clamped, f"outlier>{OUTLIER_MAX_1M}")

    if EXIT_EVAL_MODE == "CLOSE":
        p = float(bar["close"])
        hi = lo = p
    else:
        p = clamped
        hi, lo = float(bar["high"]), float(bar["low"])

    sideU = str(side).upper()
    tp_hit = sl_hit = False
    if sideU == "LONG":
        if tp_price: tp_hit = hi >= float(tp_price)
        if sl_price: sl_hit = lo <= float(sl_price)
    else:
        if tp_price: tp_hit = lo <= float(tp_price)
        if sl_price: sl_hit = hi >= float(sl_price)

    if tp_hit or sl_hit:
        reason = "TP" if tp_hit else "SL"
        trig = float(tp_price) if tp_hit else float(sl_price)
        dbg = (f"1m ohlc=({bar['open']:.6f},{bar['high']:.6f},{bar['low']:.6f},{bar['close']:.6f}) "
               f"p={p:.6f} clamp={clamped:.6f} tp={tp_price} sl={sl_price} tr={tr_pct}")
        return (True, reason, trig, dbg)

    if _trail_priority_secondary():
        if not _arm_allowed(side, tf, entry_price, p, lev, opened_ts):
            if os.getenv("REENTRY_DEBUG","0") == "1":
                log(f"[TRAIL] skip arm: tf={tf} needΔ={_tf_map_get('TRAIL_ARM_DELTA_MIN_PCT_BY_TF',tf,0)} "
                    f"needP={_tf_map_get('TRAIL_MIN_PROFIT_PCT_BY_TF',tf,0)} be={os.getenv('TRAIL_ONLY_AFTER_BREAK_EVEN','1')}")
            return (False, "NONE", p, f"tp={tp_price} sl={sl_price} tr={tr_pct} arm=False")

    tr_hit = False
    apply_lev = os.getenv("APPLY_LEV_TO_TRAIL","1") == "1"
    if apply_lev:
        curr_roe = _roe_pct(side, entry_price, p, lev)
        peak = trail_peak_roe.get(key2, curr_roe)
        trail_peak_roe[key2] = max(peak, curr_roe)
        retrace = trail_peak_roe[key2] - curr_roe
        tr_hit = (float(tr_pct or 0.0) > 0) and (retrace >= float(tr_pct))
        trail_px = p
        armed = True
        base = trail_peak_roe[key2]
    else:
        trail_px, armed, base = _compute_trail(side, float(entry_price),
                                               float(tr_pct) if tr_pct is not None else 0.0,
                                               highest_price.get(key2), lowest_price.get(key2), tf)
        if sideU == "LONG":
            if trail_px and armed: tr_hit = lo <= float(trail_px)
        else:
            if trail_px and armed: tr_hit = hi >= float(trail_px)

    dbg = (f"1m ohlc=({bar['open']:.6f},{bar['high']:.6f},{bar['low']:.6f},{bar['close']:.6f}) "
           f"p={p:.6f} clamp={clamped:.6f} armed={armed} base={base} "
           f"tp={tp_price} sl={sl_price} tr={tr_pct} trail_px={trail_px}")
    if tr_hit:
        trig = trail_px
        return (True, "TRAIL", trig, dbg)
    return (False, "NONE", p, dbg)


def _hedge_side_allowed(symbol: str, tf: str, signal: str) -> bool:
    """
    HEDGE_SIDE_POLICY에 따라 신호 허용 여부.
    BOTH(기본): 아무 제약 없음 / LONG_ONLY: BUY만 / SHORT_ONLY: SELL만
    """
    base = symbol.split("/")[0].upper()
    pol = (_SIDE_POL_BY_SYM.get(base, {}) or {}).get(tf, "BOTH")
    pol = pol.upper()
    if pol == "BOTH":
        return True
    if pol == "LONG_ONLY":
        return signal.upper() == "BUY"
    if pol == "SHORT_ONLY":
        return signal.upper() == "SELL"
    return True

def _parse_margin_by_symbol(raw: str):
    """
    예: "BTC:15m=CROSS,1h=CROSS,4h=ISOLATED;ETH:15m=ISOLATED,4h=CROSS"
    -> {'BTC': {'15m':'CROSS','1h':'CROSS','4h':'ISOLATED'}, 'ETH': {...}}
    """
    res = {}
    if not raw:
        return res
    for block in _re.split(r"[;|]\s*", raw.strip()):
        if not block or ":" not in block:
            continue
        sym, tail = block.split(":", 1)
        sym = sym.strip().upper()
        res.setdefault(sym, {})
        for pair in _re.split(r"[,\s]+", tail.strip()):
            if not pair or "=" not in pair:
                continue
            tf, val = pair.split("=", 1)
            res[sym][tf.strip()] = (val or "").strip().upper()
    return res

def _parse_tf_map_str(raw: str):
    # 예: "15m:CROSS;1h:CROSS;4h:ISOLATED"
    mp = {}
    if not raw:
        return mp
    for part in _re.split(r"[;,]\s*", raw.strip()):
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        mp[k.strip()] = (v or "").strip().upper()
    return mp

def _parse_default_by_symbol(raw: str):
    # 예: "BTC:CROSS;ETH:ISOLATED" -> {'BTC':'CROSS','ETH':'ISOLATED'}
    mp = {}
    if not raw:
        return mp
    for part in _re.split(r"[;,]\s*", raw.strip()):
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        mp[k.strip().upper()] = (v or "").strip().upper()
    return mp

_MARGIN_BY_SYMBOL = _parse_margin_by_symbol(os.getenv("MARGIN_BY_SYMBOL",""))

# 보조 ENV 병합: BTC_MARGIN_BY_TF / ETH_MARGIN_BY_TF
for _sym_env in ("BTC","ETH"):
    _raw = os.getenv(f"{_sym_env}_MARGIN_BY_TF","")
    if _raw:
        _mp = _parse_tf_map_str(_raw)
        if _mp:
            _MARGIN_BY_SYMBOL.setdefault(_sym_env, {}).update(_mp)

# 심볼 기본값
_MARGIN_DEFAULT_BY_SYMBOL = _parse_default_by_symbol(os.getenv("MARGIN_DEFAULT_BY_SYMBOL",""))
for _sym_env in ("BTC","ETH"):
    dflt = os.getenv(f"{_sym_env}_MARGIN_DEFAULT","")
    if dflt:
        _MARGIN_DEFAULT_BY_SYMBOL[_sym_env] = dflt.strip().upper()

_MARGIN_DEBUG = os.getenv("MARGIN_DEBUG","0") == "1"

def _req_margin_mode(symbol: str, tf: str) -> tuple[str, str]:
    """
    반환: (요청 마진 모드 'ISOLATED'/'CROSSED', 'source')
    우선순위: 심볼×TF > 심볼기본 > TF전역 > 전역기본
    """
    def _src(val, src):
        return (_normalize_margin(val), src)

    try:
        base = symbol.split("/")[0].upper()
    except Exception:
        base = str(symbol).upper()

    # 1) 심볼×TF
    try:
        v = _MARGIN_BY_SYMBOL.get(base, {}).get(tf)
        if v:
            return _src(v, "symbol×tf")
    except Exception:
        pass

    # 2) 심볼 기본
    try:
        v = _MARGIN_DEFAULT_BY_SYMBOL.get(base)
        if v:
            return _src(v, "symbol-default")
    except Exception:
        pass

    # 3) TF 전역
    try:
        v = TF_MARGIN.get(tf)
        if v:
            return _src(v, "tf-global")
    except Exception:
        pass

    # 4) 전역 기본
    return _src(FUT_MARGIN, "global-default")


# === Per-symbol per-TF leverage overrides ===
import re as _re  # 이미 위에서 임포트했다면 이 줄은 중복되어도 무방

def _parse_lev_by_symbol(raw: str):
    """
    예: "BTC:15m=9,1h=7,4h=5,1d=4;ETH:15m=7,1h=5,4h=4,1d=3"
    -> {'BTC': {'15m':9,'1h':7,'4h':5,'1d':4}, 'ETH': {...}}
    """
    res = {}
    if not raw:
        return res
    for block in _re.split(r"[;|]\s*", raw.strip()):
        if not block or ":" not in block:
            continue
        sym, tail = block.split(":", 1)
        sym = sym.strip().upper()
        res.setdefault(sym, {})
        for pair in _re.split(r"[,\s]+", tail.strip()):
            if not pair or "=" not in pair:
                continue
            tf, val = pair.split("=", 1)
            try:
                res[sym][tf.strip()] = int(float(val))
            except Exception:
                pass
    return res

def _parse_tf_map_int(raw: str):
    """
    예: "15m:9;1h:7;4h:5;1d:4" -> {'15m':9,'1h':7,'4h':5,'1d':4}
    """
    out = {}
    if not raw:
        return out
    for part in _re.split(r"[;,]\s*", raw.strip()):
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        try:
            out[k.strip()] = int(float(v))
        except Exception:
            pass
    return out

# ENV 로드 + 병합 (LEVERAGE_BY_SYMBOL 가 최우선, 없으면 BTC/ETH_LEVERAGE_BY_TF 병합)
_LEV_BY_SYMBOL = _parse_lev_by_symbol(os.getenv("LEVERAGE_BY_SYMBOL", ""))
for _sym_env in ("BTC", "ETH"):
    _raw = os.getenv(f"{_sym_env}_LEVERAGE_BY_TF", "")
    if _raw:
        _map = _parse_tf_map_int(_raw)
        if _map:
            _LEV_BY_SYMBOL.setdefault(_sym_env, {}).update(_map)

def _req_leverage(symbol: str, tf: str) -> int:
    """
    심볼×TF 요청 레버리지:
      1) _LEV_BY_SYMBOL (심볼×TF 오버라이드)
      2) TF_LEVERAGE[tf]
      3) FUT_LEVERAGE
    """
    try:
        base = symbol.split("/")[0].upper()
    except Exception:
        base = str(symbol).upper()
    try:
        v = _LEV_BY_SYMBOL.get(base, {}).get(tf)
        if v is not None:
            return int(v)
    except Exception:
        pass
    return int(TF_LEVERAGE.get(tf, FUT_LEVERAGE))

# (선택) 디버그 확인용
if cfg_get("DEBUG_ALLOC_LOG", "0") == "1":
    try:
        log(f"[CONF] LEV_BY_SYMBOL={_LEV_BY_SYMBOL}")
    except Exception:
        print(f"[CONF] LEV_BY_SYMBOL={_LEV_BY_SYMBOL}")

# 보조(옵션) ENV도 병합: BTC_LEVERAGE_BY_TF / ETH_LEVERAGE_BY_TF
for _sym_env in ("BTC", "ETH"):
    _raw = os.getenv(f"{_sym_env}_LEVERAGE_BY_TF", "")
    if _raw:
        mp = _parse_tf_map(_raw, int)
        if mp:
            _LEV_BY_SYMBOL.setdefault(_sym_env, {}).update(mp)

def _req_leverage(symbol: str, tf: str) -> int:
    """
    심볼×TF 요청 레버리지:
    - 1순위: LEVERAGE_BY_SYMBOL (또는 BTC/ETH_LEVERAGE_BY_TF)
    - 2순위: TF_LEVERAGE[tf]
    - 3순위: FUT_LEVERAGE(기본)
    """
    try:
        base = symbol.split("/")[0].upper()
    except Exception:
        base = str(symbol).upper()
    v = None
    try:
        v = _LEV_BY_SYMBOL.get(base, {}).get(tf)
    except Exception:
        v = None
    if v is not None:
        return int(v)
    return int(TF_LEVERAGE.get(tf, FUT_LEVERAGE))


# ==== Futures order guards (precision / min-notional / leverage clamp) ====
def _market_limits(ex, symbol):
    try:
        m = ex.market(symbol)
        limits = m.get('limits', {}) or {}
        amount = (limits.get('amount') or {})
        cost   = (limits.get('cost') or {})
        lev    = (limits.get('leverage') or {})
        return {
            'min_qty': float(amount.get('min') or 0),
            'min_cost': float(cost.get('min') or 0),
            'max_lev': float(lev.get('max') or 125),
        }
    except Exception:
        return {'min_qty': 0.0, 'min_cost': 0.0, 'max_lev': 125.0}

def _fut_amount_to_precision(ex, symbol, qty):
    try:
        return float(ex.amount_to_precision(symbol, qty))
    except Exception:
        return float(qty)

def _fut_min_notional_ok(ex, symbol, price, qty):
    try:
        m = ex.market(symbol)
        limits = m.get('limits', {}) or {}
        min_cost = float((limits.get('cost') or {}).get('min') or 0)
    except Exception:
        min_cost = 0.0
    try:
        env_min = float(os.getenv("FUT_MIN_NOTIONAL", os.getenv("MIN_NOTIONAL", "5")))
    except Exception:
        env_min = 5.0
    need = max(min_cost, env_min)
    return (float(price) * float(qty)) >= need

def _clamp(val, lo, hi):
    try:
        return max(lo, min(hi, val))
    except Exception:
        return val

def _qty_from_margin_eff(ex, symbol, price, tf=None):
    # 총자본 배분 *_margin_for_tf(tf) × 효과적 레버리지
    lev_req = int(TF_LEVERAGE.get(tf, FUT_LEVERAGE))
    limits  = _market_limits(ex, symbol)
    lev_eff = int(_clamp(lev_req, 1, int(limits.get('max_lev') or 125)))
    margin  = _margin_for_tf(tf)
    notional = float(margin) * float(lev_eff)
    if notional <= 0 or price <= 0:
        return 0.0
    return notional / float(price)

def _ensure_fut_qty(ex, symbol, price, qty):
    q = _fut_amount_to_precision(ex, symbol, max(0.0, float(qty)))
    if q <= 0:
        return 0.0
    if not _fut_min_notional_ok(ex, symbol, float(price), q):
        return 0.0
    return q


def _normalize_margin(m):
    s = (m or "").upper()
    # Binance USDT-M API는 'ISOLATED' 또는 'CROSSED' 문자열을 사용
    if s.startswith("I"):
        return "ISOLATED"
    if s.startswith("C"):   # CROSS/CROSSED 모두 허용되게 정규화
        return "CROSSED"
    return "ISOLATED"


# ===== PnL PDF 생성기 (간단 요약판) =====
async def generate_pnl_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    csv_path = "logs/futures_pnl.csv"
    if not os.path.exists(csv_path):
        return None
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(",")]
            if len(parts) >= 9:
                rows.append(parts)

    # 집계
    header = ["시각","심볼","TF","종료사유","사이드","수량","진입가","청산가","PnL(USDT)"]
    data = [header] + rows

    from collections import defaultdict
    daily = defaultdict(float)
    by_tf = defaultdict(float)
    total = 0.0
    for r in rows:
        day = r[0][:10]
        tfv = r[2]
        pnl = float(r[8])
        daily[day] += pnl
        by_tf[tfv] += pnl
        total += pnl

    out = f"logs/PNL_{datetime.now():%Y%m%d_%H%M}.pdf"
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=1.2*cm, rightMargin=1.2*cm, topMargin=1.2*cm, bottomMargin=1.2*cm)
    styles = getSampleStyleSheet()
    elems = []
    elems += [Paragraph("<b>Futures PnL Report</b>", styles['Title']), Spacer(1,0.2*cm)]
    elems += [Paragraph(f"총 손익(USDT): <b>{total:.2f}</b>", styles['Heading3']), Spacer(1,0.1*cm)]

    # 일자별 표
    day_table = [["날짜","PnL(USDT)"]] + [[d, f"{v:.2f}"] for d, v in sorted(daily.items())]
    t1 = Table(day_table, hAlign='LEFT'); t1.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.grey)]))
    elems += [t1, Spacer(1,0.3*cm)]

    # TF별 표 (신규)
    tf_table = [["TF","PnL(USDT)"]] + [[tf, f"{v:.2f}"] for tf, v in sorted(by_tf.items())]
    t_tf = Table(tf_table, hAlign='LEFT'); t_tf.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.grey)]))
    elems += [t_tf, Spacer(1,0.3*cm)]

    # Raw 표
    t2 = Table(data, hAlign='LEFT')
    t2.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.grey), ('BACKGROUND',(0,0),(-1,0),colors.whitesmoke)]))
    elems += [t2]

    doc.build(elems)
    return out



# [ANCHOR] OLDSTYLE_REPORT_BEGIN
async def _send_report_oldstyle(client, channel, symbol: str, tf: str):
    # 데이터/지표
    df = get_ohlcv(symbol, tf, limit=300)
    df = add_indicators(df)

    # 차트/리포트 산출물
    chart_files        = save_chart_groups(df, symbol, tf)           # 4장
    score_file         = plot_score_history(symbol, tf)
    perf_file          = analyze_performance_for(symbol, tf)
    performance_file   = generate_performance_stats(tf, symbol=symbol)

    # 간단 시그널 텍스트(과거 스타일): 최근 신호/스코어/가격
    closed_price = get_closed_price(df, 'close')
    try:
        sig, _, _, _, _, score, *_ = calculate_signal(df, tf, symbol)
    except Exception:
        sig, score = None, None
    head = f"📈 {symbol} · {tf} · {sig}  (score {score if score is not None else '—'})\n종가: ${closed_price:,.2f}"

    # 포지션 요약(페이퍼+실선물 통합)
    pos_lines = []
    try:
        for p in get_open_positions_iter():
            pos_lines.append(
                f"• {p['symbol']} · {p['tf']} · {p['side']} ×{int(p['lev'])}  @ ${p['entry_price']:,.2f} / qty {p['qty']:.4f}"
            )
    except Exception:
        pass
    pos_block = ("— open positions —\n" + "\n".join(pos_lines)) if pos_lines else "— open positions —\n(없음)"

    content = f"{head}\n{pos_block}"

    # 첨부 파일 구성 (존재 파일만)
    files_list = [*chart_files, score_file, perf_file, performance_file]
    pdf_path = None
    if os.getenv("PDF_REPORT_ENABLE", "1") == "1":
        try:
            pdf_path = await generate_pnl_pdf()
            if pdf_path:
                files_list.append(pdf_path)
        except Exception as e:
            log(f"[REPORT] PDF gen warn: {e}")

    files = [discord.File(p) for p in files_list if p and os.path.exists(p)]
    await channel.send(content=content, files=files, silent=True)
# [ANCHOR] OLDSTYLE_REPORT_END


# ⬇️ while True 루프 위에 따로 정의
async def send_timed_reports():
    await client.wait_until_ready()

    while not client.is_closed():
        now = datetime.now()
        if now.hour in [9, 21] and now.minute == 0:
            log("📤 자동 리포트 전송 중...")

            # ===== PnL PDF 자동 생성 & 전송 (신규) =====
            try:
                pdf = await generate_pnl_pdf()
                ch_id = int(os.getenv("PNL_REPORT_CHANNEL_ID", "0"))
                if pdf and ch_id:
                    ch = client.get_channel(ch_id)
                    if ch:
                        await ch.send(content="📊 선물 체결·PnL 요약 리포트", file=discord.File(pdf), silent=True)
            except Exception as e:
                log(f"PNL PDF send warn: {e}")


            # [ANCHOR] SEND_TIMED_REPORTS_LOOP
            # ===== 단일 리포트 채널로 모아 전송 =====
            try:
                report_ch_id = int(os.getenv("PNL_REPORT_CHANNEL_ID", "0"))
                if report_ch_id:
                    ch = client.get_channel(report_ch_id)
                else:
                    ch = None
                if not ch:
                    log("⏭ 리포트 채널(PNL_REPORT_CHANNEL_ID) 미설정 → 자동 리포트 건너뜀")
                else:
                    for symbol in ("ETH/USDT", "BTC/USDT"):
                        for tf in ('15m', '1h', '4h', '1d'):
                            await _send_report_oldstyle(client, ch, symbol, tf)
            except Exception as e:
                log(f"[AUTO_REPORT_WARN] {e}")


            await asyncio.sleep(90)  # 중복 방지

        await asyncio.sleep(60)


# ========== 동기 → 비동기 래퍼 ==========
async def safe_get_ohlcv(symbol, tf, **kwargs):
    return await asyncio.to_thread(get_ohlcv, symbol, tf, **kwargs)

async def safe_add_indicators(df):
    return await asyncio.to_thread(add_indicators, df)

# ========== 비트 이더 구분 헬퍼 ==========
def _get_channel_or_skip(asset: str, tf: str):
    """
    asset: 'ETH' 또는 'BTC'
    tf: '15m'/'1h'/'4h'/'1d'
    반환: discord.Channel 또는 None (없으면 로그 찍고 건너뜀)
    """
    mapping = CHANNEL_IDS if asset == 'ETH' else CHANNEL_BTC
    ch_id = mapping.get(tf)
    if not ch_id or ch_id == 0:
        log(f"⏭ {asset} {tf}: 채널 ID 없음 → skip")
        return None
    ch = client.get_channel(ch_id)
    if ch is None:
        log(f"❌ {asset} {tf}: 채널 객체 없음(ID:{ch_id})")
        return None
    return ch



@client.event
async def on_ready():
    log(f'✅ Logged in as {client.user}')
    if DASHBOARD_ENABLE:
        global _DASH_TASK_RUNNING
        if not _DASH_TASK_RUNNING:
            _DASH_TASK_RUNNING = True
            asyncio.create_task(_dash_loop(client))
        else:
            log("[DASH] loop already running – skip spawn")

    timeframes = ['15m', '1h', '4h', '1d']

    if getattr(client, "startup_done", False):
        return
    client.startup_done = True

    _hydrate_from_disk()
    await _sync_open_state_on_ready()
    asyncio.create_task(init_analysis_tasks())
    
   # ✅ 채널별 시작 메시지 전송 (ETH)
    for tf in timeframes:
        ch_id = CHANNEL_IDS.get(tf)
        if not ch_id or ch_id == 0:
            log(f"⏭ ETH {tf}: 채널 ID 없음 → skip")
            continue
        ch = client.get_channel(ch_id)
        if not ch:
            log(f"❌ ETH {tf}: 채널 객체 없음(ID:{ch_id})")
            continue
        await ch.send(f"🚀 [{tf}] 분석 봇이 시작되었습니다.", silent=True)
    
    # ✅ 채널별 시작 메시지 전송 (BTC)
    for tf in TIMEFRAMES_BTC:
        ch_id = CHANNEL_BTC.get(tf)
        if not ch_id or ch_id == 0:
            log(f"⏭ BTC {tf}: 채널 ID 없음 → skip")
            continue
        ch = client.get_channel(ch_id)
        if not ch:
            log(f"❌ BTC {tf}: 채널 객체 없음(ID:{ch_id})")
            continue
        await ch.send(f"🚀 [BTC {tf}] 분석 봇이 시작되었습니다.", silent=True)


    # ✅ 리포트 자동 전송 태스크 시작
    client.loop.create_task(send_timed_reports())

    while True:
        try:

            gatekeeper_heartbeat(_now_ms())

            # [ANCHOR: MARGIN_Q_SWEEP_LOOP_BEGIN]  << ADD NEW >>
            try:
                if os.getenv("MARGIN_Q_SWEEP_IN_LOOP", "1") == "1":
                    if TRADE_MODE == "futures" and FUT_EXCHANGE:
                        await _apply_all_pending_margin_switches(FUT_EXCHANGE)
            except Exception as e:
                log(f"[FUT] margin queue sweep(loop) warn: {e}")
            # [ANCHOR: MARGIN_Q_SWEEP_LOOP_END]

            # ✅ 루프 1회마다 실시간 가격 스냅샷 활용 (TF 공통)


            for tf in timeframes:
                ch_id = CHANNEL_IDS.get(tf)
                if not ch_id or ch_id == 0:
                    log(f"⏭ ETH {tf}: 채널 ID 없음 → skip")
                    continue
                channel = client.get_channel(ch_id)
                if channel is None:
                    log(f"❌ ETH {tf}: 채널 객체 없음(ID:{ch_id})")
                    continue

                # Prefetch/refresh context periodically
                try:
                    _compute_context(symbol_eth)
                except Exception as e:
                    log(f"[CTX_PREFETCH_ERR] {symbol_eth} {e}")

                df = await safe_get_ohlcv(symbol_eth, tf, limit=300)
                df = await safe_add_indicators(df)
                # === 닫힌 봉 기준값 확보 ===
                c_o, c_h, c_l, c_c = closed_ohlc(df)     # c_c = closed_close
                c_ts = closed_ts(df)                      # 닫힌 캔들 타임스탬프(초)

                # [ANCHOR: PAUSE_PRECHECK]
                now_ms = int(time.time()*1000)
                key_all = PAUSE_UNTIL.get("__ALL__", 0)
                key_tf = PAUSE_UNTIL.get((symbol_eth, tf), 0)
                if now_ms < max(key_all, key_tf):
                    log(f"⏸ {symbol_eth} {tf}: paused until {(max(key_all, key_tf))}")
                    idem_mark(symbol_eth, tf, c_ts)
                    continue

                signal, price, rsi, macd, reasons, score, weights, agree_long, agree_short, weights_detail = calculate_signal(df,tf, symbol_eth)

                # Gate opposite to context when strongly misaligned
                try:
                    st = CTX_STATE.get(symbol_eth)
                    if st:
                        bias = float(st.get("ctx_bias", 0.0))
                        if signal in ("BUY","STRONG BUY","WEAK BUY") and bias < -0.5:
                            log(f"[GATE] {symbol_eth} {tf} {signal} blocked by context {bias:.2f}")
                            continue
                        if signal in ("SELL","STRONG SELL","WEAK SELL") and bias > 0.5:
                            log(f"[GATE] {symbol_eth} {tf} {signal} blocked by context {bias:.2f}")
                            continue
                except Exception as e:
                    log(f"[GATE_CTX_ERR] {symbol_eth} {tf} {e}")

                # [ANCHOR: STORE_EXEC_SCORE]
                try: EXEC_STATE[('score', symbol_eth, tf)] = float(score)
                except: pass

                # === 환율 변환 (USD → KRW) ===
                try:
                    usd_price = float(price)
                    rate = get_usdkrw_rate()
                    krw_price = usd_price * rate if isinstance(rate, (int, float)) and rate > 0 else None
                    price_pair = f"${usd_price:,.2f}/" + (_fmt_krw(krw_price) if krw_price else "₩-")
                except Exception:
                    price_pair = f"${price}/₩-"


                LATEST_WEIGHTS[(symbol_eth, tf)] = dict(weights) if isinstance(weights, dict) else {}
                LATEST_WEIGHTS_DETAIL[(symbol_eth, tf)] = dict(weights_detail) if isinstance(weights_detail, dict) else {}

                if _len(df) == 0:
                    log(f"⏭️ ETH {tf} 생략: 캔들 데이터 없음")
                    continue

                now_str = datetime.now().strftime("%m월 %d일 %H:%M")
                key2 = (symbol_eth, tf)
                previous = previous_signal.get(key2)

                snap = await get_price_snapshot(symbol_eth)
                live_price = snap.get("mid") or snap.get("last")
                display_price = live_price if isinstance(live_price, (int, float)) else c_c
                # [ANCHOR: daily_change_unify_eth]

                daily_change_pct = calc_daily_change_pct(symbol_eth, display_price)

                last_price = float(display_price if 'display_price' in locals() else live_price)
                try:
                    set_last_price(symbol_eth, last_price)
                except Exception:
                    pass

                # === 재시작 보호: 이미 열린 포지션 보호조건 재평가 ===
                _reduced_this_cycle = False
                if TRADE_MODE == "paper":
                    for _side in ("LONG","SHORT"):
                        k = _pp_key(symbol_eth, tf, _side)
                        pos = PAPER_POS.get(k)
                        if not pos: continue
                        side = _side
                        entry = float(pos.get("entry_price") or pos.get("entry") or 0)

                        # 틱 힌트를 1분봉으로 클램프
                        clamped, bar1m = await safe_price_hint(symbol_eth)
                        # 이상치면 무시
                        if _outlier_guard(clamped, bar1m):
                            continue
                        tp_price = pos.get("tp_price")
                        sl_price = pos.get("sl_price")
                        tr_pct_eff = pos.get("eff_tr_pct") if (pos.get("eff_tr_pct") is not None) else pos.get("tr_pct")
                        lev = float(pos.get("lev") or 1.0)
                        op_ts = float(pos.get("opened_ts") or 0)/1000.0
                        ok_exit, reason, trig_px, dbg = _eval_exit(symbol_eth, tf, side, entry, clamped, tp_price, sl_price, tr_pct_eff, key2, lev, op_ts)
                        if ok_exit:

                            exec_px = await _choose_exec_price(symbol_eth, tf, reason, side, float(trig_px), bar1m)

                            info = await _paper_close(symbol_eth, tf, exec_px, reason, side=side)
                            if info:
                                await _notify_trade_exit(
                                    symbol_eth, tf,
                                    side=info["side"],
                                    entry_price=info["entry_price"],
                                    exit_price=exec_px,
                                    reason=reason, mode="paper",
                                    pnl_pct=info.get("pnl_pct"),
                                    qty=info.get("qty"),
                                    pnl_usdt=info.get("net_usdt")
                                )
                                _reduced_this_cycle = True

                            continue
                else:
                    pos = FUT_POS.get(symbol_eth)
                    if pos:
                        side = str(pos.get("side", "")).upper()
                        entry = float(pos.get("entry_price") or pos.get("entry") or 0)

                        clamped, bar1m = await safe_price_hint(symbol_eth)
                        if not _outlier_guard(clamped, bar1m):
                            tp_price = pos.get("tp_price")
                            sl_price = pos.get("sl_price")
                            tr_pct_eff = pos.get("eff_tr_pct") if (pos.get("eff_tr_pct") is not None) else pos.get("tr_pct")
                            lev = float(pos.get("lev") or 1.0)
                            op_ts = float(pos.get("opened_ts") or 0)/1000.0
                            ok_exit, reason, trig_px, dbg = _eval_exit(symbol_eth, tf, side, entry, clamped, tp_price, sl_price, tr_pct_eff, key2, lev, op_ts)
                            if ok_exit:

                                exec_px = await _choose_exec_price(symbol_eth, tf, reason, side, float(trig_px), bar1m)

                                await futures_close_all(symbol_eth, tf, exit_price=exec_px, reason=reason)
                                continue



                # Use 1m bar extremes to update trailing baselines (never raw ticks)
                _bar1m = _fetch_recent_bar_1m(symbol_eth)
                if highest_price.get(key2) is None: highest_price[key2] = float(_bar1m["high"])
                if lowest_price.get(key2)  is None: lowest_price[key2]  = float(_bar1m["low"])
                _ep = entry_data.get(key2)
                if _ep:
                    entry_price, _ = _ep
                    highest_price[key2] = max(float(highest_price.get(key2, 0.0)), float(entry_price))
                    lowest_price[key2]  = min(float(lowest_price.get(key2, 1e30)), float(entry_price))
                highest_price[key2] = max(float(highest_price.get(key2, 0.0)), float(_bar1m["high"]))
                lowest_price[key2]  = min(float(lowest_price.get(key2, 1e30)), float(_bar1m["low"]))


                # === Unified exit evaluation on 1m (ALL TFs) ===
                last_price = float(display_price if 'display_price' in locals() else live_price)
                try: set_last_price(symbol_eth, last_price)
                except Exception: pass
                if TRADE_MODE=='paper':
                    for _side in ("LONG","SHORT"):
                        pos = PAPER_POS.get(_pp_key(symbol_eth, tf, _side))
                        if not pos: continue
                        side = _side
                        entry_price = float(pos.get("entry_price") or pos.get("entry") or 0.0)
                        tp_price = pos.get("tp_price"); sl_price = pos.get("sl_price")
                        tr_pct_eff = pos.get("eff_tr_pct") if (pos.get("eff_tr_pct") is not None) else pos.get("tr_pct")
                        lev = float(pos.get("lev") or 1.0)
                        op_ts = float(pos.get("opened_ts") or 0)/1000.0
                        ok_exit, reason, trig_px, dbg = _eval_exit(symbol_eth, tf, side, entry_price, last_price, tp_price, sl_price, tr_pct_eff, key2, lev, op_ts)
                        log(f"[EXIT_CHECK] {symbol_eth} {tf} {side} -> {ok_exit} reason={reason} {dbg}")
                        if ok_exit:
                            exit_reason = reason
                            _bar = _fetch_recent_bar_1m(symbol_eth)

                            exec_px = await _choose_exec_price(symbol_eth, tf, exit_reason, side, float(trig_px), _bar)
                            info = await _paper_close(symbol_eth, tf, exec_px, exit_reason, side=side)
                            if info:
                                await _notify_trade_exit(symbol_eth, tf, side=info['side'], entry_price=info['entry_price'], exit_price=exec_px, reason=exit_reason, mode='paper', pnl_pct=info.get('pnl_pct'), qty=info.get('qty'), pnl_usdt=info.get('net_usdt'))

                                _reduced_this_cycle = True
                            continue
                else:
                    pos = FUT_POS.get(symbol_eth)
                    if pos:
                        side = str(pos.get("side","" )).upper()
                        entry_price = float(pos.get("entry_price") or pos.get("entry") or 0.0)
                        tp_price = pos.get("tp_price"); sl_price = pos.get("sl_price")
                        tr_pct_eff = pos.get("eff_tr_pct") if (pos.get("eff_tr_pct") is not None) else pos.get("tr_pct")
                        lev = float(pos.get("lev") or 1.0)
                        op_ts = float(pos.get("opened_ts") or 0)/1000.0
                        ok_exit, reason, trig_px, dbg = _eval_exit(symbol_eth, tf, side, entry_price, last_price, tp_price, sl_price, tr_pct_eff, key2, lev, op_ts)
                        log(f"[EXIT_CHECK] {symbol_eth} {tf} {side} -> {ok_exit} reason={reason} {dbg}")
                        if ok_exit:
                            exit_reason = reason
                            _bar = _fetch_recent_bar_1m(symbol_eth)
                            exec_px = await _choose_exec_price(symbol_eth, tf, exit_reason, side, float(trig_px), _bar)
                            await futures_close_all(symbol_eth, tf, exit_price=exec_px, reason=exit_reason)
                            continue

                if _reduced_this_cycle and os.getenv("PAPER_EXIT_REDUCEONLY","1") == "1":
                    log(f"[PAPER] reduce-only guard: skip any adds this cycle for {symbol_eth} {tf}")
                    return




                prev_ts = last_candle_ts_eth.get(tf)
                prev_sco = previous_score.get(tf)
                prev_bkt = previous_bucket.get(tf)
                curr_bkt = _score_bucket(score, CFG)

                trigger_mode = trigger_mode_for(tf)
                log(f"[DEBUG] {symbol_eth} live={live_price} c_close={c_c} display={display_price} tf={tf} tm={trigger_mode}")

                try:
                    set_last_price(symbol_eth, display_price if 'display_price' in locals() else live_price)
                except Exception:
                    pass

                await handle_trigger(symbol_eth, tf, trigger_mode, signal, display_price, c_ts, entry_data)

                if prev_ts == c_ts and prev_bkt == curr_bkt and (prev_sco is not None) and abs(score - prev_sco) < SCORE_DELTA[tf]:
                    log(f"⏭️ ETH {tf} 생략: 같은 캔들 + 신호 유지 + 점수Δ<{SCORE_DELTA[tf]} (Δ={abs(score - prev_sco):.2f})")
                    continue

                # 🔁 기존 신호 유지 시 알림 생략 조건 처리

                # 1. NEUTRAL 생략 조건: 별도 저장된 neutral_info에서 비교
                if signal == 'NEUTRAL':
                    prev_neutral = neutral_info.get(tf)
                    if (
                        prev_neutral
                        and isinstance(prev_neutral, tuple)
                        and len(prev_neutral) == 2
                        and all(isinstance(x, (int, float)) for x in prev_neutral)
                    ):
                        prev_price, prev_score = prev_neutral
                        if abs(price - prev_price) < 5 and score == prev_score:
                            log(f"🔁 NEUTRAL 유지 - 점수 동일 + 가격 유사 → 생략 ({tf})")
                            continue

                # 2. BUY/SELL 생략 조건 (entry_data 사용)
                if signal == previous and entry_data.get(key2):
                    prev_price, _ = entry_data.get(key2)
                    prev_score = previous_score.get(tf, None)
                    if prev_score is not None:
                        if signal == 'BUY':
                            if price > prev_price and score <= prev_score:
                                log(f"🔁 BUY 유지 - 가격 상승 + 점수 약화 → 생략 ({tf})")
                                continue
                            elif price < prev_price and score <= prev_score:
                                log(f"🔁 BUY 유지 - 가격 하락 + 점수 약화 → 생략 ({tf})")
                                continue
                        elif signal == 'SELL':
                            if price < prev_price and score >= prev_score:
                                log(f"🔁 SELL 유지 - 가격 하락 + 점수 약화 → 생략 ({tf})")
                                continue
                            elif price > prev_price and score >= prev_score:
                                log(f"🔁 SELL 유지 - 가격 상승 + 점수 약화 → 생략 ({tf})")
                                continue

                # 진입 정보 저장 (같은 방향일 경우 더 유리한 가격이면 갱신)
                if str(signal).startswith('BUY') or str(signal).startswith('SELL'):
                    update_entry = False
                    prev_entry = entry_data.get(key2)
                    # 진입 정보 저장 (같은 방향일 경우 더 유리한 가격이면 갱신)
                    if str(signal).startswith('BUY') or str(signal).startswith('SELL'):
                        update_entry = False
                        prev_entry = entry_data.get(key2)
                        if previous != signal or prev_entry is None:
                            update_entry = True
                        else:
                            prev_price, _ = prev_entry
                            if signal == 'BUY' and price < prev_price:
                                update_entry = True
                            elif signal == 'SELL' and price > prev_price:
                                update_entry = True
                        if update_entry:
                            entry_data[key2] = (price, now_str)
                            # 🔹 포지션 오픈 시 트레일링 기준점도 진입가로 초기화
                            highest_price[key2] = price if signal == 'BUY' else None
                            lowest_price[key2]  = price if signal == 'SELL' else None
                            trail_peak_roe[key2] = 0.0


                # 수익률 계산
                pnl = None
                if previous in ['BUY', 'SELL'] and signal in ['BUY', 'SELL'] and signal != previous:
                    entry_price, entry_time = entry_data.get(key2)
                    if previous == 'BUY' and signal == 'SELL':
                        pnl = ((price - entry_price) / entry_price) * 100
                    elif previous == 'SELL' and signal == 'BUY':
                        pnl = ((entry_price - price) / entry_price) * 100

                chart_files = save_chart_groups(df, symbol_eth, tf)
                df_struct = None
                struct_info = None
                struct_img = None

                # 캐시 조회(동일 캔들 재사용)
                rows = _load_ohlcv(symbol_eth, tf, limit=400)
                df_struct = _sce_build_df_from_ohlcv(rows) if rows else None
                last_ts = _df_last_ts(df_struct) if df_struct is not None else -1
                cache_ent = _struct_cache_get(symbol_eth, tf, last_ts)
                struct_info = cache_ent.get("ctx") if cache_ent else None
                struct_img  = cache_ent.get("img") if cache_ent else None

                # 구조 오버레이 이미지 생성 및 첨부 (캐시 미스시에만 렌더)
                try:
                    if struct_info is None and df_struct is not None:
                        struct_info = build_struct_context_basic(df_struct, tf)
                    if struct_img is None and df_struct is not None and struct_info is not None:
                        struct_img = render_struct_overlay(symbol_eth, tf, df_struct, struct_info)
                    # 캐시에 기록
                    if df_struct is not None and struct_info is not None:
                        _struct_cache_put(symbol_eth, tf, _df_last_ts(df_struct), struct_info, struct_img)

                    if struct_img:
                        # 오버레이를 첫 번째 첨부로(가시성↑)
                        chart_files = [struct_img] + list(chart_files)

                except Exception as _e:
                    log(f"[STRUCT_IMG_WARN] {symbol_eth} {tf} {type(_e).__name__}: {_e}")

                # ✅ entry_data가 없을 경우 None으로 초기화
                if entry_data.get(key2):
                    entry_price, entry_time = entry_data.get(key2)
                else:
                    entry_price, entry_time = None, None

                # 점수 기록 (최근 4개만)
                # ⛔ 점수기록은 실제 발송 이후에만 (중복 방지)

                # 버킷 기준 억제
                last_ts = get_closed_ts(df)
                curr_bkt  = _score_bucket(score, CFG)
                prev_bkt  = previous_bucket.get(tf)
                prev_scr  = previous_score.get(tf)
                prev_prc  = previous_price.get(tf)

                same_bucket = (curr_bkt == prev_bkt)
                same_score  = (prev_scr is not None and abs(score - prev_scr) < SCORE_DELTA.get(tf, 0.6))
                price_pct   = (abs(price - (prev_prc if prev_prc else price)) / price * 100) if (isinstance(price,(int,float)) and price>0) else 100

                if last_ts == last_candle_ts_eth.get(tf, 0) and same_bucket and same_score and price_pct < 0.5:
                    log(f"[ETH {tf}] 같은 캔들·버킷 동일·점수변화 작음·가격변화 {price_pct:.3f}% → 전송 억제")
                    continue



                msg = None  # ✅ 미리 초기화

                main_msg_pdf, summary_msg_pdf, short_msg = format_signal_message(
                    tf=tf,
                    signal=signal,
                    price=price,
                    pnl=pnl,
                    strength=reasons,
                    df=df,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    score=score,
                    weights=weights,
                    weights_detail=weights_detail,
                    prev_score_value=previous_score.get(tf),
                    agree_long=agree_long,
                    agree_short=agree_short,
                    symbol=symbol_eth,
                    daily_change_pct=daily_change_pct,          # ✅ 추가
                    recent_scores=(
                        list(score_history[tf]) +
                        ([] if (score_history[tf] and round(score,1)==score_history[tf][-1]) else [round(score,1)])
                    ),


                    live_price=display_price,  # reuse ticker for consistent short/long pricing

                    show_risk=False
                )
                struct_block = None
                # 구조 컨텍스트 섹션 프리펜드
                try:
                    struct_block = _render_struct_context_text(symbol_eth, tf, df=df_struct, ctx=struct_info)

                    legend_block = _render_struct_legend(struct_info or {}, tf)
                    main_msg_pdf = f"{struct_block}{('\n'+legend_block) if legend_block else ''}\n\n{main_msg_pdf}"

                except Exception as _e:
                    log(f"[SCE_SECT_WARN] {symbol_eth} {tf} main {type(_e).__name__}: {_e}")

                # 구조 컨텍스트 섹션 프리펜드(요약에도 동일 적용)
                try:

                    # 캐시에 ctx가 있으면 재사용
                    if struct_info is None and df_struct is not None:
                        cache_ent = _struct_cache_get(symbol_eth, tf, _df_last_ts(df_struct))
                        if cache_ent:
                            struct_info = cache_ent.get("ctx")

                    if struct_block is None:
                        struct_block = _render_struct_context_text(symbol_eth, tf, df=df_struct, ctx=struct_info)
                    legend_block = _render_struct_legend(struct_info or {}, tf)
                    summary_msg_pdf = f"{struct_block}{('\n'+legend_block) if legend_block else ''}\n\n{summary_msg_pdf}"

                except Exception as _e:
                    log(f"[SCE_SECT_WARN] {symbol_eth} {tf} summary {type(_e).__name__}: {_e}")
                # 닫힌 캔들만 사용 (iloc[-2]가 닫힌 봉)
                candle_ts = None
                if len(df) >= 2 and 'timestamp' in df:
                    # pandas Timestamp(ns) → ms
                    try:
                        candle_ts = int(df['timestamp'].iloc[-2].value // 1_000_000)
                    except Exception:
                        # 폴백: POSIX seconds → ms
                        candle_ts = int(df['timestamp'].iloc[-2].timestamp() * 1000)


                channel = _get_channel_or_skip('ETH', tf)
                if channel is None:
                    continue

                # 1) 짧은 알림(푸시용) — 첫 전송에서만
                await channel.send(content=short_msg)

                symbol_short = symbol_eth.split('/')[0]
                # 2) 분석 메시지 — 푸시에는 안 뜸
                await _discord_send_chunked(
                    channel,
                    main_msg_pdf,
                    files=[discord.File(p) for p in chart_files if p],
                    silent=True,
                    header_prefix=f"{symbol_short}-{tf}-Analysis"
                )

                # 점수기록: 실제 발송시에만(중복 방지)
                if not score_history[tf] or round(score, 1) != score_history[tf][-1]:
                    score_history[tf].append(round(score, 1))

                # 버킷 상태 업데이트
                previous_bucket[tf] = _score_bucket(score, CFG)


                # 3) 종합해석 메시지 — 길면 잘라서 전송
                await _discord_send_chunked(
                    channel,
                    summary_msg_pdf,
                    silent=True,
                    header_prefix=f"{symbol_short}-{tf}-Summary"
                )

                # NEUTRAL 상태 저장
                # 발송 후 상태 업데이트 보강

                previous_signal[key2] = signal

                previous_score[tf] = score
                previous_price[tf] = price

                # last_ts는 위에서 계산
                try:
                    last_ts  # ensure defined
                except NameError:
                    try:
                        last_ts = get_closed_ts(df)
                        if not last_ts:
                            log("⏭️ 닫힌 캔들 타임스탬프 계산 실패 → 스킵")
                            continue
                    except Exception:
                        last_val = df['timestamp'].iloc[-1]
                        last_ts = int(last_val.timestamp()) if hasattr(last_val, 'timestamp') else 0
                last_candle_ts_eth[tf] = last_ts

                if signal == 'NEUTRAL':
                    neutral_info[tf] = (price, score)
                else:
                    neutral_info[tf] = None


                log_to_csv(symbol_eth, tf, signal, display_price, rsi, macd, pnl,
                        entry_price=entry_price,
                        entry_time=entry_time,
                        score=score,
                        reasons=reasons,
                        weights=weights)

                # 발송 후 업데이트

                previous_signal[key2] = signal

                previous_score[tf] = score

            # ===== BTC 실시간 루프 (1h/4h/1d) =====

            # ✅ 루프 1회마다 실시간 가격 스냅샷 활용 (TF 공통)


            for tf in TIMEFRAMES_BTC:
                ch_id = CHANNEL_BTC.get(tf)
                if not ch_id or ch_id == 0:
                    log(f"⏭ BTC {tf}: 채널 ID 없음 → skip")
                    continue
                channel = client.get_channel(ch_id)
                if channel is None:
                    log(f"❌ BTC {tf}: 채널 객체 없음(ID:{ch_id})")
                    continue

                # Prefetch/refresh context periodically
                try:
                    _compute_context(symbol_btc)
                except Exception as e:
                    log(f"[CTX_PREFETCH_ERR] {symbol_btc} {e}")

                df = await safe_get_ohlcv(symbol_btc, tf, limit=300)
                # 신호 계산 후 즉시 닫힌 봉 값 확정
                c_o, c_h, c_l, c_c = closed_ohlc(df)
                c_ts = closed_ts(df)
                # [ANCHOR: PAUSE_PRECHECK]
                now_ms = int(time.time()*1000)
                key_all = PAUSE_UNTIL.get("__ALL__", 0)
                key_tf = PAUSE_UNTIL.get((symbol_btc, tf), 0)
                if now_ms < max(key_all, key_tf):
                    log(f"⏸ {symbol_btc} {tf}: paused until {(max(key_all, key_tf))}")
                    idem_mark(symbol_btc, tf, c_ts)
                    continue

                df = await safe_add_indicators(df)
                signal, price, rsi, macd, reasons, score, weights, agree_long, agree_short, weights_detail = calculate_signal(df,tf, symbol_btc)

                # Gate opposite to context when strongly misaligned
                try:
                    st = CTX_STATE.get(symbol_btc)
                    if st:
                        bias = float(st.get("ctx_bias", 0.0))
                        if signal in ("BUY","STRONG BUY","WEAK BUY") and bias < -0.5:
                            log(f"[GATE] {symbol_btc} {tf} {signal} blocked by context {bias:.2f}")
                            continue
                        if signal in ("SELL","STRONG SELL","WEAK SELL") and bias > 0.5:
                            log(f"[GATE] {symbol_btc} {tf} {signal} blocked by context {bias:.2f}")
                            continue
                except Exception as e:
                    log(f"[GATE_CTX_ERR] {symbol_btc} {tf} {e}")

                # [ANCHOR: STORE_EXEC_SCORE_BTC]
                try: EXEC_STATE[('score', symbol_btc, tf)] = float(score)
                except: pass

                # === 환율 변환 (USD → KRW) ===
                try:
                    usd_price = float(price)
                    rate = get_usdkrw_rate()
                    krw_price = usd_price * rate if isinstance(rate, (int, float)) and rate > 0 else None
                    price_pair = f"${usd_price:,.2f}/" + (_fmt_krw(krw_price) if krw_price else "₩-")
                except Exception:
                    price_pair = f"${price}/₩-"



                LATEST_WEIGHTS[(symbol_btc, tf)] = dict(weights) if isinstance(weights, dict) else {}
                LATEST_WEIGHTS_DETAIL[(symbol_btc, tf)] = dict(weights_detail) if isinstance(weights_detail, dict) else {}
                

                snap = await get_price_snapshot(symbol_btc)
                live_price = snap.get("chosen") or snap.get("mid") or snap.get("last")
                display_price = live_price if isinstance(live_price, (int, float)) else c_c
                # [ANCHOR: daily_change_unify_btc]

                daily_change_pct = calc_daily_change_pct(symbol_btc, display_price)

                last_price = float(display_price if 'display_price' in locals() else live_price)
                try:
                    set_last_price(symbol_btc, last_price)
                except Exception:
                    pass

                # === 재시작 보호: 이미 열린 포지션 보호조건 재평가 ===
                key2 = (symbol_btc, tf)
                _reduced_this_cycle = False
                if TRADE_MODE == "paper":
                    for _side in ("LONG","SHORT"):
                        k = _pp_key(symbol_btc, tf, _side)
                        pos = PAPER_POS.get(k)
                        if not pos: continue
                        side = _side
                        entry = float(pos.get("entry_price") or pos.get("entry") or 0)

                        clamped, bar1m = await safe_price_hint(symbol_btc)
                        if _outlier_guard(clamped, bar1m):
                            continue
                        tp_price = pos.get("tp_price")
                        sl_price = pos.get("sl_price")
                        tr_pct_eff = pos.get("eff_tr_pct") if (pos.get("eff_tr_pct") is not None) else pos.get("tr_pct")
                        lev = float(pos.get("lev") or 1.0)
                        op_ts = float(pos.get("opened_ts") or 0)/1000.0
                        ok_exit, reason, trig_px, dbg = _eval_exit(symbol_btc, tf, side, entry, clamped, tp_price, sl_price, tr_pct_eff, key2, lev, op_ts)
                        if ok_exit:

                            exec_px = await _choose_exec_price(symbol_btc, tf, reason, side, float(trig_px), bar1m)

                            info = await _paper_close(symbol_btc, tf, exec_px, reason, side=side)
                            if info:
                                await _notify_trade_exit(
                                    symbol_btc, tf,
                                    side=info["side"],
                                    entry_price=info["entry_price"],
                                    exit_price=exec_px,
                                    reason=reason, mode="paper",
                                    pnl_pct=info.get("pnl_pct"), qty=info.get("qty"), pnl_usdt=info.get("net_usdt")

                                )
                                _reduced_this_cycle = True
                            continue
                else:
                    pos = FUT_POS.get(symbol_btc)
                    if pos:
                        side = str(pos.get("side", "")).upper()
                        entry = float(pos.get("entry_price") or pos.get("entry") or 0)

                        clamped, bar1m = await safe_price_hint(symbol_btc)
                        if not _outlier_guard(clamped, bar1m):
                            tp_price = pos.get("tp_price")
                            sl_price = pos.get("sl_price")
                            tr_pct_eff = pos.get("eff_tr_pct") if (pos.get("eff_tr_pct") is not None) else pos.get("tr_pct")
                            lev = float(pos.get("lev") or 1.0)
                            op_ts = float(pos.get("opened_ts") or 0)/1000.0
                            ok_exit, reason, trig_px, dbg = _eval_exit(symbol_btc, tf, side, entry, clamped, tp_price, sl_price, tr_pct_eff, key2, lev, op_ts)
                            if ok_exit:

                                exec_px = await _choose_exec_price(symbol_btc, tf, reason, side, float(trig_px), bar1m)

                                await futures_close_all(symbol_btc, tf, exit_price=exec_px, reason=reason)
                                continue


                # Use 1m bar extremes to update trailing baselines (never raw ticks)
                _bar1m = _fetch_recent_bar_1m(symbol_btc)
                if highest_price.get(key2) is None: highest_price[key2] = float(_bar1m["high"])
                if lowest_price.get(key2)  is None: lowest_price[key2]  = float(_bar1m["low"])
                _ep = entry_data.get(key2)
                if _ep:
                    entry_price, _ = _ep
                    highest_price[key2] = max(float(highest_price.get(key2, 0.0)), float(entry_price))
                    lowest_price[key2]  = min(float(lowest_price.get(key2, 1e30)), float(entry_price))
                highest_price[key2] = max(float(highest_price.get(key2, 0.0)), float(_bar1m["high"]))
                lowest_price[key2]  = min(float(lowest_price.get(key2, 1e30)), float(_bar1m["low"]))

                # 🔽 BTC 심볼+타임프레임별 리포트/이미지 경로 생성
                score_file = plot_score_history(symbol_btc, tf)
                perf_file  = analyze_performance_for(symbol_btc, tf)
                performance_file = generate_performance_stats(tf, symbol=symbol_btc)

                # --- 게이트 (ETH와 동일 로직) ---
                key2 = (symbol_btc, tf)
                previous = previous_signal.get(key2)



                # === Unified exit evaluation on 1m (ALL TFs) ===
                last_price = float(display_price if 'display_price' in locals() else live_price)
                try: set_last_price(symbol_btc, last_price)
                except Exception: pass
                if TRADE_MODE=='paper':
                    for _side in ("LONG","SHORT"):
                        pos = PAPER_POS.get(_pp_key(symbol_btc, tf, _side))
                        if not pos: continue
                        side = _side
                        entry_price = float(pos.get("entry_price") or pos.get("entry") or 0.0)
                        tp_price = pos.get("tp_price"); sl_price = pos.get("sl_price")
                        tr_pct_eff = pos.get("eff_tr_pct") if (pos.get("eff_tr_pct") is not None) else pos.get("tr_pct")
                        lev = float(pos.get("lev") or 1.0)
                        op_ts = float(pos.get("opened_ts") or 0)/1000.0
                        ok_exit, reason, trig_px, dbg = _eval_exit(symbol_btc, tf, side, entry_price, last_price, tp_price, sl_price, tr_pct_eff, key2, lev, op_ts)
                        log(f"[EXIT_CHECK] {symbol_btc} {tf} {side} -> {ok_exit} reason={reason} {dbg}")
                        if ok_exit:
                            exit_reason = reason
                            _bar = _fetch_recent_bar_1m(symbol_btc)
                            exec_px = await _choose_exec_price(symbol_btc, tf, exit_reason, side, float(trig_px), _bar)

                            info = await _paper_close(symbol_btc, tf, exec_px, exit_reason, side=side)
                            if info:
                                await _notify_trade_exit(symbol_btc, tf, side=info['side'], entry_price=info['entry_price'], exit_price=exec_px, reason=exit_reason, mode='paper', pnl_pct=info.get('pnl_pct'), qty=info.get('qty'), pnl_usdt=info.get('net_usdt'))
                                _reduced_this_cycle = True
                            continue
                else:
                    pos = FUT_POS.get(symbol_btc)
                    if pos:
                        side = str(pos.get("side","" )).upper()
                        entry_price = float(pos.get("entry_price") or pos.get("entry") or 0.0)
                        tp_price = pos.get("tp_price"); sl_price = pos.get("sl_price")
                        tr_pct_eff = pos.get("eff_tr_pct") if (pos.get("eff_tr_pct") is not None) else pos.get("tr_pct")
                        lev = float(pos.get("lev") or 1.0)
                        op_ts = float(pos.get("opened_ts") or 0)/1000.0
                        ok_exit, reason, trig_px, dbg = _eval_exit(symbol_btc, tf, side, entry_price, last_price, tp_price, sl_price, tr_pct_eff, key2, lev, op_ts)
                        log(f"[EXIT_CHECK] {symbol_btc} {tf} {side} -> {ok_exit} reason={reason} {dbg}")
                        if ok_exit:
                            exit_reason = reason
                            _bar = _fetch_recent_bar_1m(symbol_btc)

                            exec_px = await _choose_exec_price(symbol_btc, tf, exit_reason, side, float(trig_px), _bar)
                            await futures_close_all(symbol_btc, tf, exit_price=exec_px, reason=exit_reason)
                            continue

                if _reduced_this_cycle and os.getenv("PAPER_EXIT_REDUCEONLY","1") == "1":
                    log(f"[PAPER] reduce-only guard: skip any adds this cycle for {symbol_btc} {tf}")
                    return




                prev_ts_b = last_candle_ts_btc.get(tf)
                prev_sco_b = previous_score_btc.get(tf)
                prev_bkt_b = previous_bucket_btc.get(tf)
                curr_bucket = _score_bucket(score, CFG)

                trigger_mode = trigger_mode_for(tf)
                log(f"[DEBUG] {symbol_btc} live={live_price} c_close={c_c} display={display_price} tf={tf} tm={trigger_mode}")

                try:
                    set_last_price(symbol_btc, display_price if 'display_price' in locals() else live_price)
                except Exception:
                    pass

                await handle_trigger(symbol_btc, tf, trigger_mode, signal, display_price, c_ts, entry_data)

                if prev_ts_b == c_ts and prev_bkt_b == curr_bucket and (prev_sco_b is not None) and abs(score - prev_sco_b) < SCORE_DELTA[tf]:
                    log(f"⏭️ BTC {tf} 생략: 같은 캔들 + 신호 유지 + 점수Δ<{SCORE_DELTA[tf]} (Δ={abs(score - prev_sco_b):.2f})")
                    continue

                # 1) NEUTRAL 생략
                if signal == 'NEUTRAL':
                    prev_neutral = neutral_info_btc.get(tf)
                    if (
                        prev_neutral
                        and isinstance(prev_neutral, tuple)
                        and len(prev_neutral) == 2
                        and all(isinstance(x, (int, float)) for x in prev_neutral)
                    ):
                        prev_price, prev_score = prev_neutral
                        if abs(price - prev_price) < 5 and score == prev_score:
                            log(f"🔁 NEUTRAL 유지 - 점수 동일 + 가격 유사 → 생략 (BTC {tf})")
                            continue

                # 2) BUY/SELL 생략 (entry 기준)

                # === (BTC) 진입 정보 저장 ===
                now_str_btc = datetime.now().strftime("%m월 %d일 %H:%M")
                if str(signal).startswith('BUY') or str(signal).startswith('SELL'):
                    update_entry = False
                    prev_entry = entry_data.get(key2)
                    if previous != signal or prev_entry is None:
                        update_entry = True
                    else:
                        prev_price, _ = prev_entry
                        if str(signal).startswith('BUY') and price < prev_price:
                            update_entry = True
                        elif str(signal).startswith('SELL') and price > prev_price:
                            update_entry = True
                    if update_entry:
                        entry_data[key2] = (price, now_str_btc)
                        highest_price[key2] = price
                        lowest_price[key2]  = price
                        trail_peak_roe[key2] = 0.0

                prev_entry2 = entry_data.get(key2)
                if signal == previous and prev_entry2:
                    prev_price, _ = prev_entry2
                    prev_score = previous_score_btc.get(tf, None)
                    if prev_score is not None:
                        if signal == 'BUY':
                            if price > prev_price and score <= prev_score:
                                log(f"🔁 BUY 유지 - 가격 상승 + 점수 약화 → 생략 (BTC {tf})")
                                continue
                            elif price < prev_price and score <= prev_score:
                                log(f"🔁 BUY 유지 - 가격 하락 + 점수 약화 → 생략 (BTC {tf})")
                                continue
                        elif signal == 'SELL':
                            if price < prev_price and score >= prev_score:
                                log(f"🔁 SELL 유지 - 가격 하락 + 점수 약화 → 생략 (BTC {tf})")
                                continue
                            elif price > prev_price and score >= prev_score:
                                log(f"🔁 SELL 유지 - 가격 상승 + 점수 약화 → 생략 (BTC {tf})")
                                continue

                curr_bucket = _score_bucket(score, CFG)
                ok_to_send, why = _should_notify(
                    tf, score, price, curr_bucket, c_ts,
                    last_sent_ts_btc, last_sent_bucket_btc, last_sent_score_btc, last_sent_price_btc
                )
                if not ok_to_send:
                    log(f"🔕 BTC {tf} 억제: {why}")
                    previous_bucket_btc[tf] = curr_bucket
                    previous_score_btc[tf]  = score
                    previous_price_btc[tf]  = float(price) if isinstance(price,(int,float)) else None
                    last_candle_ts_btc[tf]  = c_ts
                    continue

                _epb = entry_data.get(key2)
                _entry_price = _epb[0] if _epb else None
                _entry_time  = _epb[1] if _epb else None


                log_to_csv(
                    symbol_btc,
                    tf,
                    signal,
                    display_price,
                    rsi,
                    macd,
                    pnl=None,
                    entry_price=_entry_price,
                    entry_time=_entry_time,
                    score=score,
                    reasons=reasons,
                    weights=weights
                )

                main_msg_pdf, summary_msg_pdf, short_msg = format_signal_message(
                    tf=tf, signal=signal, price=price, pnl=None, strength=reasons, df=df,
                    score=score, weights=weights, weights_detail=weights_detail, prev_score_value=previous_score_btc.get(tf),
                    agree_long=agree_long, agree_short=agree_short, daily_change_pct=daily_change_pct,
                    symbol=symbol_btc,
                    entry_time=_entry_time,
                    entry_price=_entry_price,
                    recent_scores=list(score_history_btc.setdefault(tf, deque(maxlen=4))),
                      live_price=display_price,  # reuse ticker for consistent short/long pricing
                      show_risk=False
                  )


                chart_files = save_chart_groups(df, symbol_btc, tf)
                df_struct = None
                struct_info = None
                struct_img = None
                # 캐시 조회(동일 캔들 재사용)
                rows = _load_ohlcv(symbol_btc, tf, limit=400)
                df_struct = _sce_build_df_from_ohlcv(rows) if rows else None
                last_ts = _df_last_ts(df_struct) if df_struct is not None else -1
                cache_ent = _struct_cache_get(symbol_btc, tf, last_ts)
                struct_info = cache_ent.get("ctx") if cache_ent else None
                struct_img  = cache_ent.get("img") if cache_ent else None

                # 구조 오버레이 이미지 생성 및 첨부 (캐시 미스시에만 렌더)
                try:
                    if struct_info is None and df_struct is not None:
                        struct_info = build_struct_context_basic(df_struct, tf)
                    if struct_img is None and df_struct is not None and struct_info is not None:
                        struct_img = render_struct_overlay(symbol_btc, tf, df_struct, struct_info)
                    if df_struct is not None and struct_info is not None:
                        _struct_cache_put(symbol_btc, tf, _df_last_ts(df_struct), struct_info, struct_img)
                    if struct_img:
                        # 오버레이를 첫 번째 첨부로(가시성↑)
                        chart_files = [struct_img] + list(chart_files)
                except Exception as _e:
                    log(f"[STRUCT_IMG_WARN] {symbol_btc} {tf} {type(_e).__name__}: {_e}")

                struct_block = None
                # 구조 컨텍스트 섹션 프리펜드
                try:
                    struct_block = _render_struct_context_text(symbol_btc, tf, df=df_struct, ctx=struct_info)
                    legend_block = _render_struct_legend(struct_info or {}, tf)
                    main_msg_pdf = f"{struct_block}{('\n'+legend_block) if legend_block else ''}\n\n{main_msg_pdf}"
                except Exception as _e:
                    log(f"[SCE_SECT_WARN] {symbol_btc} {tf} main {type(_e).__name__}: {_e}")


                # 구조 컨텍스트 섹션 프리펜드(요약에도 동일 적용)
                try:
                    # 캐시에 ctx가 있으면 재사용
                    if struct_info is None and df_struct is not None:
                        cache_ent = _struct_cache_get(symbol_btc, tf, _df_last_ts(df_struct))
                        if cache_ent:
                            struct_info = cache_ent.get("ctx")
                    if struct_block is None:
                        struct_block = _render_struct_context_text(symbol_btc, tf, df=df_struct, ctx=struct_info)

                    legend_block = _render_struct_legend(struct_info or {}, tf)
                    summary_msg_pdf = f"{struct_block}{('\n'+legend_block) if legend_block else ''}\n\n{summary_msg_pdf}"

                except Exception as _e:
                    log(f"[SCE_SECT_WARN] {symbol_btc} {tf} summary {type(_e).__name__}: {_e}")

                channel = _get_channel_or_skip('BTC', tf)
                if channel is None:
                    continue


                # 1) 짧은 알림(푸시용)
                await channel.send(content=short_msg)

                symbol_short = symbol_btc.split('/')[0]
                # 2) 분석 메시지
                await _discord_send_chunked(
                    channel,
                    main_msg_pdf,
                    files=[discord.File(p) for p in chart_files if p],
                    silent=True,
                    header_prefix=f"{symbol_short}-{tf}-Analysis"
                )

                # 3) 종합해석 메시지
                await _discord_send_chunked(
                    channel,
                    summary_msg_pdf,
                    silent=True,
                    header_prefix=f"{symbol_short}-{tf}-Summary"
                )

                # 점수기록: 실제 발송시에만
                hist = score_history_btc.setdefault(tf, deque(maxlen=4))
                if not hist or round(score,1) != hist[-1]:
                    hist.append(round(score,1))

                # 발송 기록 갱신
                last_sent_ts_btc[tf]     = c_ts
                last_sent_bucket_btc[tf] = curr_bucket
                last_sent_score_btc[tf]  = score
                last_sent_price_btc[tf]  = float(price) if isinstance(price,(int,float)) else None

                # 버킷 상태 업데이트
                previous_bucket_btc[tf] = curr_bucket

                # 상태 업데이트
                if signal == 'NEUTRAL':
                    neutral_info_btc[tf] = (price, score)
                else:
                    neutral_info_btc[tf] = None

                # 상태 업데이트(손절/익절 분기에서 이미 continue 되므로 여기선 순수 신호 상태만 기록)
                previous_signal[(symbol_btc, tf)] = signal
                previous_score_btc[tf]  = score
                previous_bucket_btc[tf] = _score_bucket(score, CFG)
                previous_price_btc[tf]  = float(price) if isinstance(price,(int,float)) else c_c
                last_candle_ts_btc[tf]  = c_ts
        except Exception as e:
            log(f"⚠️ 오류 발생: {e}")

        # [ANCHOR: DAILY_RESUME_11KST]
        try:
            now_kst = datetime.now(KST)
            ymd = now_kst.strftime("%Y%m%d")
            global _LAST_RESUME_YMD
            if now_kst.hour == DAILY_RESUME_HOUR_KST and _LAST_RESUME_YMD != ymd:
                PAUSE_UNTIL.clear()
                _LAST_RESUME_YMD = ymd
                logging.info("[PAUSE] auto-resume all at 11:00 KST")
        except Exception as _:
            pass

        await asyncio.sleep(90)

# ========== 초기화 태스크 ==========
async def init_analysis_tasks():
    for symbol in ['ETH/USDT', 'BTC/USDT']:
        for tf in TIMEFRAMES_BTC:
            try:
                df = await safe_get_ohlcv(symbol, tf, limit=300)
                df = await safe_add_indicators(df)
                # 초기 리포트 전송 또는 분석 로직
            except Exception as e:
                log(f"초기화 오류 {symbol} {tf}: {e}")



async def _set_pause(symbol: str | None, tf: str | None, minutes: int | None):
    until_ms = 2**62 if minutes is None else (int(time.time()*1000) + minutes*60*1000)
    if not symbol or symbol.upper() == "ALL":
        PAUSE_UNTIL["__ALL__"] = until_ms
    else:
        key2 = (symbol.upper(), tf) if tf and tf.upper() != "ALL" else None
        if key2:
            PAUSE_UNTIL[key2] = until_ms
        else:
            for tfx in ("15m", "1h", "4h", "1d"):
                PAUSE_UNTIL[(symbol.upper(), tfx)] = until_ms

async def cmd_pause_all():
    await _set_pause("ALL", "ALL", None)


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    content = message.content.strip()
    parts = content.split()
    # using global LAST_PRICE cache defined at module scope

    # [ANCHOR: CMD_SET_GET_SAVEENV]
    if content.startswith(('!set ','!변경')):
        try:
            payload = content[5:].strip()
            if "=" in payload:
                k, v = payload.split("=", 1)
            else:
                parts2 = payload.split(None, 1)
                k, v = parts2[0], (parts2[1] if len(parts2) > 1 else "")
            cfg_set(k.strip(), v.strip())
            await message.channel.send(f"✅ set {k.strip()} = ```{v.strip()}```")
            _reload_runtime_parsed_maps()
        except Exception as e:
            await message.channel.send(f"⚠️ set error: {e}")
        return

    if content.startswith(('!get ','!값')):
        k = content[5:].strip()
        eff = cfg_get(k)
        ov = RUNTIME_CFG.get(k, None)
        await message.channel.send(f"🔎 {k}\n• effective: ```{eff}```\n• overlay: ```{ov}```")
        return

    if content.startswith(('!saveenv','!저장')):
        try:
            path = cfg_get("KEY_ENV_PATH", "key.env")
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            kv = dict([(ln.split("=",1)[0], ln) for ln in lines if "=" in ln and not ln.strip().startswith("#")])
            for k, v in RUNTIME_CFG.items():
                new = f"{k}={v}"
                if k in kv:
                    idx = lines.index(kv[k])
                    lines[idx] = new
                else:
                    lines.append(new)
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            await message.channel.send(f"💾 saved overlay to {path} ({len(RUNTIME_CFG)} keys)")
        except Exception as e:
            await message.channel.send(f"⚠️ saveenv error: {e}")
        return

    # [ANCHOR: CMD_PAUSE_RESUME]
    if content.startswith(('!pause','!정지')):
        try:
            _, *args = content.split()
            sym = args[0] if len(args) > 0 else "ALL"
            tfx = args[1] if len(args) > 1 else "ALL"
            mins = int(args[2]) if len(args) > 2 else None
            await _set_pause(sym, tfx, mins)
            await message.channel.send(f"⏸ paused {sym} {tfx} {'indefinitely' if mins is None else f'{mins}m'}")
        except Exception as e:
            await message.channel.send(f"⚠️ pause error: {e}")
        return

    if content.startswith(('!resume','!시작')):
        try:
            _, *args = content.split()
            sym = args[0] if len(args) > 0 else "ALL"
            tfx = args[1] if len(args) > 1 else "ALL"
            if sym.upper() == "ALL":
                PAUSE_UNTIL.clear()
            else:
                if tfx.upper() == "ALL":
                    for k in list(PAUSE_UNTIL.keys()):
                        if isinstance(k, tuple) and k[0] == sym.upper():
                            PAUSE_UNTIL.pop(k, None)
                else:
                    PAUSE_UNTIL.pop((sym.upper(), tfx), None)
            await message.channel.send(f"▶ resumed {sym} {tfx}")
        except Exception as e:
            await message.channel.send(f"⚠️ resume error: {e}")
        return

    # [ANCHOR: DISCORD_CMD_CAP_RESET]
    if content.lower().startswith(('!cap reset','!자본리셋')) and CAP_RESET_ALLOW:
        try:
            parts = content.split()
            amount = None
            if len(parts) >= 3:
                amount = float(parts[2])
                if amount < CAP_RESET_MIN:
                    await message.channel.send(f"⚠️ 최소 리셋 금액({CAP_RESET_MIN}) 미만입니다.")
                    return
            else:
                amount = float(CAPITAL_BASE)

            before = capital_get()
            global _CAPITAL_RT
            _CAPITAL_RT = float(amount)
            await capital_save_state()
            if CAP_RESET_LOG:
                await capital_ledger_write("RESET", capital_after=f"{capital_get():.8f}")

            msg = f"🔄 CAPITAL RESET: {before:,.2f} → {capital_get():,.2f}"
            await message.channel.send(msg)
        except Exception as e:
            await message.channel.send(f"⚠️ cap reset 실패: {e}")
        return

    if content.startswith(('!panic','!패닉')):
        _panic_on()
        if PANIC_CANCEL_OPEN:
            await cancel_all_open_orders()
        if PANIC_FORCE_CLOSE:
            await close_all_positions(reason="PANIC")
        await message.channel.send("⛔ panic ON (entries blocked)")
        return

    if content.startswith(('!unpanic','!패닉해제','!패닉off','!패닉해제해')):
        _panic_off()
        await message.channel.send("✅ panic OFF")
        return

    if content.startswith(('!limits','!제한')):
        st = _daily_state_load()
        await message.channel.send(
            f"limits — realized: {st.get('realized_usdt',0):.2f} USDT ({st.get('realized_pct',0):.2f}%), "
            f"closes: {st.get('closes',0)}, action={DAILY_LIMIT_ACTION}"
        )
        return

    if content.startswith(('!limit set','!제한 설정')):
        try:
            _,_,k,v = content.split()
            if k=='loss_usdt':
                os.environ['DAILY_MAX_LOSS_USDT'] = str(float(v))
            elif k=='loss_pct':
                os.environ['DAILY_MAX_LOSS_PCT'] = str(float(v))
            elif k=='closes':
                os.environ['DAILY_MAX_CLOSES'] = str(int(v))
            await message.channel.send(f"ok set {k}={v} (runtime)")
        except Exception as e:
            await message.channel.send(f"bad args: {e}")
        return

    # [ANCHOR: CONFIG_DUMP_HANDLER]
    if content.startswith(("!config","!설정")):
        try:
            dump = _build_config_dump_text()
            await _send_long_text_or_file(message.channel, dump, CONFIG_FILENAME)
        except Exception as e:
            await message.channel.send(f"⚠️ config dump 실패: {e}")
        return

    # [ANCHOR: CMD_CLOSE_CLOSEALL]
    if content.startswith(("!closeall","!모두청산","!전부청산","!포지션정리")):
        try:
            n = 0
            # PAPER_POS key is now "SYMBOL|TF|SIDE"
            for key, pos in list(PAPER_POS.items()):
                try:
                    sym, tf, side = key.split("|", 2)
                except Exception:
                    continue

                fallback = float(pos.get("entry_price", 0.0))
                await _paper_close(sym, tf, get_last_price(sym, fallback), "MANUAL", side=side)
                n += 1
            for tfk, sym in list(FUT_POS_TF.items()):
                await futures_close_all(sym, tfk, reason="MANUAL")
                n += 1
            # optional: clear all idempotence marks after mass close
            if CLEAR_IDEMP_ON_CLOSEALL:
                try: idem_clear_all()
                except Exception: pass
            await message.channel.send(f"🟢 전체 청산 완료 · {n}건")
        except Exception as e:
            await message.channel.send(f"⚠️ closeall error: {e}")
        return

    if content.startswith(("!close ","!청산 ")):
        try:
            parts = content.split()
            # allow: !close SYMBOL TF [SIDE]
            _, sym, tfx, *opt = parts
            side = (opt[0].upper() if opt else None)
            if TRADE_MODE == "paper":
                symU = sym.upper()
                if not side:
                    both = [s for s in ("LONG","SHORT") if PAPER_POS.get(_pp_key(symU, tfx, s))]
                    if len(both) > 1:
                        await message.channel.send(f"⚠️ need side (LONG|SHORT) for {symU} {tfx}")
                        return
                    side = (both[0] if both else None)
                await _paper_close(symU, tfx, get_last_price(symU, 0.0), "MANUAL", side=side)
            else:
                await futures_close_symbol_tf(sym.upper(), tfx)

            await message.channel.send(f"🟢 청산 완료: {sym.upper()} {tfx}" + (f" {side}" if side else ""))

        except Exception as e:
            await message.channel.send(f"⚠️ close error: {e}")
        return

    # [ANCHOR: CMD_RISK_SET]
    if content.startswith(("!risk ","!리스크 ")):
        try:
            _, sym, tfx, *rest = content.split()
            sym = sym.upper()
            args = " ".join(rest)
            def _parse_kv(s, k):
                m = re.search(rf"{k}\s*=\s*([0-9]+(\.[0-9]+)?)", s)
                return float(m.group(1)) if m else None
            tp = _parse_kv(args, "tp"); sl = _parse_kv(args, "sl"); tr = _parse_kv(args, "tr")
            # allow side=LONG|SHORT in args
            mside = re.search(r"side\s*=\s*(LONG|SHORT)", args, re.IGNORECASE)
            side = (mside.group(1).upper() if mside else None)
            if TRADE_MODE == "paper":
                cand = []
                for s in ("LONG","SHORT"):
                    k = _pp_key(sym, tfx, s)
                    if k in PAPER_POS: cand.append(k)
                if not side and len(cand) > 1:
                    await message.channel.send(f"⚠️ need side=LONG|SHORT for {sym} {tfx}")
                    return
                key = _pp_key(sym, tfx, side or ("LONG" if cand and cand[0].endswith("|LONG") else "SHORT"))
                if key in PAPER_POS:
                    pos = PAPER_POS[key]
                    if tp is not None: pos["tp_pct"] = tp
                    if sl is not None: pos["sl_pct"] = sl
                    if tr is not None: pos["tr_pct"] = tr
                    avg = float(pos.get("entry_price") or pos.get("entry") or 0.0)
                    lev = float(pos.get("lev") or 1.0)
                    eff_tp_pct, eff_sl_pct, eff_tr_pct, _ = _eff_risk_pcts(pos.get("tp_pct"), pos.get("sl_pct"), pos.get("tr_pct"), lev)
                    pos["eff_tp_pct"] = eff_tp_pct; pos["eff_sl_pct"] = eff_sl_pct; pos["eff_tr_pct"] = eff_tr_pct
                    if str(pos.get("side", "")).upper()=="LONG":
                        pos["tp_price"] = (avg*(1+(eff_tp_pct or 0)/100)) if eff_tp_pct else None
                        pos["sl_price"] = (avg*(1-(eff_sl_pct or 0)/100)) if eff_sl_pct else None
                    else:
                        pos["tp_price"] = (avg*(1-(eff_tp_pct or 0)/100)) if eff_tp_pct else None
                        pos["sl_price"] = (avg*(1+(eff_sl_pct or 0)/100)) if eff_sl_pct else None
                    PAPER_POS[key] = pos; _save_json(PAPER_POS_FILE, PAPER_POS)
            elif TRADE_MODE != "paper" and sym in FUT_POS:
                pos = FUT_POS[sym]
                if tp is not None: pos["tp_pct"] = tp
                if sl is not None: pos["sl_pct"] = sl
                if tr is not None: pos["tr_pct"] = tr
                avg = float(pos.get("entry", 0.0))
                lev = float(pos.get("lev") or 1.0)
                eff_tp_pct, eff_sl_pct, eff_tr_pct, _ = _eff_risk_pcts(pos.get("tp_pct"), pos.get("sl_pct"), pos.get("tr_pct"), lev)
                pos["eff_tp_pct"] = eff_tp_pct; pos["eff_sl_pct"] = eff_sl_pct; pos["eff_tr_pct"] = eff_tr_pct
                if str(pos.get("side","" )).upper()=="LONG":
                    pos["tp_price"] = (avg*(1+(eff_tp_pct or 0)/100)) if eff_tp_pct else None
                    pos["sl_price"] = (avg*(1-(eff_sl_pct or 0)/100)) if eff_sl_pct else None
                else:
                    pos["tp_price"] = (avg*(1-(eff_tp_pct or 0)/100)) if eff_tp_pct else None
                    pos["sl_price"] = (avg*(1+(eff_sl_pct or 0)/100)) if eff_sl_pct else None
                FUT_POS[sym] = pos; _save_json(OPEN_POS_FILE, FUT_POS)
                await _cancel_symbol_conditional_orders(sym)

                await _ensure_tp_sl_trailing(sym, tfx, get_last_price(sym, avg), pos.get("side", "LONG"))

            await message.channel.send(f"⚙️ risk updated {sym} {tfx} (tp={tp}, sl={sl}, tr={tr})")
        except Exception as e:
            await message.channel.send(f"⚠️ risk error: {e}")
        return

    if content.startswith(("!help","!도움말","!명령어")):
        lines = [
            "• 설정: !set(!변경) KEY=VALUE / !get(값) KEY / !saveenv(!저장) / !config(!설정)",
            "• 일시정지/재개: !pause(!정지) / !resume(!시작)",
            "• 청산: !close(!청산) SYMBOL TF [SIDE?] / !closeall(!모두청산|!전부청산)",
            "• 총자본: !cap reset(!자본리셋)"
            "• 리스크설정: !risk(!리스크) SYMBOL TF tp=5 sl=2.5 tr=1.8 [side=LONG|SHORT]",
            "• 제한/패닉: !limits(!제한) / !limit set(!제한 설정) / !panic(!패닉) / !unpanic(!패닉해제)",
            "• 리포트/상태: !report(!리포트) / !health / !상태 / !분석",
        ]
        await message.channel.send("\n".join(lines))
        return

    # [ANCHOR: DIAG_CMD_HEALTH]
    if content.startswith("!health"):
        try:
            import time
            tfs = ["15m","1h","4h","1d"]
            lines = [f"**HEALTH ({time.strftime('%H:%M:%S')})**"]
            def _remain_sec(ms):
                try:
                    return max(0, int((int(ms or 0) - _now_ms()) / 1000))
                except Exception:
                    return 0

            for tf in tfs:
                occ = (PAPER_POS_TF.get(tf) if 'PAPER_POS_TF' in globals() else None) or \
                      (FUT_POS_TF.get(tf) if 'FUT_POS_TF' in globals() else None)
                gate = FRAME_GATE.get(tf, {})
                cd   = COOLDOWN_UNTIL.get(tf, 0)

                obs_left = _remain_sec(gate.get("obs_until_ms"))
                tgt_left = _remain_sec(gate.get("target_until_ms"))
                if GK_DEBUG and len(gate.get("cand", [])) == 1 and obs_left == 0 and tgt_left == 0 and not gate.get("winner"):
                    log(f"[GK_WARN] {tf} single-candidate but not released — check suppression order")
                lines.append(
                    f"· {tf}: occ={occ or '-'} | cooldown={(max(0,int(cd-time.time())) if cd else 0)}s "
                    f"| gate(ts={gate.get('candle_ts_ms','-')}, cand={len(gate.get('cand',[]))}, winner={gate.get('winner','-')}, "
                    f"obs={obs_left}s, tgt={tgt_left}s)"
                )
            await message.channel.send("\n".join(lines))
        except Exception as e:
            await message.channel.send(f"health error: {e}")
        return

    # ===== PnL 리포트 생성 =====
    if (parts and parts[0] in ("!리포트","!report")) and (len(parts) == 1):
        try:
            path = await generate_pnl_pdf()
            if not path:
                await message.channel.send("PnL 기록이 없습니다.")
            else:
                await message.channel.send(file=discord.File(path))
        except Exception as e:
            await message.channel.send(f"리포트 생성 오류: {e}")
        return

    # ===== 기존 !상태 / !분석 =====
    if message.content.startswith("!상태") or message.content.startswith("!분석"):
        try:
            parts = message.content.split()
            symbol, tf = parse_symbol_tf(parts, default_symbol='ETH/USDT', default_tf='1h')
        except ValueError as ve:
            await message.channel.send(f"❌ {ve}")
            return

        df = get_ohlcv(symbol, tf, limit=300)
        df = add_indicators(df)

        df_1d = get_ohlcv(symbol, '1d', limit=300)
        signal, price, rsi, macd, reasons, score, weights, agree_long, agree_short, weights_detail = calculate_signal(df,tf, symbol)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        snap = await get_price_snapshot(symbol)
        live_price_val = snap.get("chosen") or snap.get("mid") or snap.get("last")
        display_price  = live_price_val if isinstance(live_price_val, (int, float)) else price

        main_msg_pdf, summary_msg_pdf, short_msg = format_signal_message(
            tf=tf,
            signal=signal,
            price=price,
            pnl=None,
            strength=reasons,
            df=df,
            entry_time=None,
            entry_price=None,
            score=score,
            weights=weights,
            weights_detail=weights_detail,
            prev_score_value=None,
            agree_long=agree_long,
            agree_short=agree_short,
            symbol=symbol,
            live_price=display_price,
            show_risk=False
            )
        

        chart_files = save_chart_groups(df, symbol, tf)  # 분할 4장
        await _discord_send_chunked(
            message.channel,
            main_msg_pdf,
            files=[discord.File(p) for p in chart_files if p],
            silent=True,
            header_prefix=f"{symbol.split('/')[0]}-{tf}-Analysis"
        )

        await _discord_send_chunked(
            message.channel,
            summary_msg_pdf,
            silent=True,
            header_prefix=f"{symbol.split('/')[0]}-{tf}-Summary"
        )


    # ===== PDF 리포트 =====
    elif message.content.startswith("!리포트"):
        parts = message.content.split()

        # 사용법: !리포트 ETH 1d  |  !리포트 BTC 15m
        if len(parts) >= 3:
            sym_in = parts[1].upper()
            tf     = parts[2].lower()
            symbol = "ETH/USDT" if sym_in in ("ETH","ETH/USDT") else ("BTC/USDT" if sym_in in ("BTC","BTC/USDT") else sym_in)
            await _send_report_oldstyle(client, message.channel, symbol, tf)
        else:
            # 인자 없으면 PnL PDF만(과거 동작 유지)

            try:
                pdf = await generate_pnl_pdf()
                if pdf:
                    await message.channel.send(content="📊 선물 체결·PnL 요약 리포트", file=discord.File(pdf), silent=True)
            except Exception as e:
                await message.channel.send(f"리포트 생성 오류: {e}")

    # ===== 신호 이력 조회 =====
    elif message.content.startswith("!이력"):
        tf = parts[1] if len(parts) > 1 else "1h"
        import csv, glob
        rows = []

        # 1) 우선 통합 로그가 있으면 그걸 사용
        if os.path.exists("logs/signals.csv"):
            try:
                with open("logs/signals.csv", "r", encoding="utf-8") as f:
                    rdr = csv.DictReader(f)
                    for row in rdr:
                        if row.get("timeframe") == tf:
                            rows.append(row)
            except Exception as e:
                await message.channel.send(f"❌ 로그 읽기 오류: {e}")
                return
        else:
            # 2) 통합 로그가 없으면 심볼별 파일을 합쳐서 사용
            candidates = glob.glob(f"logs/signals_*_{tf}.csv")
            for fp in candidates:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        for line in f.readlines():
                            rows.append({"raw": line.strip()})
                except Exception:
                    pass

        if not rows:
            await message.channel.send("❌ 해당 타임프레임의 이력이 없습니다.")
        else:
            if "raw" in rows[0]:
                lines = [r["raw"] for r in rows][-10:]
            else:
                lines = [
                    f"{r.get('datetime')},{r.get('timeframe')},{r.get('signal')},{r.get('price')},{r.get('rsi')},{r.get('macd')},{r.get('entry_price')},{r.get('entry_time')},{r.get('pnl')},{r.get('score')},{r.get('reasons')},{r.get('weights')}"
                    for r in rows
                ][-10:]
            history_msg = "📜 최근 신호 이력\n" + "\n".join(lines)
            await message.channel.send(f"```{history_msg}```")


    # ===== 지표 요약 =====
    elif message.content.startswith("!지표"):
        tf = parts[1] if len(parts) > 1 else "1h"
        symbol = 'ETH/USDT'  # 기본 심볼
        df = get_ohlcv(symbol, tf)
        df = add_indicators(df)
        signal, price, rsi, macd, reasons, score, weights, agree_long, agree_short, weights_detail = calculate_signal(df, tf, symbol)

        summary = "\n".join(reasons)
        await message.channel.send(f"📊 {tf} 주요 지표 상태:\n```{summary}```")

    # ===== 설정 조회 =====
    elif message.content.startswith("!설정"):
        cfg_text = "\n".join([f"{k}: {v}" for k, v in CFG.items()])
        await message.channel.send(f"⚙ 현재 설정값:\n```{cfg_text}```")

    # fallthrough to command router
    router = getattr(client, "process_commands", None)
    if callable(router):
        await router(message)

# (NEW) reparse helper
def _reload_runtime_parsed_maps():
    global _STRENGTH_W, _STRENGTH_BUCKETS, _MTF_F, _FULL_ON_ALL, _DEBUG_ALLOC
    global SCALE_ENABLE, SCALE_MAX_LEGS, SCALE_UP_DELTA, SCALE_DN_DELTA, SCALE_STEP_PCT, SCALE_REDUCE_PCT
    global SCALE_MIN_ADD_NOTIONAL, SCALE_REALLOCATE_BRACKETS, SCALE_LOG
    global _SLIP_BY_SYMBOL, _TP_BY_SYMBOL, _SL_BY_SYMBOL, _TRAIL_BY_SYMBOL
    try:
        _STRENGTH_W = _parse_kv_map(cfg_get("STRENGTH_WEIGHTS", ""), to_float=True)
        if not _STRENGTH_W:
            _STRENGTH_W = {
                'STRONG_BUY':0.80, 'BUY':0.55, 'WEAK_BUY':0.30,
                'STRONG_SELL':0.80, 'SELL':0.55, 'WEAK_SELL':0.30
            }
        _BUCKET = cfg_get("STRENGTH_BUCKETS", "80:STRONG,60:BASE,0:WEAK")
        _STRENGTH_BUCKETS = sorted(
            [(int(k), v.upper()) for k, v in _parse_kv_map(_BUCKET, to_float=False, upper_key=False).items()],
            key=lambda x: -x[0]
        )
        _MTF_F = _parse_kv_map(cfg_get("MTF_FACTORS", "ALL_ALIGN:1.00,MAJ_ALIGN:1.25,SOME_ALIGN:1.10,NO_ALIGN:0.85,MAJ_OPPOSE:0.60,ALL_OPPOSE:0.40"), to_float=True)
        _FULL_ON_ALL = (cfg_get("FULL_ALLOC_ON_ALL_ALIGN", "1") == "1")
        _DEBUG_ALLOC = (cfg_get("DEBUG_ALLOC_LOG", "0") == "1")

        SCALE_ENABLE = (cfg_get("SCALE_ENABLE", "1") == "1")
        SCALE_MAX_LEGS = int(cfg_get("SCALE_MAX_LEGS", "3"))
        SCALE_UP_DELTA = _parse_pct_map2(cfg_get("SCALE_UP_SCORE_DELTA", "15m:0.5,1h:0.6,4h:0.6,1d:0.7"))
        SCALE_DN_DELTA = _parse_pct_map2(cfg_get("SCALE_DOWN_SCORE_DELTA", "15m:0.6,1h:0.7,4h:0.8,1d:1.0"))
        SCALE_STEP_PCT = _parse_pct_map2(cfg_get("SCALE_STEP_PCT", "15m:0.25,1h:0.25,4h:0.25,1d:0.25"))
        SCALE_REDUCE_PCT = _parse_pct_map2(cfg_get("SCALE_REDUCE_PCT", "15m:0.20,1h:0.20,4h:0.20,1d:0.20"))
        SCALE_MIN_ADD_NOTIONAL = float(cfg_get("SCALE_MIN_ADD_NOTIONAL_USDT", "15"))
        SCALE_REALLOCATE_BRACKETS = (cfg_get("SCALE_REALLOCATE_BRACKETS", "1") == "1")
        SCALE_LOG = (cfg_get("SCALE_LOG", "1") == "1")

        _SLIP_BY_SYMBOL   = _parse_float_by_symbol(cfg_get("SLIPPAGE_BY_SYMBOL", ""))
        _TP_BY_SYMBOL     = _parse_float_by_symbol(cfg_get("TP_PCT_BY_SYMBOL", ""))
        _SL_BY_SYMBOL     = _parse_float_by_symbol(cfg_get("SL_PCT_BY_SYMBOL", ""))
        _TRAIL_BY_SYMBOL  = _parse_float_by_symbol(cfg_get("TRAIL_PCT_BY_SYMBOL", ""))
    except Exception as e:
        logging.warning(f"[RUNTIME_RELOAD_WARN] {e}")




if __name__ == "__main__":
    exchange = GLOBAL_EXCHANGE

    asyncio.run(capital_load_state())
    log(f"[BOOT] CAPITAL: restored={int(bool(CAPITAL_PERSIST))} base={CAPITAL_BASE:,.2f} now={capital_get():,.2f}")
    log(f"[BOOT] ALLOC_UPNL mode={ALLOC_UPNL_MODE}, use={ALLOC_USE_UPNL}, w+={ALLOC_UPNL_W_POS}, w-={ALLOC_UPNL_W_NEG}, alpha={ALLOC_UPNL_EMA_ALPHA}, clamp={ALLOC_UPNL_CLAMP_PCT}%")

    import time
    while True:
        try:
            # discord.py는 기본 재접속 로직이 있지만,
            # 예외로 런루프가 죽을 때를 대비해 바깥에서 감싸 재시작
            client.run(TOKEN, log_handler=None)
        except KeyboardInterrupt:
            log("⏹ 수동 종료")
            break
        except Exception as e:
            log(f"⚠️ Discord client crashed: {e}. 5초 후 재시도...")
            time.sleep(5)

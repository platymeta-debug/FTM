# signal_bot.py
import ccxt
import pandas as pd
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ★ 비대화형 백엔드 (파일 저장 전용)
import matplotlib.pyplot as plt
import platform
import os, sys, logging
import discord
from dotenv import load_dotenv
load_dotenv("key.env")  # 같은 폴더의 key.env 읽기 (.env로 바꾸면 load_dotenv()만 써도 됨)
import json, uuid
import asyncio  # ✅ 이 줄을 꼭 추가
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
PRICE_SNAPSHOT = {}  # {symbol: {"ts": ms, "last": float|None, "bid": float|None, "ask": float|None, "mid": float|None, "chosen": float|None}}
PRICE_SNAPSHOT_TTL_MS = 500  # 동일 틱 처리용 짧은 TTL

async def get_price_snapshot(symbol: str) -> dict:
    """
    전 TF(15m/1h/4h/1d)에서 동일하게 쓸 '현재가 스냅샷'을 만든다.
    chosen = mid(가능) 또는 last(대체). 실패 시 None.
    """
    now_ms = int(time.time() * 1000)
    rec = PRICE_SNAPSHOT.get(symbol)
    if rec and (now_ms - rec.get("ts", 0) < PRICE_SNAPSHOT_TTL_MS):
        return rec

    # 1) 선물 모드면 선물 인스턴스 우선, 2) 아니면 스팟 티커
    last = bid = ask = mid = mark = chosen = None
    try:
        ex = FUT_EXCHANGE if (TRADE_MODE == "futures" and FUT_EXCHANGE) else None
        if ex:
            t = await _post(ex.fetch_ticker, symbol)
            last = float(t.get('last') or 0) or None
            bid  = float(t.get('bid')  or 0) or None
            ask  = float(t.get('ask')  or 0) or None
            try:
                mark = float(
                    t.get('markPrice')
                    or (t.get('info', {}).get('markPrice') if isinstance(t.get('info'), dict) else 0)
                ) or None
            except Exception:
                mark = None
        else:
            last = float(fetch_live_price(symbol) or 0) or None
    except Exception:
        pass

    if bid and ask:
        try:
            mid = (bid + ask) / 2.0
        except Exception:
            mid = None

    chosen = mid or last
    PRICE_SNAPSHOT[symbol] = {"ts": now_ms, "last": last, "bid": bid, "ask": ask, "mid": mid, "mark": mark, "chosen": chosen}
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
            await maybe_execute_trade(symbol, tf, signal, last_price=display_price, candle_ts=c_ts)
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

    # [ANCHOR: PROTECTIVE_CHECK_BEFORE_ENTRY]
    key = f"{symbol}|{tf}"
    pos = (PAPER_POS or {}).get(key)
    if pos:
        side  = str(pos.get("side", "")).upper()
        entry = float(pos.get("entry_price") or pos.get("entry") or 0)
        snap_curr = await get_price_snapshot(symbol)
        curr = snap_curr.get("mark") or snap_curr.get("mid") or snap_curr.get("last") or last_price
        hit, reason = _eval_tp_sl(side, float(entry), float(curr), tf)
        if hit:
            info = _paper_close(symbol, tf, float(last_price), reason)
            if info:
                await _notify_trade_exit(symbol, tf, side=info["side"], entry_price=info["entry_price"], exit_price=float(last_price), reason=(reason or "TP/SL"), mode="paper", pnl_pct=info.get("pnl_pct"))
            log(f"⏭ {symbol} {tf}: exited by {reason}, skip new entry this tick")
            log(f"⏭ {symbol} {tf}: skip reason=PROTECT")
            return
        else:
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

    if tf not in IGNORE_OCCUPANCY_TFS and PAPER_POS_TF.get(tf):
        log(f"⏭ {symbol} {tf}: skip reason=OCCUPIED")
        return

    PAPER_POS_TF[tf] = symbol
    _save_json(PAPER_POS_TF_FILE, PAPER_POS_TF)

    # [ANCHOR: AVOID_OVERWRITE_OPEN_POS]  (REPLACED)
    existing_paper = (PAPER_POS or {}).get(key)
    has_paper = existing_paper is not None
    fut_qty, fut_side = await _fut_get_open_qty_side(symbol)
    has_futures = fut_qty > 0

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
        if same_side and (cur_score - last_score) >= up_thr and legs < int(SCALE_MAX_LEGS):
            add_base = tf_base_cap * step_pct
            used_base = float((existing_paper or {}).get("used_base_margin") or 0.0)
            add_base = max(0.0, min(add_base, max_cap - used_base))
            notional_add = add_base * lev_used
            if notional_add >= SCALE_MIN_ADD_NOTIONAL and add_base > 0.0:
                if TRADE_MODE == "paper":
                    add_eff = add_base
                    add_qty = (add_eff * lev_used) / float(last_price)
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

                    avg = float(existing_paper["entry_price"])
                    tp_pct = float(existing_paper.get("tp_pct", 0.0))
                    sl_pct = float(existing_paper.get("sl_pct", 0.0))
                    if existing_paper.get("side") == "LONG":
                        existing_paper["tp_price"] = (avg * (1 + tp_pct/100.0)) if tp_pct>0 else None
                        existing_paper["sl_price"] = (avg * (1 - sl_pct/100.0)) if sl_pct>0 else None
                    else:
                        existing_paper["tp_price"] = (avg * (1 - tp_pct/100.0)) if tp_pct>0 else None
                        existing_paper["sl_price"] = (avg * (1 + sl_pct/100.0)) if sl_pct>0 else None

                    PAPER_POS[key] = existing_paper
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

            if did_scale and SCALE_LOG:
                log(f"🔼 scale-in {symbol} {tf}: +{add_base:.2f} base (lev×{lev_used}) at {last_price:.2f} (Δscore={cur_score-last_score:.2f})")

        # SCALE-OUT (same side + weakening)
        elif same_side and (last_score - cur_score) >= dn_thr:
            if TRADE_MODE == "paper":
                cur_qty = float(existing_paper.get("qty") or 0.0)
                red_qty = cur_qty * red_pct
                info = _paper_reduce(symbol, tf, red_qty, float(last_price)) if red_qty>0 else None
                if info: did_scale = True
            else:
                # reduceOnly partial close based on current futures qty
                red_qty = fut_qty * red_pct
                closed = await _fut_reduce(symbol, red_qty, "LONG" if exec_signal=="BUY" else "SHORT") if red_qty>0 else 0.0
                if closed > 0:
                    did_scale = True

                    fp = FUT_POS.get(symbol, {})
                    fp_qty = max(0.0, float(fp.get("qty", 0.0)) - closed)
                    fp["qty"] = fp_qty
                    FUT_POS[symbol] = fp
                    _save_json(OPEN_POS_FILE, FUT_POS)

                    await _fut_rearm_brackets(symbol, tf, float(last_price), "LONG" if exec_signal=="BUY" else "SHORT")

            if did_scale and SCALE_LOG:
                log(f"🔽 scale-out {symbol} {tf}: -{red_pct*100:.1f}% qty at {last_price:.2f} (Δscore={last_score-cur_score:.2f})")

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


    alloc = _preview_allocation_and_qty(
        symbol=symbol,
        tf=tf,
        signal=exec_signal,
        price=float(last_price),
        ex=None
    )
    base_margin = alloc["base_margin"]
    eff_margin  = alloc["eff_margin"]
    lev_used    = alloc["lev_used"]
    qty         = alloc["qty"]


    side = "LONG" if exec_signal == "BUY" else "SHORT"


    PAPER_POS[key] = {
        "side": side,
        "entry": float(last_price),
        "entry_price": float(last_price),
        "qty": qty,
        "eff_margin": eff_margin,
        "lev": lev_used,
        "ts_ms": int(time.time()*1000),
        "high": float(last_price),
        "low": float(last_price),

    }
    # (NEW) persist risk to paper JSON and CSV
    tp_pct = _req_tp_pct(symbol, tf, (take_profit_pct or {}))
    sl_pct = _req_sl_pct(symbol, tf, (HARD_STOP_PCT or {}))
    tr_pct = _req_trail_pct(symbol, tf, (trailing_stop_pct or {}))
    slip   = _req_slippage_pct(symbol, tf)
    eff_tp_pct, eff_sl_pct, eff_tr_pct, _src = _eff_risk_pcts(tp_pct, sl_pct, tr_pct, lev_used)
    if PAPER_POS[key]["side"] == "LONG":
        tp_price = (float(last_price)*(1+(eff_tp_pct or 0)/100)) if eff_tp_pct else None
        sl_price = (float(last_price)*(1-(eff_sl_pct or 0)/100)) if eff_sl_pct else None
    else:
        tp_price = (float(last_price)*(1-(eff_tp_pct or 0)/100)) if eff_tp_pct else None
        sl_price = (float(last_price)*(1+(eff_sl_pct or 0)/100)) if eff_sl_pct else None
    tr_pct_eff = eff_tr_pct
    PAPER_POS[key].update({
        "tp_pct": tp_pct, "sl_pct": sl_pct, "tr_pct": tr_pct,
        "tp_price": tp_price, "sl_price": sl_price,
        "lev": float(lev_used or 1.0),
        "eff_tp_pct": eff_tp_pct, "eff_sl_pct": eff_sl_pct, "eff_tr_pct": tr_pct_eff,
        "risk_mode": RISK_INTERPRET_MODE,
        "slippage_pct": slip
    })
    _save_json(PAPER_POS_FILE, PAPER_POS)
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
        f"tp_price={(tp_price if tp_price else '')}", f"sl_price={(sl_price if sl_price else '')}"
    ])
    if PAPER_CSV_OPEN_LOG:
        _log_trade_csv(symbol, tf, "OPEN", side, qty, last_price, extra=extra)

    # [ANCHOR: POSITION_OPEN_HOOK]
    # initialize trailing baseline at entry (per (symbol, tf))
    try:
        k2 = (symbol, tf)
        entry_price = float(last_price)
        if str(side).upper() == "LONG":
            highest_price[k2] = entry_price
            lowest_price.pop(k2, None)
        else:
            lowest_price[k2] = entry_price
            highest_price.pop(k2, None)
    except Exception:
        pass
    previous_signal[(symbol, tf)] = exec_signal
    entry_data[(symbol, tf)] = (float(last_price), datetime.now().strftime("%m월 %d일 %H:%M"))

    if TRADE_MODE == "paper" and PAPER_STRICT_NONZERO and (not base_margin or not eff_margin or not qty):
        logging.warning("[PAPER_WARN] zero allocation on paper entry: check PART A")

    await _notify_trade_entry(
        symbol, tf, exec_signal,
        mode="paper", price=float(last_price),
        qty=qty,
        base_margin=base_margin, eff_margin=eff_margin,
        lev_used=lev_used,
        score=EXEC_STATE.get(('score', symbol, tf))
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
        return max(0.0, TOTAL_CAPITAL_USDT * pct)
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
SCALE_REALLOCATE_BRACKETS = (cfg_get("SCALE_REALLOCATE_BRACKETS", "1") == "1")  # re-arm TP/SL/Trail after scaling
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
    eff_margin = base_margin * frac

    # 레버리지
    req_lev = int(_req_leverage(symbol, tf))                         # ← 변경
    lev_used = req_lev
    if ex:
        try:
            limits = _market_limits(ex, symbol)
            lev_used = int(_clamp(req_lev, 1, int(limits.get('max_lev') or 125)))
        except Exception:
            pass

    # 수량(미리보기)
    qty = 0.0
    try:
        if price and eff_margin and lev_used:
            qty = (float(eff_margin) * float(lev_used)) / float(price)
            # ex가 있으면 정밀도 반올림
            if ex:
                qty = _fut_amount_to_precision(ex, symbol, qty)
            # min notional은 알림에선 강제하지 않음(체결로직에서 이미 검증)
    except Exception:
        qty = 0.0

    return {
        'base_margin': float(base_margin),
        'eff_margin': float(eff_margin),
        'sf': float(sf),
        'mf': float(mf),
        'all_align': bool(all_align),
        'lev_used': int(lev_used),
        'qty': float(qty)
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

# repair hi/lo baselines on boot (applies to all TFs)
try:
    for key, pos in (PAPER_POS or {}).items():
        sym, tf = key.split("|", 1)
        side = str(pos.get("side","" )).upper()
        entry = float(pos.get("entry_price") or 0.0)
        k2 = (sym, tf)
        if side == "LONG":
            highest_price[k2] = max(float(highest_price.get(k2, 0.0)), entry)
        elif side == "SHORT":
            lowest_price[k2]  = min(float(lowest_price.get(k2, 1e30)), entry)
except Exception: pass


def _has_open_position(symbol: str, tf: str, mode: str) -> bool:
    if mode == "paper":
        return PAPER_POS.get(f"{symbol}|{tf}") is not None
    pos = FUT_POS.get(symbol)
    try:
        return bool(pos and abs(float(pos.get("qty", 0))) > 0)
    except Exception:
        return False


def _paper_close(symbol: str, tf: str, exit_price: float, exit_reason: str = ""):
    key = f"{symbol}|{tf}"
    pos = PAPER_POS.pop(key, None)
    if not pos:
        return None
    _save_json(PAPER_POS_FILE, PAPER_POS)
    if PAPER_POS_TF.get(tf) == symbol:
        PAPER_POS_TF.pop(tf, None)
        _save_json(PAPER_POS_TF_FILE, PAPER_POS_TF)
    side = pos.get("side", "")
    entry = float(pos.get("entry_price") or pos.get("entry") or 0.0)
    pnl_pct = None
    try:
        if entry > 0 and exit_price > 0:
            gross = ((exit_price - entry) / entry) * 100.0 if side == "LONG" else ((entry - exit_price) / entry) * 100.0
            pnl_pct = gross
    except Exception:
        pnl_pct = None
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
            _log_trade_csv(symbol, tf, "CLOSE", side, float((pos or {}).get('qty',0.0)), float(exit_price), extra=extra)

    except Exception as e:
        log(f"[CSV_CLOSE_WARN] paper {symbol} {tf}: {e}")
    # IDEMP: allow re-entry after manual/forced close
    try: idem_clear_symbol_tf(symbol, tf)
    except Exception: pass
    return {"side": side, "entry_price": entry, "pnl_pct": pnl_pct}

# [ANCHOR: PAPER_PARTIAL_CLOSE_BEGIN]
def _paper_reduce(symbol: str, tf: str, reduce_qty: float, exit_price: float):
    key = f"{symbol}|{tf}"
    pos = PAPER_POS.get(key)
    if not pos or reduce_qty <= 0: return None
    side = pos.get("side","")
    qty_old = float(pos.get("qty",0.0))
    if qty_old <= 0: return None
    reduce_qty = min(reduce_qty, qty_old)
    entry = float(pos.get("entry_price") or pos.get("entry") or 0.0)
    pnl_usdt = (exit_price - entry) * reduce_qty if side=="LONG" else (entry - exit_price) * reduce_qty
    qty_new = qty_old - reduce_qty
    if qty_new <= 0: return _paper_close(symbol, tf, exit_price)
    eff_margin_old = float(pos.get("eff_margin") or 0.0)
    eff_margin_new = eff_margin_old * (qty_new/qty_old)
    pos["qty"] = qty_new
    pos["eff_margin"] = eff_margin_new
    pos["last_update_ms"] = int(time.time()*1000)
    PAPER_POS[key] = pos
    _save_json(PAPER_POS_FILE, PAPER_POS)
    return {"pnl": pnl_usdt, "qty_closed": reduce_qty, "qty_left": qty_new}
# [ANCHOR: PAPER_PARTIAL_CLOSE_END]

# [ANCHOR: HYDRATE_FROM_DISK_BEGIN]
def _hydrate_from_disk():
    try:
        # 페이퍼 포지션/점유 복원
        global PAPER_POS, PAPER_POS_TF
        if 'PAPER_POS' in globals():
            for k, v in (PAPER_POS or {}).items():
                try:
                    sym, tf = k.split("|", 1)
                    # TF 점유가 비어있으면 복원
                    if not PAPER_POS_TF.get(tf):
                        PAPER_POS_TF[tf] = sym
                except Exception:
                    continue
        _save_json(PAPER_POS_TF_FILE, PAPER_POS_TF)
    except Exception as e:
        log(f"[HYDRATE] warn: {e}")
# [ANCHOR: HYDRATE_FROM_DISK_END]

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


# ==========================
#   USDT-M Futures Engine
# ==========================
AUTO_TRADE   = os.getenv("AUTO_TRADE", "0") == "1"
TRADE_MODE   = os.getenv("TRADE_MODE", "paper")   # 'paper' | 'spot' | 'futures'
EXCHANGE_ID  = os.getenv("EXCHANGE_ID", "binanceusdm")
SANDBOX      = os.getenv("SANDBOX", "1") == "1"

FUT_MGN_USDT = float(os.getenv("FUT_MGN_USDT", "10"))    # 1회 진입 증거금(USDT)
FUT_LEVERAGE = int(os.getenv("LEVERAGE", "3"))
FUT_MARGIN   = os.getenv("MARGIN_TYPE", "ISOLATED").upper()  # ISOLATED|CROSS
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

async def _stop_market(ex, symbol, side, stop_price, closePosition=True, positionSide=None):
    params = {'stopPrice': float(stop_price), 'closePosition': bool(closePosition), 'reduceOnly': True}
    try:
        import uuid, time
        if getattr(ex, 'id', '') in ('binanceusdm', 'binance'):
            params['newClientOrderId'] = f"sb-sl-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
    except Exception:
        pass
    if HEDGE_MODE and positionSide:
        params['positionSide'] = positionSide  # 'LONG' or 'SHORT'
    return await _post(ex.create_order, symbol, 'STOP_MARKET', side, None, None, params)

async def _tp_market(ex, symbol, side, stop_price, closePosition=True, positionSide=None):
    params = {'stopPrice': float(stop_price), 'closePosition': bool(closePosition), 'reduceOnly': True}
    try:
        import uuid, time
        if getattr(ex, 'id', '') in ('binanceusdm', 'binance'):
            params['newClientOrderId'] = f"sb-tp-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
    except Exception:
        pass
    if HEDGE_MODE and positionSide:
        params['positionSide'] = positionSide
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
    try:
        await _cancel_symbol_conditional_orders(symbol)
    except Exception as e:
        logging.warning(f"[FUT_REARM_CANCEL_WARN] {symbol}: {e}")
    try:
        await _place_protect_orders(FUT_EXCHANGE, symbol, tf, side, float(last_price))
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



async def _place_protect_orders(ex, symbol, tf, side, entry_price):
    """
    TP/SL/Trailing — 심볼×TF 오버라이드 반영.
    듀얼(헤지) 모드면 positionSide 명시.
    """
    # 이미 상단에서 환경변수들을 파싱해서 만든 소문자 dict들을 사용
    tp_map = (take_profit_pct or {})
    sl_map = (HARD_STOP_PCT or {})
    tr_map = (trailing_stop_pct or {})  # <-- 대문자 금지

    tp_pct = _req_tp_pct(symbol, tf, tp_map)
    sl_pct = _req_sl_pct(symbol, tf, sl_map)
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

async def _notify_trade_entry(symbol: str, tf: str, signal: str, *,
                              mode: str,              # 'futures' or 'spot' or 'paper'
                              price: float, qty: float|None,
                              base_margin: float|None=None, eff_margin: float|None=None,
                              lev_used: int|None=None,
                              score: float|None=None):
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

        # 배분/사용비율(총자본 → TF배정 → 강도×MTF → 최종)
        total_cap = None
        try:
            total_cap = float(TOTAL_CAPITAL_USDT)
        except Exception:
            pass

        alloc_pct = None
        use_frac  = None
        notional  = None
        if base_margin:
            if total_cap and total_cap > 0:
                alloc_pct = base_margin / total_cap                  # TF 기본배정 비율
            if eff_margin and base_margin > 0:
                use_frac = float(eff_margin) / float(base_margin)    # 강도×MTF 적용 후 실제 사용 비율
        if eff_margin and lev_used:
            try:
                notional = float(eff_margin) * int(lev_used)         # 레버리지 적용 노치오날
            except Exception:
                pass

        # 헤더/요약
        title = "🟢 **진입 (BUY)**" if signal == "BUY" else "🔴 **진입 (SELL)**"
        mode_text = '🧪 페이퍼' if mode=='paper' else ('선물' if mode=='futures' else '현물')
        lines = [
            f"{title} 〔{symbol} · {tf}〕",
            f"• 모드/가격: {mode_text} / {_fmt_usd(price)}" + (f" / 레버리지 ×{int(lev_used)}" if lev_used else ""),
        ]

        # 강도/MTF
        if strength_label and sf is not None:
            lines.append(f"• 강도: {strength_label} (계수 ×{sf:.2f})")
        if mf is not None:
            lines.append(f"• 상위TF: {align_text} (계수 ×{mf:.2f}" + (" · ALL" if all_align else "") + ")")

        # 배분 브레이크다운
        # ① 총자본 → TF배정
        if total_cap and alloc_pct is not None:
            lines.append(f"• 배분(1): 총자본 {_fmt_usd(total_cap)} → TF배정 {_fmt_usd(base_margin)} ({_fmt_pct(alloc_pct)})")
        elif base_margin:
            lines.append(f"• 배분(1): TF배정 {_fmt_usd(base_margin)}")

        # ② 강도×MTF 적용(최종 사용비율/금액)
        if eff_margin is not None:
            # 표시: “강도×MTF = ×sf ×mf → 사용 {_fmt_pct(use_frac)} = {_fmt_usd(eff_margin)}”
            sf_txt = f"×{sf:.2f}" if sf is not None else "-"
            mf_txt = f"×{mf:.2f}" if mf is not None else "-"
            use_txt = _fmt_pct(use_frac) if use_frac is not None else "-"
            lines.append(f"• 배분(2): 강도×MTF = {sf_txt} {mf_txt} → 사용 {use_txt} = {_fmt_usd(eff_margin)}")
            # ⚠️ 설명: 최종 사용비율은 기본적으로 min(1.00, sf*mf)로 100%를 넘지 않도록 안전 클램프되어 있습니다.
            # 메시지에는 1.25 같은 중간 계수가 보일 수 있지만, 실제 사용 비율은 100%를 초과하지 않습니다.

        # 수량/노치오날
        if qty is not None or notional is not None:
            qtxt = f"{_fmt_qty(qty)}" if qty is not None else "-"
            ntx  = _fmt_usd(notional) if notional is not None else "-"
            lines.append(f"• 수량/노치오날: {qtxt} @ {_fmt_usd(price)} / {ntx}")
        
        # [ANCHOR: entry_risk_prices]
        try:
            show_price  = os.getenv("ENTRY_SHOW_RISK_PRICE","1") == "1"
            show_pct    = os.getenv("ENTRY_SHOW_RISK_PERCENT","0") == "1"

            tpv = _req_tp_pct(symbol, tf, (take_profit_pct or {}))
            slv = _req_sl_pct(symbol, tf, (HARD_STOP_PCT or {}))
            trv = _req_trail_pct(symbol, tf, (trailing_stop_pct or {}))
            sv  = _req_slippage_pct(symbol, tf)

            if show_pct:
                lines.append(f"• Risk: TP {tpv:.2f}% / SL {slv:.2f}% / TR {trv:.2f}% / Slippage {sv:.2f}%")

            if show_price:
                eff_tp_pct, eff_sl_pct, eff_tr_pct, _src = _eff_risk_pcts(tpv, slv, trv, lev_used)
                if signal == "BUY":
                    tp_price = price*(1+(eff_tp_pct or 0)/100) if eff_tp_pct else None
                    sl_price = price*(1-(eff_sl_pct or 0)/100) if eff_sl_pct else None
                else:
                    tp_price = price*(1-(eff_tp_pct or 0)/100) if eff_tp_pct else None
                    sl_price = price*(1+(eff_sl_pct or 0)/100) if eff_sl_pct else None

                tp_price_fmt = _fmt_usd(tp_price) if tp_price else "-"
                sl_price_fmt = _fmt_usd(sl_price) if sl_price else "-"
                tr_pct_eff = eff_tr_pct if eff_tr_pct is not None else 0.0
                _lev_show = f" ×{float(lev_used or 1.0):.0f}"
                _tp_pct_price = eff_tp_pct if eff_tp_pct is not None else tpv
                _sl_pct_price = eff_sl_pct if eff_sl_pct is not None else slv
                _tp_pct_margin = (float(tpv) if tpv is not None else (_tp_pct_price*(float(lev_used or 1.0))))
                _sl_pct_margin = (float(slv) if slv is not None else (_sl_pct_price*(float(lev_used or 1.0))))
                lines.append(
                    f"• Risk (price): TP: {tp_price_fmt} (+{_tp_pct_price:.2f}% | +{_tp_pct_margin:.2f}% on margin{_lev_show}) / "
                    f"SL: {sl_price_fmt} (-{_sl_pct_price:.2f}% | -{_sl_pct_margin:.2f}% on margin{_lev_show}) / "
                    f"TR: {tr_pct_eff:.2f}% (percent trail)"
                )
        except Exception:
            pass

        await ch.send("\n".join(lines))
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
                             status: str | None = None):
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
        info = _paper_close(symbol_eth, tf, float(exit_price), action_reason)
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
        info = _paper_close(symbol, tf, xp, action_reason)
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
    base_margin = _margin_for_tf(tf)  # TOTAL_CAPITAL_USDT * ALLOC_TF[tf] or fallback(FUT_MGN_USDT)

    # 3) 강도 가중
    sf = _strength_factor(exec_signal, local_score)

    # 4) 상위 TF 바이어스
    mf, all_align = _mtf_factor(symbol, tf, exec_signal)

    # 5) 최종 증거금 비율
    frac = min(1.0, sf * mf)
    if all_align and _FULL_ON_ALL:
        frac = 1.0

    eff_margin = base_margin * frac

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

        tp_pct = _req_tp_pct(symbol, tf, (take_profit_pct or {}))
        sl_pct = _req_sl_pct(symbol, tf, (HARD_STOP_PCT or {}))
        tr_pct = _req_trail_pct(symbol, tf, (trailing_stop_pct or {}))
        lev = _req_leverage(symbol, tf)
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
            f"tp_price={(tp_price if tp_price else '')}", f"sl_price={(sl_price if sl_price else '')}"
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
        except Exception:
            pass
        previous_signal[(symbol, tf)] = 'BUY' if side == 'LONG' else 'SELL'
        entry_data[(symbol, tf)] = (float(last), datetime.now().strftime("%m월 %d일 %H:%M"))

        # 보호 주문(TP/SL) 동시 등록
        await _place_protect_orders(ex, symbol, tf, side, float(last))

        # (체결 후) 디스코드 알림
        try:
            base_margin = _margin_for_tf(tf)
            req_lev = int(TF_LEVERAGE.get(tf, FUT_LEVERAGE))
            limits  = _market_limits(ex, symbol)
            lev_used = int(_clamp(req_lev, 1, int(limits.get('max_lev') or 125)))
            await _notify_trade_entry(
                symbol, tf, exec_signal, mode="futures",
                price=float(last), qty=float(qty),
                base_margin=float(base_margin),
                eff_margin=float(
                    base_margin * min(
                        1.0,
                        _strength_factor(signal, EXEC_STATE.get(('score', symbol, tf)))
                        * _mtf_factor(symbol, tf, signal)[0]
                    )
                ),
                lev_used=lev_used,
                score=EXEC_STATE.get(('score', symbol, tf))
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

TF_LEVERAGE = _parse_tf_map(os.getenv("LEVERAGE_BY_TF", ""), int)   # 예: {'15m':7,'1h':5,...}
TF_MARGIN   = _parse_tf_map(os.getenv("MARGIN_BY_TF", ""), lambda x: x.upper())                  # 예: {'15m':'ISOLATED','4h':'CROSS',...}

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
               tr_pct: float|None, key2: tuple):
    """
    Returns: (should_exit: bool, reason: str, trigger_price: float, dbg: str)
    - Uses 1m bar and EXIT_EVAL_MODE (TOUCH | CLOSE)
    - Price is sanitized/clamped; outlier-guarded
    - Trailing uses _compute_trail() (armed + base guard)
    """
    clamped, bar = _sanitize_exit_price(symbol, last_price_hint)
    if _outlier_guard(clamped, bar):
        return (False, "OUTLIER_SKIP", clamped, f"outlier>{OUTLIER_MAX_1M}")
    # Compute trailing price with arming (helpers already enforce hp>=entry / lp<=entry)
    trail_px, armed, base = _compute_trail(side, float(entry_price),
                                           float(tr_pct) if tr_pct is not None else 0.0,
                                           highest_price.get(key2), lowest_price.get(key2), tf)
    # Select representative price for evaluation mode
    if EXIT_EVAL_MODE == "CLOSE":
        p = float(bar["close"])
        hi, lo = float(bar["close"]), float(bar["close"])
    else:  # TOUCH
        p = clamped
        hi, lo = float(bar["high"]), float(bar["low"])
    sideU = str(side).upper()
    tp_hit = sl_hit = tr_hit = False
    if sideU == "LONG":
        if tp_price: tp_hit = hi >= float(tp_price)
        if sl_price: sl_hit = lo <= float(sl_price)
        if trail_px and armed: tr_hit = lo <= float(trail_px)
    else:  # SHORT
        if tp_price: tp_hit = lo <= float(tp_price)
        if sl_price: sl_hit = hi >= float(sl_price)
        if trail_px and armed: tr_hit = hi >= float(trail_px)
    # conflict resolution: SL > TRAIL > TP (more conservative first)
    reason = None
    if   sl_hit: reason = "SL"
    elif tr_hit: reason = "TRAIL"
    elif tp_hit: reason = "TP"
    dbg = (f"1m ohlc=({bar['open']:.6f},{bar['high']:.6f},{bar['low']:.6f},{bar['close']:.6f}) "
           f"p={p:.6f} clamp={clamped:.6f} armed={armed} base={base} "
           f"tp={tp_price} sl={sl_price} tr={tr_pct} trail_px={trail_px}")
    if reason:
        trig = (float(sl_price) if reason=="SL" else (float(trail_px) if reason=="TRAIL" else float(tp_price)))
        return (True, reason, trig, dbg)
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
        env_min = float(os.getenv("FUT_MIN_NOTIONAL", "5"))
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

            # (이하 기존 ETH/BTC 루프 계속)


            timeframes = ['15m', '1h', '4h', '1d']

            # ===== ETH 루프 =====
            for tf in timeframes:
                try:
                    ch_id = CHANNEL_IDS.get(tf)  # 또는 CHANNEL_BTC.get(tf)
                    if not ch_id or ch_id == 0:
                        log(f"⏭ 채널 ID 없음: ETH {tf} 건너뜀")
                        continue
                    channel = client.get_channel(ch_id)
                    if channel is None:
                        log(f"❌ 채널 객체 없음: ETH {tf} (ID: {ch_id})")
                        continue

                    symbol_eth = 'ETH/USDT'
                    key2 = _key2(symbol_eth, tf)

                    # === Closed-candle snapshot (ETH) ===
                    # (use closed candle to avoid intra-candle spikes)
                    # 아래 df는 직후에 get_ohlcv로 다시 채워지므로, prelude는 get_ohlcv 이후 위치로 옮겨질 수 있음.


                    # 1) 데이터/지표 준비
                    df = get_ohlcv(symbol_eth, tf, limit=300)
                    df = add_indicators(df)  # 차트 함수가 지표 컬럼을 사용하므로 미리 계산

                    chart_files   = save_chart_groups(df, symbol_eth, tf)   # 4장

                    score_file     = plot_score_history(symbol_eth, tf)
                    perf_file      = analyze_performance_for(symbol_eth, tf)
                    performance_file = generate_performance_stats(tf, symbol=symbol_eth)

                    # 📌 닫힌 캔들의 종가(게이팅·신호 결정에 사용)
                    closed_price = get_closed_price(df, 'close')
                    if closed_price is None:
                        log("⏭️ 닫힌 캔들 종가 없음 → 스킵")
                        continue

                    # 📌 닫힌 캔들 기준 타임스탬프(초)
                    last_ts = get_closed_ts(df)
                    if not last_ts:
                        log("⏭️ 닫힌 캔들 타임스탬프 계산 실패 → 스킵")
                        continue


                    # 3) 일봉 변동률 계산
                    if _len(df) == 0:
                        log(f"⏭️ {symbol_eth} {tf} 보고서 생략: 데이터 없음")
                        continue

                    snap = await get_price_snapshot(symbol_eth)  # ETH/USDT
                    live_price = snap.get("mid") or snap.get("last")
                    display_price = live_price if isinstance(live_price, (int, float)) else closed_price
                    # [ANCHOR: daily_change_unify_eth_alt]
                    daily_change_pct = calc_daily_change_pct(symbol_eth, display_price)


                    
                    # 📍 ETH 진입 정보 주입

                    _ep = entry_data.get(key2)

                    entry_price_local = _ep[0] if _ep else None
                    entry_time_local  = _ep[1] if _ep else None


                    # Ichimoku 이미지 준비 (없으면 None)
                    ichimoku_file = save_ichimoku_chart(df, symbol_eth, tf)  # 실패 시 함수가 None 반환

                    main_msg_pdf, summary_msg_pdf, _ = format_signal_message(
                        tf, signal, closed_price, pnl, reasons, df,
                        entry_time=entry_price_local and entry_time_local,
                        entry_price=entry_price_local,
                        score=score,
                        weights=weights, weights_detail=weights_detail,
                        prev_score_value=previous_score.get(tf),
                        agree_long=agree_long, agree_short=agree_short,
                        symbol=symbol_eth,
                        daily_change_pct=daily_change_pct,
                        score_history=score_history.get(tf),
                        recent_scores=score_history.get(tf),

                        live_price=display_price,

                        show_risk=False
                    )


                    # --- 강도/MTF 상태 기록 & 메시지 보강(ETH) ---
                    _record_signal(symbol_eth, tf, signal, score)

                    sf = _strength_factor(signal, score)
                    mf, all_align = _mtf_factor(symbol_eth, tf, signal)
                    align_text, agree_cnt, oppose_cnt = _mtf_alignment_text(symbol_eth, tf, signal)
                    strength_label = _strength_label(signal, score)

                    addon = (
                        f"\n• 강도: {strength_label} (×{sf:.2f})"
                        f"\n• 상위TF: {align_text} (×{mf:.2f})"
                    )
                    # ETH의 이 경로는 short_msg를 안 쓰므로 메인/요약에만 반영
                    main_msg_pdf = addon + "\n" + main_msg_pdf
                    summary_msg_pdf = addon + "\n" + summary_msg_pdf


                    snap = await get_price_snapshot(symbol_eth)  # ETH/USDT
                    display_price = snap.get("mid") or snap.get("last") or closed_price


                    pdf_path = generate_pdf_report(
                        df=df, tf=tf, symbol=symbol_eth,
                        signal=signal, price=display_price, score=score,
                        reasons=reasons, weights=weights,
                        agree_long=agree_long, agree_short=agree_short,
                        now=datetime.now(),
                        chart_imgs=chart_files, ichimoku_img=ichimoku_file,
                        daily_change_pct=daily_change_pct,
                        discord_message=(main_msg_pdf + "\n\n" + summary_msg_pdf),
                        entry_price=entry_price_local, entry_time=entry_time_local
                    )
                    

                    # 현재 버킷(BUY/NEUTRAL/SELL)
                    curr_bucket = _score_bucket(score, CFG)
                    price_eth_now   = curr_price_eth
                    # 게이팅 판정
                    ok_to_send, why = _should_notify(
                        tf, score, closed_price, curr_bucket, last_ts,
                        last_sent_ts_eth, last_sent_bucket_eth, last_sent_score_eth, last_sent_price_eth
                    )

                    if not ok_to_send:
                        log(f"🔕 ETH {tf} 억제: {why}")
                        # 계산값(이전 상태)만 업데이트하고 전송은 생략해도 됨 — 선택
                        previous_bucket[tf] = curr_bucket
                        previous_score[tf]  = score
                        previous_price[tf]  = closed_price   # 📌 닫힌 캔들 종가로 저장
                        last_candle_ts_eth[tf] = last_ts     # 📌 닫힌 캔들 ts로 저장
                        continue

                    # 6) 전송
                    # 보고서 안내 문구
                    content = f"📄 {datetime.now():%m월 %d일 %p %I시} {symbol_eth} {tf} 보고서입니다."
                    files = [p for p in [*(chart_files or []), ichimoku_file, pdf_path, score_file, perf_file, performance_file] if p and os.path.exists(p)]       
                    await channel.send(
                        content=main_msg_pdf,
                        files=[discord.File(p) for p in chart_files if p],
                        silent=True
                    )

                except Exception as e:
                    # 채널이 None일 수 있어 안전 가드
                    try:
                        await channel.send(f"❌ ETH PDF 생성 실패: {e}")
                    except Exception:
                        log(f"❌ ETH PDF 생성 실패(채널 전송 불가): {e}")
                    
                    # 📌 전송 성공 후 마지막 전송 상태 업데이트 (닫힌 기준)
                    last_sent_ts_eth[tf]     = last_ts
                    last_sent_bucket_eth[tf] = curr_bucket
                    last_sent_score_eth[tf]  = score
                    last_sent_price_eth[tf]  = closed_price


            # ===== BTC 루프 (교체) =====
            for tf in TIMEFRAMES_BTC:
                try:
                    # 0) 채널 확인
                    channel = _get_channel_or_skip('BTC', tf)  # 없으면 로그 남기고 skip
                    if channel is None:
                        continue

                    symbol_btc = 'BTC/USDT'

                    # 1) 데이터/지표
                    df = await safe_get_ohlcv(symbol_btc, tf, limit=300)
                    df = await safe_add_indicators(df)

                    # 닫힌 캔들 기준 타임스탬프/가격 (게이팅·리포팅 공용)
                    c_ts = get_closed_ts(df)
                    if not c_ts:
                        log(f"⏭️ 닫힌 캔들 ts 없음: BTC {tf} → skip")
                        continue
                    c_c  = get_closed_price(df, 'close')
                    if c_c is None:
                        log(f"⏭️ 닫힌 캔들 종가 없음: BTC {tf} → skip")
                        continue

                    # 2) 신호 계산 (ETH와 동일 시그니처)
                    signal, price, rsi, macd, reasons, score, weights, agree_long, agree_short, weights_detail = \
                        calculate_signal(df, tf, symbol_btc)


                    snap = await get_price_snapshot(symbol_btc)  # BTC/USDT
                    live_price = snap.get("mid") or snap.get("last")
                    display_price = live_price if isinstance(live_price, (int, float)) else c_c
                    # [ANCHOR: daily_change_unify_btc]
                    daily_change_pct = calc_daily_change_pct(symbol_btc, display_price)


                    # 4) 진입 정보 (없으면 None)
                    _epb = entry_data.get((symbol_btc, tf))  # (entry_price, entry_time)
                    entry_price_local = _epb[0] if _epb else None
                    entry_time_local  = _epb[1] if _epb else None

                    # 5) 이미지 준비 (각 함수가 내부적으로 plt.close 처리)
                    ichimoku_file    = save_ichimoku_chart(df, symbol_btc, tf)
                    chart_files      = save_chart_groups(df, symbol_btc, tf)           # 묶음 차트
                    score_file       = plot_score_history(symbol_btc, tf)              # 점수 히스토리
                    perf_file        = analyze_performance_for(symbol_btc, tf)         # 누적 성과 그래프
                    performance_file = generate_performance_stats(tf, symbol=symbol_btc)

                    # 6) 메시지 (요약/본문/짧은 알림)
                    main_msg_pdf, summary_msg_pdf, short_msg = format_signal_message(
                        tf=tf, signal=signal, price=c_c, pnl=None, strength=reasons, df=df,
                        entry_time=entry_time_local, entry_price=entry_price_local,
                        score=score, weights=weights, weights_detail=weights_detail,
                        prev_score_value=previous_score_btc.get(tf),
                        agree_long=agree_long, agree_short=agree_short,
                        recent_scores=list(score_history_btc.setdefault(tf, deque(maxlen=4))),
                        daily_change_pct=daily_change_pct,
                        symbol=symbol_btc,

                        live_price=display_price,

                        show_risk=False
                    )

                    display_price = sanitize_price_for_tf(symbol_btc, tf, c_c)

                    # (선택) PDF 생성 — 파일 목록에 같이 첨부
                    try:
                        display_price = sanitize_price_for_tf(symbol_btc, tf, c_c)
                        pdf_path = generate_pdf_report(
                            df=df, tf=tf, symbol=symbol_btc,
                            signal=signal, price=display_price, score=score,
                            reasons=reasons, weights=weights,
                            agree_long=agree_long, agree_short=agree_short,
                            now=datetime.now(),
                            chart_imgs=chart_files, ichimoku_img=ichimoku_file,
                            daily_change_pct=daily_change_pct,
                            discord_message=(main_msg_pdf + "\n\n" + summary_msg_pdf),
                            entry_price=entry_price_local, entry_time=entry_time_local
                        )
                    except Exception as e:
                        log(f"PDF 생성 경고: {e}")
                        pdf_path = None

                    # 7) 알림 억제(게이팅)
                    curr_bucket = _score_bucket(score, CFG)
                    trigger_mode = trigger_mode_for(tf)
                    await handle_trigger(symbol_btc, tf, trigger_mode, signal, display_price, c_ts, entry_data)
                    ok_to_send, why = _should_notify(
                        tf, score, c_c, curr_bucket, c_ts,
                        last_sent_ts_btc, last_sent_bucket_btc, last_sent_score_btc, last_sent_price_btc
                    )
                    if not ok_to_send:
                        log(f"🔕 BTC {tf} 억제: {why}")
                        previous_bucket_btc[tf] = curr_bucket
                        previous_score_btc[tf]  = score
                        previous_price_btc[tf]  = float(c_c)
                        last_candle_ts_btc[tf]  = c_ts
                        continue

                    # 8) 디스코드 전송
                    try:
                        await channel.send(content=short_msg)
                        files_to_send = [p for p in [*(chart_files or []), ichimoku_file, score_file, perf_file, performance_file, pdf_path] if p and os.path.exists(p)]
                        await channel.send(
                            content=main_msg_pdf,
                            files=[discord.File(p) for p in files_to_send] if files_to_send else None,
                            silent=True
                        )
                        if len(summary_msg_pdf) > 1900:
                            summary_msg_pdf = summary_msg_pdf[:1900] + "\n...(이하 생략)"
                        await channel.send(summary_msg_pdf, silent=True)
                    except Exception as e:
                        log(f"❌ BTC 전송 오류: {e}")

                    # 9) 상태 업데이트(‘발송 성공’ 시점)
                    hist = score_history_btc.setdefault(tf, deque(maxlen=4))
                    if not hist or round(score, 1) != hist[-1]:
                        hist.append(round(score, 1))

                    previous_signal[(symbol_btc, tf)] = signal
                    previous_score_btc[tf]  = score
                    previous_bucket_btc[tf] = curr_bucket
                    previous_price_btc[tf]  = float(c_c)
                    last_candle_ts_btc[tf]  = c_ts

                    last_sent_ts_btc[tf]     = c_ts
                    last_sent_bucket_btc[tf] = curr_bucket
                    last_sent_score_btc[tf]  = score
                    last_sent_price_btc[tf]  = float(c_c)

                except Exception as e:
                    log(f"⚠️ BTC 루프 오류: {e}")

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
                k = f"{symbol_eth}|{tf}"
                pos = PAPER_POS.get(k) if TRADE_MODE == "paper" else (FUT_POS.get(symbol_eth) if TRADE_MODE == "futures" else None)
                if pos:
                    side = pos.get("side")
                    entry = float(pos.get("entry_price") or pos.get("entry") or 0)
                    hit, reason = _eval_tp_sl(side, entry, float(snap.get("mark") or last_price), tf)
                    if hit:
                        if TRADE_MODE == "paper":
                            info = _paper_close(symbol_eth, tf, last_price, reason)
                            if info:
                                await _notify_trade_exit(symbol_eth, tf, side=info["side"], entry_price=info["entry_price"], exit_price=last_price, reason=reason, mode="paper", pnl_pct=info.get("pnl_pct"))
                        elif TRADE_MODE == "futures":
                            await futures_close_all(symbol_eth, tf, exit_price=last_price, reason=reason)
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
                pos = PAPER_POS.get(f"{symbol_eth}|{tf}") if TRADE_MODE=='paper' else FUT_POS.get(symbol_eth)
                if pos:
                    side = str(pos.get("side","" )).upper()
                    entry_price = float(pos.get("entry_price") or pos.get("entry") or 0.0)
                    tp_price = pos.get("tp_price"); sl_price = pos.get("sl_price")
                    tr_pct_eff = pos.get("eff_tr_pct") if (pos.get("eff_tr_pct") is not None) else pos.get("tr_pct")
                    ok_exit, reason, trig_px, dbg = _eval_exit(symbol_eth, tf, side, entry_price, last_price, tp_price, sl_price, tr_pct_eff, key2)
                    log(f"[EXIT_CHECK] {symbol_eth} {tf} {side} -> {ok_exit} reason={reason} {dbg}")
                    if ok_exit:
                        exit_reason = reason  # 'SL' | 'TRAIL' | 'TP'
                        if TRADE_MODE=='paper':
                            info = _paper_close(symbol_eth, tf, float(trig_px), exit_reason)
                            if info:
                                await _notify_trade_exit(symbol_eth, tf, side=info['side'], entry_price=info['entry_price'], exit_price=float(trig_px), reason=exit_reason, mode='paper', pnl_pct=info.get('pnl_pct'))
                        else:
                            await futures_close_all(symbol_eth, tf, exit_price=float(trig_px), reason=exit_reason)
                        continue




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


                # 수익률 계산
                pnl = None
                if previous in ['BUY', 'SELL'] and signal in ['BUY', 'SELL'] and signal != previous:
                    entry_price, entry_time = entry_data.get(key2)
                    if previous == 'BUY' and signal == 'SELL':
                        pnl = ((price - entry_price) / entry_price) * 100
                    elif previous == 'SELL' and signal == 'BUY':
                        pnl = ((entry_price - price) / entry_price) * 100

                chart_files = save_chart_groups(df, symbol_eth, tf)

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

                # 2) 분석 메시지 — 푸시에는 안 뜸
                await channel.send(
                    content=main_msg_pdf,
                    files=[discord.File(p) for p in chart_files if p],
                    silent=True
                )

                # 점수기록: 실제 발송시에만(중복 방지)
                if not score_history[tf] or round(score, 1) != score_history[tf][-1]:
                    score_history[tf].append(round(score, 1))

                # 버킷 상태 업데이트
                previous_bucket[tf] = _score_bucket(score, CFG)


                # 3) 종합해석 메시지 — 길면 잘라서 전송
                if len(summary_msg_pdf) > 1900:            # ← summary_msg → summary_msg_pdf
                    summary_msg_pdf = summary_msg_pdf[:1900] + "\n...(이하 생략)"
                await channel.send(summary_msg_pdf, silent=True)

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
                live_price = snap.get("mid") or snap.get("last")
                display_price = live_price if isinstance(live_price, (int, float)) else c_c
                # [ANCHOR: daily_change_unify_btc]

                daily_change_pct = calc_daily_change_pct(symbol_btc, display_price)

                last_price = float(display_price if 'display_price' in locals() else live_price)
                try:
                    set_last_price(symbol_btc, last_price)
                except Exception:
                    pass

                # === 재시작 보호: 이미 열린 포지션 보호조건 재평가 ===
                k = f"{symbol_btc}|{tf}"
                key2 = (symbol_btc, tf)
                pos = PAPER_POS.get(k) if TRADE_MODE == "paper" else (FUT_POS.get(symbol_btc) if TRADE_MODE == "futures" else None)
                if pos:
                    side = pos.get("side")
                    entry = float(pos.get("entry_price") or pos.get("entry") or 0)
                    hit, reason = _eval_tp_sl(side, entry, float(snap.get("mark") or last_price), tf)
                    if hit:
                        if TRADE_MODE == "paper":
                            info = _paper_close(symbol_btc, tf, last_price, reason)
                            if info:
                                await _notify_trade_exit(symbol_btc, tf, side=info["side"], entry_price=info["entry_price"], exit_price=last_price, reason=reason, mode="paper", pnl_pct=info.get("pnl_pct"))
                        elif TRADE_MODE == "futures":
                            await futures_close_all(symbol_btc, tf, exit_price=last_price, reason=reason)
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
                pos = PAPER_POS.get(f"{symbol_btc}|{tf}") if TRADE_MODE=='paper' else FUT_POS.get(symbol_btc)
                if pos:
                    side = str(pos.get("side","" )).upper()
                    entry_price = float(pos.get("entry_price") or pos.get("entry") or 0.0)
                    tp_price = pos.get("tp_price"); sl_price = pos.get("sl_price")
                    tr_pct_eff = pos.get("eff_tr_pct") if (pos.get("eff_tr_pct") is not None) else pos.get("tr_pct")
                    ok_exit, reason, trig_px, dbg = _eval_exit(symbol_btc, tf, side, entry_price, last_price, tp_price, sl_price, tr_pct_eff, key2)
                    log(f"[EXIT_CHECK] {symbol_btc} {tf} {side} -> {ok_exit} reason={reason} {dbg}")
                    if ok_exit:
                        exit_reason = reason  # 'SL' | 'TRAIL' | 'TP'
                        if TRADE_MODE=='paper':
                            info = _paper_close(symbol_btc, tf, float(trig_px), exit_reason)
                            if info:
                                await _notify_trade_exit(symbol_btc, tf, side=info['side'], entry_price=info['entry_price'], exit_price=float(trig_px), reason=exit_reason, mode='paper', pnl_pct=info.get('pnl_pct'))
                        else:
                            await futures_close_all(symbol_btc, tf, exit_price=float(trig_px), reason=exit_reason)
                        continue



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


                channel = _get_channel_or_skip('BTC', tf)
                if channel is None:
                    continue

                chart_files = save_chart_groups(df, symbol_btc, tf)
                # 1) 짧은 알림(푸시용)
                await channel.send(content=short_msg)

                # 2) 분석 메시지
                await channel.send(
                    content=main_msg_pdf,
                    files=[discord.File(p) for p in chart_files if p],
                    silent=True
                )

                # 3) 종합해석 메시지
                if len(summary_msg_pdf) > 1900:
                    summary_msg_pdf = summary_msg_pdf[:1900] + "\n...(이하 생략)"
                await channel.send(summary_msg_pdf, silent=True)

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


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    content = message.content.strip()
    parts = content.split()
    # using global LAST_PRICE cache defined at module scope

    # [ANCHOR: CMD_SET_GET_SAVEENV]
    if content.startswith("!set "):
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

    if content.startswith("!get "):
        k = content[5:].strip()
        eff = cfg_get(k)
        ov = RUNTIME_CFG.get(k, None)
        await message.channel.send(f"🔎 {k}\n• effective: ```{eff}```\n• overlay: ```{ov}```")
        return

    if content.startswith("!saveenv"):
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
    if content.startswith("!pause"):
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

    if content.startswith("!resume"):
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

    # [ANCHOR: CMD_CLOSE_CLOSEALL]
    if content.startswith("!closeall"):
        try:
            n = 0
            # PAPER_POS key is "SYMBOL|TF" (string). Split safely.
            for key, pos in list(PAPER_POS.items()):
                try:
                    sym, tf = key.split("|", 1)
                except Exception:
                    continue

                fallback = float(pos.get("entry_price", 0.0))
                _paper_close(sym, tf, get_last_price(sym, fallback), "MANUAL")
                n += 1
            for tfk, sym in list(FUT_POS_TF.items()):
                await futures_close_all(sym, tfk, reason="MANUAL")
                n += 1
            # optional: clear all idempotence marks after mass close
            if CLEAR_IDEMP_ON_CLOSEALL:
                try: idem_clear_all()
                except Exception: pass
            await message.channel.send(f"🟢 closed all ({n})")
        except Exception as e:
            await message.channel.send(f"⚠️ closeall error: {e}")
        return

    if content.startswith("!close "):
        try:
            _, sym, tfx = content.split()
            if TRADE_MODE == "paper":

                _paper_close(sym.upper(), tfx, get_last_price(sym.upper(), 0.0), "MANUAL")

            else:
                await futures_close_symbol_tf(sym.upper(), tfx)
            await message.channel.send(f"🟢 closed {sym.upper()} {tfx}")
        except Exception as e:
            await message.channel.send(f"⚠️ close error: {e}")
        return

    # [ANCHOR: CMD_RISK_SET]
    if content.startswith("!risk "):
        try:
            _, sym, tfx, *rest = content.split()
            sym = sym.upper()
            args = " ".join(rest)
            def _parse_kv(s, k):
                m = re.search(rf"{k}\s*=\s*([0-9]+(\.[0-9]+)?)", s)
                return float(m.group(1)) if m else None
            tp = _parse_kv(args, "tp"); sl = _parse_kv(args, "sl"); tr = _parse_kv(args, "tr")
            key = f"{sym}|{tfx}"
            if TRADE_MODE == "paper" and key in PAPER_POS:
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

    if content.startswith("!help"):
        lines = [
            "• !set KEY=VALUE / !get KEY / !saveenv",
            "• !pause [SYMBOL|ALL] [TF|ALL] [mins?] / !resume [SYMBOL|ALL] [TF|ALL]",
            "• !close SYMBOL TF / !closeall",
            "• !risk SYMBOL TF tp=5 sl=2.5 tr=1.8",
        ]
        await message.channel.send("\n".join(lines))
        return

    # [ANCHOR: DIAG_CMD_CONFIG]
    if content.startswith("!config"):
        try:
            lines = [
                f"• ENABLE_OBSERVE: {cfg_get('ENABLE_OBSERVE','1')}",
                f"• ENABLE_COOLDOWN: {cfg_get('ENABLE_COOLDOWN','1')}",
                f"• STRONG_BYPASS_SCORE: {cfg_get('STRONG_BYPASS_SCORE','0.8')}",
                f"• GK_TTL_HOLD_SEC: {cfg_get('GK_TTL_HOLD_SEC','0.8')}",
                f"• GATEKEEPER_OBS_SEC: {cfg_get('GATEKEEPER_OBS_SEC','15m:20,1h:25,4h:40,1d:60')}",
                f"• WAIT_TARGET_ENABLE: {cfg_get('WAIT_TARGET_ENABLE','0')}",
                f"• TARGET_SCORE_BY_TF: {cfg_get('TARGET_SCORE_BY_TF')}",
                f"• WAIT_TARGET_SEC: {cfg_get('WAIT_TARGET_SEC')}",
                f"• TARGET_WAIT_MODE: {cfg_get('TARGET_WAIT_MODE','SOFT')}",
                f"• IGNORE_OCCUPANCY_TFS: {cfg_get('IGNORE_OCCUPANCY_TFS','')}",
                f"• TRADE_MODE: {cfg_get('TRADE_MODE','paper')}",
                f"• ROUTE_ALLOW: {cfg_get('ROUTE_ALLOW','*')}",
                f"• ROUTE_DENY: {cfg_get('ROUTE_DENY','')}",
            ]
            # [ANCHOR: CONFIG_EXT]
            lines.append(f"• STRENGTH_WEIGHTS: {cfg_get('STRENGTH_WEIGHTS')}")
            lines.append(f"• STRENGTH_BUCKETS: {cfg_get('STRENGTH_BUCKETS')}")
            lines.append(f"• MTF_FACTORS: {cfg_get('MTF_FACTORS')}")
            lines.append(f"• FULL_ALLOC_ON_ALL_ALIGN: {cfg_get('FULL_ALLOC_ON_ALL_ALIGN','1')}")
            lines.append(f"• SCALE_ENABLE: {cfg_get('SCALE_ENABLE')}")
            lines.append(f"• SCALE_MAX_LEGS: {cfg_get('SCALE_MAX_LEGS')}")
            lines.append(f"• SCALE_UP_SCORE_DELTA: {cfg_get('SCALE_UP_SCORE_DELTA')}")
            lines.append(f"• SCALE_DOWN_SCORE_DELTA: {cfg_get('SCALE_DOWN_SCORE_DELTA')}")
            lines.append(f"• SCALE_STEP_PCT: {cfg_get('SCALE_STEP_PCT')}")
            lines.append(f"• SCALE_REDUCE_PCT: {cfg_get('SCALE_REDUCE_PCT')}")
            lines.append(f"• SCALE_MIN_ADD_NOTIONAL_USDT: {cfg_get('SCALE_MIN_ADD_NOTIONAL_USDT')}")
            lines.append(f"• SCALE_REALLOCATE_BRACKETS: {cfg_get('SCALE_REALLOCATE_BRACKETS')}")
            lines.append(f"• SLIPPAGE_BY_SYMBOL: {cfg_get('SLIPPAGE_BY_SYMBOL')}")
            lines.append(f"• TP_PCT_BY_SYMBOL: {cfg_get('TP_PCT_BY_SYMBOL')}")
            lines.append(f"• SL_PCT_BY_SYMBOL: {cfg_get('SL_PCT_BY_SYMBOL')}")
            lines.append(f"• TRAIL_PCT_BY_SYMBOL: {cfg_get('TRAIL_PCT_BY_SYMBOL')}")
            lines.append(f"• EXIT_RESOLUTION: {EXIT_RESOLUTION}")
            lines.append(f"• EXIT_EVAL_MODE: {EXIT_EVAL_MODE}")
            lines.append(f"• EXIT_PRICE_SOURCE: {EXIT_PRICE_SOURCE}")
            lines.append(f"• OUTLIER_MAX_1M: {OUTLIER_MAX_1M}")
            lines.append(f"• RISK_INTERPRET_MODE: {RISK_INTERPRET_MODE}")
            lines.append(f"• APPLY_LEV_TO_TRAIL: {int(APPLY_LEV_TO_TRAIL)}")
            lines.append(f"• PAPER_CSV_CLOSE_LOG: {int(PAPER_CSV_CLOSE_LOG)}")
            lines.append(f"• FUTURES_CSV_CLOSE_LOG: {int(FUTURES_CSV_CLOSE_LOG)}")
            lines.append(f"• CLEAR_IDEMP_ON_CLOSEALL: {int(CLEAR_IDEMP_ON_CLOSEALL)}")
            lines.append(f"• DEFAULT_PAUSE: {cfg_get('DEFAULT_PAUSE','1')}")
            lines.append(f"• AFTER_CLOSE_PAUSE: {cfg_get('AFTER_CLOSE_PAUSE','1')}")
            lines.append(f"• DAILY_RESUME_HOUR_KST: {cfg_get('DAILY_RESUME_HOUR_KST','11')}")
            await message.channel.send("**CONFIG**\n" + "\n".join(lines))
        except Exception as e:
            await message.channel.send(f"config error: {e}")
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
        live_price_val = snap.get("mid") or snap.get("last")
        display_price = live_price_val if isinstance(live_price_val, (int, float)) else price

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
        await message.channel.send(
            content=main_msg_pdf,
            files=[discord.File(p) for p in chart_files if p],
            silent=True
        )


        if len(summary_msg_pdf) > 1900:
            summary_msg_pdf = summary_msg_pdf[:1900] + "\n...(이하 생략)"
        await message.channel.send(summary_msg_pdf, silent=True) # ← 조용히


    # ===== PDF 리포트 =====
    elif message.content.startswith("!리포트"):
        parts = message.content.split()
        try:
            symbol, tf = parse_symbol_tf(parts, default_symbol='ETH/USDT', default_tf='1h')
        except ValueError as ve:
            await message.channel.send(f"❌ {ve}")
            return

        if generate_pdf_report is None:
            await message.channel.send("❌ PDF 모듈 임포트에 실패했습니다. (generate_pdf_report=None)")
            return

        # [ANCHOR: REPORT_PRICE_SNAPSHOT_BEGIN]
        # 리포트에서도 전 TF와 동일한 '현재가 스냅샷'을 사용
        try:
            snap = await get_price_snapshot(symbol)
            report_price = snap.get("mid") or snap.get("last")
        except Exception:
            report_price = None
        try_live = None
        try_close = None
        # 마지막 보루: 스냅샷 실패 시 실시간/종가로 대체
        if not isinstance(report_price, (int, float)):
            try:
                try_live = fetch_live_price(symbol)
                report_price = try_live
            except Exception:
                try_live = None
        if not isinstance(report_price, (int, float)):
            # df가 있으면 마지막 종가로
            try:
                _df_tmp = get_ohlcv(symbol, tf, limit=2)
                if _df_tmp is not None and len(_df_tmp) > 0:
                    try_close = float(_df_tmp['close'].iloc[-1])
                    report_price = try_close
            except Exception:
                pass
        # [ANCHOR: REPORT_PRICE_SNAPSHOT_END]
        log(f"[REPORT] {symbol} {tf} price(report/live/close)={report_price}/{try_live}/{try_close}")

        df = get_ohlcv(symbol, tf, limit=300)
        df = add_indicators(df)

        # 분할 차트 생성 (PDF/첨부 둘 다 사용)
        chart_files   = save_chart_groups(df, symbol, tf)
        ichimoku_file = save_ichimoku_chart(df, symbol, tf)

        df_1d = get_ohlcv(symbol, '1d', limit=300)
        signal, price, rsi, macd, reasons, score, weights, agree_long, agree_short, weights_detail = calculate_signal(df,tf, symbol)

        # 일봉 변동률 계산

        display_price = report_price
        # [ANCHOR: daily_change_unify_eth_alt]
        daily_change_pct = calc_daily_change_pct(symbol, display_price)



        main_msg_pdf, summary_msg_pdf, _short_msg_pdf = format_signal_message(
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
            daily_change_pct=daily_change_pct,
            live_price=report_price,
            show_risk=False
        )
        msg_for_pdf = f"{main_msg_pdf}\n\n{summary_msg_pdf}"

        # 🟢 심볼·타임프레임 인자를 다른 함수에도 반영
        score_file = plot_score_history(symbol, tf)
        perf_file  = analyze_performance_for(symbol, tf)
        performance_file = generate_performance_stats(tf, symbol=symbol)

        display_price = sanitize_price_for_tf(symbol, tf, price)

        pdf_path = generate_pdf_report(
            df=df,
            tf=tf,
            symbol=symbol,
            signal=signal,
            price=report_price,
            score=score,
            reasons=reasons,
            weights=weights,
            agree_long=agree_long,
            agree_short=agree_short,
            now=datetime.now(),
            chart_imgs=chart_files,                 # ✅ 분할차트 리스트
            ichimoku_img=ichimoku_file,             # ✅ 이치모쿠
            discord_message=msg_for_pdf,
            daily_change_pct=daily_change_pct
        )


        # 심볼별 로그 저장

        log_to_csv(symbol, tf, signal, report_price, rsi, macd, None, None, None, score, reasons, weights)

        # 빈 메시지 가드 적용
        # 보고서 안내 문구
        content = f"📄 요청하신 {symbol} {tf} 보고서입니다."
        files = [p for p in [*chart_files, ichimoku_file, pdf_path, score_file, perf_file, performance_file] if p and os.path.exists(p)]
        await message.channel.send(content=content, files=[discord.File(p) for p in files] if files else None)



    # ===== 신호 이력 조회 =====
    elif message.content.startswith("!이력"):
        tf = parts[1] if len(parts) > 1 else "1h"
        import csv, glob, os
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
    # [ANCHOR: BOOT_ENV_SUMMARY]
    try:
        log("[BOOT] ENV SUMMARY: "
            f"OBS={os.getenv('GATEKEEPER_OBS_SEC','-')}, "
            f"COOLDOWN={os.getenv('POST_EXIT_COOLDOWN_SEC','-')}, "
            f"WAIT_TARGET={os.getenv('WAIT_TARGET_ENABLE','0')}/"
            f"{os.getenv('TARGET_SCORE_BY_TF','-')}/"
            f"{os.getenv('WAIT_TARGET_SEC','-')}/"
            f"{os.getenv('TARGET_WAIT_MODE','-')}, "
            f"IGNORE_OCCUPANCY_TFS={os.getenv('IGNORE_OCCUPANCY_TFS','')}, "
            f"TRADE_MODE={os.getenv('TRADE_MODE','paper')}"
        )
    except Exception as _e:
        log(f"[BOOT] ENV SUMMARY warn: {_e}")
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


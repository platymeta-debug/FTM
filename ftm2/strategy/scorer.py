# [ANCHOR:SCORER]
from __future__ import annotations
import math


def _clamp01(x): return max(0.0, min(1.0, x))
def _scale01(x, lo, hi):
    if hi == lo: return 0.0
    return _clamp01((x - lo) / (hi - lo))

def _vboost(vol, vol_ma, wv):
    ratio = 0.0
    if vol_ma and vol_ma > 0:
        ratio = max(0.5, min(2.0, (vol or 0.0) / vol_ma))
    return 1.0 + (wv * (ratio - 1.0))  # 0.5~2.0 -> 1.0±wv

def regime_trend(row, cfg) -> bool:
    adx_ok = (row.get('adx', 0.0) >= cfg.REGIME_ADX_MIN)
    ema_ok = ( (row.get('ema_fast',0.0) - row.get('ema_slow',0.0)) > 0 )
    cloud_ok = (row.get('close',0.0) > min(row.get('senkou_a',0.0), row.get('senkou_b',0.0)))
    return adx_ok and (ema_ok or cloud_ok)

def mtf_bias(mtf_ctx: dict[str, dict] | None) -> int:
    """
    mtf_ctx: {'5m': {'ema_fast':..,'ema_slow':..,'senkou_a':..,'senkou_b':..,'close':..}, '15m': {...}}
    return: +1 (동의), -1 (반대), 0 (중립)
    """
    if not mtf_ctx: return 0
    votes = 0
    for tf, r in mtf_ctx.items():
        if r is None: continue
        ema_ok = (r.get('ema_fast',0.0) - r.get('ema_slow',0.0)) > 0
        cloud_ok = r.get('close',0.0) > min(r.get('senkou_a',0.0), r.get('senkou_b',0.0))
        if ema_ok or cloud_ok: votes += 1
        elif (r.get('close',0.0) < max(r.get('senkou_a',0.0), r.get('senkou_b',0.0)) and (r.get('ema_fast',0.0) - r.get('ema_slow',0.0)) < 0):
            votes -= 1
    if votes > 0: return +1
    if votes < 0: return -1
    return 0

def score_row(row: dict, cfg, mtf_ctx: dict | None = None):
    # safety
    atr = max(row.get('atr', 0.0), 1e-9)
    vol_boost = _vboost(row.get('volume'), row.get('vol_ma', 0.0), cfg.SCORE_W_VOL)

    # --- regime ---
    is_trend = regime_trend(row, cfg)

    # --- Trend components ---
    ema_spread = (row.get('ema_fast',0.0) - row.get('ema_slow',0.0)) / atr
    long_tr  = _scale01( ema_spread,  0.0, 1.5)
    short_tr = _scale01(-ema_spread,  0.0, 1.5)

    # ichimoku (0..1 within)
    cloud_long  = 1.0 if row.get('close',0.0) > max(row.get('senkou_a',0.0), row.get('senkou_b',0.0)) else 0.0
    cloud_short = 1.0 if row.get('close',0.0) < min(row.get('senkou_a',0.0), row.get('senkou_b',0.0)) else 0.0
    tk_long  = 1.0 if row.get('tenkan',0.0) > row.get('kijun',0.0) else 0.0
    tk_short = 1.0 if row.get('tenkan',0.0) < row.get('kijun',0.0) else 0.0

    # kama slope
    kama_slope = ((row.get('kama',0.0) - row.get('kama_prev', row.get('kama',0.0))) / atr)
    long_tr += 0.2 * _scale01(kama_slope, 0.0, 0.8)
    short_tr+= 0.2 * _scale01(-kama_slope,0.0, 0.8)

    # supertrend direction
    if row.get('st_dir', 0) == 1: long_tr += 0.15
    if row.get('st_dir', 0) == -1: short_tr += 0.15

    # clip
    long_tr = _clamp01(0.4*cloud_long + 0.4*tk_long + 0.2*_clamp01(long_tr))
    short_tr= _clamp01(0.4*cloud_short+ 0.4*tk_short+ 0.2*_clamp01(short_tr))

    # --- Momentum ---
    rsi = row.get('rsi', 50.0)
    macd_hist = row.get('macd_hist', 0.0)
    # hist slope proxy: current - prev
    macd_hist_prev = row.get('macd_hist_prev', macd_hist)
    d_hist = macd_hist - macd_hist_prev

    long_mom  = _clamp01( 0.6*_scale01(rsi, 50, 70) + 0.4*_scale01(d_hist, 0.0, 1.5*atr) )
    short_mom = _clamp01( 0.6*_scale01(50-(rsi-50), 50, 70) + 0.4*_scale01(-d_hist, 0.0, 1.5*atr) )

    # CCI
    cci = row.get('cci', 0.0)
    long_mom  += 0.2*_scale01( cci, 0.0, 150)
    short_mom += 0.2*_scale01(-cci, 0.0, 150)
    long_mom  = _clamp01(long_mom)
    short_mom = _clamp01(short_mom)

    # --- Breakout ---
    brk_long = 0.0; brk_short = 0.0
    if row.get('close',0.0) >= (row.get('donch_hi') or float('inf')):
        brk_long += 0.2
    if row.get('close',0.0) <= (row.get('donch_lo') or -float('inf')):
        brk_short += 0.2

    # --- Mean Reversion (BB Z & VWAP) ---
    z = row.get('z', 0.0)
    long_mr  = _scale01(-z, 0.5, 2.0)
    short_mr = _scale01( z, 0.5, 2.0)
    # vwap pullback: price near vwap & trend direction -> small bonus
    vwap_ref = row.get('vwap_day') or row.get('vwap_roll')
    if vwap_ref:
        dist = abs(row.get('close',0.0) - vwap_ref) / max(1e-9, row.get('atr',1.0))
        near = 1.0 - _clamp01(dist/1.5)  # within ~1.5*ATR
        if is_trend and ema_spread>0:
            long_mr += 0.2*near
        if is_trend and ema_spread<0:
            short_mr+= 0.2*near
    # regime gate
    if is_trend:
        long_mr *= 0.5; short_mr *= 0.5

    # --- Volume/Flow (OBV slope) ---
    obv_slope = row.get('obv_slope', 0.0)
    long_mom  += 0.15*_scale01( obv_slope, 0.0, 0.5)
    short_mom += 0.15*_scale01(-obv_slope,0.0, 0.5)
    long_mom  = _clamp01(long_mom)
    short_mom = _clamp01(short_mom)

    # --- combine by weights (regime-aware) ---
    wT, wM, wR, wB = cfg.SCORE_W_TREND, cfg.SCORE_W_MOM, cfg.SCORE_W_MR, cfg.SCORE_W_BRK
    if is_trend:
        # trend regime: trend/mom/brk 우세
        wT, wM, wR, wB = 0.45, 0.25, 0.05, 0.20
    else:
        # range regime: mean reversion 우세
        wT, wM, wR, wB = 0.20, 0.25, 0.40, 0.10

    long_score  = _clamp01(wT*long_tr  + wM*long_mom  + wR*long_mr  + wB*brk_long)
    short_score = _clamp01(wT*short_tr + wM*short_mom + wR*short_mr + wB*brk_short)

    # --- MTF confluence ---
    if cfg.MTF_USE and mtf_ctx:
        bias = mtf_bias(mtf_ctx)
        if bias > 0:
            long_score  = _clamp01(long_score  * cfg.MTF_CONFLUENCE_BOOST)
            short_score = _clamp01(short_score * cfg.MTF_CONTRA_DAMP)
        elif bias < 0:
            short_score = _clamp01(short_score * cfg.MTF_CONFLUENCE_BOOST)
            long_score  = _clamp01(long_score  * cfg.MTF_CONTRA_DAMP)

    # --- volume boost (multiplicative) ---
    long_score  = _clamp01(long_score  * vol_boost)
    short_score = _clamp01(short_score * vol_boost)

    return int(round(long_score*100)), int(round(short_score*100)), is_trend

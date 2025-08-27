# [ANCHOR:INDICATORS_CORE]
from __future__ import annotations
import pandas as pd
import numpy as np

# ---------- basic MAs ----------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

# ---------- ranges / ATR / ADX ----------
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - prev_close).abs(),
        (df['low']  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return true_range(df).rolling(n).mean()

def adx(df: pd.DataFrame, n: int=14) -> pd.Series:
    high, low = df['high'], df['low']
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atrn = tr.rolling(n).mean()
    plus_di  = 100 * (pd.Series(plus_dm, index=df.index).rolling(n).mean() / (atrn + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(n).mean() / (atrn + 1e-12))
    dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12))
    return dx.rolling(n).mean()

# ---------- momentum ----------
def rsi(series: pd.Series, n: int=14) -> pd.Series:
    delta = series.diff()
    up = pd.Series(np.where(delta>0, delta, 0.0), index=series.index)
    dn = pd.Series(np.where(delta<0, -delta, 0.0), index=series.index)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100/(1+rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ef = ema(series, fast); es = ema(series, slow)
    macd = ef - es
    sig  = ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def cci(df: pd.DataFrame, n: int=20) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    sma = tp.rolling(n).mean()
    md  = (tp - sma).abs().rolling(n).mean()
    return (tp - sma) / (0.015 * (md + 1e-12))

# ---------- mean reversion ----------
def bollinger(df: pd.DataFrame, n: int=20, k: float=2.0):
    mid = df['close'].rolling(n).mean()
    std = df['close'].rolling(n).std(ddof=0)
    up  = mid + k*std
    dn  = mid - k*std
    z   = (df['close'] - mid) / (std.replace(0, np.nan))
    return mid, up, dn, z

# ---------- breakout ----------
def donchian(df: pd.DataFrame, n: int=20):
    upper = df['high'].rolling(n).max()
    lower = df['low'].rolling(n).min()
    return upper, lower

# ---------- Ichimoku ----------
def ichimoku(df: pd.DataFrame, tenkan=9, kijun=26, senkoub=52):
    tenkan_sen = (df['high'].rolling(tenkan).max() + df['low'].rolling(tenkan).min())/2
    kijun_sen  = (df['high'].rolling(kijun).max() + df['low'].rolling(kijun).min())/2
    senkou_a   = ((tenkan_sen + kijun_sen)/2)
    senkou_b   = (df['high'].rolling(senkoub).max() + df['low'].rolling(senkoub).min())/2
    chikou_span = df['close'].shift(-kijun)  # 관습상 뒤로 시프트, 실시간 비교는 동일 시점 사용
    return tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_span

# ---------- KAMA ----------
def kama(series: pd.Series, er_period=10, fast=2, slow=30) -> pd.Series:
    s = series.copy()
    change = s.diff(er_period).abs()
    vol = s.diff().abs().rolling(er_period).sum()
    er = change / (vol + 1e-12)
    fast_sc = 2/(fast+1); slow_sc = 2/(slow+1)
    sc = (er*(fast_sc - slow_sc) + slow_sc)**2
    out = s.copy()
    out.iloc[:er_period] = s.iloc[:er_period]
    for i in range(er_period, len(s)):
        out.iloc[i] = out.iloc[i-1] + sc.iloc[i] * (s.iloc[i] - out.iloc[i-1])
    return out

# ---------- VWAP ----------
def vwap_rolling(df: pd.DataFrame, n: int=500) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close'])/3.0
    pv = (tp * df['volume']).rolling(n).sum()
    vv = df['volume'].rolling(n).sum()
    return pv / (vv.replace(0, np.nan))

def vwap_daily_anchor(df: pd.DataFrame, anchor_hour: int=0) -> pd.Series:
    # UTC 기준 anchor_hour 시점 이후 누적
    idx = df.index
    day_key = (idx.tz_convert("UTC").floor("D") + pd.to_timedelta(anchor_hour, unit="h"))
    grp = day_key
    tp = (df['high'] + df['low'] + df['close'])/3.0
    pv = (tp * df['volume']).groupby(grp).cumsum()
    vv = df['volume'].groupby(grp).cumsum()
    return pv / (vv.replace(0, np.nan))

# ---------- Volume/Flow ----------
def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['close'].diff().fillna(0.0))
    return (direction * df['volume']).fillna(0.0).cumsum()

# ---------- Supertrend ----------
def supertrend(df: pd.DataFrame, atr_len=10, mult=3.0):
    hl2 = (df['high'] + df['low']) / 2.0
    atrv = atr(df, atr_len)
    upper = hl2 + mult*atrv
    lower = hl2 - mult*atrv
    dir = pd.Series(index=df.index, dtype=int)
    if len(df) > 0: dir.iloc[0] = 1
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper.iloc[i-1]:
            dir.iloc[i] = 1
        elif df['close'].iloc[i] < lower.iloc[i-1]:
            dir.iloc[i] = -1
        else:
            dir.iloc[i] = dir.iloc[i-1]
            if dir.iloc[i] == 1:
                lower.iloc[i] = max(lower.iloc[i], lower.iloc[i-1])
            else:
                upper.iloc[i] = min(upper.iloc[i], upper.iloc[i-1])
    return dir, upper, lower

# ---------- master ----------
def add_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()
    # 보장: DatetimeIndex (UTC)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex (UTC).")

    # moving averages & trend
    df['ema_fast']  = ema(df['close'], cfg.EMA_FAST)
    df['ema_slow']  = ema(df['close'], cfg.EMA_SLOW)
    df['ema_trend'] = ema(df['close'], cfg.EMA_TREND)
    df['kama']      = kama(df['close'], cfg.KAMA_ER, cfg.KAMA_FAST, cfg.KAMA_SLOW)

    # volatility / strength
    df['atr'] = atr(df, cfg.ATR_LEN)
    df['adx'] = adx(df, cfg.ADX_LEN)

    # momentum
    df['rsi'] = rsi(df['close'], cfg.RSI_LEN)
    df['macd'], df['macd_sig'], df['macd_hist'] = macd(df['close'])
    df['cci'] = cci(df, cfg.CCI_LEN)

    # mean reversion
    df['bb_mid'], df['bb_up'], df['bb_dn'], df['z'] = bollinger(df, cfg.BB_LEN, cfg.BB_K)
    df['vwap_roll'] = vwap_rolling(df, cfg.VWAP_ROLL_N)
    df['vwap_day']  = vwap_daily_anchor(df, cfg.VWAP_DAILY_ANCHOR_HOUR)

    # breakout & structure
    df['donch_hi'], df['donch_lo'] = donchian(df, cfg.DONCHIAN_LEN)

    # ichimoku
    df['tenkan'], df['kijun'], df['senkou_a'], df['senkou_b'], df['chikou'] = ichimoku(
        df, cfg.ICHI_TENKAN, cfg.ICHI_KIJUN, cfg.ICHI_SENKOUB
    )

    # volume / flow
    df['vol_ma'] = df['volume'].rolling(cfg.VOL_MA_LEN).mean()
    df['obv']    = obv(df)
    df['obv_slope'] = (df['obv'] - df['obv'].shift(cfg.OBV_SLOPE_LEN)) / max(1, cfg.OBV_SLOPE_LEN)

    # supertrend
    df['st_dir'], df['st_up'], df['st_lo'] = supertrend(df, cfg.ST_ATR_LEN, cfg.ST_MULT)
    return df

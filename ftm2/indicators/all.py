# ftm2/indicators/all.py
from __future__ import annotations
import numpy as np
import pandas as pd

def add_indicators(
    df: pd.DataFrame,
    *,
    ema_fast: int = 50,
    ema_slow: int = 200,
    rsi_len: int = 14,
    atr_len: int = 14,
    adx_len: int = 14,
    cci_len: int = 20,
    kama_len: int = 10,
) -> pd.DataFrame:
    """
    df: 반드시 'open','high','low','close','volume' 컬럼 보유
    반환: 지표 컬럼을 추가한 동일 DataFrame (in-place 스타일)
    """
    # 타입 강제
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float)

    # EMA
    df["ema_fast"] = close.ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=ema_slow, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/rsi_len, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_len, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50.0)

    # ATR
    tr = np.maximum(high - low,
         np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()))
    df["atr"] = tr.rolling(atr_len).mean()

    # ADX (간단 구현)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_n = tr.rolling(adx_len).sum()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(adx_len).sum() / tr_n)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(adx_len).sum() / tr_n)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.rolling(adx_len).mean()

    # CCI
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(cci_len).mean()
    mad = (tp - sma_tp).abs().rolling(cci_len).mean()
    df["cci"] = (tp - sma_tp) / (0.015 * mad)

    # OBV
    obv_step = np.where(close > close.shift(), vol,
                np.where(close < close.shift(), -vol, 0.0))
    df["obv"] = pd.Series(obv_step, index=df.index).cumsum()

    # VWAP(누적 방식)
    df["vwap"] = (tp * vol).cumsum() / vol.cumsum()

    # Ichimoku (표준 파라미터)
    conv = (high.rolling(9).max() + low.rolling(9).min()) / 2.0
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2.0
    span_a = ((conv + base) / 2.0).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2.0).shift(26)
    chikou = close.shift(-26)
    df["tenkan"] = conv
    df["kijun"] = base
    df["senkou_a"] = span_a
    df["senkou_b"] = span_b
    df["chikou"] = chikou

    # KAMA(간단 구현)
    n = kama_len
    change = close.diff(n).abs()
    volatility = close.diff().abs().rolling(n).sum()
    er = (change / volatility).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    fast = 2 / (2 + 1)     # 2
    slow = 2 / (30 + 1)    # 30
    sc = (er * (fast - slow) + slow) ** 2
    kama = pd.Series(index=df.index, dtype=float)
    if len(close) > 0:
        kama.iloc[0] = close.iloc[0]
        for i in range(1, len(close)):
            prev = kama.iloc[i-1]
            kama.iloc[i] = prev + sc.iloc[i] * (close.iloc[i] - prev)
    df["kama"] = kama

    return df

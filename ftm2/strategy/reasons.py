from __future__ import annotations

"""지표 해석 함수 모음."""


def interpret_rsi(v: float) -> str:
    if v < 30:
        return "과매도"
    if v < 40:
        return "약한 과매도"
    if v < 55:
        return "중립"
    if v < 70:
        return "약한 과매수"
    return "과매수"


def interpret_adx(adx: float, plus_di: float, minus_di: float) -> str:
    if adx < 15:
        return "추세 약함"
    if adx < 25:
        return "추세 형성 가능"
    dir_txt = "상승" if plus_di > minus_di else "하락"
    return f"추세 강함 · {dir_txt} 우위"


def interpret_bb(close: float, mid: float, up: float, dn: float) -> str:
    if close > up:
        return "밴드 상단 돌파"
    if close < dn:
        return "밴드 하단 이탈"
    if close > mid:
        return "중앙선 위"
    return "중앙선 아래"


def interpret_ema(close: float, ema50: float, ema200: float) -> str:
    if close > ema50 > ema200:
        return "상승 정배열"
    if close < ema50 < ema200:
        return "하락 역배열"
    return "혼조"


def interpret_cci(v: float) -> str:
    if v > 100:
        return "강한 양의 모멘텀"
    if v < -100:
        return "강한 음의 모멘텀"
    return "중립"


def interpret_obv_slope(s: float) -> str:
    if s > 0:
        return "OBV 증가"
    if s < 0:
        return "OBV 감소"
    return "OBV 정체"


def interpret_kama_slope(s: float) -> str:
    if s > 0:
        return "KAMA 상승"
    if s < 0:
        return "KAMA 하락"
    return "KAMA 정체"

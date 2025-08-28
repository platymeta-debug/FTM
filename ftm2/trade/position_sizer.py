# [ANCHOR:POSITION_SIZER]
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import os

@dataclass
class SizingDecision:
    side: Literal["LONG","SHORT"]
    qty: float
    entry_type: Literal["market","limit"]
    limit_offset_ticks: int
    sl: float
    tp: float
    reason: str
    is_add: bool = False  # 추가진입 여부

def _tier_weight(score: int, bins: list[int], weights: list[float]) -> float:
    w = 0.0
    for b, wt in zip(bins, weights):
        if score >= b: w = wt
    return max(0.25, w)  # 최저 0.25x

def _risk_qty_usdm(atr: float, cfg) -> float:
    # 손절에 닿을 때 손실액이 RISK_TARGET_USDT가 되도록 qty 산출
    stop_dist = atr * cfg.SL_MULT
    if stop_dist <= 0: return 0.0
    return cfg.RISK_TARGET_USDT / stop_dist

def sizing_decision(
    symbol: str,
    side: str,
    score_long: int,
    score_short: int,
    price: float,
    atr: float,
    filters,            # ExchangeFilters
    pos_state,          # 현재 보유 포지션 {side, qty, entryPrice, adds:int, last_add_price}
    cfg,
    is_trend: bool,
    mtf_bias: int       # +1/-1/0
) -> Optional[SizingDecision]:
    # 1) 점수→비율 (레짐/MTF 반영)
    bins = [int(b) for b in cfg.TIER_BINS]
    weights = [float(w) for w in cfg.TIER_WEIGHTS]

    raw_weight = _tier_weight(score_long if side=="LONG" else score_short, bins, weights)
    # 레짐/MTF 보정
    if is_trend and side=="LONG": raw_weight *= 1.10
    if is_trend and side=="SHORT": raw_weight *= 1.10
    if mtf_bias>0 and side=="LONG": raw_weight *= 1.05
    if mtf_bias<0 and side=="SHORT": raw_weight *= 1.05

    # 2) 기본 수량 (ATR 위험단위 기반)
    base_qty = _risk_qty_usdm(atr, cfg)
    target_qty = base_qty * raw_weight

    # 3) 추가진입 규칙
    is_add = False
    if pos_state and pos_state.get("side")==side:
        # 유리한 방향으로 STEP_ATR 만큼 이동했거나, 추세 유지 하에 PULLBACK에서만 허용
        adds = int(pos_state.get("adds",0))
        if adds < cfg.SCALE_IN_MAX_ADDS:
            entry = float(pos_state.get("entryPrice", price))
            moved_favor = (price - entry)/atr if side=="LONG" else (entry - price)/atr
            pullback = (entry - price)/atr if side=="LONG" else (price - entry)/atr
            can_add = moved_favor >= cfg.SCALE_IN_STEP_ATR
            if cfg.DCA_USE_PULLBACK and not can_add:
                can_add = pullback >= cfg.PULLBACK_ADD_ATR and is_trend
            if can_add:
                # add는 현재 수량의 50~100%까지 (과도 확장 방지)
                target_qty = max(target_qty, float(pos_state.get("qty",0))*0.5)
                is_add = True
            else:
                return None  # 조건 불충족 시 미진입
        else:
            return None

    min_env = float(os.getenv(f"MIN_QTY_{symbol}", "0"))
    target_qty = max(target_qty, min_env)

    # 4) SL/TP (ATR 기반, TP는 R multiple)
    sl = price - cfg.SL_MULT*atr if side=="LONG" else price + cfg.SL_MULT*atr
    tp = price + cfg.TP_MULT*atr if side=="LONG" else price - cfg.TP_MULT*atr

    # 5) 엔트리 타입
    entry_type = cfg.ENTRY_ORDER
    limit_offset_ticks = cfg.LIMIT_OFFSET_TICKS

    # 6) 필터 정량화 (호출은 router에서 수행)
    return SizingDecision(
        side=side, qty=target_qty, entry_type=entry_type,
        limit_offset_ticks=limit_offset_ticks, sl=sl, tp=tp,
        reason=f"{'추가진입' if is_add else '초기진입'}: weight={raw_weight:.2f}, atr={atr:.4f}",
        is_add=is_add
    )

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

    # 6) 최소 명목가 충족(있으면 보정, 아니면 미진입)
    try:
        from ftm2.exchange.quantize import ExchangeFilters
        if isinstance(filters, ExchangeFilters):
            filters.use(symbol)
            if not filters.min_ok(price, target_qty):
                if cfg.ORDER_SCALE_TO_MIN:
                    q_min = filters.min_qty_for(price, symbol=symbol)
                    if q_min and q_min > 0:
                        target_qty = float(filters.q_qty(symbol, q_min))
                    else:
                        return None
                else:
                    return None
            if target_qty <= 0:
                return None
    except Exception:
        pass

    # 6) 필터 정량화 (호출은 router에서 수행)
    return SizingDecision(
        side=side, qty=target_qty, entry_type=entry_type,
        limit_offset_ticks=limit_offset_ticks, sl=sl, tp=tp,
        reason=f"{'추가진입' if is_add else '초기진입'}: weight={raw_weight:.2f}, atr={atr:.4f}",
        is_add=is_add
    )

# [ANCHOR:SIZE_ENTRY_TICKET_AWARE]
from decimal import Decimal

class PositionSizer:
    def __init__(self, cfg, filters):
        self.cfg = cfg
        self.filters = filters

    def size_entry(self, sym: str, ticket, account):
        """
        ticket: SetupTicket (entry_px, stop_px, side, score)
        account: 계정 정보 (availableBalance 등)
        """
        px = Decimal(str(ticket.entry_px))
        sl = Decimal(str(ticket.stop_px))
        dist = abs(px - sl)

        risk_pct = Decimal(str(self.cfg.RISK_PCT_OVERRIDE.get(sym, self.cfg.RISK_PCT_DEFAULT)))
        boost = Decimal("1.0") + Decimal(str(max(0, abs(getattr(ticket, 'score', 0)) - 60))) / Decimal("200")
        risk_pct = (risk_pct * min(boost, Decimal("1.5"))).quantize(Decimal("0.0001"))

        eq = Decimal(str(getattr(account, 'equity_usdt', 0)))
        risk_usdt = (eq * risk_pct / Decimal("100")).quantize(Decimal("0.01"))

        if dist <= 0 or risk_usdt <= 0:
            return 0.0

        lev = Decimal(str(self.cfg.LEVERAGE_OVERRIDE.get(sym, self.cfg.LEVERAGE_DEFAULT)))
        qty = (risk_usdt / dist).quantize(Decimal("0.0000001"))
        qty = min(qty, Decimal(str(self.cfg.MAX_QTY_OVERRIDE.get(sym, self.cfg.MAX_QTY_DEFAULT))))
        qty = max(qty, Decimal("0"))

        q = self.filters.q_qty(sym, float(qty))
        min_ok = self.filters.min_ok(float(px), q)
        if not min_ok:
            q_min = self.filters.min_qty_for(float(px), symbol=sym)
            if self.cfg.ORDER_SCALE_TO_MIN:
                q = float(q_min)
            else:
                return 0.0

        return float(q)

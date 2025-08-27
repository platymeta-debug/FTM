# [ANCHOR:ORDER_ROUTER]
from __future__ import annotations
import time, math
from typing import Literal, Optional
from ftm2.exchange.binance_client import BinanceClient
from ftm2.notify.discord_bot import send_log, send_trade
from ftm2.trade.position_sizer import SizingDecision
from ftm2.exchange.quantize import ExchangeFilters

def _cid(sym: str, side: str) -> str:
    return f"FTM2_{int(time.time()*1000)}_{sym}_{side}"

def quantize(filters: ExchangeFilters, price: float, qty: float):
    return filters.q_price(price), filters.q_qty(qty)

class OrderRouter:
    def __init__(self, cfg, filters: ExchangeFilters):
        self.cfg = cfg
        self.filters = filters
        self.bx = BinanceClient()

    def place_entry(self, symbol: str, dec: SizingDecision, mark_price: float):
        p, q = dec.sl, dec.qty
        entry_price = mark_price
        # limit 오더면 틱 오프셋 반영
        if dec.entry_type == "limit":
            tick = self.filters.tick_size(symbol)
            entry_price = mark_price * (1 - dec.limit_offset_ticks*tick) if dec.side=="LONG" else mark_price * (1 + dec.limit_offset_ticks*tick)
        q_price, q_qty = quantize(self.filters, entry_price, q)
        if q_qty <= 0:
            send_log(f"❗[{symbol}] 수량이 0으로 정량화됨(필터 위반)."); return None

        side = "BUY" if dec.side=="LONG" else "SELL"
        params = dict(symbol=symbol, side=side, type="MARKET" if dec.entry_type=="market" else "LIMIT",
                      quantity=q_qty, newClientOrderId=_cid(symbol, side))
        if dec.entry_type=="limit":
            params.update(price=q_price, timeInForce=self.cfg.TIME_IN_FORCE)
        try:
            od = self.bx.new_order(**params)
            send_trade(f"✅ 진입 주문 전송: {symbol} {dec.side} 수량 {q_qty} / {dec.reason}")
            return od
        except Exception as e:
            send_log(f"🚫 진입 주문 실패: {symbol} {e}")
            return None

    def place_brackets(self, symbol: str, side: str, qty: float, entry_price: float, sl: float, tp: float):
        reduce_side = "SELL" if side=="LONG" else "BUY"
        sl_price, sl_qty = quantize(self.filters, sl, qty)
        tp_price, tp_qty = quantize(self.filters, tp, qty)
        # SL: 시장가(스톱), TP: 리밋(reduceOnly)
        # Futures는 동시에 존재 가능. closePosition 모드는 전체 청산용.
        try:
            if self.cfg.SL_ORDER=="market":
                self.bx.new_order(symbol=symbol, side=reduce_side, type="STOP_MARKET",
                                  stopPrice=sl_price, reduceOnly=True,
                                  workingType=self.cfg.WORKING_TYPE, newClientOrderId=_cid(symbol,"SL"))
            if self.cfg.TP_ORDER=="limit":
                self.bx.new_order(symbol=symbol, side=reduce_side, type="TAKE_PROFIT",
                                  price=tp_price, stopPrice=tp_price, timeInForce=self.cfg.TIME_IN_FORCE,
                                  reduceOnly=True, workingType=self.cfg.WORKING_TYPE, newClientOrderId=_cid(symbol,"TP"))
            send_trade(f"📎 브래킷 설정: SL≈{sl_price}, TP≈{tp_price} (reduceOnly)")
        except Exception as e:
            send_log(f"⚠️ 브래킷 설정 실패: {e}")

    # 추적손절(트레일) 계산 헬퍼 — R 단위
    def trail_price(self, entry: float, atr: float, side: str, r_unreal: float, cfg):
        if r_unreal < cfg.TRAIL_START_R: return None
        # 스텝 R 당 SL을 R_BACK만큼 좁힘
        steps = math.floor((r_unreal - cfg.TRAIL_START_R)/cfg.TRAIL_STEP_R)
        tighten = cfg.TRAIL_BACK_R + steps*0.1
        tighten = min(tighten, cfg.SL_MULT)  # 너무 과도하게 좁히지 않음
        if side=="LONG":
            return entry + (cfg.SL_MULT - tighten)*atr
        else:
            return entry - (cfg.SL_MULT - tighten)*atr


def close_position_all(symbol: str) -> str:
    """간단한 reduceOnly 시장가 청산."""
    bx = BinanceClient()
    try:
        # 양방향 모두 reduceOnly 시장가 시도
        bx.new_order(symbol=symbol, side="BUY", type="MARKET", reduceOnly=True)
        bx.new_order(symbol=symbol, side="SELL", type="MARKET", reduceOnly=True)
        send_trade(f"🔻 {symbol} 전량 청산 주문 전송")
        return f"{symbol} 청산 주문 전송"
    except Exception as e:
        send_log(f"⚠️ {symbol} 청산 실패: {e}")
        return f"{symbol} 청산 실패: {e}"

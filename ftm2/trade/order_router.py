# [ANCHOR:ORDER_ROUTER]
from __future__ import annotations
import time, math
from typing import Literal, Optional
from ftm2.exchange.binance_client import BinanceClient
from ftm2.notify import dispatcher
from ftm2.notify.dispatcher import send_signal
from ftm2.trade.position_sizer import SizingDecision
from ftm2.exchange.quantize import ExchangeFilters
from ftm2.strategy.trace import DecisionTrace

CSV = None


def log_decision(trace: DecisionTrace):
    print(
        f"[DECISION][{trace.symbol}] dir={trace.direction} "
        f"score={trace.decision_score:+.1f} reasons={trace.reasons} gates={trace.gates}"
    )

    if "ENTER" in trace.reasons:
        send_signal(
            f"{trace.symbol} 진입 신호 → {trace.direction} / {trace.decision_score:+.1f}"
        )
    else:
        send_signal(
            f"{trace.symbol} 의도만: {trace.direction} / {trace.decision_score:+.1f} "
            f"/ 사유: {', '.join(trace.reasons)}"
        )


def _cid(sym: str, side: str) -> str:
    return f"FTM2_{int(time.time()*1000)}_{sym}_{side}"

def quantize(filters: ExchangeFilters, price: float, qty: float):
    return filters.q_price(price), filters.q_qty(qty)

class OrderRouter:
    def __init__(self, cfg, filters: ExchangeFilters, rt=None, market=None, notify=dispatcher):
        self.cfg = cfg
        self.filters = filters
        self.bx = BinanceClient()
        self.live_allowed = False
        self.rt = rt
        self.market = market
        self.notify = notify

    def allow_live(self):
        self.live_allowed = True

    def _live_guard(self, symbol: str, qty: float, price: float, trace: DecisionTrace | None = None):
        if self.cfg.TRADE_MODE != "live" or not self.cfg.LIVE_GUARD_ENABLE:
            return True
        if not self.live_allowed:
            if trace:
                trace.reasons.append("live trading disabled")
            return False
        notional = qty * price
        if trace:
            trace.gates.update({
                "MIN_NOTIONAL_USDT": self.cfg.LIVE_MIN_NOTIONAL_USDT,
                "NOTIONAL_USDT": notional,
            })
        if notional < self.cfg.LIVE_MIN_NOTIONAL_USDT:
            if trace:
                trace.reasons.append("below min notional")

            return False
        return True

    def place_entry(self, symbol: str, dec: SizingDecision, mark_price: float, trace: DecisionTrace | None = None):

        q = dec.qty

        entry_price = mark_price
        # limit 오더면 틱 오프셋 반영
        if dec.entry_type == "limit":
            tick = self.filters.tick_size(symbol)
            if dec.side == "LONG":
                entry_price = mark_price * (1 - dec.limit_offset_ticks * tick)
            else:
                entry_price = mark_price * (1 + dec.limit_offset_ticks * tick)

        self.filters.use(symbol)
        for need in ("q_price", "q_qty", "min_ok", "min_qty_for"):
            if not hasattr(self.filters, need):
                raise RuntimeError(f"ExchangeFilters has no {need}")
        q_price, q_qty = quantize(self.filters, entry_price, q)


        if self.rt and self.market:
            from ftm2.trade.gates import pre_trade_gates
            ok, reasons = pre_trade_gates(self.rt, self.cfg, self.market, symbol, [])
            if trace:
                trace.gates.update(dict(reasons))
            if not ok:
                if trace:
                    trace.reasons.append("pre_trade_gate")
                    log_decision(trace)
                return None

        # --- 최소 명목가 보정/검증 ---
        if not self.filters.min_ok(entry_price, q_qty):
            if self.cfg.ORDER_SCALE_TO_MIN:
                q_min = self.filters.min_qty_for(entry_price, symbol=symbol)
                if q_min and q_min > 0:
                    q_qty = self.filters.q_qty(symbol, q_min)
            else:
                if trace:
                    trace.reasons.append("below min notional")
                    log_decision(trace)
                ch = "trades" if self.cfg.SEND_SKIP_TO_TRADES else "logs"
                self.notify.send_once(
                    key=f"skip_min_{symbol}",
                    text=f"❌ 최소 명목 미달로 스킵: {symbol} px~{entry_price:.2f} qty_req≥{self.filters.min_qty_for(entry_price, symbol)}",
                    channel=ch,
                    ttl_ms=self.cfg.NOTIFY_THROTTLE_MS,
                )
                return None

        # 보정 후에도 0이면 스킵
        if q_qty <= 0:
            if trace:
                trace.reasons.append("qty quantized zero")
                log_decision(trace)
            ch = "trades" if self.cfg.SEND_SKIP_TO_TRADES else "logs"
            self.notify.send_once(
                key=f"skip_qty0_{symbol}",
                text=f"❌ 진입 스킵: {symbol} {dec.side} — 수량이 0",
                channel=ch,
                ttl_ms=self.cfg.NOTIFY_THROTTLE_MS,
            )

            return None

        if not self._live_guard(symbol, float(q_qty), float(q_price), trace):
            if trace:
                log_decision(trace)
            return None

        side = "BUY" if dec.side=="LONG" else "SELL"
        params = dict(symbol=symbol, side=side, type="MARKET" if dec.entry_type=="market" else "LIMIT",
                      quantity=float(q_qty), newClientOrderId=_cid(symbol, side))
        if dec.entry_type=="limit":
            params.update(price=float(q_price), timeInForce=self.cfg.TIME_IN_FORCE)
        try:
            print(f"[ORDER][TRY] {symbol} {dec.side} qty={float(q_qty)}")
            od = self.bx.new_order(**params)
            print(f"[ORDER][RESP] {od}")
            self.notify.send_trade(
                f"✅ 진입 주문 전송: {symbol} {dec.side} 수량 {float(q_qty)} / {dec.reason}"
            )

            if trace:
                trace.reasons.append("ENTER")
                log_decision(trace)
            if CSV:
                CSV.log("ORDER_NEW", symbol=symbol, side=dec.side, price=float(q_price), qty=float(q_qty),
                        sl=dec.sl, tp=dec.tp, leverage=self.cfg.LEVERAGE, margin=self.cfg.MARGIN_TYPE,
                        reason=dec.reason,
                        route={"slippage":0, "post_only":self.cfg.POST_ONLY, "reduce_only":False, "type":params.get("type")})
            return od
        except Exception as e:
            print(f"[ORDER][ERR] {e}")
            self.notify.send_trade(f"❌ 진입 주문 실패: {symbol} {e}")
            if trace:
                trace.reasons.append("order failed")
                log_decision(trace)
            return None

    def place_brackets(self, symbol: str, side: str, qty: float, entry_price: float, sl: float, tp: float):
        self.filters.use(symbol)
        for need in ("q_price", "q_qty", "min_ok"):
            if not hasattr(self.filters, need):
                raise RuntimeError(f"ExchangeFilters has no {need}")
        reduce_side = "SELL" if side=="LONG" else "BUY"
        sl_price, sl_qty = quantize(self.filters, sl, qty)
        tp_price, tp_qty = quantize(self.filters, tp, qty)
        # SL: 시장가(스톱), TP: 리밋(reduceOnly)
        # Futures는 동시에 존재 가능. closePosition 모드는 전체 청산용.
        try:
            if self.cfg.SL_ORDER=="market":
                self.bx.new_order(symbol=symbol, side=reduce_side, type="STOP_MARKET",
                                  stopPrice=sl_price, reduceOnly=True,
                                  workingType=self.cfg.WORKING_PRICE, newClientOrderId=_cid(symbol,"SL"))
            if self.cfg.TP_ORDER=="limit":
                self.bx.new_order(symbol=symbol, side=reduce_side, type="TAKE_PROFIT",
                                    price=tp_price, stopPrice=tp_price, timeInForce=self.cfg.TIME_IN_FORCE,
                                    reduceOnly=True, workingType=self.cfg.WORKING_PRICE, newClientOrderId=_cid(symbol,"TP"))
            self.notify.send_trade(f"📎 브래킷 설정: SL≈{sl_price}, TP≈{tp_price} (reduceOnly)")
        except Exception as e:
            self.notify.send_trade(f"⚠️ 브래킷 설정 실패: {e}")

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
        dispatcher.send_trade(f"🔻 {symbol} 전량 청산 주문 전송")
        if CSV:
            CSV.log("POSITION_CLOSE", symbol=symbol, side="", exit="", realized="", fee="", roe="", elapsed_sec="", reason="close_all")
        return f"{symbol} 청산 주문 전송"
    except Exception as e:
        dispatcher.send_trade(f"⚠️ {symbol} 청산 실패: {e}")
        return f"{symbol} 청산 실패: {e}"

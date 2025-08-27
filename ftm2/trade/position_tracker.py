# [ANCHOR:M5_POSITION_TRACKER]
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from time import time
from ftm2.trade.pnl_calc import upnl_usdm, initial_margin, roe_pct

@dataclass
class PositionState:
    symbol: str
    side: str                 # LONG/SHORT
    qty: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    upnl: float = 0.0
    roe: float = 0.0
    initial_margin: float = 0.0
    leverage: float = 1.0
    margin_type: str = "ISOLATED"
    realized_pnl: float = 0.0
    fee_paid: float = 0.0
    adds: int = 0
    tp_price: float = 0.0
    sl_price: float = 0.0
    liq_price: float = 0.0
    last_fill_ts: float = 0.0
    last_mark_ts: float = 0.0
    last_update_ts: float = 0.0

@dataclass
class AccountState:
    wallet_balance: float = 0.0
    available_balance: float = 0.0
    total_upnl: float = 0.0
    equity: float = 0.0
    funding_fee_total: float = 0.0   # (옵션) 펀딩 누적
    maker_commission: float = 0.0
    taker_commission: float = 0.0

class PositionTracker:
    def __init__(self):
        self.pos: Dict[str, PositionState] = {}         # key: f"{symbol}:{side}"
        self.account = AccountState()
        self.msg_ids: Dict[str, int] = {}         # symbol -> discord message id
        self._disabled_edit = False
        self._last_upnl: Dict[str, float] = {}

    def key(self, symbol: str, side: str) -> str: return f"{symbol}:{side}"

    def set_snapshot(self, symbol: str, side: str, qty: float, entry: float,
                     lev: float, margin_type: str, liq: float=0.0):
        k = self.key(symbol, side)
        ps = self.pos.get(k) or PositionState(symbol, side)
        ps.qty, ps.entry_price = qty, entry
        ps.leverage, ps.margin_type = lev or 1.0, margin_type
        ps.liq_price = liq
        ps.last_update_ts = time()
        self.pos[k] = ps

    def apply_fill(self, symbol: str, side: str, price: float, qty_delta: float,
                   realized: float=0.0, fee: float=0.0):
        k = self.key(symbol, side)
        ps = self.pos.get(k) or PositionState(symbol, side)
        old_qty, old_entry = ps.qty, ps.entry_price
        new_qty = old_qty + qty_delta
        if new_qty != 0:
            # 가중 평균
            ps.entry_price = ((old_entry*old_qty) + (price*qty_delta)) / new_qty if old_qty != 0 else price
        ps.qty = new_qty
        ps.realized_pnl += realized
        ps.fee_paid += fee
        ps.last_fill_ts = time()
        ps.last_update_ts = ps.last_fill_ts
        if ps.qty == 0:
            # 포지션 종료 시 엔트리 리셋(선택)
            ps.entry_price = 0.0
        self.pos[k] = ps

    def update_mark(self, symbol: str, side: str, mark: float):
        k = self.key(symbol, side)
        ps = self.pos.get(k)
        if not ps: return
        ps.mark_price = mark
        ps.upnl = upnl_usdm(ps.entry_price, mark, ps.qty, side)
        ps.initial_margin = initial_margin(ps.entry_price, ps.qty, ps.leverage)
        ps.roe = roe_pct(ps.upnl, ps.initial_margin)
        ps.last_mark_ts = time()
        ps.last_update_ts = ps.last_mark_ts

    def set_brackets(self, symbol: str, side: str, sl: float, tp: float):
        k = self.key(symbol, side)
        ps = self.pos.get(k)
        if not ps: return
        ps.sl_price, ps.tp_price = sl, tp
        ps.last_update_ts = time()

    def set_liq(self, symbol: str, side: str, liq: float):
        k = self.key(symbol, side)
        ps = self.pos.get(k)
        if ps:
            ps.liq_price = liq
            ps.last_update_ts = time()

    def set_account_balance(self, wallet: float, avail: float):
        self.account.wallet_balance = wallet
        self.account.available_balance = avail
        # total_upnl는 외부에서 sum으로 갱신

    def recompute_totals(self):
        self.account.total_upnl = sum(p.upnl for p in self.pos.values())
        self.account.equity = self.account.wallet_balance + self.account.total_upnl

    def get_symbol_view(self, symbol: str) -> Optional[PositionState]:
        # 우선 순위: LONG 보유 > SHORT 보유; 없으면 None
        for side in ("LONG","SHORT"):
            ps = self.pos.get(self.key(symbol, side))
            if ps and ps.qty != 0: return ps
        return None

    def should_edit(self, symbol: str, pnl_change_bps: int) -> bool:
        """UPNL 변동률 임계치(bps) 체크. 0이면 항상 True."""
        if pnl_change_bps <= 0: return True
        ps = self.get_symbol_view(symbol)
        cur = ps.upnl if ps else 0.0
        last = self._last_upnl.get(symbol, None)
        self._last_upnl[symbol] = cur
        if last is None: return True
        denom = max(abs(ps.initial_margin), 1e-9) if ps else 1.0
        delta_bps = abs(cur - last) / denom * 1e4
        return delta_bps >= pnl_change_bps

    def disable_edits(self, b: bool): self._disabled_edit = b
    def edits_disabled(self) -> bool: return self._disabled_edit

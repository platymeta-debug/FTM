from dataclasses import dataclass
from typing import Dict


@dataclass
class Position:
    qty: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0


class PositionTracker:
    def __init__(self):
        self.positions: Dict[str, Position] = {}

    def update_fill(self, symbol: str, qty: float, price: float):
        pos = self.positions.setdefault(symbol, Position())
        pos.entry_price = price
        pos.qty += qty

    def update_mark(self, symbol: str, mark_price: float):
        pos = self.positions.get(symbol)
        if pos:
            pos.unrealized_pnl = (mark_price - pos.entry_price) * pos.qty

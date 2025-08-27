from decimal import Decimal, ROUND_DOWN


class SymbolFilter:
    def __init__(self, tickSize: str, stepSize: str, minNotional: str):
        self.tick = Decimal(tickSize)
        self.step = Decimal(stepSize)
        self.min_notional = Decimal(minNotional)

    def q_price(self, px: Decimal) -> Decimal:
        """Apply PRICE_FILTER tickSize."""
        return (px / self.tick).to_integral_value(rounding=ROUND_DOWN) * self.tick

    def q_qty(self, qty: Decimal) -> Decimal:
        """Apply LOT_SIZE stepSize."""
        return (qty / self.step).to_integral_value(rounding=ROUND_DOWN) * self.step

    def min_ok(self, price: Decimal, qty: Decimal) -> bool:
        """Check MIN_NOTIONAL."""
        return (price * qty) >= self.min_notional


class ExchangeFilters:
    def __init__(self, symbol_map: dict[str, SymbolFilter]):
        self.map = symbol_map

    @classmethod
    def from_exchange_info(cls, info: dict) -> "ExchangeFilters":
        m: dict[str, SymbolFilter] = {}
        for s in info.get("symbols", []):
            sym = s["symbol"]
            pf = next(f for f in s["filters"] if f["filterType"] == "PRICE_FILTER")
            lf = next(f for f in s["filters"] if f["filterType"] == "LOT_SIZE")
            mn = next(
                (f for f in s["filters"] if f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL")),
                None,
            )
            min_notional = (mn or {}).get("notional", (mn or {}).get("minNotional", "0.0"))
            m[sym] = SymbolFilter(pf["tickSize"], lf["stepSize"], min_notional)
        return cls(m)

    def for_symbol(self, symbol: str) -> SymbolFilter:
        return self.map[symbol]

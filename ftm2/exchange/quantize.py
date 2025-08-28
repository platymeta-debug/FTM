from decimal import Decimal, ROUND_DOWN


class SymbolFilter:
    def __init__(self, tickSize: str, stepSize: str, minNotional: str):
        self.tick = Decimal(tickSize)
        self.step = Decimal(stepSize)
        self.min_notional = Decimal(minNotional)

    def q_price(self, px: Decimal) -> Decimal:
        return (px / self.tick).to_integral_value(rounding=ROUND_DOWN) * self.tick

    def q_qty(self, qty: Decimal) -> Decimal:
        return (qty / self.step).to_integral_value(rounding=ROUND_DOWN) * self.step

    def min_ok(self, price: Decimal, qty: Decimal) -> bool:
        return (price * qty) >= self.min_notional


class ExchangeFilters:
    def __init__(self, symbol_map: dict[str, SymbolFilter]):
        self.map = symbol_map
        self._last_symbol: str | None = None

    @classmethod
    def from_exchange_info(cls, info: dict) -> "ExchangeFilters":
        m: dict[str, SymbolFilter] = {}
        for s in info.get("symbols", []):
            sym = s["symbol"]
            pf = next(f for f in s["filters"] if f["filterType"] == "PRICE_FILTER")
            lf = next(f for f in s["filters"] if f["filterType"] == "LOT_SIZE")
            mn = next((f for f in s["filters"] if f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL")), None)
            min_notional = (mn or {}).get("notional", (mn or {}).get("minNotional", "0.0"))
            m[sym] = SymbolFilter(pf["tickSize"], lf["stepSize"], min_notional)
        return cls(m)

    def for_symbol(self, symbol: str) -> SymbolFilter:
        self._last_symbol = symbol
        return self.map[symbol]

    def use(self, symbol: str) -> "ExchangeFilters":
        self._last_symbol = symbol
        return self

    def _coerce_decimal(self, x) -> Decimal:
        return x if isinstance(x, Decimal) else Decimal(str(x))

    def _resolve_symbol(self, symbol_kw, n_args_expected):
        if symbol_kw:
            return symbol_kw
        if self._last_symbol:
            return self._last_symbol
        raise TypeError("심볼 컨텍스트 미설정. filters.q_price('SYM', price)로 호출하거나, 먼저 filters.use('SYM')를 호출하세요.")

    def q_price(self, *args, **kwargs) -> Decimal:
        symbol_kw = kwargs.get("symbol")
        if len(args) == 2 and isinstance(args[0], str):
            symbol, price = args[0], args[1]
        elif len(args) == 1:
            symbol = self._resolve_symbol(symbol_kw, 1)
            price = args[0]
        else:
            raise TypeError("q_price('SYM', price) 또는 q_price(price, symbol='SYM') 형태로 호출하세요.")
        return self.map[symbol].q_price(self._coerce_decimal(price))

    def q_qty(self, *args, **kwargs) -> Decimal:
        symbol_kw = kwargs.get("symbol")
        if len(args) == 2 and isinstance(args[0], str):
            symbol, qty = args[0], args[1]
        elif len(args) == 1:
            symbol = self._resolve_symbol(symbol_kw, 1)
            qty = args[0]
        else:
            raise TypeError("q_qty('SYM', qty) 또는 q_qty(qty, symbol='SYM') 형태로 호출하세요.")
        return self.map[symbol].q_qty(self._coerce_decimal(qty))

    def min_ok(self, *args, **kwargs) -> bool:
        symbol_kw = kwargs.get("symbol")
        if len(args) == 3 and isinstance(args[0], str):
            symbol, price, qty = args
        elif len(args) == 2:
            symbol = self._resolve_symbol(symbol_kw, 2)
            price, qty = args
        else:
            raise TypeError("min_ok('SYM', price, qty) 또는 min_ok(price, qty, symbol='SYM') 형태로 호출하세요.")
        return self.map[symbol].min_ok(self._coerce_decimal(price), self._coerce_decimal(qty))

    def tick_size(self, symbol: str) -> Decimal:
        return self.map[symbol].tick

    def step_size(self, symbol: str) -> Decimal:
        return self.map[symbol].step

    q_px = q_price
    quantize_price = q_price
    quantize_qty = q_qty

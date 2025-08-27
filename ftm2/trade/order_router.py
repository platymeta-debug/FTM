from decimal import Decimal

from ..exchange.binance_client import BinanceClient
from ..exchange.quantize import ExchangeFilters


class OrderRouter:
    def __init__(self, client: BinanceClient, filters: ExchangeFilters):
        self.client = client
        self.filters = filters

    def market_order(self, symbol: str, side: str, qty: Decimal):
        f = self.filters.for_symbol(symbol)
        q = f.q_qty(qty)
        self.client.trade.new_order(
            symbol=symbol,
            side=side.upper(),
            type="MARKET",
            quantity=str(q),
        )

    def limit_order(self, symbol: str, side: str, qty: Decimal, price: Decimal):
        f = self.filters.for_symbol(symbol)
        q = f.q_qty(qty)
        p = f.q_price(price)
        self.client.trade.new_order(
            symbol=symbol,
            side=side.upper(),
            type="LIMIT",
            price=str(p),
            quantity=str(q),
            timeInForce="GTC",
        )

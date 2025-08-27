from decimal import Decimal

from ..config.settings import load_env_chain
from ..exchange.quantize import ExchangeFilters

CFG = load_env_chain()


def atr_position_size(atr_value: float, filters: ExchangeFilters, symbol: str) -> Decimal:
    """Size position based on ATR and target risk in USDT."""
    risk = Decimal(str(CFG.RISK_TARGET_USDT))
    atr_d = Decimal(str(atr_value)) * Decimal(str(CFG.ATR_MULT))
    if atr_d == 0:
        return Decimal("0")
    qty = risk / atr_d
    return filters.for_symbol(symbol).q_qty(qty)

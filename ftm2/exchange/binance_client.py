from binance_sdk_derivatives_trading_usds_futures import (
    ApiClient,
    Configuration,
    TradeApi,
    MarketDataApi,
    AccountApi,
)

from ..config.settings import load_env_chain
from .quantize import ExchangeFilters

CFG = load_env_chain()

BASE_URL = (
    "https://fapi.binance.com" if CFG.MODE == "live" else "https://testnet.binancefuture.com"
)


def make_client() -> ApiClient:
    conf = Configuration(host=BASE_URL)
    conf.api_key["X-MBX-APIKEY"] = CFG.BINANCE_API_KEY
    return ApiClient(conf)


class BinanceClient:
    """Thin wrapper around Binance USDⓈ-M Futures client."""

    def __init__(self):
        self.client = make_client()
        self.trade = TradeApi(self.client)
        self.market = MarketDataApi(self.client)
        self.account = AccountApi(self.client)
        self.filters: ExchangeFilters | None = None

    def sync_time(self) -> dict:
        """Fetch server time for timestamp synchronization."""
        return self.market.time().to_dict()

    def load_exchange_info(self) -> dict:
        """Load exchange information and parse trading filters."""
        info = self.market.exchange_information().to_dict()
        self.filters = ExchangeFilters.from_exchange_info(info)
        return info

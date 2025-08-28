
from __future__ import annotations
import os, time, hmac, hashlib
from typing import Any, Dict
from urllib.parse import urlencode
import httpx, websockets

from ftm2.config.settings import load_env_chain
from ftm2.exchange.quantize import ExchangeFilters
from ftm2.notify.discord_bot import send_trade

CFG = load_env_chain()


HEADERS = {"X-MBX-APIKEY": CFG.BINANCE_API_KEY}

FATAL_USER_ERRORS = {-1013, -1111, -1113, -4164, -2019}
RETRYABLE_STATUS = {429, 418, 500, 502, 503, 504}


class BinanceClient:
    def __init__(self):
        # 데이터/거래 엔드포인트 분리
        self.REST_MARKET_BASE = (
            "https://fapi.binance.com" if CFG.DATA_FEED == "live" else "https://testnet.binancefuture.com"
        )
        self.WS_MARKET_BASE = (
            "wss://fstream.binance.com" if CFG.DATA_FEED == "live" else "wss://fstream.binancefuture.com"
        )
        self.REST_TRADE_BASE = (
            "https://fapi.binance.com" if CFG.TRADE_MODE == "live" else "https://testnet.binancefuture.com"
        )
        self.WS_USER_BASE = (
            "wss://fstream.binance.com/ws" if CFG.TRADE_MODE == "live" else "wss://fstream.binancefuture.com/ws"
        )
        self.session_market = httpx.Client(base_url=self.REST_MARKET_BASE, timeout=CFG.HTTP_TIMEOUT_S)
        self.session_trade = httpx.Client(base_url=self.REST_TRADE_BASE, timeout=CFG.HTTP_TIMEOUT_S, headers=HEADERS)
        self.filters: ExchangeFilters | None = None

    # --- low-level helpers ---
    def trade_rest_signed(self, method: str, path: str, params: Dict[str, Any] | None = None):
        params = params or {}
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = CFG.RECV_WINDOW_MS
        q = urlencode(params, doseq=True)
        sig = hmac.new(
            CFG.BINANCE_API_SECRET.encode(), q.encode(), hashlib.sha256
        ).hexdigest()
        url = f"{path}?{q}&signature={sig}"
        return self.session_trade.request(method, url)

    def market_rest(self, method: str, path: str, params: Dict[str, Any] | None = None):
        return self.session_market.request(method, path, params=params)

    def ws_connect_market(self, url_path: str):
        return websockets.connect(self.WS_MARKET_BASE + url_path, ping_interval=150)

    def ws_connect_user(self, url_path: str):
        return websockets.connect(self.WS_USER_BASE + url_path, ping_interval=150)

    # --- public ---
    def server_time(self) -> Dict[str, Any]:
        r = self.market_rest("GET", "/fapi/v1/time")
        r.raise_for_status()
        return r.json()

    def load_exchange_info(self) -> Dict[str, Any]:
        r = self.market_rest("GET", "/fapi/v1/exchangeInfo")
        r.raise_for_status()
        info = r.json()
        self.filters = ExchangeFilters.from_exchange_info(info)
        return info

    # --- user data stream (listenKey) ---
    def create_listen_key(self) -> str:
        r = self.session_trade.post("/fapi/v1/listenKey")
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError:
            print(f"[LISTEN_KEY][ERR] status={r.status_code} body={r.text}")
            raise
        return r.json()["listenKey"]

    def keepalive_listen_key(self, listen_key: str):
        r = self.session_trade.put("/fapi/v1/listenKey", params={"listenKey": listen_key})
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError:
            return r
        return r
    def delete_listen_key(self, listen_key: str) -> None:
        r = self.session_trade.delete("/fapi/v1/listenKey", params={"listenKey": listen_key})
        r.raise_for_status()

    # --- trading (example: query account, place order) ---
    def account(self) -> Dict[str, Any]:
        r = self.trade_rest_signed("GET", "/fapi/v2/account")
        r.raise_for_status()
        return r.json()

    def new_order(self, **kwargs) -> Dict[str, Any]:
        # expects kwargs like: symbol, side, type, quantity, price?, timeInForce?, positionSide?, workingType=MARK_PRICE, ...
        for attempt in range(1, CFG.RETRY_MAX + 1):
            r = self.trade_rest_signed("POST", "/fapi/v1/order", kwargs)
            try:
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as e:
                body = r.text
                code = None
                msg = body
                try:
                    j = r.json()
                    code = j.get("code")
                    msg = j.get("msg", body)
                    body = f"code={code} msg={msg}"
                except Exception:
                    pass
                symbol = kwargs.get("symbol")
                send_trade(
                    f"❌ 진입 주문 실패: {symbol} status={r.status_code} code={code} msg={msg}"
                )
                if code in FATAL_USER_ERRORS or r.status_code == 400:
                    raise RuntimeError(
                        f"HTTP {r.status_code} {e.request.url} body={body}"
                    ) from e
                if r.status_code in RETRYABLE_STATUS and attempt < CFG.RETRY_MAX:
                    backoff_ms = (
                        CFG.BACKOFF_429_MS if r.status_code in (418, 429) else CFG.BACKOFF_NET_MS
                    )
                    time.sleep(backoff_ms / 1000)
                    continue
                raise RuntimeError(
                    f"HTTP {r.status_code} {e.request.url} body={body}"
                ) from e

    # [ANCHOR:M5_BINANCE_REST]
    def get_account_v2(self):
        return self.trade_rest_signed("GET", "/fapi/v2/account").json()

    def get_position_risk(self, symbol: str|None=None):
        params = {"symbol": symbol} if symbol else {}
        return self.trade_rest_signed("GET", "/fapi/v2/positionRisk", params=params).json()

    def get_commission_rate(self, symbol: str):
        return self.trade_rest_signed("GET", "/fapi/v1/commissionRate", params={"symbol": symbol}).json()

    # [ANCHOR:M5P_INCOME_API]
    def get_income(self, incomeType: str|None=None, startTime: int|None=None):
        params = {}
        if incomeType: params["incomeType"] = incomeType
        if startTime: params["startTime"] = startTime
        return self.trade_rest_signed("GET", "/fapi/v1/income", params=params).json()



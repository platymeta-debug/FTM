
from __future__ import annotations
import os, time, hmac, hashlib
from typing import Any, Dict
from urllib.parse import urlencode
import httpx

from ..config.settings import load_env_chain
from .quantize import ExchangeFilters

CFG = load_env_chain()


REST_BASE = "https://fapi.binance.com" if CFG.MODE == "live" else "https://testnet.binancefuture.com"  # docs
HEADERS = {"X-MBX-APIKEY": CFG.BINANCE_API_KEY}


class BinanceClient:
    def __init__(self):
        self.session = httpx.Client(base_url=REST_BASE, timeout=CFG.HTTP_TIMEOUT_S, headers=HEADERS)
        self.filters: ExchangeFilters | None = None

    # --- low-level signed helper (HMAC SHA256) ---
    def _signed(self, method: str, path: str, params: Dict[str, Any] | None = None):
        params = params or {}
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = CFG.RECV_WINDOW_MS
        q = urlencode(params, doseq=True)
        sig = hmac.new(
            CFG.BINANCE_API_SECRET.encode(), q.encode(), hashlib.sha256
        ).hexdigest()
        url = f"{path}?{q}&signature={sig}"
        return self.session.request(method, url)

    # --- public ---
    def server_time(self) -> Dict[str, Any]:
        r = self.session.get("/fapi/v1/time")
        r.raise_for_status()
        return r.json()

    def load_exchange_info(self) -> Dict[str, Any]:
        r = self.session.get("/fapi/v1/exchangeInfo")
        r.raise_for_status()
        info = r.json()
        self.filters = ExchangeFilters.from_exchange_info(info)
        return info

    # --- user data stream (listenKey) ---
    def create_listen_key(self) -> str:
        r = self.session.post("/fapi/v1/listenKey")
        r.raise_for_status()
        return r.json()["listenKey"]

    def keepalive_listen_key(self, listen_key: str) -> None:
        r = self.session.put("/fapi/v1/listenKey", params={"listenKey": listen_key})
        r.raise_for_status()

    def delete_listen_key(self, listen_key: str) -> None:
        r = self.session.delete("/fapi/v1/listenKey", params={"listenKey": listen_key})
        r.raise_for_status()

    # --- trading (example: query account, place order) ---
    def account(self) -> Dict[str, Any]:
        r = self._signed("GET", "/fapi/v2/account")
        r.raise_for_status()
        return r.json()

    def new_order(self, **kwargs) -> Dict[str, Any]:
        # expects kwargs like: symbol, side, type, quantity, price?, timeInForce?, positionSide?, workingType=MARK_PRICE, ...
        r = self._signed("POST", "/fapi/v1/order", kwargs)
        # Rate-limit headers visible via r.headers.get("X-MBX-USED-WEIGHT-1m")
        r.raise_for_status()
        return r.json()



# [ANCHOR:TIME_DRIFT]
import time


class TimeGuard:
    def __init__(self, client, warn_cb):
        self.client = client
        self.warn = warn_cb
        self.offset_ms = 0

    async def sync(self):
        try:
            sv = await self.client.server_time()  # binance /fapi/v1/time
            self.offset_ms = int(sv) - int(time.time() * 1000)
        except Exception as e:
            self.warn(f"time sync fail: {e}")

    def now_ms(self):
        return int(time.time() * 1000 + self.offset_ms)

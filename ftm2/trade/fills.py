# [ANCHOR:WAIT_FILL_ACCUM]
import time, asyncio


class FillWaiter:
    def __init__(self, client):
        self.client = client

    async def wait_accum(self, sym, order_id, want_qty, timeout_ms, poll_ms=150):
        t0 = time.time() * 1000
        got = 0.0
        while (time.time() * 1000 - t0) < timeout_ms:
            od = await self.client.get_order(symbol=sym, orderId=order_id)
            exq = float(od.get("executedQty") or 0.0)
            got = max(got, exq)
            if od.get("status") in ("FILLED", "PARTIALLY_FILLED"):
                if got >= want_qty * 0.999:  # 99.9% 이상
                    return got, True, od
            await asyncio.sleep(poll_ms / 1000.0)
        return got, False, None

# [ANCHOR:ENSURE]
from __future__ import annotations
import asyncio
from typing import Literal

from ftm2.notify import dispatcher


async def ensure(
    client,
    sym: str,
    *,
    leverage: int,
    margin_type: Literal["ISOLATED", "CROSSED"] = "ISOLATED",
) -> None:
    """Ensure leverage and margin type are set on the exchange.

    If the current settings differ, attempt to update once with a retry.
    Failures are logged via emit_once.
    """
    for attempt in range(2):
        try:
            cur = await client.get_leverage(sym)
            cur_mode = await client.get_margin_type(sym)
            if cur != leverage:
                await client.set_leverage(sym, leverage)
            if cur_mode.upper() != margin_type.upper():
                await client.set_margin_mode(sym, margin_type)
            return
        except Exception as e:
            if attempt == 0:
                await asyncio.sleep(0.5)
                continue
            dispatcher.emit_once(
                f"levsync_fail_{sym}",
                "error",
                f"leverage sync failed for {sym}: {e}",
                ttl_ms=60000,
            )
            return


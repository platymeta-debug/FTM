from typing import Literal

async def enforce_leverage_and_margin(
    client,
    sym: str,
    *,
    leverage: int,
    margin_type: Literal["ISOLATED", "CROSSED"] = "ISOLATED",
    notify=None,
):
    _ensure = globals().get("ensure")
    if callable(_ensure):
        try:
            return await _ensure(
                client=client, sym=sym, leverage=leverage, margin_type=margin_type, notify=notify
            )
        except TypeError:
            return await _ensure(sym, leverage=leverage, margin_type=margin_type)
    try:
        if hasattr(client, "set_margin_type"):
            await client.set_margin_type(sym, margin_type)
        if hasattr(client, "set_leverage"):
            await client.set_leverage(sym, int(leverage))
        if notify is not None and hasattr(notify, "emit_once"):
            notify.emit_once(
                f"lev_sync_ok:{sym}", "system",
                f"레버리지/마진 동기화 완료: {sym} lev={leverage}, mode={margin_type}",
                60_000,
            )
        return {"symbol": sym, "leverage": int(leverage), "margin_type": margin_type}
    except Exception as e:
        if notify is not None and hasattr(notify, "emit_once"):
            notify.emit_once(
                f"lev_sync_err:{sym}", "error",
                f"레버리지/마진 동기화 실패: {sym} -> {e}",
                60_000,
            )
        raise

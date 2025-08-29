# ------------------ Compatibility wrapper for app.py ------------------
from typing import Literal

async def enforce_leverage_and_margin(
    client,
    sym: str,
    *,
    leverage: int,
    margin_type: Literal["ISOLATED", "CROSSED"] = "ISOLATED",
    notify=None,
):
    """
    Compat wrapper for app.py:
    - 같은 모듈에 ensure(...)가 있으면 그걸 호출 (시그니처 차이도 흡수)
    - 없으면 client API(set_margin_type/set_leverage 또는 fapiPrivatePOST)로 직접 동기화
    - 성공/실패는 notify.emit_once(있으면)로 1회만 알림
    """
    # 1) 같은 모듈의 ensure(...)가 있으면 우선 위임
    _ensure = globals().get("ensure")
    if callable(_ensure):
        try:
            # 가장 보편적인 시그니처
            return await _ensure(
                client=client,
                sym=sym,
                leverage=leverage,
                margin_type=margin_type,
                notify=notify,
            )
        except TypeError:
            # 다른 시그니처로 정의되어 있다면 최소 인자만 전달
            return await _ensure(sym, leverage=leverage, margin_type=margin_type)

    # 2) 직접 동기화 (fallback)
    try:
        # margin type 먼저
        if hasattr(client, "set_margin_type"):
            await client.set_margin_type(sym, margin_type)
        elif hasattr(client, "fapiPrivatePOST"):
            await client.fapiPrivatePOST(
                "/fapi/v1/marginType",
                {"symbol": sym, "marginType": margin_type},
            )

        # leverage 다음
        if hasattr(client, "set_leverage"):
            await client.set_leverage(sym, int(leverage))
        elif hasattr(client, "fapiPrivatePOST"):
            await client.fapiPrivatePOST(
                "/fapi/v1/leverage",
                {"symbol": sym, "leverage": int(leverage)},
            )

        # 1회 알림(선택)
        try:
            if notify is not None and hasattr(notify, "emit_once"):
                notify.emit_once(
                    f"lev_sync_ok:{sym}",
                    "system",
                    f"레버리지/마진 동기화 완료: {sym} lev={leverage}, mode={margin_type}",
                    60_000,
                )
        except Exception:
            pass

        return {"symbol": sym, "leverage": int(leverage), "margin_type": margin_type}

    except Exception as e:
        try:
            if notify is not None and hasattr(notify, "emit_once"):
                notify.emit_once(
                    f"lev_sync_err:{sym}",
                    "error",
                    f"레버리지/마진 동기화 실패: {sym} -> {e}",
                    60_000,
                )
        except Exception:
            pass
        raise

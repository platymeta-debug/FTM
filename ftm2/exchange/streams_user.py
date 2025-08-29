# [ANCHOR:M5_USER_STREAM]
import asyncio, json, time, websockets
from ftm2.notify.discord_bot import edit_trade_card
from ftm2.trade.position_tracker import PositionTracker
from ftm2.exchange.binance_client import BinanceClient
from ftm2.notify import dispatcher
from ftm2.journal.events import JEvent
from ftm2.dashboard.pnl_rollup import on_realized

TRACKER: PositionTracker | None = None
CSV = None
LEDGER = None
RT = None

async def user_stream(bx: BinanceClient, tracker: PositionTracker, cfg):
    global TRACKER
    TRACKER = tracker
    listen_key = bx.create_listen_key()

    async def on_message(raw: str):
        msg = json.loads(raw)
        if 'e' in msg:
            await on_user_event(msg, bx, cfg, listen_key)

    async def keepalive():
        while True:
            try:
                bx.keepalive_listen_key(listen_key)
                dispatcher.emit_once("lk_ok", "system", "🔑 listenKey keepalive", 3600000)
            except Exception as e:
                dispatcher.emit("error", f"🔑 keepalive err: {e}")
            await asyncio.sleep(30*60)

    asyncio.create_task(keepalive())

    # [ANCHOR:WS_BACKOFF]
    backoff = 1
    url = f"{bx.WS_USER_BASE}/{listen_key}"
    while True:
        try:
            async with websockets.connect(url, ping_interval=15, ping_timeout=10, close_timeout=5) as ws:
                dispatcher.emit_once("ws_user_ok", "system", "👤 USER_WS connected", 60000)
                if RT and getattr(RT, "journal", None):
                    RT.journal.write(JEvent.now("WS", symbol="", message="USER_WS connected"))
                if RT: RT.ws["user_ok"] = True
                backoff = 1
                async for raw in ws:
                    await on_message(raw)
        except Exception as e:
            dispatcher.emit_once("ws_user_re", "error", f"⚠️ USER_WS reconnecting: {e}", 60000)
            if RT and getattr(RT, "journal", None):
                RT.journal.write(JEvent.now("WS", symbol="", message=f"USER_WS reconnect: {type(e).__name__}"))
            if RT: RT.ws["user_ok"] = False
            await asyncio.sleep(min(60, backoff))
            backoff = min(60, backoff * 2)
            try:
                listen_key = bx.create_listen_key()
            except Exception:
                pass
            url = f"{bx.WS_USER_BASE}/{listen_key}"
            continue

async def on_user_event(evt, bx: BinanceClient, cfg, listen_key: str):

    et = evt.get("e")
    if et == "ORDER_TRADE_UPDATE":
        x = evt.get("o", {})
        sym = x.get("s"); side = "LONG" if x.get("S")=="BUY" else "SHORT"
        ex_type = x.get("x")         # TRADE/NEW/CANCELED/EXPIRED
        st = x.get("X")              # FILLED/PARTIALLY_FILLED
        last_price = float(x.get("L", 0) or 0)
        last_qty   = float(x.get("l", 0) or 0)
        realized   = float(x.get("rp", 0) or 0)
        fee        = float(x.get("n", 0) or 0)
        if ex_type == "TRADE" and last_qty:
            prev = TRACKER.pos.get(TRACKER.key(sym, side)) if hasattr(TRACKER, 'pos') else None
            TRACKER.apply_fill(sym, side, last_price, last_qty if side=="LONG" else -last_qty, realized, fee)
            await edit_trade_card(sym, TRACKER, cfg, force=True)
            # [ANCHOR:M5P_USERSTREAM_LOG]
            win = None
            ps = TRACKER.get_symbol_view(sym) if TRACKER.get_symbol_view(sym) else None
            if ps:
                win = (ps.upnl >= 0)
            if LEDGER:
                LEDGER.on_realized(realized, fee, win)
            if CSV:
                CSV.log("ORDER_FILL", symbol=sym, side=side, price=last_price,
                        qty=last_qty if side=="LONG" else -last_qty,
                        realized=realized, fee=fee, reason="fill", order_id=x.get("i"), client_id=x.get("c"),
                        entry=ps.entry_price if ps else "", mark=ps.mark_price if ps else "",
                        wallet=TRACKER.account.wallet_balance, equity=TRACKER.account.equity, avail=TRACKER.account.available_balance)
            ps_after = TRACKER.pos.get(TRACKER.key(sym, side)) if hasattr(TRACKER, 'pos') else None
            if ps_after and ps_after.qty == 0 and prev and abs(prev.qty) > 0:
                realized_total = ps_after.realized_pnl
                qty_closed = abs(prev.qty)
                if RT and getattr(RT, "journal", None):
                    RT.journal.write(JEvent.now("CLOSE", symbol=sym, side=side, qty=qty_closed, price=last_price, realized=realized_total))
                    RT.journal.write(JEvent.now("PNL_REALIZED", symbol=sym, realized=realized_total))
                if RT:
                    on_realized(RT, realized_total)
    elif et == "ACCOUNT_UPDATE":
        a = evt.get("a", {})
        # 지갑/가용잔고
        for b in a.get("B", []):
            if b.get("a") == "USDT":
                wallet = float(b.get("wb", 0) or 0); avail = float(b.get("cw", 0) or 0)
                TRACKER.set_account_balance(wallet, avail)
        # 포지션 벡터
        for p in a.get("P", []):
            sym = p.get("s"); ps = p.get("ps")  # BOTH/LONG/SHORT (hedge)
            side = "LONG" if ps in ("LONG","BOTH") and float(p.get("pa",0))>0 else "SHORT"
            qty = abs(float(p.get("pa", 0) or 0))
            entry = float(p.get("ep", 0) or 0)
            lev = float(p.get("l", 1) or 1)
            mt = "ISOLATED" if p.get("mt")=="isolated" else "CROSSED"
            # liq는 AccountUpdate에 없음 → 주기적 REST로 보정
            TRACKER.set_snapshot(sym, side, qty, entry, lev, mt)
            await edit_trade_card(sym, TRACKER, cfg, force=True)
        if CSV:
            CSV.log("SNAPSHOT", symbol=sym if 'sym' in locals() else "", reason="account_update",
                    wallet=TRACKER.account.wallet_balance, equity=TRACKER.account.equity, avail=TRACKER.account.available_balance)
    elif et == "listenKeyExpired":
        try:
            bx.delete_listen_key(listen_key)
            print("[USER_WS] listenKey deleted")
        except Exception as e:
            print("[USER_WS] listenKey delete failed", e)
        raise RuntimeError("listenKeyExpired")

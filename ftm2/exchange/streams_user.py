# [ANCHOR:M5_USER_STREAM]
import asyncio, json, time
from ftm2.notify.discord_bot import send_log, edit_trade_card
from ftm2.trade.position_tracker import PositionTracker
from ftm2.exchange.binance_client import BinanceClient

TRACKER: PositionTracker | None = None

async def keepalive_loop(bx: BinanceClient, listen_key: str, interval_sec: int):
    while True:
        await asyncio.sleep(interval_sec)
        try:
            r = bx.keepalive_listen_key(listen_key)
            if r and r.status_code == 200:
                print("[USER_WS] keepalive OK")
            else:
                print("[USER_WS] keepalive ERR → recreate")
                break
        except Exception as e:
            print("[USER_WS] keepalive exception", e); break

async def user_stream(bx: BinanceClient, tracker: PositionTracker, cfg):
    global TRACKER
    TRACKER = tracker
    while True:
        lk = ""
        keep = None
        try:
            lk = bx.create_listen_key()
            print(f"[USER_WS] listenKey={lk[:8]}… created")
            keep = asyncio.create_task(keepalive_loop(bx, lk, cfg.LISTENKEY_KEEPALIVE_SEC))
            url = f"{bx.WS_USER_BASE}/{lk}"
            async with bx.ws_connect(url) as ws:
                print("[USER_WS] connected")
                async for raw in ws:
                    msg = json.loads(raw)
                    if 'e' in msg:
                        await on_user_event(msg, bx, cfg)
        except Exception as e:
            print("[USER_WS] stream error → reconnect", e)
        finally:
            try:
                if lk: bx.delete_listen_key(lk); print("[USER_WS] listenKey deleted")
            except: pass
            if keep and not keep.done():
                keep.cancel()
            await asyncio.sleep(1.0)

async def on_user_event(evt, bx: BinanceClient, cfg):
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
            TRACKER.apply_fill(sym, side, last_price, last_qty if side=="LONG" else -last_qty, realized, fee)
            await edit_trade_card(sym, TRACKER, cfg, force=True)
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

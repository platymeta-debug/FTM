# [ANCHOR:M5_RECONCILER]
import asyncio
from ftm2.notify.discord_bot import send_log

async def resync_loop(bx, tracker, cfg, symbols):
    while True:
        try:
            # 지갑/가용
            acc = bx.get_account_v2()
            for a in acc.get("assets", []):
                if a.get("asset")=="USDT":
                    tracker.set_account_balance(float(a.get("walletBalance",0) or 0),
                                                float(a.get("availableBalance",0) or 0))
                    break
            # 포지션
            pr = bx.get_position_risk()
            for p in pr:
                sym = p.get("symbol"); side = "LONG" if float(p.get("positionAmt",0))>0 else "SHORT"
                qty = abs(float(p.get("positionAmt",0) or 0))
                entry = float(p.get("entryPrice",0) or 0)
                lev   = float(p.get("leverage",1) or 1)
                mt    = "ISOLATED" if p.get("isolated")==True else "CROSSED"
                liq   = float(p.get("liquidationPrice",0) or 0)
                if qty>0:
                    tracker.set_snapshot(sym, side, qty, entry, lev, mt, liq)
                    tracker.set_liq(sym, side, liq)
            tracker.recompute_totals()
        except Exception as e:
            send_log(f"⚠️ 재동기화 실패: {e}")
        await asyncio.sleep(cfg.REST_RESYNC_SEC)

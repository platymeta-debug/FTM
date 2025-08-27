import asyncio, time

async def income_poll_loop(bx, ledger, csv, cfg):
    last_ts = 0
    while True:
        try:
            rows = bx.get_income(startTime=last_ts)
            for r in rows:
                it = r.get("incomeType")
                t  = int(r.get("time",0))
                amt= float(r.get("income",0) or 0.0)
                sym= r.get("symbol","")
                last_ts = max(last_ts, t+1)
                if it == "FUNDING_FEE":
                    ledger.on_funding(amt)
                    csv.log("FUNDING_FEE", symbol=sym, realized="", fee="", funding_cum=ledger.stats.funding, reason="poll")
                elif it == "COMMISSION":
                    # 커미션도 참고용으로 적재
                    csv.log("COMMISSION_FEE", symbol=sym, fee=amt, reason="poll")
        except Exception:
            pass
        await asyncio.sleep(cfg.INCOME_POLL_SEC)


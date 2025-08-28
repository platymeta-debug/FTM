# [ANCHOR:WEB_REPORTS]
import os, csv, glob, sqlite3, datetime as dt
from fastapi import APIRouter, Depends, Query
from ftm2.web.auth import verify
router = APIRouter(prefix="/reports", tags=["reports"])

def _csv_path(day=None, base=None):
    base = base or os.getenv("JOURNAL_DIR","./logs/journal")
    if not day: day = dt.datetime.now().strftime("%Y%m%d")
    return os.path.join(base, f"{day}_trades.csv")

@router.get("/daily")
def daily(day: str | None = Query(default=None), _: None = Depends(verify)):
    path = _csv_path(day)
    if not os.path.exists(path): return {"rows": [], "path": path}
    with open(path, newline="", encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    return {"rows": r, "path": path}

@router.get("/summary")
def summary(_: None = Depends(verify)):
    base = os.getenv("JOURNAL_DIR","./logs/journal")
    files = sorted(glob.glob(os.path.join(base,"*_trades.csv")))
    wins=0.0; losses=0.0; nwin=0; nloss=0; trades=0
    for p in files:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["kind"]=="PNL_REALIZED":
                    trades += 1
                    v = float(row.get("realized") or 0.0)
                    if v>=0: wins += v; nwin+=1
                    else: losses += v; nloss+=1
    pf = (wins/abs(losses)) if losses!=0 else float("inf")
    wr = nwin/max(1,(nwin+nloss))
    exp = (wins/max(1,nwin)) * wr - (abs(losses)/max(1,nloss)) * (1-wr) if (nwin+nloss)>0 else 0.0
    return {"trades":trades, "winrate":wr, "profit_factor":pf, "expectancy":exp}

@router.get("/sql")
def sql(q: str, _: None = Depends(verify)):
    if os.getenv("DISABLE_SQL_API","true").lower() in ("1","true","yes"):
        return {"error":"sql api disabled"}
    if not os.getenv("JOURNAL_SQLITE_ENABLE","false").lower() in ("1","true","yes"):
        return {"error":"sqlite disabled"}
    db = os.getenv("JOURNAL_SQLITE_PATH","./logs/journal/trades.db")
    if not os.path.exists(db): return {"error":"db not found"}
    if ";" in q.lower(): return {"error":"semicolon not allowed"}
    con = sqlite3.connect(db); cur = con.cursor()
    cur.execute(q); cols=[d[0] for d in cur.description or []]; rows=cur.fetchall()
    con.close()
    return {"columns": cols, "rows": rows}

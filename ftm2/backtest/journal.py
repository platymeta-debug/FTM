# [ANCHOR:JOURNAL_CSV]
def write_csv(trades, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_entry","ts_exit","symbol","side","qty","entry","exit","sl","tp1","tp2","pnl","roe"])
        for t in trades:
            w.writerow([t.ts_in, t.ts_out, t.sym, t.side, t.qty, t.px_in, t.px_out, t.sl, t.tp1, t.tp2, t.pnl, t.roe])

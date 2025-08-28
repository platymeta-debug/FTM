# [ANCHOR:DATA_LOADER]
def load_klines(symbol, tf, path_csv):
    import csv
    rows = []
    with open(path_csv, "r", newline="") as f:
        for ts, o, h, l, c, v in csv.reader(f):
            rows.append({"ts": int(ts), "o": float(o), "h": float(h), "l": float(l), "c": float(c), "v": float(v)})
    return rows

import itertools, json, random, os, time
from dataclasses import dataclass

try:
    from dotenv import dotenv_values
except Exception:
    dotenv_values = None


@dataclass
class TuneResult:
    params: dict
    metrics: dict


def _parse_range(spec: str):
    # "60:80:5" -> [60,65,70,75,80] / "0.8,1.0,1.2" -> [0.8,1.0,1.2]
    if ":" in spec:
        a, b, s = spec.split(":")
        a = float(a)
        b = float(b)
        s = float(s)
        vals = []
        v = a
        while v <= b + 1e-9:
            vals.append(round(v, 10))
            v += s
        return vals
    return [float(x) if "." in x else int(x) for x in spec.split(",")]


def _space_from_env(cfg):
    sp={}
    # 1) .env.tune가 있으면 우선 사용
    tune_map = {}
    if dotenv_values and os.path.exists('.env.tune'):
        tune_map = dotenv_values('.env.tune')
    for k in cfg.TUNE_PARAMS:
        spec = (tune_map.get(k) if tune_map else None) or os.getenv(k)
        if not spec:
            continue
        sp[k] = _parse_range(spec)
    return sp


def _product_or_random(space, limit, seed=42):
    keys = list(space.keys())
    grid = list(itertools.product(*[space[k] for k in keys]))
    random.Random(seed).shuffle(grid)
    return keys, grid[: min(limit, len(grid))]


def run_sweep(cfg, bt_engine_factory):
    space = _space_from_env(cfg)
    keys, combos = _product_or_random(space, cfg.TUNE_LIMIT, cfg.TUNE_SEED)
    best = None
    rows = []
    for vals in combos:
        params = dict(zip(keys, vals))
        for k, v in params.items():
            setattr(cfg, k, v if not isinstance(v, float) or v % 1 else int(v))
        bt = bt_engine_factory(cfg)
        metrics = bt.run(cfg.BT_START_IDX, cfg.BT_END_IDX)
        ok = _constraints_ok(cfg, metrics)
        rows.append(TuneResult(params=params, metrics=metrics).__dict__)
        if ok and _is_better(cfg, metrics, best.metrics if best else None):
            best = TuneResult(params=params, metrics=metrics)
    return best, rows


def _constraints_ok(cfg, m):
    return (
        m["pf"] >= cfg.TUNE_PF_MIN
        and m["sharpe"] >= cfg.TUNE_SHARPE_MIN
        and cfg.TUNE_WINRATE_MIN <= m["winrate"] <= cfg.TUNE_WINRATE_MAX
        and m.get("trades", 50) >= cfg.TUNE_TRADES_MIN
    )


def _is_better(cfg, m, m_best):
    if m_best is None:
        return True
    key = cfg.TUNE_OBJECTIVE
    if m[key] != m_best[key]:
        return m[key] > m_best[key]
    if m["pf"] != m_best["pf"]:
        return m["pf"] > m_best["pf"]
    return m["sharpe"] > m_best["sharpe"]


def export_results(rows, out_csv, best_json):
    import csv

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        all_keys = set()
        for r in rows:
            all_keys |= set(r["params"].keys())
        header = ["winrate", "pf", "expectancy", "sharpe", "trades"] + sorted(all_keys)
        w.writerow(header)
        for r in rows:
            m = r["metrics"]
            p = r["params"]
            w.writerow(
                [
                    m.get("winrate"),
                    m.get("pf"),
                    m.get("expectancy"),
                    m.get("sharpe"),
                    m.get("trades"),
                ]
                + [p.get(k, "") for k in sorted(all_keys)]
            )
    if rows:
        best = max(
            rows,
            key=lambda r: (r["metrics"]["expectancy"], r["metrics"]["pf"], r["metrics"]["sharpe"]),
        )
        with open(best_json, "w") as f:
            json.dump({"best": best}, f, indent=2)

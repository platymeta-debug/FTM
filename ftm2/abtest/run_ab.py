# [ANCHOR:AB_RUNNER]
import json, copy, os
from ftm2.backtest.engine import BacktestEngine
from ftm2.config.settings import cfg as base_cfg
from ftm2.backtest.loader import load_dataset
from ftm2.backtest.metrics import compute as metrics_compute


def apply_overrides(cfg, overrides: dict):
    for k,v in overrides.items():
        if hasattr(cfg,k):
            setattr(cfg,k,v)
        else:
            try:
                cur = getattr(cfg,k); cur.update(v)
            except Exception:
                setattr(cfg,k,v)
    return cfg

def build_ab_cfgs():
    A = json.loads(os.getenv("AB_PARAMS_A","{}"))
    B = json.loads(os.getenv("AB_PARAMS_B","{}"))
    cfgA = copy.deepcopy(base_cfg); cfgB = copy.deepcopy(base_cfg)
    apply_overrides(cfgA, A); apply_overrides(cfgB, B)
    return cfgA, cfgB

def run_variant(cfg):
    data = load_dataset(cfg)
    analysis, router = build_analysis_and_router_for_backtest(cfg, data)  # type: ignore[name-defined]
    bt = BacktestEngine(cfg, data, analysis, router, fees_bps=cfg.BT_FEES_BPS)
    res = bt.run(cfg.BT_START_IDX, cfg.BT_END_IDX)
    return res

if __name__=="__main__":
    cfgA, cfgB = build_ab_cfgs()
    resA = run_variant(cfgA); resB = run_variant(cfgB)
    print(json.dumps({"A":resA,"B":resB,"diff":{
        "expectancy": (resB["expectancy"]-resA["expectancy"]),
        "pf": (resB["pf"]-resA["pf"]),
        "sharpe": (resB["sharpe"]-resA["sharpe"])
    }}, indent=2))

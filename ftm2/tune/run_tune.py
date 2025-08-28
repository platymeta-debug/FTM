import os
from ftm2.tune.sweep import run_sweep, export_results
from ftm2.backtest.engine import BacktestEngine
from ftm2.config.settings import cfg as base_cfg


def bt_factory(cfg):
    from ftm2.backtest.loader import load_dataset
    data = load_dataset(cfg)
    analysis, router = build_analysis_and_router_for_backtest(cfg, data)  # type: ignore[name-defined]
    return BacktestEngine(cfg, data, analysis, router, fees_bps=cfg.BT_FEES_BPS)


if __name__ == "__main__":
    best, rows = run_sweep(base_cfg, bt_factory)
    export_results(
        rows,
        os.getenv("TUNE_EXPORT_CSV", "./reports/tune_results.csv"),
        os.getenv("TUNE_EXPORT_BEST", "./reports/tune_best.json"),
    )
    print("[TUNE] done.")

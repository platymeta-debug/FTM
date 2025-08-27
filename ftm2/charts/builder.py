# [ANCHOR:M6_CHARTS_BUILDER_POLICY]
import os, time, glob
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _prune_rotate(dir_path: str, keep: int):
    files = sorted(glob.glob(os.path.join(dir_path, "*.png")))
    if len(files) > keep:
        for f in files[:-keep]:
            try:
                os.remove(f)
            except Exception:
                pass


def _name_overwrite(sym: str, tf: str, panel: int):
    return f"{sym}_{tf}_current_panel{panel}.png"


def _name_rotate(sym: str, tf: str, panel: int):
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{sym}_{tf}_{ts}_panel{panel}.png"


def render_analysis_charts(cfg, snapshot, out_dir: str) -> list[str]:
    """
    저장 정책:
    - overwrite: 고정 파일명으로 덮어쓰기 (폴더에 파일 2개만 유지)
    - rotate: 타임스탬프 파일명, CHART_KEEP_PER_SYMBOL 만큼만 남기고 자동 삭제
    - none: 디스크 저장 안 함(Discord 업로드 직전에 temp 파일 쓰고 즉시 삭제하고 싶으면 처리)
    """
    tf = "1m"   # 기본 패널 기준 TF
    sym = snapshot.symbol
    base_dir = os.path.join(cfg.CHART_DIR, sym.lower(), tf)
    _ensure_dir(base_dir)

    # 파일명 결정
    if cfg.CHART_MODE == "overwrite":
        fn1 = _name_overwrite(sym, tf, 1)
        fn2 = _name_overwrite(sym, tf, 2)
    elif cfg.CHART_MODE == "rotate":
        fn1 = _name_rotate(sym, tf, 1)
        fn2 = _name_rotate(sym, tf, 2)
    else:  # none
        fn1 = _name_overwrite(sym, tf, 1)
        fn2 = _name_overwrite(sym, tf, 2)

    p1 = os.path.join(base_dir, fn1)
    p2 = os.path.join(base_dir, fn2)

    indicators = getattr(snapshot, "indicators", {}) or {}
    df = indicators.get(tf)
    if df is None:
        df = indicators.get("main")

    if df is None or len(df) == 0:
        return []

    # Panel 1
    plt.figure()
    plt.plot(df["close"])
    for k in ("ema_fast", "ema_slow", "tenkan", "kijun", "kama", "vwap"):
        if k in df.columns:
            plt.plot(df[k])
    plt.title("Panel 1")
    plt.savefig(p1)
    plt.close()

    # Panel 2
    plt.figure()
    for k in ("rsi", "adx", "cci", "obv"):
        if k in df.columns:
            plt.plot(df[k])
    plt.title("Panel 2")
    plt.savefig(p2)
    plt.close()

    # rotate면 프루닝
    if cfg.CHART_MODE == "rotate":
        _prune_rotate(base_dir, cfg.CHART_KEEP_PER_SYMBOL)
    elif cfg.CHART_MODE == "none":
        pass  # 필요하면 즉시 삭제(Discord 업로드 후)

    return [p1, p2]


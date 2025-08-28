# [ANCHOR:M6_CHARTS_BUILDER_POLICY]
import os
import matplotlib.pyplot as plt


def render_analysis_charts(snapshot, out_dir: str) -> list[str]:
    """간단한 분석 차트 4장 렌더링.

    가격 패널 1장(캔들 대신 종가 라인) + EMA/Bollinger 오버레이,
    RSI/ADX/CCI 패널 각각 1장씩.
    """
    sym = getattr(snapshot, "symbol", "UNK")
    base_dir = os.path.join(out_dir, sym.lower())
    os.makedirs(base_dir, exist_ok=True)

    indicators = getattr(snapshot, "indicators", {}) or {}
    df = indicators.get("15m")
    if df is None:
        df = next(iter(indicators.values()), None)
    if df is None or len(df) == 0:
        return []

    paths: list[str] = []

    price_png = os.path.join(base_dir, f"{sym}_price.png")
    plt.figure()
    plt.plot(df["close"], label="close")
    for k in ("ema_fast", "ema_slow", "ema50", "ema200"):
        if k in df.columns:
            plt.plot(df[k], label=k)
    if "bb_up" in df.columns and "bb_dn" in df.columns:
        plt.plot(df["bb_up"], label="bb_up")
        plt.plot(df["bb_dn"], label="bb_dn")
    plt.title("Price")
    plt.legend()
    plt.savefig(price_png)
    plt.close()
    paths.append(price_png)

    # RSI panel
    rsi_png = os.path.join(base_dir, f"{sym}_rsi.png")
    plt.figure()
    if "rsi" in df.columns:
        plt.plot(df["rsi"], label="rsi")
    plt.title("RSI")
    plt.savefig(rsi_png)
    plt.close()
    paths.append(rsi_png)

    # ADX panel
    adx_png = os.path.join(base_dir, f"{sym}_adx.png")
    plt.figure()
    if "adx" in df.columns:
        plt.plot(df["adx"], label="adx")
    plt.title("ADX")
    plt.savefig(adx_png)
    plt.close()
    paths.append(adx_png)

    # CCI panel
    cci_png = os.path.join(base_dir, f"{sym}_cci.png")
    plt.figure()
    if "cci" in df.columns:
        plt.plot(df["cci"], label="cci")
    plt.title("CCI")
    plt.savefig(cci_png)
    plt.close()
    paths.append(cci_png)

    return paths


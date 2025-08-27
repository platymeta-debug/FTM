import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _plot_indicators(ax, df: pd.DataFrame, indicators: dict | None):
    if not indicators:
        return
    if "sma_fast" in indicators and "sma_slow" in indicators:
        ax.plot(df.index, indicators["sma_fast"], linewidth=1.2, alpha=0.9, label="SMA Fast")
        ax.plot(df.index, indicators["sma_slow"], linewidth=1.2, alpha=0.9, label="SMA Slow")
        ax.legend(loc="best", fontsize=8)


def render_chart(symbol: str, df: pd.DataFrame, indicators: dict | None, out_dir: str) -> str:
    """Render chart with optional indicators and return output path."""
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{symbol}_chart.png")

    fig = plt.figure(figsize=(10, 4), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(df.index, df["close"], linewidth=1.0)
    _plot_indicators(ax, df, indicators)
    ax.set_title(symbol)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

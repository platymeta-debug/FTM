from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def render_analysis_charts(snapshot, out_dir: str) -> List[str]:
    """Render placeholder analysis charts and return list of image paths."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = Path(out_dir) / f"{snapshot.symbol}_{snapshot.tfs[0]}_{ts}"
    paths = []
    for i in (1, 2):
        fig, ax = plt.subplots()
        ax.set_title(f"Panel {i}")
        ax.plot([0, 1], [0, 1])
        p = f"{base}_panel{i}.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths

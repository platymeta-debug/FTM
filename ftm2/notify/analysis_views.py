from __future__ import annotations
import discord

import time
from ftm2.strategy.compat import to_viewdict


def build_analysis_embed(cfg, snapshot, divergence_bps: float, next_eta_sec: int) -> discord.Embed:
    """Build a rich analysis embed for Discord."""
    snap = to_viewdict(snapshot)
    title = f"{snap.get('symbol')} 분석"
    desc = (
        f"총점 {snap.get('total_score', 0):.1f} / 방향: {snap.get('direction')} / "
        f"신뢰도: {snap.get('confidence', 0):.2f} / 괴리: {divergence_bps:.2f}bps"
    )
    emb = discord.Embed(title=title, description=desc)

    order = ["15m", "4h", "1h", "1d"]
    contribs = snap.get("contribs", {})
    for tf in order:
        cs = contribs.get(tf) or []
        if not cs:
            continue
        lines = [f"• {c['name']}: {c['text']} ({c['score']:+.1f})" for c in cs[:8]]
        emb.add_field(name=f"{tf} 해석", value="\n".join(lines) if lines else "-", inline=False)

    now_txt = time.strftime("%H:%M:%S", time.localtime())
    emb.set_footer(text=f"마지막 갱신 {now_txt} | 다음 갱신까지 {next_eta_sec}s")
    return emb

from __future__ import annotations
import discord



def build_analysis_embed(view: dict, divergence_bps: float, next_s: int) -> discord.Embed:
    """Build a rich analysis embed for Discord from lightweight view data."""
    title = f"{view.get('symbol')} 분석"
    desc = (
        f"총점 {view.get('total_score', 0):.1f} / 방향: {view.get('direction')} / "
        f"신뢰도: {view.get('confidence', 0):.2f} / 괴리: {divergence_bps:.2f}bps"
    )
    emb = discord.Embed(title=title, description=desc)

    order = ["15m", "4h", "1h", "1d"]

    contribs = view.get("contribs", {})
    for tf in order:
        cs = contribs.get(tf) or []
        if not cs:
            continue
        lines = [f"• {c['name']}: {c['text']} ({c['score']:+.1f})" for c in cs[:8]]
        emb.add_field(name=f"{tf} 해석", value="\n".join(lines) if lines else "-", inline=False)


    return emb

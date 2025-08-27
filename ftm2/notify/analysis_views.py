from __future__ import annotations
import discord

from ftm2.analysis.state import AnalysisSnapshot


def build_analysis_embed(cfg, snapshot: AnalysisSnapshot, divergence_bps: float, next_eta_sec: int) -> discord.Embed:
    """Build a simple analysis embed for Discord."""
    title = f"{snapshot.symbol} 분석"
    desc = f"총점: {snapshot.total_score:.1f} / 방향: {snapshot.direction}\n" \
           f"신뢰도: {snapshot.confidence:.2f} / 괴리: {divergence_bps:.2f}bps"
    emb = discord.Embed(title=title, description=desc)
    emb.set_footer(text=f"다음 갱신까지 {next_eta_sec}s")
    return emb

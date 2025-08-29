"""Templates for signal v2 cards."""

from __future__ import annotations


def position_card(
    sym: str,
    side: str,
    qty: float,
    leverage: float,
    mode: str,
    avg_px: float,
    notional: float,
    init_margin: float,
    sl: float,
    tp1: tuple[float, float],
    tp2: tuple[float, float],
    rr: float,
    unrealized: float,
    realized_today: float,
) -> str:
    """Render a position card text block.

    Args:
        sym: Symbol name.
        side: LONG/SHORT side.
        qty: Position size.
        leverage: Leverage value.
        mode: Margin mode (iso/cross).
        avg_px: Average entry price.
        notional: Current notional value.
        init_margin: Approx initial margin.
        sl: Stop loss price.
        tp1: Tuple of (price, pct).
        tp2: Tuple of (price, pct).
        rr: Risk reward ratio.
        unrealized: Unrealized PnL.
        realized_today: Today's realized PnL.
    """

    t1_px, t1_pct = tp1
    t2_px, t2_pct = tp2
    return (
        f"{sym} | {side} x{qty} (Lvg {leverage}x {mode})\n"
        f"AvgPx {avg_px:.4f}, Notional {notional:.2f}, InitialMargin {init_margin:.2f}\n"
        f"SL @{sl:.4f}, TP1 @{t1_px:.4f}/{t1_pct:.1f}%, TP2 @{t2_px:.4f}/{t2_pct:.1f}%, RR {rr:.2f}\n"
        f"Unrealized {unrealized:.2f}, Today Realized {realized_today:.2f}"
    )


def analysis_card(
    score: int,
    trend: str,
    divergence: str,
    atr: float,
    rr_entry: float,
    invalidation: float,
    entry_window: str,
    cooldown: str,
    last_change: str,
) -> str:
    """Render an analysis card text block."""

    return (
        f"Score {score} | HTF {trend} | Div {divergence} | ATR {atr:.4f}\n"
        f"RR@entry {rr_entry:.2f} | Inval {invalidation:.4f}\n"
        f"Entry {entry_window} | Cooldown {cooldown} | Last {last_change}"
    )


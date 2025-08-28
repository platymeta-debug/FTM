# [ANCHOR:BT_METRICS]
def compute(trades):
    import math
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    wr = len(wins) / max(1, len(trades))
    avg_w = sum(wins) / max(1, len(wins))
    avg_l = abs(sum(losses) / max(1, len(losses)))
    pf = (sum(wins) / max(1, abs(sum(losses)))) if losses else float("inf")
    exp = wr * avg_w - (1 - wr) * avg_l
    ret = [t.roe for t in trades]
    mu = sum(ret) / max(1, len(ret))
    sigma = (sum((r - mu) ** 2 for r in ret) / max(1, len(ret))) ** 0.5
    sharpe = mu / max(1e-9, sigma)
    return {"winrate": wr, "avg_win": avg_w, "avg_loss": avg_l, "pf": pf, "expectancy": exp, "sharpe": sharpe}

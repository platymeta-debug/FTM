import time
from dataclasses import dataclass


@dataclass
class SyncIssue:
    code: str
    detail: str


class SyncGuard:
    def __init__(self, cfg, market, notify):
        self.cfg = cfg
        self.market = market
        self.notify = notify

    async def verify_after_fill(self, sym: str, rt, bracket, analysis_views) -> bool:
        issues: list[SyncIssue] = []
        pos = rt.positions.get(sym)
        if not pos or abs(pos.qty) <= 1e-12:
            issues.append(SyncIssue("NO_POS", "포지션 없음"))
        if sym in rt.active_ticket:
            issues.append(SyncIssue("TICKET_NOT_CONSUMED", "체결 후 티켓 미소모"))
        sl, tps = await bracket.current_brackets(sym)
        if not sl:
            issues.append(SyncIssue("NO_SL", "SL 미설정"))
        if not tps:
            issues.append(SyncIssue("NO_TP", "TP 미설정"))
        if pos and pos.leverage != int(
            self.cfg.LEVERAGE_OVERRIDE.get(sym, self.cfg.LEVERAGE_DEFAULT)
        ):
            issues.append(
                SyncIssue(
                    "LEV_MISMATCH",
                    f"now={pos.leverage} cfg={self.cfg.LEVERAGE_OVERRIDE.get(sym)}",
                )
            )
        if hasattr(analysis_views, "active_empty") and analysis_views.active_empty(sym):
            issues.append(SyncIssue("ANLS_INACTIVE", "분석 ACTIVE 섹션 비어있음"))
        if issues:
            msg = " | ".join(f"{x.code}:{x.detail}" for x in issues)
            self.notify.emit("error", f"🧭 SYNC {sym}: {msg}")
            return False
        return True

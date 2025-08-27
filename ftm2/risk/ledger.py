from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date
from zoneinfo import ZoneInfo
import time

def _now_tz(tz: str):
    try: return datetime.now(ZoneInfo(tz))
    except: return datetime.utcnow()

@dataclass
class DailyStats:
    day: str = ""
    start_equity: float = 0.0
    realized: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    trades: int = 0
    wins: int = 0
    max_dd: float = 0.0
    peak_equity: float = 0.0

    @property
    def net(self) -> float:
        return self.realized - self.fees + self.funding

class DailyLedger:
    def __init__(self, cfg, csv_logger):
        self.cfg = cfg
        self.csv = csv_logger
        self.stats = DailyStats()
        self.curr_day = ""
        self.cooldown_until = 0.0

    def rollover_if_needed(self, equity_now: float):
        d = _now_tz(self.cfg.REPORT_TZ).strftime("%Y-%m-%d")
        if self.curr_day == "":
            self.curr_day = d
            self.stats = DailyStats(day=d, start_equity=equity_now, peak_equity=equity_now)
            return
        if d != self.curr_day:
            # 전일 요약 기록
            s = self.stats
            self.csv.log("DAILY_SUMMARY",
                         equity=s.start_equity + s.net,
                         wallet="", avail="",
                         realized=s.realized, fee=s.fees, funding_cum=s.funding,
                         reason="rollover", day=s.day, elapsed_sec="")
            # 새 일자 시작
            self.curr_day = d
            self.stats = DailyStats(day=d, start_equity=equity_now, peak_equity=equity_now)

    def on_equity_tick(self, equity_now: float):
        s = self.stats
        if equity_now > s.peak_equity:
            s.peak_equity = equity_now
        dd = max(0.0, s.peak_equity - equity_now)
        if dd > s.max_dd:
            s.max_dd = dd

    def on_realized(self, realized: float, fee: float, win: bool|None=None):
        s = self.stats
        s.realized += realized
        s.fees += fee
        s.trades += 1
        if win is True: s.wins += 1

    def on_funding(self, amount: float):
        self.stats.funding += amount

class LossCutController:
    def __init__(self, cfg, ledger: DailyLedger, tracker, router, notify, csv_logger):
        self.cfg = cfg
        self.ledger = ledger
        self.tracker = tracker
        self.router = router
        self.notify = notify
        self.csv = csv_logger

    def _limit_breached(self):
        s = self.ledger.stats
        # 금액 기준
        if self.cfg.LOSS_CUT_DAILY_USDT > 0 and (-s.net) >= self.cfg.LOSS_CUT_DAILY_USDT:
            return True
        # 퍼센트 기준(시작자본 대비)
        if self.cfg.LOSS_CUT_DAILY_PCT > 0 and s.start_equity > 0:
            loss_pct = max(0.0, -s.net) / s.start_equity * 100.0
            if loss_pct >= self.cfg.LOSS_CUT_DAILY_PCT:
                return True
        return False

    async def check_and_fire(self):
        # 쿨다운 중이면 통과
        if time.time() < self.ledger.cooldown_until:
            return
        if not self._limit_breached():
            return
        # 발동
        action = self.cfg.LOSS_CUT_ACTION
        self.ledger.cooldown_until = time.time() + self.cfg.LOSS_CUT_COOLDOWN_MIN * 60
        self.csv.log("LOSS_CUT", reason=action, equity=self.tracker.account.equity)
        await self.notify(f"⚠️ 손실컷 발동: 정책={action}")
        if action == "close_all":
            # 모든 심볼 reduceOnly 시장가
            for sym in set(k.split(":")[0] for k in self.tracker.pos.keys()):
                try:
                    await self.router.close_all(sym)
                except Exception as e:
                    await self.notify(f"청산 오류 {sym}: {e}")
        elif action == "pause_only":
            self.tracker.disable_edits(True)
            await self.notify("알림/편집이 일시중지되었습니다.")

    def reset_cooldown(self):
        self.ledger.cooldown_until = 0.0


from __future__ import annotations
import asyncio, csv, os, io
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

DEFAULT_HEADER = [
    # 타임/구분
    "ts_iso","day","event","session_id","profile","strategy_ver",
    # 주문/포지션 공통
    "symbol","side","reason","order_id","client_id","position_id",
    # 가격/수량/손익
    "price","qty","notional","mark","entry","exit","upnl","roe",
    "realized","fee","fee_cum","funding_cum",
    # 계정/리스크
    "wallet","equity","avail","leverage","margin","sl","tp","liq",
    # 분석/점수(미기입시 공란)
    "score_total","score.rsi","score.adx","score.atr","score.ema","score.ichimoku",
    "score.kama","score.vwap","score.cci","score.obv","score.corr","score.mtf",
    # 지표 스냅샷
    "ind.rsi","ind.adx","ind.atr","ind.ema50","ind.ema200",
    "ind.kijun","ind.tenkan","ind.senkouA","ind.senkouB",
    "ind.kama","ind.vwap","ind.cci","ind.obv",
    # MTF 부스트/상태
    "mtf.h1.boost","mtf.h4.boost","mtf.d1.boost","trend_state",
    # 라우팅/위험
    "risk.tier","risk.leverage","risk.max_loss","risk.daily_cut",
    "risk.bracket_r","risk.size_pct",
    "route.slippage","route.post_only","route.reduce_only","route.type",
    # 기타
    "elapsed_sec"
]

def _tz_now(tz: str) -> datetime:
    try: return datetime.now(ZoneInfo(tz))
    except: return datetime.utcnow()

def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        kk = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, kk))
        else:
            out[kk] = v
    return out

class CsvLogger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base = Path(cfg.CSV_DIR)
        self.base.mkdir(parents=True, exist_ok=True)
        self.file_path = self.base / cfg.CSV_FILE   # 단일 파일
        self.header = DEFAULT_HEADER[:]             # 고정 슈퍼셋
        self.q: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._writer_task: asyncio.Task | None = None
        self._w = None  # csv.writer
        self._fh = None # file handle
        self.session_id = f"S{_tz_now(cfg.REPORT_TZ).strftime('%Y%m%d')}-{os.getpid()}"

    async def start(self):
        self._ensure_open()
        self._writer_task = asyncio.create_task(self._writer_loop())

    def _ensure_open(self):
        new = not self.file_path.exists() or self.file_path.stat().st_size == 0
        self._fh = open(self.file_path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        if new:
            self._w.writerow(self.header)
            self._fh.flush()

    async def _writer_loop(self):
        flush_interval = max(1, int(self.cfg.CSV_FLUSH_SEC))
        buf = []
        last = asyncio.get_event_loop().time()
        while True:
            try:
                item = await asyncio.wait_for(self.q.get(), timeout=flush_interval)
                buf.append(item)
            except asyncio.TimeoutError:
                pass
            now = asyncio.get_event_loop().time()
            if buf and (now - last) >= flush_interval:
                for row in buf:
                    self._w.writerow(row)
                self._fh.flush()
                buf.clear()
                last = now

    def _row_from_event(self, ev: dict) -> list:
        # 공통 필드 + 낯선 키는 무시(슈퍼셋 헤더 기준)
        flat = _flatten(ev)
        row = []
        for col in self.header:
            row.append(flat.get(col, ""))
        return row

    def log(self, event: str, **payload):
        """비동기 큐에 적재. payload는 dict/중첩 dict 가능."""
        ts = _tz_now(self.cfg.REPORT_TZ)
        base = {
            "ts_iso": ts.isoformat(timespec="seconds"),
            "day": ts.strftime("%Y-%m-%d"),
            "event": event,
            "session_id": self.session_id,
            "profile": os.getenv("ENV_PROFILE",""),
        }
        base.update(payload or {})
        row = self._row_from_event(base)
        try:
            self.q.put_nowait(row)
        except asyncio.QueueFull:
            # 드랍 대신 즉시 동기 기록(최후 수단)
            self._w.writerow(row); self._fh.flush()

    async def stop(self):
        if self._writer_task:
            self._writer_task.cancel()
        try:
            if self._fh: self._fh.close()
        except: pass


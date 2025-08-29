import time
class OpsBoard:
    def __init__(self, cfg, notify, collector, renderer):
        self.cfg, self.notify = cfg, notify
        self.collect = collector
        self.render  = renderer
        self._last_edit = 0
        self._card = None

    async def tick(self, rt, market, bracket, guard=None):
        now = time.time()
        if now - self._last_edit < self.cfg.DASH_EDIT_MIN_MS/1000: return
        ops = self.collect(rt, self.cfg, market, bracket, guard)
        text = self.render(ops)
        try:
            if self._card and (now - self._card["created_at"] < self.cfg.DASH_LIFETIME_MIN*60):
                await self.notify.dc.edit(self._card["id"], text)
            else:
                # 채널 이름/별칭/ID 모두 허용, dispatcher가 알아서 해석
                mid = await self.notify.dc.send(self.cfg.CHANNEL_SIGNALS, text)
                self._card = {"id": mid, "created_at": now}
            self._last_edit = now
        except Exception as e:
            # 채널 설정 이상 등 모든 예외 → logs로 폴백 1회
            try:
                await self.notify.dc.send(self.cfg.CHANNEL_LOGS, f"[DASH_FALLBACK] {type(e).__name__}: {e}\n{text}")
            except Exception:
                # 마지막 안전장치: 스팸 방지용 내부 emit
                self.notify.emit("error", f"dash tick err(fallback): {type(e).__name__}: {e}")

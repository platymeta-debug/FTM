import time
from ftm2.notify.discord_bot import upsert


class OpsBoard:
    def __init__(self, cfg, notify, collector, renderer):
        self.cfg, self.notify = cfg, notify
        self.collect = collector
        self.render = renderer
        self._last_edit = 0

    async def tick(self, rt, market, bracket, guard=None):
        now = time.time()
        if now - self._last_edit < self.cfg.DASH_EDIT_MIN_MS / 1000:
            return
        ops = self.collect(rt, self.cfg, market, bracket, guard)
        payload = self.render(ops)
        if payload == getattr(self, "_last_payload", None):
            return  # 내용 변동 없으면 전송 안 함
        self._last_payload = payload
        await upsert(self.cfg.CHANNEL_SIGNALS, payload, sticky_key="ops_board")
        self._last_edit = now


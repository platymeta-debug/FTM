import time
from ftm2.notify.discord_bot import upsert


from ftm2.notify import discord_bot


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
        text = self.render(ops)
        # [ANCHOR:DASH_UPSERT_ONLY_IF_CHANGED]
        if text == getattr(self, "_last_payload", None):
            return
        try:
            await discord_bot.upsert(
                self.cfg.CHANNEL_SIGNALS,
                text,
                sticky_key="ops_board",
            )
            self._last_payload = text
            self._last_edit = now
        except Exception as e:
            try:
                await self.notify.dc.send(

                    self.cfg.CHANNEL_LOGS,
                    f"[DASH_FALLBACK] {type(e).__name__}: {e}\n{text}",
                    sticky_key="ops_board_err",
                )
            except Exception:
                self.notify.emit(
                    "error", f"dash tick err(fallback): {type(e).__name__}: {e}"
                )



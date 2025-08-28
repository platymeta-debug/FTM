# [ANCHOR:OFFLINE_REPLAY]
class OfflineReplayer:
    def __init__(self, cfg, feed):
        self.cfg, self.feed = cfg, feed
        self.analysis = getattr(feed, 'analysis', None)

    async def run(self):
        for bar in self.feed:
            if hasattr(self.analysis, 'on_bar'):
                await self.analysis.on_bar(bar)

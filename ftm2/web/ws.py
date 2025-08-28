# [ANCHOR:WEB_WSHUB]
import json, asyncio
class WSHub:
    def __init__(self): self._clients=set(); self._lock=asyncio.Lock()
    async def register(self, ws):
        await ws.accept()
        async with self._lock: self._clients.add(ws)
    async def unregister(self, ws):
        async with self._lock:
            if ws in self._clients: self._clients.remove(ws)
    async def broadcast_json(self, obj):
        dead=[]
        for ws in list(self._clients):
            try:
                await ws.send_text(json.dumps(obj, ensure_ascii=False))
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.unregister(ws)

# [ANCHOR:WEB_APP]
import os, asyncio, uvicorn
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ftm2.dashboard.collect import collect as dash_collect
from ftm2.web.auth import verify
from ftm2.web.ws import WSHub
from ftm2.web.ipguard import require_ip_allow


app = FastAPI(title="FTM2 Web", version="1.0")
hub = WSHub()


# [ANCHOR:WEB_READONLY_FLAG]
READONLY = os.getenv("WEB_READONLY","true").lower() in ("1","true","yes")

@app.middleware("http")
async def readonly_guard(request, call_next):
    if READONLY and request.method not in ("GET","HEAD","OPTIONS"):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error":"readonly mode"}, status_code=403)
    return await call_next(request)


def init(app, cfg, rt, market, bracket, notify):
    origins = [o.strip() for o in os.getenv("WEB_CORS", "*").split(",")]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    app.mount("/ui", StaticFiles(directory=os.getenv("WEB_STATIC_DIR","./ftm2/web/static"), html=True), name="ui")


    # [ANCHOR:WEB_SAFE_ROUTES]
    DISABLE_AB = os.getenv("DISABLE_AB_API","true").lower() in ("1","true","yes")
    DISABLE_SQL = os.getenv("DISABLE_SQL_API","true").lower() in ("1","true","yes")

    from ftm2.web.reports import router as reports_router
    app.include_router(reports_router)

    if not DISABLE_AB:
        from ftm2.web.ab import router as ab_router
        app.include_router(ab_router)


    @app.get("/api/health")
    async def health():
        return {"ok": True}

    @app.get("/api/state")
    async def state(_: None = Depends(verify), __: None = Depends(require_ip_allow)):

        ops = dash_collect(rt, cfg, market, bracket, None)
        return ops.__dict__

    @app.get("/api/positions")
    async def positions(_: None = Depends(verify), __: None = Depends(require_ip_allow)):
        return {k: v.__dict__ if hasattr(v,"__dict__") else v for k,v in rt.positions.items()}

    @app.get("/api/tickets")
    async def tickets(_: None = Depends(verify), __: None = Depends(require_ip_allow)):

        return {k: vars(v) for k,v in rt.active_ticket.items()}

    @app.websocket("/ws")
    async def ws(ws: WebSocket):
        token = ws.query_params.get("token")
        if not verify(token=token, direct=True):
            await ws.close(code=4401); return
        await hub.register(ws)
        try:
            while True:
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            await hub.unregister(ws)

    async def broadcaster():
        while True:
            try:
                ops = dash_collect(rt, cfg, market, bracket, None)
                await hub.broadcast_json({"type":"ops", "data": ops.__dict__})
            except Exception as e:
                notify.emit("error", f"web broadcast err: {type(e).__name__}: {e}")
            await asyncio.sleep(cfg.WEB_PUSH_INTERVAL_S)

    return broadcaster

def run_standalone(cfg, rt, market, bracket, notify):
    init(app, cfg, rt, market, bracket, notify)
    uvicorn.run(app, host=os.getenv("WEB_HOST","0.0.0.0"), port=int(os.getenv("WEB_PORT","8088")))

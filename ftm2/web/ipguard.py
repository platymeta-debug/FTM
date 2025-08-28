# [ANCHOR:IP_GUARD]
import os
from fastapi import Request, HTTPException

ALLOW = {x.strip() for x in os.getenv("WEB_IP_ALLOW","" ).split(",") if x.strip()}

async def require_ip_allow(request: Request):
    if not ALLOW:
        return
    ip = request.client.host
    if ip not in ALLOW:
        raise HTTPException(403, "ip not allowed")

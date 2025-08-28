# [ANCHOR:WEB_AUTH]
import os
from fastapi import Header, HTTPException

def verify(authorization: str | None = Header(default=None), token: str | None = None, direct: bool=False):
    want = os.getenv("WEB_TOKEN","devtoken")
    got = token or (authorization.split("Bearer ")[-1] if authorization else None)
    if got != want:
        if direct: return False
        raise HTTPException(status_code=401, detail="invalid token")
    return None

# [ANCHOR:API_RETRY]
import asyncio, random


def need_retry(exc_text: str, status: int):
    if status in (500, 502, 503, 504):
        return True
    if status == 429:
        return True
    if "Internal error" in exc_text:
        return True
    return False


def is_min_notional(exc_text: str):
    t = exc_text.lower()
    return ("min notional" in t) or ("minnominal" in t) or ("-1013" in t and "filter" in t)


async def with_retry(coro_factory, *, tries: int, base_ms: int):
    err = None
    for i in range(tries):
        try:
            return await coro_factory()
        except Exception as e:
            txt = str(e)
            status = getattr(e, "status", getattr(e, "code", 0)) or 0
            if not need_retry(txt, status):
                raise
            delay = (base_ms / 1000.0) * (2 ** i) + random.uniform(0, 0.2)
            await asyncio.sleep(delay)
            err = e
    raise err

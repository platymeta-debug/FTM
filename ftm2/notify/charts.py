def _should_render(last_ts, now_ts, *, force_first: bool, cooldown_s: int, delta_ok: bool):
    if force_first and not last_ts:
        return True
    if cooldown_s > 0 and last_ts and (now_ts - last_ts) < cooldown_s * 1000:
        return False
    return bool(delta_ok)


__all__ = ["_should_render"]


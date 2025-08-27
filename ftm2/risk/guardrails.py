from time import time


class Guardrails:
    """Basic placeholders for risk checks and cooldowns."""

    def __init__(self, cooldown_s: int = 15):
        self.cooldown_s = cooldown_s
        self._last_ts = 0.0

    def allow(self) -> bool:
        now = time()
        if now - self._last_ts >= self.cooldown_s:
            self._last_ts = now
            return True
        return False

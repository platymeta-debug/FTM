from time import time


class TimeSync:
    """Maintain offset between local and server time."""

    def __init__(self):
        self.offset = 0.0

    def sync(self, server_ms: int):
        self.offset = server_ms / 1000 - time()

    def time(self) -> float:
        return time() + self.offset

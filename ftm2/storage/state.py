import json
from pathlib import Path


class State:
    def __init__(self, path: str = "state.json"):
        self.path = Path(path)
        self.data = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2))

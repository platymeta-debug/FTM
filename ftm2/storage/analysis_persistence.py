from __future__ import annotations
import json
from pathlib import Path

FILE_PATH = Path("storage/analysis_cards.json")


def load_analysis_cards() -> dict:
    if FILE_PATH.exists():
        try:
            return json.loads(FILE_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_analysis_cards(data: dict) -> None:
    FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FILE_PATH.write_text(json.dumps(data))

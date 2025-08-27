# [ANCHOR:M5_PERSIST]
from __future__ import annotations
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # ftm2/
STORE = ROOT / "storage" / "trade_cards.json"
STORE.parent.mkdir(parents=True, exist_ok=True)


def load_trade_cards() -> dict:
    if not STORE.exists(): return {}
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_trade_cards(data: dict):
    tmp = STORE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STORE)


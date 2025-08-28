import glob
import json
import os
import time
from typing import Optional

SNAP_DIR = "./logs/snapshots"


def dump_state(rt, cfg) -> str:
    os.makedirs(SNAP_DIR, exist_ok=True)
    snap = {
        "ts": time.time(),
        "tickets": {k: vars(v) for k, v in rt.active_ticket.items()},
        "idem": {f"{k[0]}|{k[1]}": v for k, v in getattr(rt, "idem_hit", {}).items()},
        "cooldown": getattr(rt, "cooldown_until", {}),
        "cfg": {
            "LEVERAGE_OVERRIDE": cfg.LEVERAGE_OVERRIDE,
            "MARGIN_MODE_OVERRIDE": cfg.MARGIN_MODE_OVERRIDE,
            "RISK_PCT_OVERRIDE": cfg.RISK_PCT_OVERRIDE,
        },
    }
    path = os.path.join(SNAP_DIR, f"ftm_state_{int(snap['ts'])}.json")
    with open(path, "w") as f:
        json.dump(snap, f, indent=2)
    return path


def latest_state() -> Optional[str]:
    xs = sorted(glob.glob(os.path.join(SNAP_DIR, "ftm_state_*.json")))
    return xs[-1] if xs else None


def load_state(rt, cfg, path: Optional[str] = None) -> bool:
    path = path or latest_state()
    if not path:
        return False
    with open(path) as f:
        s = json.load(f)
    rt.active_ticket.clear()
    for k, v in s.get("tickets", {}).items():
        from ftm2.analysis.types import SetupTicket

        rt.active_ticket[k] = SetupTicket(**v)
    rt.idem_hit = {tuple(x.split("|")): y for x, y in s.get("idem", {}).items()}
    rt.cooldown_until = s.get("cooldown", {})
    for k, v in s.get("cfg", {}).items():
        if hasattr(cfg, k):
            attr = getattr(cfg, k)
            if isinstance(attr, dict):
                attr.update(v)
            else:
                setattr(cfg, k, v)
    return True

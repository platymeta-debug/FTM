# [ANCHOR:CID_BUILD]
import hashlib, time

def build_cid(prefix: str, sym: str, ticket_id: str, bar_ts: int):
    raw = f"{prefix}|{sym}|{ticket_id}|{bar_ts}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:10]
    return f"{prefix}_{sym}_{ticket_id}_{bar_ts}_{h}"

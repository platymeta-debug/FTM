import os, csv, sqlite3, time, threading
from .events import JEvent

class Journal:
    def __init__(self, cfg, tz="Asia/Seoul"):
        self.cfg = cfg
        self.tz  = tz
        self.session = f"{int(time.time())}"
        self._csv_lock = threading.Lock()
        self._ensure_dirs()

        self._db = None
        if cfg.JOURNAL_SQLITE_ENABLE:
            self._db = sqlite3.connect(cfg.JOURNAL_SQLITE_PATH, check_same_thread=False)
            self._migrate()

    def _ensure_dirs(self):
        os.makedirs(self.cfg.JOURNAL_DIR, exist_ok=True)

    def _csv_path(self):
        d = time.strftime("%Y%m%d")
        return os.path.join(self.cfg.JOURNAL_DIR, f"{d}_trades.csv")

    def _migrate(self):
        cur = self._db.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS journal(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, kind TEXT, symbol TEXT, side TEXT, qty REAL, price REAL,
            order_id TEXT, ticket_id TEXT, message TEXT, entry REAL, mark REAL,
            sl REAL, tp1 REAL, tp2 REAL, upnl REAL, roe REAL, realized REAL,
            lev INTEGER, mode TEXT, session TEXT
        );
        """)
        self._db.commit()

    def write(self, ev: JEvent):
        # CSV
        if self.cfg.JOURNAL_CSV_ENABLE:
            with self._csv_lock:
                path = self._csv_path()
                is_new = not os.path.exists(path)
                with open(path,"a",newline="",encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(ev.to_row().keys()))
                    if is_new: w.writeheader()
                    row = ev.to_row(); row["session"] = self.session
                    w.writerow(row)
        # SQLite
        if self._db:
            cur = self._db.cursor()
            r = ev.to_row(); r["session"] = self.session
            cur.execute("""INSERT INTO journal
                (ts,kind,symbol,side,qty,price,order_id,ticket_id,message,entry,mark,sl,tp1,tp2,upnl,roe,realized,lev,mode,session)
                VALUES(:ts,:kind,:symbol,:side,:qty,:price,:order_id,:ticket_id,:message,:entry,:mark,:sl,:tp1,:tp2,:upnl,:roe,:realized,:lev,:mode,:session)""", r)
            self._db.commit()

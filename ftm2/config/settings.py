from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv, dotenv_values


PROTECT_KEYS = {"BINANCE_API_KEY", "BINANCE_API_SECRET", "DISCORD_TOKEN"}

def _norm_key(k: str) -> str:
    return k.lstrip("\ufeff").strip() if k else k

def _apply_env_from(p: Path) -> bool:
    if not p.exists():
        return False
    data = dotenv_values(p)
    ok = False
    for k, v in data.items():
        nk = _norm_key(k)
        if nk in PROTECT_KEYS and (v is None or v == ""):
            continue
        if v is not None:
            os.environ[nk] = v
            ok = True
    return ok


ENV_FILES_ORDER = [
    "token.env", ".env", ".env.strategy", ".env.risk", ".env.trade", ".env.notify"
]

def _load_env_series(root: Path, profile: str | None):
    loaded = []
    for f in ENV_FILES_ORDER:
        p = root / f

        if _apply_env_from(p):
            loaded.append(str(p))
        if profile and f.startswith(".env"):
            pp = root / f"{f}.{profile}"
            if _apply_env_from(pp):

                loaded.append(str(pp))
    print(f"[ENV] loaded={len(loaded)} files={loaded}")
    return loaded

class Settings(BaseModel):
    MODE: str = os.getenv("MODE", "testnet")  # testnet | live
    # 민감키는 import 시점 os.getenv 사용을 피한다
    BINANCE_API_KEY: str | None = None
    BINANCE_API_SECRET: str | None = None
    RECV_WINDOW_MS: int = int(os.getenv("RECV_WINDOW_MS", "5000"))
    HTTP_TIMEOUT_S: float = float(os.getenv("HTTP_TIMEOUT_S", "5"))

    SYMBOLS: list[str] = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
    INTERVAL: str = os.getenv("INTERVAL", "1m")
    LOOKBACK: int = int(os.getenv("LOOKBACK", "300"))

    USE_CORR: bool = os.getenv("USE_CORR", "true").lower() == "true"
    CORR_LOOKBACK: int = int(os.getenv("CORR_LOOKBACK", "240"))

    LEVERAGE: int = int(os.getenv("LEVERAGE", "5"))
    MARGIN_TYPE: str = os.getenv("MARGIN_TYPE", "ISOLATED")
    HEDGE_MODE: bool = os.getenv("HEDGE_MODE", "false").lower() == "true"
    RISK_MODE: str = os.getenv("RISK_MODE", "atr_unit")
    RISK_TARGET_USDT: float = float(os.getenv("RISK_TARGET_USDT", "30"))
    ATR_LOOKBACK: int = int(os.getenv("ATR_LOOKBACK", "14"))
    ATR_MULT: float = float(os.getenv("ATR_MULT", "1.5"))
    SL_MULT: float = float(os.getenv("SL_MULT", "1.5"))
    TP_MULT: float = float(os.getenv("TP_MULT", "3.0"))
    MAX_POS_PER_SIDE: int = int(os.getenv("MAX_POS_PER_SIDE", "1"))
    MAX_DAILY_LOSS_USDT: float = float(os.getenv("MAX_DAILY_LOSS_USDT", "200"))
    MAX_OPEN_ORDERS: int = int(os.getenv("MAX_OPEN_ORDERS", "10"))
    LOSS_CUT_DAILY_USDT: float = float(os.getenv("LOSS_CUT_DAILY_USDT", "0.0"))
    LOSS_CUT_DAILY_PCT: float = float(os.getenv("LOSS_CUT_DAILY_PCT", "0.0"))
    LOSS_CUT_ACTION: str = os.getenv("LOSS_CUT_ACTION", "close_all")
    LOSS_CUT_COOLDOWN_MIN: int = int(os.getenv("LOSS_CUT_COOLDOWN_MIN", "60"))
    REPORT_TZ: str = os.getenv("REPORT_TZ", "UTC")

    # --- Indicator lengths/params ---
    EMA_FAST: int = int(os.getenv("EMA_FAST", "20"))
    EMA_SLOW: int = int(os.getenv("EMA_SLOW", "50"))
    EMA_TREND: int = int(os.getenv("EMA_TREND", "200"))
    RSI_LEN: int = int(os.getenv("RSI_LEN", "14"))
    ATR_LEN: int = int(os.getenv("ATR_LEN", "14"))
    ADX_LEN: int = int(os.getenv("ADX_LEN", "14"))
    BB_LEN: int = int(os.getenv("BB_LEN", "20"))
    BB_K: float = float(os.getenv("BB_K", "2.0"))
    DONCHIAN_LEN: int = int(os.getenv("DONCHIAN_LEN", "20"))
    CCI_LEN: int = int(os.getenv("CCI_LEN", "20"))
    KAMA_ER: int = int(os.getenv("KAMA_ER", "10"))
    KAMA_FAST: int = int(os.getenv("KAMA_FAST", "2"))
    KAMA_SLOW: int = int(os.getenv("KAMA_SLOW", "30"))
    ST_ATR_LEN: int = int(os.getenv("ST_ATR_LEN", "10"))
    ST_MULT: float = float(os.getenv("ST_MULT", "3.0"))
    VOL_MA_LEN: int = int(os.getenv("VOL_MA_LEN", "20"))
    OBV_SLOPE_LEN: int = int(os.getenv("OBV_SLOPE_LEN", "20"))
    VWAP_ROLL_N: int = int(os.getenv("VWAP_ROLL_N", "500"))
    VWAP_DAILY_ANCHOR_HOUR: int = int(os.getenv("VWAP_DAILY_ANCHOR_HOUR", "0"))  # UTC 기준 0시 앵커
    ICHI_TENKAN: int = int(os.getenv("ICHI_TENKAN", "9"))
    ICHI_KIJUN: int = int(os.getenv("ICHI_KIJUN", "26"))
    ICHI_SENKOUB: int = int(os.getenv("ICHI_SENKOUB", "52"))
    REGIME_ADX_MIN: int = int(os.getenv("REGIME_ADX_MIN", "20"))

    # --- Score weights (sum ~1.0, regime별 내부에서 배분) ---
    SCORE_W_TREND: float = float(os.getenv("SCORE_W_TREND", "0.40"))
    SCORE_W_MOM: float = float(os.getenv("SCORE_W_MOM", "0.25"))
    SCORE_W_MR: float = float(os.getenv("SCORE_W_MR", "0.25"))
    SCORE_W_BRK: float = float(os.getenv("SCORE_W_BRK", "0.10"))
    SCORE_W_VOL: float = float(os.getenv("SCORE_W_VOL", "0.05"))

    # --- Thresholds ---
    ENTRY_SCORE: int = int(os.getenv("ENTRY_SCORE", "70"))
    EXIT_SCORE: int = int(os.getenv("EXIT_SCORE", "50"))
    OPPOSITE_MAX: int = int(os.getenv("OPPOSITE_MAX", "40"))
    COOLDOWN_S: int = int(os.getenv("COOLDOWN_S", "15"))

    # --- MTF (multi-timeframe) ---
    MTF_USE: bool = os.getenv("MTF_USE", "true").lower() == "true"
    MTF_HIGHERS: list[str] = os.getenv("MTF_HIGHERS", "5m,15m").split(",")
    MTF_CONFLUENCE_BOOST: float = float(os.getenv("MTF_CONFLUENCE_BOOST", "1.08"))
    MTF_CONTRA_DAMP: float = float(os.getenv("MTF_CONTRA_DAMP", "0.88"))


    # --- Discord (KR only) ---
    DISCORD_TOKEN: str | None = None
    DISCORD_GUILD_ID: int | None = None
    DISCORD_CHANNEL_SIGNALS: int | None = None
    DISCORD_CHANNEL_TRADES: int | None = None
    DISCORD_CHANNEL_LOGS: int | None = None
    DISCORD_PREFIX: str = "!"
    DISCORD_TEST_ON_BOOT: bool = True
    DISCORD_UPDATE_INTERVAL_S: int = 5
    PNL_CHANGE_BPS: int = int(os.getenv("PNL_CHANGE_BPS", "5"))
    EMBED_DECIMALS_PRICE: int = int(os.getenv("EMBED_DECIMALS_PRICE", "2"))
    EMBED_DECIMALS_QTY: int = int(os.getenv("EMBED_DECIMALS_QTY", "6"))
    EMBED_DECIMALS_USDT: int = int(os.getenv("EMBED_DECIMALS_USDT", "2"))
    EMBED_SHOW_FUNDING: bool = os.getenv("EMBED_SHOW_FUNDING", "false").lower() == "true"
    EMBED_SHOW_LIQ: bool = os.getenv("EMBED_SHOW_LIQ", "true").lower() == "true"
    EMBED_SHOW_EQUITY: bool = os.getenv("EMBED_SHOW_EQUITY", "true").lower() == "true"

    # --- User stream / REST sync ---
    LISTENKEY_KEEPALIVE_SEC: int = int(os.getenv("LISTENKEY_KEEPALIVE_SEC", "1800"))
    REST_RESYNC_SEC: int = int(os.getenv("REST_RESYNC_SEC", "45"))
    WALLET_REFRESH_SEC: int = int(os.getenv("WALLET_REFRESH_SEC", "20"))
    INCOME_POLL_SEC: int = int(os.getenv("INCOME_POLL_SEC", "90"))


    # --- 진입/라우팅 ---
    ENTRY_ORDER: str = os.getenv("ENTRY_ORDER", "market")
    LIMIT_OFFSET_TICKS: int = int(os.getenv("LIMIT_OFFSET_TICKS", "2"))
    BRACKET_MODE: str = os.getenv("BRACKET_MODE", "reduce")
    TP_ORDER: str = os.getenv("TP_ORDER", "limit")
    SL_ORDER: str = os.getenv("SL_ORDER", "market")
    WORKING_TYPE: str = os.getenv("WORKING_TYPE", "MARK_PRICE")
    POST_ONLY: bool = os.getenv("POST_ONLY", "false").lower() == "true"
    TIME_IN_FORCE: str = os.getenv("TIME_IN_FORCE", "GTC")
    MAX_SLIPPAGE_BPS: int = int(os.getenv("MAX_SLIPPAGE_BPS", "10"))
    RETRY_MAX: int = int(os.getenv("RETRY_MAX", "3"))
    BACKOFF_429_MS: int = int(os.getenv("BACKOFF_429_MS", "800"))
    BACKOFF_NET_MS: int = int(os.getenv("BACKOFF_NET_MS", "500"))
    KILL_SWITCH_ON: bool = os.getenv("KILL_SWITCH_ON", "true").lower() == "true"

    # --- 점수 기반 "비율적" 포지션/추가진입 ---
    TIER_BINS: list[str] = os.getenv("TIER_BINS", "60,70,80,90").split(",")
    TIER_WEIGHTS: list[str] = os.getenv("TIER_WEIGHTS", "0.5,1.0,1.5,2.0").split(",")
    SCALE_IN_MAX_ADDS: int = int(os.getenv("SCALE_IN_MAX_ADDS", "2"))
    SCALE_IN_STEP_ATR: float = float(os.getenv("SCALE_IN_STEP_ATR", "0.8"))
    PULLBACK_ADD_ATR: float = float(os.getenv("PULLBACK_ADD_ATR", "0.7"))
    DCA_USE_PULLBACK: bool = os.getenv("DCA_USE_PULLBACK", "true").lower() == "true"
    TRAIL_START_R: float = float(os.getenv("TRAIL_START_R", "1.5"))
    TRAIL_STEP_R: float = float(os.getenv("TRAIL_STEP_R", "0.5"))
    TRAIL_BACK_R: float = float(os.getenv("TRAIL_BACK_R", "0.8"))
    CSV_ENABLE: bool = os.getenv("CSV_ENABLE", "true").lower() == "true"
    CSV_DIR: str = os.getenv("CSV_DIR", "storage/csv")
    CSV_FILE: str = os.getenv("CSV_FILE", "trades_all.csv")
    CSV_FLUSH_SEC: int = int(os.getenv("CSV_FLUSH_SEC", "2"))
    CSV_MARK_SNAPSHOT_SEC: int = int(os.getenv("CSV_MARK_SNAPSHOT_SEC", "0"))
    
def load_env_chain() -> Settings:
    root = Path(__file__).resolve().parents[2]
    profile = os.getenv("ENV_PROFILE", None)
    _load_env_series(root, profile)
    _load_env_series(Path.cwd(), profile)
    s = Settings()
    for k in ("BINANCE_API_KEY","BINANCE_API_SECRET","DISCORD_TOKEN"):
        v = os.getenv(k)
        if v is not None and v != "":
            setattr(s, k, v)

    for k in ("DISCORD_GUILD_ID","DISCORD_CHANNEL_SIGNALS","DISCORD_CHANNEL_TRADES","DISCORD_CHANNEL_LOGS"):
        v = os.getenv(k)
        if v and v.strip().isdigit():
            setattr(s, k, int(v.strip()))
    s.DISCORD_PREFIX = os.getenv("DISCORD_PREFIX", s.DISCORD_PREFIX)
    s.DISCORD_TEST_ON_BOOT = os.getenv("DISCORD_TEST_ON_BOOT", "true").lower() == "true"
    try:
        s.DISCORD_UPDATE_INTERVAL_S = int(os.getenv("DISCORD_UPDATE_INTERVAL_S", str(s.DISCORD_UPDATE_INTERVAL_S)))
    except:
        pass
    ak = s.BINANCE_API_KEY or ""
    dt = s.DISCORD_TOKEN or ""
    print(f"[ENV][CHK] BINANCE_API_KEY={(ak[:4]+'…') if ak else 'EMPTY'}  DISCORD_TOKEN={(dt[:6]+'…') if dt else 'EMPTY'}")
    print(f"[ENV][DISCORD] guild={s.DISCORD_GUILD_ID}  logs={s.DISCORD_CHANNEL_LOGS}  trades={s.DISCORD_CHANNEL_TRADES}  signals={s.DISCORD_CHANNEL_SIGNALS}")
    return s


from pydantic import BaseModel
import os


class Settings(BaseModel):
    MODE: str = os.getenv("MODE", "testnet")  # testnet | live
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    RECV_WINDOW_MS: int = int(os.getenv("RECV_WINDOW_MS", "5000"))
    HTTP_TIMEOUT_S: float = float(os.getenv("HTTP_TIMEOUT_S", "5"))

    SYMBOLS: list[str] = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
    INTERVAL: str = os.getenv("INTERVAL", "1m")
    LOOKBACK: int = int(os.getenv("LOOKBACK", "300"))

    USE_CORR: bool = os.getenv("USE_CORR", "true").lower() == "true"
    CORR_LOOKBACK: int = int(os.getenv("CORR_LOOKBACK", "240"))

    LEVERAGE: int = int(os.getenv("LEVERAGE", "5"))
    RISK_MODE: str = os.getenv("RISK_MODE", "atr_unit")
    RISK_TARGET_USDT: float = float(os.getenv("RISK_TARGET_USDT", "30"))
    ATR_LOOKBACK: int = int(os.getenv("ATR_LOOKBACK", "14"))
    ATR_MULT: float = float(os.getenv("ATR_MULT", "1.5"))
    SL_MULT: float = float(os.getenv("SL_MULT", "1.5"))
    TP_MULT: float = float(os.getenv("TP_MULT", "3.0"))
    MAX_POS_PER_SIDE: int = int(os.getenv("MAX_POS_PER_SIDE", "1"))

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

    # --- MTF (multi-timeframe) ---
    MTF_USE: bool = os.getenv("MTF_USE", "true").lower() == "true"
    MTF_HIGHERS: list[str] = os.getenv("MTF_HIGHERS", "5m,15m").split(",")
    MTF_CONFLUENCE_BOOST: float = float(os.getenv("MTF_CONFLUENCE_BOOST", "1.08"))
    MTF_CONTRA_DAMP: float = float(os.getenv("MTF_CONTRA_DAMP", "0.88"))

    DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
    DISCORD_CHANNEL_SIGNALS: int = int(os.getenv("DISCORD_CHANNEL_SIGNALS", "0"))
    DISCORD_CHANNEL_TRADES: int = int(os.getenv("DISCORD_CHANNEL_TRADES", "0"))
    DISCORD_CHANNEL_LOGS: int = int(os.getenv("DISCORD_CHANNEL_LOGS", "0"))
    DISCORD_UPDATE_INTERVAL_S: float = float(os.getenv("DISCORD_UPDATE_INTERVAL_S", "5"))


def load_env_chain() -> Settings:
    """Load settings from environment chain.

    token.env and .env are assumed to be loaded by runtime infra before
    this function executes.
    """
    return Settings()

from typing import Optional
from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv

class Settings(BaseModel):
    MODE: str = os.getenv("MODE", "testnet")  # legacy field
    TRADE_MODE: str = os.getenv("TRADE_MODE", MODE)  # testnet | live
    DATA_FEED: str = os.getenv("DATA_FEED", "live")  # live | testnet
    WORKING_PRICE: str = os.getenv("WORKING_PRICE", "MARK_PRICE")
    LIVE_GUARD_ENABLE: bool = os.getenv("LIVE_GUARD_ENABLE", "true").lower() == "true"
    LIVE_MIN_NOTIONAL_USDT: float = float(os.getenv("LIVE_MIN_NOTIONAL_USDT", "10"))
    LIVE_CONFIRM_MODE: str = os.getenv("LIVE_CONFIRM_MODE", "auto")
    CONFIRM_TIMEOUT_S: int = int(os.getenv("CONFIRM_TIMEOUT_S", "15"))
    ANALYZE_INTERVAL_S: int = int(os.getenv("ANALYZE_INTERVAL_S", "30"))
    ANALYSIS_TF: str = os.getenv("ANALYSIS_TF", "1m,5m,1h")
    TF_WEIGHTS: str = os.getenv("TF_WEIGHTS", "1m:0.7,5m:0.2,1h:0.1")
    MTF_ALIGN_MIN: int = int(os.getenv("MTF_ALIGN_MIN", "2"))
    ENTRY_TH: int = int(os.getenv("ENTRY_TH", "60"))
    EXIT_TH: int = int(os.getenv("EXIT_TH", "40"))
    COOLDOWN_S: int = int(os.getenv("COOLDOWN_S", "180"))
    MAX_DIVERGENCE_BPS: int = int(os.getenv("MAX_DIVERGENCE_BPS", "15"))
    # [ANCHOR:M6_SETTINGS_CHART]
    CHART_TFS: str = os.getenv("CHART_TFS", "15m,4h")
    CHART_PRICE_OVERLAYS: str = os.getenv("CHART_PRICE_OVERLAYS", "ema50,ema200,bb20_2")
    CHART_PANELS_15m: str = os.getenv("CHART_PANELS_15m", "price,RSI,ROTATE")
    CHART_PANELS_4h: str = os.getenv("CHART_PANELS_4h", "price,ADXDI,ROTATE")
    CHART_ROTATE_SET: str = os.getenv("CHART_ROTATE_SET", "CCI,OBV,KAMA")
    CHART_ATTACH_MAX: int = int(os.getenv("CHART_ATTACH_MAX", "4"))
    CHART_ZOOM_LAST_N: int = int(os.getenv("CHART_ZOOM_LAST_N", "250"))

    CHART_MODE: str = os.getenv("CHART_MODE", "overwrite")
    CHART_DIR: str = os.getenv("CHART_DIR", "storage/charts")
    CHART_KEEP_PER_SYMBOL: int = int(os.getenv("CHART_KEEP_PER_SYMBOL", "10"))
    CHART_MIN_INTERVAL_S: int = int(os.getenv("CHART_MIN_INTERVAL_S", "120"))
    CHART_MIN_SCORE_DELTA: float = float(os.getenv("CHART_MIN_SCORE_DELTA", "2.0"))
    CHART_MIN_DIVERGENCE_BPS: float = float(os.getenv("CHART_MIN_DIVERGENCE_BPS", "2.0"))
    CHART_FORCE_N_CYCLES: int = int(os.getenv("CHART_FORCE_N_CYCLES", "10"))
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
    DISCORD_CHANNEL_ANALYSIS_BTC: Optional[int] = None
    DISCORD_CHANNEL_ANALYSIS_ETH: Optional[int] = None
    DISCORD_PREFIX: str = "!"
    DISCORD_TEST_ON_BOOT: bool = True
    DISCORD_UPDATE_INTERVAL_S: int = 5
    TRADE_HEARTBEAT_S: int = int(os.getenv("TRADE_HEARTBEAT_S", "30"))
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
    """
    로드 순서 (나중에 로드한 값이 우선; override=True):
    token.env → .env → .env.strategy → .env.risk → .env.trade → .env.notify → .env.local
    """
    ROOT = Path(__file__).resolve().parents[2]

    candidates = [
        ROOT / "token.env",
        ROOT / ".env",
        ROOT / ".env.strategy",
        ROOT / ".env.risk",
        ROOT / ".env.trade",
        ROOT / ".env.notify",
        ROOT / ".env.local",
    ]

    loaded, missing = [], []
    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p, override=True)
            loaded.append(str(p))
        else:
            missing.append(str(p))

    print(f"[ENV] loaded={len(loaded)} files={loaded}")
    if missing:
        print(f"[ENV][MISS] {missing}")

    field_values = {}
    field_names = getattr(Settings, "model_fields", getattr(Settings, "__fields__", {}))
    for k in field_names:
        v = os.getenv(k)
        if v is not None:
            field_values[k] = v

    s = Settings(**field_values)

    print(
        f"[ENV][DUMP] ANALYSIS_TF={s.ANALYSIS_TF}  TF_WEIGHTS={s.TF_WEIGHTS}  "
        f"CHART_TFS={s.CHART_TFS}  CHART_MODE={s.CHART_MODE}"
    )

    for k in ["DISCORD_GUILD_ID", "DISCORD_CHANNEL_LOGS", "DISCORD_CHANNEL_TRADES",
              "DISCORD_CHANNEL_SIGNALS", "DISCORD_CHANNEL_ANALYSIS_BTC", "DISCORD_CHANNEL_ANALYSIS_ETH",
              "CHART_ATTACH_MAX", "CHART_KEEP_PER_SYMBOL", "CHART_MIN_INTERVAL_S",
              "CHART_FORCE_N_CYCLES", "MTF_ALIGN_MIN", "ENTRY_TH", "EXIT_TH", "COOLDOWN_S"]:
        v = os.getenv(k, None)
        if v is not None:
            try:
                setattr(s, k, int(v))
            except Exception:
                pass

    return s


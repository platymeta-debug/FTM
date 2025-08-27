from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv

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

    DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
    DISCORD_CHANNEL_SIGNALS: int = int(os.getenv("DISCORD_CHANNEL_SIGNALS", "0"))
    DISCORD_CHANNEL_TRADES: int = int(os.getenv("DISCORD_CHANNEL_TRADES", "0"))
    DISCORD_CHANNEL_LOGS: int = int(os.getenv("DISCORD_CHANNEL_LOGS", "0"))
    DISCORD_UPDATE_INTERVAL_S: float = float(os.getenv("DISCORD_UPDATE_INTERVAL_S", "5"))
    
def load_env_chain() -> Settings:
    # 리포 루트에서 token.env → .env 순서로 로드(.env가 덮어씀)
    root = Path(__file__).resolve().parents[2]  # .../FTM
    load_dotenv(root / "token.env", override=False)
    load_dotenv(root / ".env", override=True)
    # 실행 디렉터리에 같은 이름의 파일이 있으면 추가로 로드(옵션)
    load_dotenv("token.env", override=False)
    load_dotenv(".env", override=True)
    return Settings()

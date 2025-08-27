# run_ftm2.py (리포 루트= ...\FTM-main\FTM)
import sys
from pathlib import Path
import runpy
from dotenv import load_dotenv

if __name__ == "__main__":
    ROOT = Path(__file__).parent.resolve()
    # token.env/.env를 run 이전에 로드(override=True로 확실히 덮기)
    load_dotenv(ROOT / "token.env", override=True)
    load_dotenv(ROOT / ".env", override=True)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # 모듈 실행 (python -m ftm2.app 과 동일)
    runpy.run_module("ftm2.app", run_name="__main__")

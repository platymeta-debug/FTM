import sys
from pathlib import Path
import runpy


if __name__ == "__main__":
    # 루트 경로를 sys.path 최우선에 보장
    ROOT = Path(__file__).parent.resolve()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # 패키지 모듈을 __main__으로 실행 (python -m ftm2.app 과 동일)
    runpy.run_module("ftm2.app", run_name="__main__")


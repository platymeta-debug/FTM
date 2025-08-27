# [ANCHOR:M6_CHART_JANITOR]
import os, glob, asyncio


async def run_chart_janitor(cfg):
    # 10분마다 심볼별 디렉토리를 훑고, rotate 정책에 맞춰 추가 정리
    while True:
        if cfg.CHART_MODE == "rotate":
            for d in glob.glob(os.path.join(cfg.CHART_DIR, "*", "1m")):
                files = sorted(glob.glob(os.path.join(d, "*.png")))
                if len(files) > cfg.CHART_KEEP_PER_SYMBOL:
                    for f in files[:-cfg.CHART_KEEP_PER_SYMBOL]:
                        try:
                            os.remove(f)
                        except Exception:
                            pass
        await asyncio.sleep(600)


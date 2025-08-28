class Scoring:
    def __init__(self, cfg, snap):
        self.cfg = cfg
        self.snap = snap

    # [SCORE_STANDARDIZE_AND_MTF]
    def mtf_score(self, sym: str):
        # tf별 지표 스코어(0..1)와 ADX, 추세방향을 이미 갖고 있다고 가정
        # 예: self.snap[tf].score01, self.snap[tf].adx, self.snap[tf].trend in {"UP","DOWN","FLAT"}
        TFs = self.cfg.SCORING_TFS  # ["1m","15m","1h","4h"]
        weights = {}
        raw = {}
        for tf in TFs:
            s01 = self.snap[sym][tf].score01  # 0..1
            adx = max(5.0, min(40.0, getattr(self.snap[sym][tf], "adx", 5.0) or 5.0))
            w = adx / 40.0  # 0.125..1.0
            raw[tf] = s01
            weights[tf] = w

        # HTF(1h/4h)와 LTF(1m/15m) 방향 일치 보정
        htf_dir = self.snap[sym]["1h"].trend, self.snap[sym]["4h"].trend
        ltf_dir = self.snap[sym]["1m"].trend, self.snap[sym]["15m"].trend
        align_bonus = 0.0
        if all(d == "UP" for d in htf_dir) and any(d == "UP" for d in ltf_dir):
            align_bonus = +0.05
        elif all(d == "DOWN" for d in htf_dir) and any(d == "DOWN" for d in ltf_dir):
            align_bonus = -0.05

        # 가중 평균
        wsum = sum(weights.values()) or 1.0
        score01 = sum(raw[tf] * weights[tf] for tf in TFs) / wsum
        score01 = min(1.0, max(0.0, score01 + align_bonus))

        # –100..+100 매핑 (0.5=중립)
        score = int(round((score01 - 0.5) * 200))
        return score


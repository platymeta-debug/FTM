class Scoring:
    def __init__(self, cfg, snap):
        self.cfg = cfg
        self.snap = snap

    # [SCORE_STANDARDIZE_AND_MTF]
    def mtf_score(self, sym: str):
        # tf별 지표 스코어(0..1)와 ADX, 추세방향을 이미 갖고 있다고 가정
        # 예: self.snap[tf].score01, self.snap[tf].adx, self.snap[tf].trend in {"UP","DOWN","FLAT"}
        TFs = self.cfg.SCORING_TFS  # ["1m","15m","1h","4h"]
        weights: dict[str, float] = {}
        raw: dict[str, float] = {}
        for tf in TFs:
            s01 = self.snap[sym][tf].score01  # 0..1
            adx = max(5.0, min(40.0, getattr(self.snap[sym][tf], "adx", 5.0) or 5.0))
            w = (adx / 40.0) ** 1.25  # [ANCHOR:ADX_WEIGHT]
            raw[tf] = s01
            weights[tf] = w

        # [ANCHOR:REGIME_INJECT]
        from ftm2.analysis.regime import atr_percentile, classify_regime
        atr_pct = atr_percentile(self.snap[sym][self.cfg.ENTRY_TF].atr_hist, self.cfg.REGM_ATR_LOOKBACK)
        regime = classify_regime(atr_pct)
        reg_adj = {"LOW": 0.85, "NORMAL": 1.0, "HIGH": 0.9, "EXTREME": 0.75}[regime]

        # [ANCHOR:HTF_CONSENSUS]
        htf_ok_up = all(self.snap[sym][tf].trend == "UP" for tf in ("1h", "4h"))
        htf_ok_down = all(self.snap[sym][tf].trend == "DOWN" for tf in ("1h", "4h"))
        ltf_bias_up = any(self.snap[sym][tf].trend == "UP" for tf in ("1m", "15m"))
        ltf_bias_down = any(self.snap[sym][tf].trend == "DOWN" for tf in ("1m", "15m"))

        align_bonus = 0.0
        if htf_ok_up and ltf_bias_up:
            align_bonus = +0.06
        elif htf_ok_down and ltf_bias_down:
            align_bonus = -0.06

        # [ANCHOR:REGIME_WEIGHT_APPLY]
        wsum = sum(weights.values()) or 1.0
        score01 = min(1.0, max(0.0, (sum(raw[tf] * weights[tf] for tf in TFs) / wsum)))
        score01 = 0.5 + (score01 - 0.5) * reg_adj
        score01 = min(1.0, max(0.0, score01 + align_bonus))

        score = int(round((score01 - 0.5) * 200))

        # [ANCHOR:CONFIDENCE]
        adx_avg = sum(getattr(self.snap[sym][tf], "adx", 0) or 0 for tf in TFs) / len(TFs)
        cons = 1.0 if (htf_ok_up or htf_ok_down) else 0.5
        reg = {"LOW": 1.0, "NORMAL": 0.9, "HIGH": 0.8, "EXTREME": 0.7}[regime]
        confidence = max(0.0, min(1.0, 0.4 * (adx_avg / 40.0) + 0.4 * cons + 0.2 * reg))
        # [ANCHOR:SCORE_STATE_CACHE]
        self.last_conf = confidence
        self.last_regime = regime
        return score, confidence, regime


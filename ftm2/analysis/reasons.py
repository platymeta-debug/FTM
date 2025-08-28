# [ANCHOR:WHY_REASONS]

def top_reasons(snap, score, conf, regime):
    rs = []
    # 예시 규칙 (필요 시 맞춤)
    if getattr(snap, "adx_avg", 0) >= 20:
        rs.append(f"ADX평균 {getattr(snap, 'adx_avg'):.1f}↑(추세품질)")
    if getattr(snap, "htf_consensus", "") in ("UP", "DOWN"):
        rs.append(f"HTF 컨센서스 {snap.htf_consensus}")
    if getattr(snap, "rr", 0) >= 1.2:
        rs.append(f"RR≈{snap.rr:.2f} (목표/손절)")
    if getattr(snap, "div_filter_pass", True) == False:
        rs.append("다이버전스 필터 통과 실패")  # 발급 시엔 제외
    rs.append(f"Regime={regime}, Confidence={conf:.2f}, Score={score:.0f}")
    return rs[:3]

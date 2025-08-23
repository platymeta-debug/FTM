# generate_pdf_report.py (최종 개선 버전)
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

# ===== 한글 폰트 등록 =====
def _register_korean_font():
    try:
        if os.name == "nt":  # Windows
            malgun = r"C:\Windows\Fonts\malgun.ttf"
            if os.path.exists(malgun):
                pdfmetrics.registerFont(TTFont("KOR", malgun))
                return "KOR"
        for p in (
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/Library/Fonts/NanumGothic.ttf",
            os.path.expanduser("~/Library/Fonts/NanumGothic.ttf"),
        ):
            if os.path.exists(p):
                pdfmetrics.registerFont(TTFont("KOR", p))
                return "KOR"
    except Exception:
        pass
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
        return "HYSMyeongJo-Medium"
    except Exception:
        return "Helvetica"

FONT_NAME = _register_korean_font()

# ===== 스타일 헬퍼 =====
def _para(text, size=11, bold=False, color=colors.black, space_after=4):
    style = ParagraphStyle(
        name="KStyle",
        parent=getSampleStyleSheet()["Normal"],
        fontName=FONT_NAME,
        fontSize=size,
        textColor=color,
        spaceAfter=space_after,
        leading=size + 2,
    )
    if bold:
        text = f"<b>{text}</b>"
    return Paragraph(text, style)

def _safe_num(x, fmt="{:.2f}"):
    try:
        return fmt.format(float(x))
    except Exception:
        return str(x) if pd.notna(x) else "-"

# 상단 유틸
def _symtag(symbol: str) -> str:
    return symbol.replace('/', '').lower()

# ===== 로그 데이터 로드 =====
def _load_logs(tf, symbol=None):
    import glob
    if symbol:
        symtag = _symtag(symbol)
        fp = f"logs/signals_{symtag}_{tf}.csv"
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            return df[df['timeframe'] == tf] if 'timeframe' in df.columns else df
    # 폴백: 예전 통합 로그
    if os.path.exists("logs/signals.csv"):
        df = pd.read_csv("logs/signals.csv")
        return df[df['timeframe'] == tf]
    return pd.DataFrame()

# ===== 최근 신호 테이블 =====
def _recent_signals_table(df, n=5):
    if df.empty:
        return [["데이터 없음"]]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime', ascending=False).head(n)
    rows = [["일시", "신호", "진입가", "수익률", "점수"]]
    for _, row in df.iterrows():
        rows.append([
            row['datetime'].strftime("%m-%d %H:%M"),
            row['signal'],
            _safe_num(row.get('entry_price')),
            f"{_safe_num(row.get('pnl'))}%" if pd.notna(row.get('pnl')) else "-",
            _safe_num(row.get('score'), "{:.1f}")
        ])
    return rows

# ===== 성과 요약 =====
def _performance_summary(df):
    if df.empty or 'pnl' not in df:
        return None

    # pnl을 숫자로 변환 (% 기호 제거)
    df['pnl'] = pd.to_numeric(df['pnl'].astype(str).str.replace('%', ''), errors='coerce')
    df = df.dropna(subset=['pnl'])
    if len(df) == 0:
        return None

    total_trades = len(df)
    cumulative_return = df['pnl'].sum()

    # ⚠️ division by zero 방지
    win_rate = (df['pnl'] > 0).sum() / total_trades * 100 if total_trades > 0 else 0.0
    avg_return = df['pnl'].mean() if total_trades > 0 else 0.0
    max_gain = df['pnl'].max() if total_trades > 0 else 0.0
    max_loss = df['pnl'].min() if total_trades > 0 else 0.0

    return [
        ["누적 수익률", f"{cumulative_return:.2f}%"],
        ["승률", f"{win_rate:.2f}%"],
        ["평균 수익률", f"{avg_return:.2f}%"],
        ["최대 수익률", f"{max_gain:.2f}%"],
        ["최대 손실률", f"{max_loss:.2f}%"]
    ]


# ===== 지표별 성과 =====
def _reason_performance(df, top_n=5):
    if df.empty or 'reasons' not in df:
        return [["데이터 없음", "", "", ""]]

    reason_stats = defaultdict(list)

    # 이유별로 pnl 수집
    for _, row in df.iterrows():
        for reason in str(row['reasons']).replace('"', '').split(" | "):
            if reason:
                try:
                    pnl = float(str(row['pnl']).replace('%', ''))
                    reason_stats[reason].append(pnl)
                except ValueError:
                    pass

    rows_data = []
    for reason, pnls in reason_stats.items():
        if pnls:
            count = len(pnls)
            # ⚠️ division by zero 방지
            win_rate = sum(1 for p in pnls if p > 0) / count * 100 if count > 0 else 0.0
            avg_return = sum(pnls) / count if count > 0 else 0.0
            rows_data.append([reason, avg_return, win_rate, count])

    if not rows_data:
        return [["데이터 없음", "", "", ""]]

    # 평균 수익률 기준 내림차순 정렬
    rows_data.sort(key=lambda x: x[1], reverse=True)

    rows = [["지표", "평균 수익률", "승률", "횟수"]]
    for reason, avg_return, win_rate, count in rows_data[:top_n]:
        rows.append([reason, f"{avg_return:.2f}%", f"{win_rate:.1f}%", str(count)])

    return rows


# 구조 스냅샷 표 생성
def _struct_snapshot_table(struct_info: dict|None):
    """
    struct_info 예시: {
        "levels":[("ATH",4870.4),("PH",4820.2),...],
        "nearest":{"res":("PH",4820.2,0.45),"sup":("PL",4380.5,0.72)},
        "atr": 145.1
    }
    """
    if not struct_info:
        return [["데이터 없음"]]
    rows = [["유형","값","거리(ATR)","메모"]]
    price = None
    try:  # 가격을 reasons 문자열에서 추출하거나 생략
        pass
    except Exception:
        pass
    near = struct_info.get("nearest") or {}
    res, sup = near.get("res"), near.get("sup")
    for t, v in struct_info.get("levels", [])[:10]:
        note = []
        if res and v == res[1]: note.append("상단 최근접")
        if sup and v == sup[1]: note.append("하단 최근접")
        rows.append([str(t), f"{float(v):.2f}", "-", ", ".join(note)])
    if res:
        rows.append(["최근접저항", f"{float(res[1]):.2f}", f"{float(res[2]):.2f}", ""])
    if sup:
        rows.append(["최근접지지", f"{float(sup[1]):.2f}", f"{float(sup[2]):.2f}", ""])
    return rows

def _struct_legend_pdf(enable_env: str = "STRUCT_LEGEND_ENABLE") -> list:
    """구조 해석 가이드(표준 문구). ENV로 on/off 가능."""
    import os
    if os.getenv(enable_env, "1") != "1":
        return []
    lines = [
        "🔎 구조 해석 가이드",
        "• 수평레벨: 가격↔레벨 거리(ATR배수)가 작을수록 반대포지션 위험↑",
        "• 추세선: 하락선 위 종가마감=돌파, 상승선 아래 종가마감=이탈",
        "• 회귀채널: 상단=롱 익절/숏 관심, 하단=숏 익절/분할매수 관심",
        "• 피보채널: 0.382/0.618/1.0 접촉 시 반응·돌파 체크",
        "• 컨플루언스: 다중 레벨이 ATR×ε 이내로 겹치면 신뢰도↑",
    ]
    p = _para("\n".join(lines), size=9, leading=12)
    return [Spacer(1, 0.2*cm), p, Spacer(1, 0.4*cm)]

# ===== PDF 생성 =====
def generate_pdf_report(
    df, tf, signal, price, score, reasons, weights,
    agree_long, agree_short, now=None,
    output_path=None, chart_imgs=None, chart_img=None, ichimoku_img=None,
    daily_change_pct=None, discord_message=None,
    symbol=None,
    entry_price=None, entry_time=None,
    struct_info=None, struct_img=None
):

    logs_df = _load_logs(tf, symbol=symbol)

    # 경로 설정
    ts = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M")
    symtag = _symtag(symbol) if symbol else "ethusdt"
    if output_path is None:
        base_dir = os.path.join("reports", symtag, tf)
        os.makedirs(base_dir, exist_ok=True)
        output_path = os.path.join(base_dir, f"report_{symtag}_{tf}_{ts}.pdf")

    if chart_imgs is None:
        # 과거 호환: 단일 chart_img만 주던 경우
        if chart_img is None:
            chart_img = os.path.join("images", f"chart_{symtag}_{tf}.png")
        chart_imgs = [chart_img]

    if ichimoku_img is None:
        ichimoku_img = os.path.join("images", f"ichimoku_{symtag}_{tf}.png")


    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=1.4*cm, rightMargin=1.4*cm,
                            topMargin=1.4*cm, bottomMargin=1.4*cm)
    elements = []

    # 제목
    tf_kor = {"15m":"15분봉","1h":"1시간봉","4h":"4시간봉","1d":"일봉"}.get(tf, tf)
    title = f"■ {symbol or 'ETH/USDT'} {tf_kor} 리포트 - {ts.replace('_',' ')}"
    elements.append(_para(title, size=16, bold=True, space_after=10))
    if entry_price is not None and entry_time:
        elements.append(_para(f"진입 시점: {entry_time} ({_safe_num(entry_price)})"))

    # ✅ 디스코드 메세지 형태 추가
    if discord_message:
        elements.append(_para(discord_message.replace("\n", "<br/>"), size=9))
        elements.append(Spacer(1, 0.5*cm))
    
    # ◼ 구조 스냅샷(있을 때만)
    if struct_info or struct_img:
        elements.append(_para("◼ 구조 스냅샷", size=13, bold=True))
        if struct_info:
            st = Table(_struct_snapshot_table(struct_info))
            st.setStyle(TableStyle([
                ("FONTNAME",(0,0),(-1,-1),FONT_NAME),
                ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
                ("BOX",(0,0),(-1,-1),0.6,colors.black),
                ("INNERGRID",(0,0),(-1,-1),0.4,colors.black),
            ]))
            elements += [st, Spacer(1, 0.5*cm)]
        if struct_img and os.path.exists(struct_img):
            elements += [Image(struct_img, width=16*cm, height=8*cm), Spacer(1, 0.4*cm)]

        # 구조 해석 가이드(표준 문구)
        elements += _struct_legend_pdf("STRUCT_PDF_SHOW_LEGEND")

    # 최근 신호
    elements.append(_para("◼ 최근 신호 이력", size=13, bold=True))
    recent_tbl = Table(_recent_signals_table(logs_df))
    recent_tbl.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),FONT_NAME),
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
        ("BOX",(0,0),(-1,-1),0.6,colors.black),
        ("INNERGRID",(0,0),(-1,-1),0.4,colors.black),
    ]))
    elements += [recent_tbl, Spacer(1,0.5*cm)]

    # 성과 요약
    perf_summary = _performance_summary(logs_df)
    if perf_summary:
        elements.append(_para("◼ 성과 요약", size=13, bold=True))
        perf_tbl = Table([["지표","값"], *perf_summary])
        perf_tbl.setStyle(TableStyle([
            ("FONTNAME",(0,0),(-1,-1),FONT_NAME),
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("BOX",(0,0),(-1,-1),0.6,colors.black),
            ("INNERGRID",(0,0),(-1,-1),0.4,colors.black),
        ]))
        elements += [perf_tbl, Spacer(1,0.5*cm)]

    # 지표별 성과 분석
    elements.append(_para("◼ 지표별 성과 분석", size=13, bold=True))
    reason_tbl = Table(_reason_performance(logs_df))
    reason_tbl.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),FONT_NAME),
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
        ("BOX",(0,0),(-1,-1),0.6,colors.black),
        ("INNERGRID",(0,0),(-1,-1),0.4,colors.black),
    ]))
    elements += [reason_tbl, Spacer(1,0.5*cm)]

    
    elements.append(PageBreak()) # 페이지 나눔

    # 차트 이미지 삽입 수정
    from reportlab.platypus import Image as RLImage, KeepInFrame
    MAX_W = 18*cm   # A4 여백 고려 최대 폭
    MAX_H = 18*cm   # 충분히 크게

    def _fit_image(path, max_w=MAX_W, max_h=MAX_H):
        img = RLImage(path)
        try:
            img._restrictSize(max_w, max_h)   # 비율 유지하며 축소
            img.hAlign = 'CENTER'
            return img
        except Exception:
            return KeepInFrame(max_w, max_h, [RLImage(path)], mode='shrink')

    img_list = []
    # 분할차트들
    if chart_imgs:
        img_list.extend([p for p in chart_imgs if p])
    # 이치모쿠 추가(있으면)
    if ichimoku_img:
        img_list.append(ichimoku_img)

    for img_path in img_list:
        if os.path.exists(img_path):
            elements += [_fit_image(img_path), Spacer(1, 0.4*cm)]


    # 점수 히스토리(있으면)
    score_img_path = f"logs/score_history_{symtag}_{tf}.png"
    if os.path.exists(score_img_path):
        elements += [Image(score_img_path, width=18*cm, height=6*cm), Spacer(1, 0.3*cm)]

    # 점수 히스토리
    score_img_path = f"logs/score_history_{symtag}_{tf}.png"
    if os.path.exists(score_img_path):
        elements += [Image(score_img_path, width=16*cm, height=5*cm), Spacer(1,0.3*cm)]

    # 꼬리말
    elements.append(_para("※ 이 리포트는 자동화된 참고용 분석이며, 실제 매매 판단은 투자자 본인의 책임입니다.", size=9, color=colors.grey))

    doc.build(elements)
    return output_path

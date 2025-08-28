import time

def on_realized(rt, amount):
    # 일자 바뀌면 리셋
    today = time.strftime("%Y-%m-%d")
    if getattr(rt, "_pnl_day", None) != today:
        rt._pnl_day = today
        rt.daily_realized = 0.0
        rt.loss_streak = 0
    rt.daily_realized += amount
    if amount < 0:
        rt.loss_streak += 1
    else:
        rt.loss_streak = 0

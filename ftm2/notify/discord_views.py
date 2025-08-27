# [ANCHOR:M5_DISCORD_VIEWS]
import time, discord
from ftm2.notify.discord_bot import send_log

def fmt(v, n): 
    try: return f"{v:.{n}f}"
    except: return str(v)

def build_trade_embed(cfg, symbol: str, ps, acct) -> discord.Embed:
    # 색상: 이익/손실/중립
    color = 0x2ecc71 if ps and ps.upnl>=0.0 else 0xe74c3c if ps else 0x95a5a6
    title = f"{symbol} — {'🟢 LONG' if (ps and ps.side=='LONG') else '🔴 SHORT' if ps else 'CLOSED'}"
    if ps and ps.qty:
        title += f" × {fmt(ps.qty, cfg.EMBED_DECIMALS_QTY)} (격리x{int(ps.leverage)})"
    emb = discord.Embed(title=title, color=color)
    now = time.strftime("%H:%M:%S", time.localtime())

    if ps:
        emb.add_field(name="진입가 / 마크가", value=f"{fmt(ps.entry_price,cfg.EMBED_DECIMALS_PRICE)} / {fmt(ps.mark_price,cfg.EMBED_DECIMALS_PRICE)}", inline=True)
        emb.add_field(name="UPNL / ROE", value=f"{fmt(ps.upnl,cfg.EMBED_DECIMALS_USDT)} USDT / {fmt(ps.roe,2)}%", inline=True)
        emb.add_field(name="실현손익 / 수수료", value=f"{fmt(ps.realized_pnl,cfg.EMBED_DECIMALS_USDT)} / {fmt(ps.fee_paid,cfg.EMBED_DECIMALS_USDT)}", inline=True)
        emb.add_field(name="SL / TP", value=f"{fmt(ps.sl_price,cfg.EMBED_DECIMALS_PRICE)} / {fmt(ps.tp_price,cfg.EMBED_DECIMALS_PRICE)}", inline=True)
        if cfg.EMBED_SHOW_LIQ and ps.liq_price:
            emb.add_field(name="청산가", value=f"{fmt(ps.liq_price,cfg.EMBED_DECIMALS_PRICE)}", inline=True)
    else:
        emb.add_field(name="상태", value="✅ 포지션 없음(CLOSED)", inline=False)

    if cfg.EMBED_SHOW_EQUITY:
        emb.add_field(name="총자본(지갑+UPNL)", value=f"{fmt(acct.equity,cfg.EMBED_DECIMALS_USDT)}", inline=True)
        emb.add_field(name="지갑/가용", value=f"{fmt(acct.wallet_balance,cfg.EMBED_DECIMALS_USDT)} / {fmt(acct.available_balance,cfg.EMBED_DECIMALS_USDT)}", inline=True)
    emb.set_footer(text=f"마지막 갱신 {now}")
    return emb

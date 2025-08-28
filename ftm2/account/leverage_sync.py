# [ANCHOR:LEV_SYNC_ENFORCE]
async def enforce_leverage_and_margin(client, cfg, symbols):
    for sym in symbols:
        mode = cfg.MARGIN_MODE_OVERRIDE.get(sym, cfg.MARGIN_MODE_DEFAULT).upper()
        lev = int(cfg.LEVERAGE_OVERRIDE.get(sym, cfg.LEVERAGE_DEFAULT))
        try:
            await client.set_margin_mode(sym, mode)
        except Exception:
            pass
        try:
            await client.set_leverage(sym, lev)
        except Exception:
            pass
        if hasattr(client, "notify"):
            client.notify.emit("system", f"⚙️ {sym} margin={mode} lev={lev} 적용")

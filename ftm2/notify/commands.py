from typing import Any

ALLOWED_ROLES = set()


def init_commands(cfg, dc_bot):
    global ALLOWED_ROLES
    ALLOWED_ROLES = {r.strip() for r in cfg.DISCORD_ALLOWED_ROLES.split(",") if r.strip()}

    @dc_bot.slash_command(name="auto", description="오토트레이드 on/off/status")
    async def auto(ctx, action: str):
        if not _allowed(ctx):
            return await ctx.respond("권한 없음", ephemeral=True)
        action = action.lower()
        if action == "on":
            cfg.AUTOTRADE_SWITCH.set(True)
            await ctx.respond("✅ AutoTrade ON", ephemeral=True)
        elif action == "off":
            cfg.AUTOTRADE_SWITCH.set(False)
            await ctx.respond("🛑 AutoTrade OFF", ephemeral=True)
        else:
            await ctx.respond(
                f"AutoTrade: {'ON' if cfg.AUTOTRADE_SWITCH.get() else 'OFF'}",
                ephemeral=True,
            )

    @dc_bot.slash_command(name="lev", description="심볼별 레버리지 설정")
    async def lev(ctx, symbol: str, leverage: int):
        if not _allowed(ctx):
            return await ctx.respond("권한 없음", ephemeral=True)
        symbol = symbol.upper()
        cfg.LEVERAGE_OVERRIDE[symbol] = leverage
        await ctx.respond(
            f"⚙️ {symbol} 레버리지={leverage} 적용 요청", ephemeral=True
        )
        try:
            await ctx.bot.binance.set_leverage(symbol, leverage)
            await ctx.respond(f"✅ {symbol} 레버리지 적용 완료", ephemeral=True)
        except Exception as e:  # pragma: no cover - network call
            await ctx.respond(
                f"⚠️ 실패: {type(e).__name__}: {e}", ephemeral=True
            )

    @dc_bot.slash_command(name="mode", description="심볼 격리/교차 설정")
    async def mode(ctx, symbol: str, margin: str):
        if not _allowed(ctx):
            return await ctx.respond("권한 없음", ephemeral=True)
        symbol = symbol.upper()
        margin = margin.upper()
        cfg.MARGIN_MODE_OVERRIDE[symbol] = margin
        try:
            await ctx.bot.binance.set_margin_mode(symbol, margin)
            await ctx.respond(
                f"✅ {symbol} margin={margin} 적용", ephemeral=True
            )
        except Exception as e:  # pragma: no cover
            await ctx.respond(f"⚠️ 실패: {e}", ephemeral=True)

    @dc_bot.slash_command(name="risk", description="심볼별 리스크% 설정(계정 대비)")
    async def risk(ctx, symbol: str, pct: float):
        if not _allowed(ctx):
            return await ctx.respond("권한 없음", ephemeral=True)
        symbol = symbol.upper()
        cfg.RISK_PCT_OVERRIDE[symbol] = pct
        await ctx.respond(
            f"✅ {symbol} risk%={pct} 저장", ephemeral=True
        )

    @dc_bot.slash_command(name="close", description="포지션 청산(심볼/all)")
    async def close(ctx, symbol: str):
        if not _allowed(ctx):
            return await ctx.respond("권한 없음", ephemeral=True)
        sym_list = [symbol.upper()] if symbol.lower() != "all" else list(cfg.SYMBOLS)
        for sym in sym_list:
            try:
                await ctx.bot.binance.new_order(
                    symbol=sym,
                    side=("SELL" if ctx.bot.rt.is_long(sym) else "BUY"),
                    type="MARKET",
                    reduceOnly=True,
                )
            except Exception as e:  # pragma: no cover
                await ctx.respond(f"{sym} 실패: {e}", ephemeral=True)
        await ctx.respond("💥 청산 요청 완료", ephemeral=True)

    @dc_bot.slash_command(name="cancel", description="열린 주문 취소(심볼/all)")
    async def cancel(ctx, symbol: str):
        if not _allowed(ctx):
            return await ctx.respond("권한 없음", ephemeral=True)
        sym_list = [symbol.upper()] if symbol.lower() != "all" else list(cfg.SYMBOLS)
        for sym in sym_list:
            await ctx.bot.binance.cancel_all_open_orders(sym)
        await ctx.respond("🧹 오더 취소 완료", ephemeral=True)

    @dc_bot.slash_command(name="dump", description="상태 스냅샷 저장")
    async def dump(ctx):
        if not _allowed(ctx):
            return await ctx.respond("권한 없음", ephemeral=True)
        from ftm2.recover.snapshot import dump_state

        p = dump_state(ctx.bot.rt, ctx.bot.cfg)
        await ctx.respond(f"🧩 snapshot: {p}", ephemeral=True)

    @dc_bot.slash_command(
        name="panic", description="전량 청산 + 오토트레이드 OFF"
    )
    async def panic(ctx):
        if not _allowed(ctx):
            return await ctx.respond("권한 없음", ephemeral=True)
        cfg.AUTOTRADE_SWITCH.set(False)
        for sym, pos in ctx.bot.rt.positions.items():
            q = abs(pos.qty)
            if q > 0:
                await ctx.bot.binance.new_order(
                    symbol=sym,
                    side=("SELL" if pos.qty > 0 else "BUY"),
                    type="MARKET",
                    quantity=str(q),
                    reduceOnly=True,
                )
        await ctx.respond(
            "⛔ PANIC: 전량 청산 + AutoTrade OFF", ephemeral=False
        )

    @dc_bot.slash_command(
        name="stop", description="안전 종료(오토트레이드 OFF, WS 정리)"
    )
    async def stop(ctx):
        if not _allowed(ctx):
            return await ctx.respond("권한 없음", ephemeral=True)
        cfg.AUTOTRADE_SWITCH.set(False)
        await ctx.bot.ws_close_all()
        await ctx.respond("🛑 안전 종료 요청 완료", ephemeral=False)


def _allowed(ctx: Any) -> bool:
    if not ALLOWED_ROLES:
        return True
    roles = {r.name for r in getattr(ctx.author, "roles", [])}
    return bool(roles & ALLOWED_ROLES)

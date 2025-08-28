async def handle_interaction(ctx, rt, risk, bracket, client):
    cid = ctx.custom_id
    sym = cid.split("_")[-1].upper()
    if cid.startswith("btn_be_"):
        await risk.maybe_move_to_breakeven(
            sym,
            rt.positions[sym],
            rt.positions[sym].entry_price,
            (await bracket.current_sl(sym)),
        )
        return await ctx.respond("BE 이동 요청", ephemeral=True)
    if cid.startswith("btn_half_"):
        q = abs(rt.positions[sym].qty) / 2
        if q > 0:
            await client.new_order(
                symbol=sym,
                side=("SELL" if rt.is_long(sym) else "BUY"),
                type="MARKET",
                quantity=str(q),
                reduceOnly=True,
            )
        return await ctx.respond("50% 청산", ephemeral=True)
    if cid.startswith("btn_flat_"):
        q = abs(rt.positions[sym].qty)
        if q > 0:
            await client.new_order(
                symbol=sym,
                side=("SELL" if rt.is_long(sym) else "BUY"),
                type="MARKET",
                quantity=str(q),
                reduceOnly=True,
            )
        return await ctx.respond("전량 청산", ephemeral=True)
    if cid.startswith("btn_ctp1_"):
        await bracket.cancel_tp1(sym)
        return await ctx.respond("TP1 취소", ephemeral=True)

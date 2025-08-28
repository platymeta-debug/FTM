import time
from ftm2.notify import dispatcher
from ftm2.journal.events import JEvent


def log_decision(*args, **kwargs):
    pass

# [ANCHOR:ROUTER_TICKET_FLOW]
class OrderRouter:
    def __init__(
        self,
        cfg,
        client,
        sizer=None,
        market=None,
        fills=None,
        account=None,
        rt=None,
        bracket=None,
        risk=None,
        notify=dispatcher,
        cards=None,
        analysis_views=None,
        sync_guard=None,
    ):

        self.cfg = cfg
        self.client = client
        self.sizer = sizer
        self.market = market
        self.fills = fills
        self.account = account
        self.rt = rt
        self.bracket = bracket
        self.risk = risk
        self.notify = notify
        self.cards = cards
        self.analysis_views = analysis_views
        self.sync_guard = sync_guard

    async def place_entry(self, sym: str):
        tk = self.rt.active_ticket.get(sym)
        if not tk:
            self.notify.emit("gate_skip", f"📡 {sym} 티켓없음 → 진입 금지")
            return False

        qty = self.sizer.size_entry(sym, tk, account=self.account.snapshot())
        if qty <= 0:
            self.notify.emit("gate_skip", f"📡 {sym} 사이징=0 → 스킵")
            return False

        bar_ts = self.market.bar_open_ts(sym, self.cfg.ENTRY_TF) if self.market else 0
        if self.rt.idem_hit.get((sym, tk.side)) == bar_ts:
            self.notify.emit("gate_skip", f"📡 {sym} 동일바 중복 → 스킵")
            return False
        if time.time() < self.rt.cooldown_until.get(sym, 0):
            self.notify.emit("gate_skip", f"📡 {sym} 쿨다운 중 → 스킵")
            return False


        resp = await self.client.new_order(symbol=sym, side=("BUY" if tk.side == "LONG" else "SELL"),
                                           type="MARKET", quantity=str(qty))
        self.notify.emit("order_submitted", f"📡 ✅ {sym} {tk.side} qty={qty} (ticket={tk.id})")
        if self.rt and getattr(self.rt, "journal", None):
            self.rt.journal.write(JEvent.now("ORDER_SUBMIT", symbol=sym, side=tk.side, qty=qty,
                                            price=None, order_id=resp.get("orderId"), ticket_id=tk.id))

        ok = True
        if self.fills and hasattr(self.fills, "wait_fill"):
            try:
                ok = await self.fills.wait_fill(sym, resp.get("orderId"), timeout=self.cfg.FILL_TIMEOUT_SEC)
            except Exception:
                ok = False
        pos = None
        if self.account and hasattr(self.account, "fetch_position"):
            pos = await self.account.fetch_position(sym, hydrate=True)
        if not ok and (not pos or abs(getattr(pos, "qty", 0)) < 1e-12):
            self.notify.emit("order_failed", f"📡 ⏱️ {sym} 체결 확인 실패")
            return False
        if pos and self.rt and getattr(self.rt, "journal", None):
            self.rt.journal.write(JEvent.now("FILL", symbol=sym, side=tk.side,
                                            qty=abs(pos.qty), price=pos.entry_price,
                                            entry=pos.entry_price, mark=pos.mark_price,
                                            lev=pos.leverage, mode=pos.margin_mode,
                                            ticket_id=tk.id))


        tps = []
        sl = tk.stop_px
        if self.bracket:
            tps = await self.bracket.place_from_ticket(sym, tk, abs(getattr(pos, "qty", qty)))
            if self.rt and getattr(self.rt, "journal", None):
                self.rt.journal.write(JEvent.now("SL_SET", symbol=sym, sl=sl))
                for px, q in tps:
                    self.rt.journal.write(JEvent.now("TP_SET", symbol=sym, tp1=px, qty=q))

        if self.sync_guard:
            ok = await self.sync_guard.verify_after_fill(
                sym, self.rt, self.bracket, self.analysis_views
            )
            if not ok and self.cards and pos:
                await self.cards.upsert_trade_card(sym, pos, sl, tps, force=True)
                if self.analysis_views and hasattr(self.analysis_views, "render_active"):
                    entry_px = getattr(
                        self.rt.positions.get(sym, pos), "entry_price", tk.entry_px
                    )
                    text = self.analysis_views.render_active(
                        sym,
                        side=tk.side,
                        entry=entry_px,
                        stop=sl,
                        tps=[px for px, _ in tps],
                    )
                    self.notify._upsert_sticky(
                        self.cfg.CHANNEL_SIGNALS,
                        f"analysis_{sym}",
                        text,
                        lifetime_min=self.cfg.ANALYSIS_LIFETIME_MIN,
                    )

        if self.risk and pos:
            self.risk.on_open(sym, getattr(pos, "entry_price", tk.entry_px))


        self.rt.idem_hit[(sym, tk.side)] = bar_ts
        self.rt.cooldown_until[sym] = time.time() + self.cfg.ENTRY_COOLDOWN_SEC
        self.rt.active_ticket.pop(sym, None)

        if self.cards and pos:
            await self.cards.upsert_trade_card(sym, pos, sl, tps)
        if self.analysis_views and hasattr(self.analysis_views, "render_active"):
            text = self.analysis_views.render_active(sym, side=tk.side,
                entry=getattr(pos, "entry_price", tk.entry_px), stop=sl,
                tps=[px for px, _ in tps])
            self.notify._upsert_sticky(self.cfg.CHANNEL_SIGNALS, f"analysis_{sym}", text,
                                       lifetime_min=self.cfg.ANALYSIS_LIFETIME_MIN)
        self.notify.emit("fill", f"💹 {sym} 체결: {tk.side} ×{abs(getattr(pos, 'qty', qty)):.6f} @~{getattr(pos, 'entry_price', tk.entry_px):.2f}  SL {sl:.2f}  TP {', '.join(f'{px:.2f}' for px,_ in tps)}")
        return True

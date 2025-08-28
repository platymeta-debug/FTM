import time
from ftm2.notify import dispatcher


def log_decision(*args, **kwargs):
    pass

# [ANCHOR:ROUTER_TICKET_FLOW]
class OrderRouter:
    def __init__(self, cfg, client, sizer, market, fills, account, rt, bracket, notify=dispatcher, cards=None, analysis_views=None):
        self.cfg = cfg
        self.client = client
        self.sizer = sizer
        self.market = market
        self.fills = fills
        self.account = account
        self.rt = rt
        self.bracket = bracket
        self.notify = notify
        self.cards = cards
        self.analysis_views = analysis_views

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

        tps = []
        sl = tk.stop_px
        if self.bracket:
            tps = await self.bracket.place_from_ticket(sym, tk, abs(getattr(pos, "qty", qty)))

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

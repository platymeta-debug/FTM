import time
from ftm2.notify import dispatcher
from ftm2.journal.events import JEvent
from ftm2.trade.order_fsm import OFSM, OState
from ftm2.trade.cid import build_cid
from ftm2.exchange.retry import with_retry, is_min_notional
from ftm2.exchange.timeguard import TimeGuard


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

        # [ANCHOR:ROUTER_INJECT_FILTERS]
        self.filters = getattr(self.client, "filters", None)
        if hasattr(self, "sizer") and self.sizer is not None:
            setattr(self.sizer, "filters", self.filters)

    async def place_entry(self, sym: str):
        # 0) 티켓 게이트
        tk = self.rt.active_ticket.get(sym)
        if not tk:
            self.notify.emit("gate_skip", f"📡 {sym} 티켓없음 → 진입 금지")
            return False

        # 1) 사이징
        qty = self.sizer.size_entry(sym, tk, account=self.account.snapshot())
        if qty <= 0:
            self.notify.emit("gate_skip", f"📡 {sym} 사이징=0 → 스킵")
            return False
        # 1-1) 최소명목 자동보정(옵션)
        if self.cfg.ORDER_AUTOSCALE_TO_MIN:
            qty = self.sizer.autoscale_min(sym, self.market.mark(sym), qty)

        # 2) 아이뎀/쿨다운
        bar_ts = self.market.bar_open_ts(sym, self.cfg.ENTRY_TF)
        if self.rt.idem_hit.get((sym, tk.side)) == bar_ts:
            self.notify.emit("gate_skip", f"📡 {sym} 동일바 중복 → 스킵")
            return False
        if time.time() < self.rt.cooldown_until.get(sym, 0):
            self.notify.emit("gate_skip", f"📡 {sym} 쿨다운 중 → 스킵")
            return False

        # 3) CID/FSM/시간가드
        cid = build_cid("FTM2", sym, tk.id, bar_ts)
        fsm = OFSM(sym, cid)
        fsm.to(OState.NEW)
        tg = TimeGuard(self.client, lambda m: self.notify.emit("system", "timeguard: " + m))
        await tg.sync()

        # 4) 주문 전송 with 재시도/백오프(+최소명목 자동보정 재시도)
        async def _send():
            return await self.client.new_order(
                symbol=sym,
                side=("BUY" if tk.side == "LONG" else "SELL"),
                type="MARKET",
                quantity=str(qty),
                newClientOrderId=cid,
                timestamp=tg.now_ms(),
                recvWindow=self.cfg.ORDER_RECV_WINDOW_MS,
            )

        try:
            resp = await with_retry(
                _send, tries=self.cfg.ORDER_RETRIES, base_ms=self.cfg.ORDER_BACKOFF_MS
            )
        except Exception as e:
            if is_min_notional(str(e)) and self.cfg.ORDER_AUTOSCALE_TO_MIN:
                qty = self.sizer.autoscale_min(sym, self.market.mark(sym), qty)
                resp = await self.client.new_order(
                    symbol=sym,
                    side=("BUY" if tk.side == "LONG" else "SELL"),
                    type="MARKET",
                    quantity=str(qty),
                    newClientOrderId=cid,
                )
            else:
                self.notify.emit("order_failed", f"📡 {sym} 전송실패: {e}")
                return False

        self.notify.emit(
            "order_submitted", f"📡 ✅ {sym} {tk.side} qty={qty} (ticket={tk.id})"
        )

        # 5) 체결 대기 (부분체결 포함)
        got, ok, od = await self.fills.wait_accum(
            sym,
            resp["orderId"],
            float(qty),
            timeout_ms=self.cfg.FILL_TOTAL_TIMEOUT_MS,
            poll_ms=self.cfg.FILL_POLL_MS,
        )
        if not ok:
            if self.cfg.CANCEL_ON_TIMEOUT:
                try:
                    await self.client.cancel_order(symbol=sym, orderId=resp["orderId"])
                except Exception:
                    pass
            self.notify.emit(
                "order_failed",
                f"📡 ⏱️ {sym} 체결 타임아웃 got={got:.6f}/{qty}",
            )
            return False

        if got < float(qty) * (1 - self.cfg.PARTIAL_TOL_PCT / 100):
            fsm.to(OState.PARTIAL)
            self.notify.emit(
                "order_submitted", f"📡 {sym} 부분체결 {got:.6f}/{qty:.6f}"
            )
        qty_filled = got

        # 6) 포지션 하이드레이션 & 브래킷
        pos = await self.account.fetch_position(sym, hydrate=True)
        if not pos or abs(pos.qty) < 1e-12:
            self.notify.emit("order_failed", f"📡 {sym} 체결 확인 실패")
            return False
        self.rt.positions[sym] = pos
        tps = await self.bracket.place_from_ticket(sym, tk, abs(pos.qty))
        sl = tk.stop_px
        fsm.to(OState.BRACKETS_SET)

        # 7) 상태/쿨다운/아이뎀
        self.rt.idem_hit[(sym, tk.side)] = bar_ts
        self.rt.cooldown_until[sym] = time.time() + self.cfg.ENTRY_COOLDOWN_SEC
        self.rt.active_ticket.pop(sym, None)

        # 카드/리스크/분석뷰 업데이트
        if self.risk:
            self.risk.on_open(sym, getattr(pos, "entry_price", tk.entry_px))
        if self.cards:
            await self.cards.upsert_trade_card(sym, pos, sl, tps)
        if self.analysis_views and hasattr(self.analysis_views, "render_active"):
            text = self.analysis_views.render_active(
                sym,
                side=tk.side,
                entry=getattr(pos, "entry_price", tk.entry_px),
                stop=sl,
                tps=[px for px, _ in tps],
            )
            self.notify._upsert_sticky(
                self.cfg.CHANNEL_SIGNALS,
                f"analysis_{sym}",
                text,
                lifetime_min=self.cfg.ANALYSIS_LIFETIME_MIN,
            )

        # 8) 알림
        self.notify.emit(
            "fill",
            f"💹 {sym} 체결: {tk.side} ×{abs(pos.qty):.6f} @~{pos.entry_price:.2f}  SL {sl:.2f}  TP {', '.join(f'{px:.2f}' for px,_ in tps)}",
        )
        # [ANCHOR:ATOMIC_NOTIFY_FILL]
        from ftm2.notify.receipt import build_receipt
        rec, txt = build_receipt(sym, self.rt.positions[sym], tk, sl, tps)
        self.notify.emit_once(f"receipt_{sym}", "fill", txt, ttl_ms=60000)
        if hasattr(self.rt, "journal"):
            from ftm2.journal.events import JEvent
            self.rt.journal.write(
                JEvent.now(
                    "INFO",
                    symbol=sym,
                    message="RECEIPT",
                    entry=rec.get("entry"),
                    sl=sl,
                    tp1=(rec["tps"][0] if rec["tps"] else None),
                )
            )
        return True

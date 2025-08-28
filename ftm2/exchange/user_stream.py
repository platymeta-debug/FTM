class UserStream:
    def __init__(self, account, notify):
        self.account = account
        self.notify = notify
        self._seen_fills: set[tuple[str, int, float]] = set()

    async def on_execution_report(self, report):
        # [ANCHOR:FILL_DEDUP]
        if report.get("X") in ("FILLED", "PARTIALLY_FILLED"):
            oid = report.get("i")
            sym = report.get("s")
            filled_qty = float(report.get("l", 0) or report.get("z", 0))
            key = (sym, oid, filled_qty)
            if filled_qty > 0 and key not in self._seen_fills:
                self._seen_fills.add(key)
                pos = await self.account.fetch_position(sym, hydrate=True)
                qty = float(pos.qty)
                side = "LONG" if qty > 0 else "SHORT" if qty < 0 else "FLAT"
                self.notify.emit(
                    "fill",
                    f"💹 {sym} 진입: {side} x{abs(qty):.6f} @~{pos.entry_price:.2f}",
                )

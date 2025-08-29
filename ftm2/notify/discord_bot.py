# [ANCHOR:DISCORD_BOT]
import asyncio, os, traceback, time
from pathlib import Path
from typing import Optional
import discord
from ftm2.signals.dedupe import should_emit
from ftm2.config.settings import load_env_chain
from ftm2.notify.discord_views import build_trade_embed
from ftm2.notify.analysis_views import build_analysis_embed
from ftm2.storage.persistence import load_trade_cards, save_trade_cards
from ftm2.storage.analysis_persistence import load_analysis_cards, save_analysis_cards
from ftm2.trade.position_tracker import PositionTracker
from ftm2.notify import dispatcher

_send_queue: "asyncio.Queue[tuple[str,str]]" = asyncio.Queue()
_cfg = None
_client: Optional[discord.Client] = None
_ch_logs = _ch_trades = _ch_signals = None
_ch_analysis: dict[str, discord.TextChannel] = {}
_analysis_cards: dict[str, dict] = {}


# 외부에서 주입할 훅(한국어 명령용)
_hooks = {
    "force_flat": None,   # def func(symbol:str) -> str
    "kill_switch": None,  # def func(on:bool) -> str
    "get_status": None,   # def func() -> str
    "get_signal": None,   # def func(symbol:str) -> str
    "close_all": None,
    "tracker_ref": None,
}

CSV = None
LEDGER = None
SETTINGS = load_env_chain()


def register_hooks(**kwargs):
    _hooks.update({k: v for k, v in kwargs.items() if k in _hooks})


def register_tracker(tracker: PositionTracker):
    _hooks["tracker_ref"] = tracker

def inject_tracker(tracker: PositionTracker):
    global _tracker_ref
    _tracker_ref = tracker

def TRACKER_REF():
    return _tracker_ref

# 동기 호출 가능: 내부 큐에 적재
def send_log(text: str, embed=None):    _send_queue.put_nowait(("logs", text, embed))
def send_trade(text: str, embed=None):  _send_queue.put_nowait(("trades", text, embed))
def send_signal(text: str, embed=None): _send_queue.put_nowait(("signals", text, embed))

# [UPSERT_MSG]
from ftm2.notify import dispatcher

_last_payload_hash = {}

def _payload_hash(txt: str) -> int:
    return hash(txt.replace("\r\n", "\n").strip())

async def upsert(channel_key_or_name: str, text: str, *, dedupe_ms=3000, max_age_edit_s=3300, sticky_key=None):
    now = time.time()*1000
    ph = _payload_hash(text)
    key = sticky_key or f"{channel_key_or_name}::default"


async def upsert(channel_key_or_name, text, *, sticky_key=None, dedupe_ms=2000, max_age_edit_s=3300):

    now = time.time() * 1000
    k = (channel_key_or_name, text)
    if now - _last_emit_cache.get(k, 0) < dedupe_ms:
        return None
    _last_emit_cache[k] = now

    key = sticky_key or channel_key_or_name
    mid_store = getattr(upsert, "_store", {})
    if not hasattr(upsert, "_store"):
        upsert._store = mid_store
    store = mid_store.setdefault(key, {"id": None, "ts": 0})

    try:
        if last_mid and (time.time() - (last_ts/1000.0) < max_age_edit_s):
            mid = await dispatcher.dc.edit(last_mid, text)
            _last_payload_hash[key] = (now, last_mid, ph)
            return mid
        else:
            mid = await dispatcher.dc.send(channel_key_or_name, text)
            _last_payload_hash[key] = (now, mid, ph)
            return mid
    except Exception:
        # 수정 실패(30046 등) -> 새로 보냄
        mid = await dispatcher.dc.send(channel_key_or_name, text)
        _last_payload_hash[key] = (now, mid, ph)
        return mid


async def send_signal_to_discord(sym: str, side: str, score: float, reasons: list[str], img_path: str | None = None):
    text = f"{sym} {side} score={score:.1f} reasons={', '.join(reasons or [])}"
    try:
        if _ch_signals:
            files = [discord.File(img_path)] if img_path and os.path.exists(img_path) else None
            await _ch_signals.send(content=f"📡 {text}", files=files)
        else:
            print(f"[SIGNAL][DRY] {text}")
    except Exception:
        traceback.print_exc()


async def publish_signal(sym: str, side: str, score: float, reasons: list[str], candle_open_ts: int, img_path: str | None = None):
    allow_intent_only = getattr(SETTINGS, "ALLOW_INTENT_ONLY", False)
    min_reason_cnt = getattr(SETTINGS, "MIN_REASON_COUNT", 1)
    if not allow_intent_only and reasons == ["INTENT"]:
        return
    if reasons and len(reasons) < max(min_reason_cnt, 1):
        return

    emit = should_emit(
        sym,
        side,
        score,
        reasons,
        candle_open_ts,
        enter_th=getattr(SETTINGS, "ENTER_TH", 60),
        cooldown_sec=getattr(SETTINGS, "COOLDOWN_SEC", 300),
        score_bucket=getattr(SETTINGS, "SCORE_BUCKET", 5),
        edge_trigger=getattr(SETTINGS, "EDGE_TRIGGER", True),
    )
    if not emit:
        return

    await send_signal_to_discord(sym, side, score, reasons, img_path if img_path and os.path.exists(img_path) else None)

async def _sender_loop():
    global _ch_logs, _ch_trades, _ch_signals
    while True:
        chan, text, embed = await _send_queue.get()
        try:
            if chan=="logs" and _ch_logs:    await _ch_logs.send(content=f"🧩 {text}", embed=embed)
            elif chan=="trades" and _ch_trades: await _ch_trades.send(content=f"💹 {text}", embed=embed)
            elif chan=="signals" and _ch_signals: await _ch_signals.send(content=f"📡 {text}", embed=embed)
            else:
                print(f"[DISCORD][DRY] {chan}: {text}")  # 채널 미설정 시 콘솔로
        except Exception:
            traceback.print_exc()

# [ANCHOR:M5_TRADE_CARD]
_persist_loaded = False
_last_edit_ts: dict[str, float] = {}
_analysis_loaded = False


def _load_persist(tracker: PositionTracker):
    global _persist_loaded
    if _persist_loaded: return
    data = load_trade_cards()
    if isinstance(data, dict):
        tracker.msg_ids.update({k: int(v) for k, v in data.items() if str(v).isdigit()})
    _persist_loaded = True


def _load_analysis_persist():
    global _analysis_loaded, _analysis_cards
    if _analysis_loaded:
        return
    data = load_analysis_cards()
    if isinstance(data, dict):
        _analysis_cards.update(data)
    _analysis_loaded = True



async def ensure_trade_card(symbol: str, tracker: PositionTracker, cfg):
    global _ch_trades
    if not _ch_trades: return None

    _load_persist(tracker)

    if symbol in tracker.msg_ids:
        try:
            msg = await _ch_trades.fetch_message(tracker.msg_ids[symbol])
            return msg
        except:
            pass

    ps = tracker.get_symbol_view(symbol)
    tracker.recompute_totals()
    emb = build_trade_embed(cfg, symbol, ps, tracker.account)
    msg = await _ch_trades.send(embed=emb)
    tracker.msg_ids[symbol] = msg.id
    save_trade_cards(tracker.msg_ids)
    return msg


async def edit_trade_card(symbol: str, tracker: PositionTracker, cfg, force: bool=False):
    if tracker.edits_disabled(): return
    now = time.time()
    last = _last_edit_ts.get(symbol, 0)
    if not force and (now - last) < cfg.DISCORD_UPDATE_INTERVAL_S:
        return
    if not force and not tracker.should_edit(symbol, cfg.PNL_CHANGE_BPS):
        hb = getattr(cfg, "TRADE_HEARTBEAT_S", 30)
        if (now - last) < hb:
            return


    msg = await ensure_trade_card(symbol, tracker, cfg)
    if not msg: return
    ps = tracker.get_symbol_view(symbol)
    tracker.recompute_totals()
    emb = build_trade_embed(cfg, symbol, ps, tracker.account)
    try:
        await msg.edit(embed=emb)
        _last_edit_ts[symbol] = now
    except Exception as e:
        print("[DISCORD][EDIT_ERR]", e)


# [ANCHOR:M6_ANALYSIS_MSG_API]
async def update_analysis(
    symbol: str,
    snapshot,
    divergence_bps: float,
    interval_s: int,
    view: dict | None = None,
):
    """Update analysis messages for a symbol.

    snapshot: Snapshot 객체 (차트 렌더용)
    view:     dict 가공본 (텍스트 임베드용)
    """

    from ftm2.charts.registry import render_ready, should_render
    from ftm2.charts.builder import render_analysis_charts


    if view is None:
        try:
            from ftm2.strategy.compat import to_viewdict  # type: ignore
        except Exception:
            to_viewdict = None  # type: ignore
        if to_viewdict:
            view = to_viewdict(snapshot)
        else:
            tf_scores = getattr(snapshot, "tf_scores", {})
            total = 0.0
            if isinstance(tf_scores, dict):
                try:
                    total = sum(float(v) for v in tf_scores.values())
                except Exception:
                    total = 0.0
            view = {
                "symbol": getattr(snapshot, "symbol", symbol),
                "decision_score": getattr(snapshot, "total_score", 0.0),
                "total_score": total,
                "direction": getattr(snapshot, "direction", "NEUTRAL"),
                "confidence": getattr(snapshot, "confidence", 0.0),
                "tf_scores": tf_scores,
            }

    _load_analysis_persist()
    ch = _ch_analysis.get(symbol)
    if not ch:
        return



    ready, info = should_render(_cfg, snapshot)
    if not ready:
        print(f"[CHART][SKIP] {symbol} cause={info.get('cause')}")
    else:
        Path(_cfg.CHART_DIR).mkdir(parents=True, exist_ok=True)
        try:
            paths = render_analysis_charts(snapshot, _cfg.CHART_DIR)
            if paths:
                print(f"[CHART][RENDER] {symbol} saved={paths}")

                ids = _analysis_cards.get(symbol, {})
                old_id = ids.get("chart")
                if old_id:
                    try:
                        old = await ch.fetch_message(old_id)
                        await old.delete()
                    except Exception:
                        pass
                files = [discord.File(p) for p in paths if os.path.exists(p)]
                new_msg = await ch.send(files=files)
                print(
                    f"[DISCORD][CHART][SEND] {symbol} msg_id={new_msg.id} paths={paths}"
                )

                ids["chart"] = new_msg.id
                _analysis_cards[symbol] = ids
                save_analysis_cards(_analysis_cards)
                if _cfg.CHART_MODE == "none":
                    for p in paths:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
        except Exception as e:
            send_log(f"[CHART][ERR] save failed: {e}")

    embed = build_analysis_embed(view, divergence_bps, interval_s)
    now_txt = time.strftime("%H:%M:%S", time.localtime())
    embed.set_footer(text=f"마지막 갱신 {now_txt} | 다음 갱신까지 {interval_s}s")


    ids = _analysis_cards.get(symbol, {})
    text_id = ids.get("text")
    if text_id:
        try:
            msg = await ch.fetch_message(text_id)
            await msg.edit(embed=embed)
        except Exception:
            msg = await ch.send(embed=embed)
            ids["text"] = msg.id
            _analysis_cards[symbol] = ids
            save_analysis_cards(_analysis_cards)
    else:
        msg = await ch.send(embed=embed)
        ids["text"] = msg.id
        _analysis_cards[symbol] = ids
        save_analysis_cards(_analysis_cards)


async def _resolve_guild_and_channels(client: discord.Client):
    global _ch_logs, _ch_trades, _ch_signals
    gid = _cfg.DISCORD_GUILD_ID
    guild = client.get_guild(gid) if gid else None
    if not guild and gid:
        try:
            guild = await client.fetch_guild(gid)
        except discord.Forbidden:
            print(f"[DISCORD][ERR] 길드({gid}) 접근 권한 없음."); return False
        except discord.NotFound:
            print(f"[DISCORD][ERR] 길드({gid})를 찾을 수 없음."); return False
        except Exception as e:
            print(f"[DISCORD][ERR] 길드({gid}) 조회 실패: {e}"); return False
    if not guild:
        print("[DISCORD][ERR] 길드 ID 미설정 또는 미접근. .env.notify의 DISCORD_GUILD_ID 확인")
        return False

    async def bind_channel(cid: int, label: str):
        ch = client.get_channel(cid)
        if ch is None:
            try:
                ch = await client.fetch_channel(cid)
            except discord.Forbidden:
                print(f"[DISCORD][ERR] {label} 채널({cid}) 권한 없음."); return None
            except discord.NotFound:
                print(f"[DISCORD][ERR] {label} 채널({cid}) 없음(잘못된 ID)."); return None
            except Exception as e:
                print(f"[DISCORD][ERR] {label} 채널({cid}) 조회 실패: {e}"); return None
        return ch

    _ch_logs = await bind_channel(_cfg.DISCORD_CHANNEL_LOGS, "logs") if _cfg.DISCORD_CHANNEL_LOGS else None
    _ch_trades = await bind_channel(_cfg.DISCORD_CHANNEL_TRADES, "trades") if _cfg.DISCORD_CHANNEL_TRADES else None
    _ch_signals = await bind_channel(_cfg.DISCORD_CHANNEL_SIGNALS, "signals") if _cfg.DISCORD_CHANNEL_SIGNALS else None
    if _cfg.DISCORD_CHANNEL_ANALYSIS_BTC:
        ch = await bind_channel(_cfg.DISCORD_CHANNEL_ANALYSIS_BTC, "analysis_btc")
        if ch:
            _ch_analysis["BTCUSDT"] = ch
    if _cfg.DISCORD_CHANNEL_ANALYSIS_ETH:
        ch = await bind_channel(_cfg.DISCORD_CHANNEL_ANALYSIS_ETH, "analysis_eth")
        if ch:
            _ch_analysis["ETHUSDT"] = ch

    print(f"[DISCORD] 연결 완료. guild={guild.name} "
          f"logs={'OK' if _ch_logs else 'X'} trades={'OK' if _ch_trades else 'X'} signals={'OK' if _ch_signals else 'X'} analysis={len(_ch_analysis)}")
    return True

async def _on_ready(client: discord.Client):
    ok = await _resolve_guild_and_channels(client)
    if ok and _cfg.DISCORD_TEST_ON_BOOT:
        # 이제 채널 바인딩이 끝났으니 반드시 채널로 전송됨
        send_log("FTM2 봇이 부팅되었습니다. (테스트 메시지)")
        send_log(f"환경: MODE={_cfg.MODE}, 심볼={_cfg.SYMBOLS}, 인터벌={_cfg.INTERVAL}, 프로파일={os.getenv('ENV_PROFILE','-')}")


async def _handle_message(msg: discord.Message):
    if msg.author.bot: return
    if not msg.content.startswith(_cfg.DISCORD_PREFIX): return
    args = msg.content[len(_cfg.DISCORD_PREFIX):].strip().split()
    if not args: return
    cmd = args[0]
    # --- 한국어 명령 ---
    if cmd == "상태":
        f = _hooks.get("get_status")
        text = f() if f else "상태 조회 훅이 아직 연결되지 않았습니다."
        await msg.channel.send(f"🧭 {text}")
    elif cmd == "킬스위치" and len(args)>=2:
        on = args[1] in ("켜","on","ON","true","TRUE")
        f = _hooks.get("kill_switch")
        text = f(on) if f else "킬스위치 훅 미연결"
        await msg.channel.send(f"🛑 {text}")
    elif cmd == "신호" and len(args)>=2:
        sym = args[1].upper()
        f = _hooks.get("get_signal")
        text = f(sym) if f else f"{sym} 신호 훅 미연결"
        await msg.channel.send(f"📡 {text}")
    elif cmd == "모드":
        await msg.channel.send(f"⚙️ TRADE_MODE={_cfg.TRADE_MODE} DATA_FEED={_cfg.DATA_FEED} WORKING_PRICE={_cfg.WORKING_PRICE}")
    elif cmd == "포지션":

        tr = _hooks.get("tracker_ref")
        if not tr:
            await msg.channel.send("트래커가 아직 연결되지 않았습니다."); return
        lines=[]
        for k, ps in tr.pos.items():
            if ps.qty==0: continue
            lines.append(f"{ps.symbol} {ps.side} × {ps.qty:.6f} | 진입 {ps.entry_price:.2f} UPNL {ps.upnl:.2f} ROE {ps.roe:.2f}%")
        await msg.channel.send("📊 현재 포지션\n" + ("\n".join(lines) if lines else "포지션 없음"))
    elif cmd == "자본":
        tr = _hooks.get("tracker_ref")
        if not tr:
            await msg.channel.send("트래커가 아직 연결되지 않았습니다."); return
        tr.recompute_totals()
        a = tr.account
        await msg.channel.send(f"💼 총자본: {a.equity:.2f} USDT (지갑 {a.wallet_balance:.2f} / 가용 {a.available_balance:.2f} / UPNL {a.total_upnl:.2f})")
    elif cmd == "일손익":
        s = LEDGER.stats
        await msg.channel.send(
          f"📆 {s.day} 일손익 요약\n"
          f"- 실현손익: {s.realized:.2f} USDT\n"
          f"- 수수료: {s.fees:.2f} USDT\n"
          f"- 펀딩: {s.funding:.2f} USDT\n"
          f"- 순익: {s.net:.2f} USDT\n"
          f"- 거래수/승률: {s.trades}/{(s.wins/max(1,s.trades))*100:.1f}%\n"
          f"- 최대낙폭: {s.max_dd:.2f} USDT"
        )
    elif cmd == "손실컷해제":
        LEDGER.cooldown_until = 0.0
        await msg.channel.send("⏰ 손실컷 쿨다운을 해제했습니다.")
    elif cmd == "csv스냅샷" and len(args)>=2:
        opt = args[1].lower()
        if opt == "on":
            _cfg.CSV_MARK_SNAPSHOT_SEC = max(5, _cfg.CSV_MARK_SNAPSHOT_SEC or 0)
            await msg.channel.send(f"📝 포지션 스냅샷 활성화({_cfg.CSV_MARK_SNAPSHOT_SEC}s)")
        else:
            _cfg.CSV_MARK_SNAPSHOT_SEC = 0
            await msg.channel.send("📝 포지션 스냅샷 비활성화")
    elif cmd == "청산" and len(args)>=2:
        sym = args[1].upper()
        f = _hooks.get("close_all")
        text = f(sym) if f else "청산 라우터가 아직 연결되지 않았습니다."
        await msg.channel.send(f"🔻 {text}")
    elif cmd == "로그테스트":
        send_log("이것은 로그 테스트입니다.")
        await msg.add_reaction("✅")
    elif cmd == "채널테스트":
        send_log("logs 채널 테스트 메시지")
        send_trade("trades 채널 테스트 메시지")
        send_signal("signals 채널 테스트 메시지")
        await msg.channel.send("📨 테스트 메시지 전송 시도 완료 (권한/바인딩 확인은 콘솔 로그 참조)")
    elif cmd == "디버그":
        await msg.channel.send(f"🔧 디버그: "
                               f"GUILD_ID={_cfg.DISCORD_GUILD_ID}, "
                               f"LOGS={_cfg.DISCORD_CHANNEL_LOGS}, "
                               f"TRADES={_cfg.DISCORD_CHANNEL_TRADES}, "
                               f"SIGNALS={_cfg.DISCORD_CHANNEL_SIGNALS}")
    else:
        await msg.channel.send("❓ 지원하지 않는 명령입니다. (상태, 포지션, 자본, 일손익, 손실컷해제, csv스냅샷 on|off, 청산 심볼, 킬스위치 켜|꺼, 신호 심볼, 로그테스트)")

async def start_notifier(cfg):
    """앱 루프 내에서 호출: 디스코드 클라이언트 + 송신 루프 실행"""
    global _cfg, _client
    _cfg = cfg
    # 송신 루프는 항상 가동 (토큰 없어도 콘솔 출력)
    asyncio.create_task(_sender_loop())

    token = (cfg.DISCORD_TOKEN or os.getenv("DISCORD_TOKEN") or "").strip()
    if not token:
        print("[DISCORD] 토큰 없음. 콘솔 로그만 출력합니다.")
        return
    print(f"[DISCORD] 토큰 감지: {token[:6]}… (길이={len(token)})")


    intents = discord.Intents.default()
    intents.message_content = True  # 포털에서 Message Content Intent 켜야 함
    _client = discord.Client(intents=intents)

    @_client.event
    async def on_ready():
        await _on_ready(_client)

    @_client.event
    async def on_message(message: discord.Message):
        await _handle_message(message)

    try:
        await _client.start(token)

    except Exception:
        traceback.print_exc()

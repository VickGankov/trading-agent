#!/usr/bin/env python3
"""
dashboard.py - Streamlit trading dashboard.

Usage:
    .venv/bin/streamlit run dashboard.py
"""

import os
import sys
import json
import glob
import math
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
load_dotenv()

try:
    import outcomes as _outcomes_mod
    _OUTCOMES_AVAILABLE = True
except Exception:
    _OUTCOMES_AVAILABLE = False

try:
    from pwa_patch import ensure_pwa_patch
    ensure_pwa_patch()
except Exception:
    pass  # PWA install prompt just won't be available — app still works fine

JOURNAL_DIR = Path(__file__).parent / "journal"
SCRIPTS_DIR = Path(__file__).parent / "scripts"
PYTHON = str(Path(__file__).parent / ".venv" / "bin" / "python")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Trading Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Typography */
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* Metric deltas */
    [data-testid="stMetricDelta"] { font-size:0.8em; }
    [data-testid="stMetricValue"] { font-size:1.4em; font-weight:600; }
    [data-testid="stMetricLabel"] { font-size:0.75em; text-transform:uppercase;
                                    letter-spacing:0.05em; color:#a0aec0; }

    /* Analysis cards */
    .analysis-card {
        background: #161b2e;
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .analysis-card.buy  { border-left: 4px solid #00d4aa; }
    .analysis-card.skip { border-left: 4px solid #ff4b4b; }
    .analysis-card.opts  { border-left: 4px solid #f6ad55; }
    .analysis-card.daily { border-left: 4px solid #9f7aea; background:#16112e; border-color:#4a3070; }

    /* Daily play sector badge */
    .sector-badge {
        display:inline-block; padding:4px 12px; border-radius:12px;
        background:rgba(159,122,234,0.15); color:#9f7aea;
        font-size:0.8em; font-weight:600; margin-right:8px;
    }
    .catalyst-box {
        background:#1a1530; border:1px solid #3d2f6e; border-radius:8px;
        padding:10px 14px; margin:10px 0; font-size:0.88em;
    }
    .catalyst-label { font-size:0.68em; text-transform:uppercase;
                      letter-spacing:0.06em; color:#9f7aea; margin-bottom:4px; }
    .hist-reaction {
        background:#0f1020; border-left:3px solid #9f7aea;
        padding:8px 12px; border-radius:0 6px 6px 0;
        font-size:0.85em; color:#a0aec0; margin:8px 0; font-style:italic;
    }

    /* Verdict pill */
    .verdict-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .verdict-bull { background: rgba(0,212,170,0.15); color: #00d4aa; }
    .verdict-bear { background: rgba(255,75,75,0.15);  color: #ff4b4b; }
    .verdict-neut { background: rgba(160,174,192,0.15); color: #a0aec0; }

    /* Level badges (entry / stop / target) */
    .level-grid { display:flex; gap:12px; margin:12px 0; flex-wrap:wrap; }
    .level-box {
        flex:1; min-width:80px;
        background:#1e2130;
        border-radius:8px;
        padding:10px 14px;
        text-align:center;
    }
    .level-label { font-size:0.7em; text-transform:uppercase;
                   letter-spacing:0.06em; color:#718096; margin-bottom:4px; }
    .level-value { font-size:1.25em; font-weight:700; color:#e2e8f0; }
    .level-sub   { font-size:0.75em; color:#a0aec0; margin-top:2px; }
    .level-stop   .level-value { color:#ff4b4b; }
    .level-target .level-value { color:#00d4aa; }

    /* Options structure block */
    .opts-structure {
        background:#1a2035;
        border-radius:8px;
        padding:10px 14px;
        font-family:monospace;
        font-size:0.9em;
        color:#f6ad55;
        margin:10px 0;
    }
    .opts-row { display:flex; gap:16px; margin-top:10px; flex-wrap:wrap; }
    .opts-item { flex:1; min-width:120px; }
    .opts-item-label { font-size:0.7em; text-transform:uppercase;
                       letter-spacing:0.06em; color:#718096; }
    .opts-item-value { font-size:0.95em; color:#e2e8f0; margin-top:2px; }

    /* Tech strip */
    .tech-strip { display:flex; gap:10px; flex-wrap:wrap; margin:10px 0 16px; }
    .tech-box {
        background:#1e2130; border-radius:8px; padding:8px 14px;
        text-align:center; min-width:90px; flex:1;
    }
    .tech-label { font-size:0.68em; text-transform:uppercase;
                  letter-spacing:0.05em; color:#718096; }
    .tech-value { font-size:1.1em; font-weight:600; color:#e2e8f0; margin-top:2px; }

    /* Divider */
    hr { border-color:#2d3250 !important; margin:1.5rem 0; }

    /* Options 4-metric grid */
    .opts-metrics {
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:8px;
        margin:14px 0 10px;
    }
    .opts-metric-box {
        background:#1e2130;
        border-radius:8px;
        padding:10px 14px;
    }
    .opts-metric-label {
        font-size:0.68em;
        text-transform:uppercase;
        letter-spacing:0.06em;
        color:#718096;
        margin-bottom:3px;
    }
    .opts-metric-value {
        font-size:1.2em;
        font-weight:700;
        color:#e2e8f0;
    }
    .opts-metric-sub {
        font-size:0.72em;
        color:#718096;
        margin-top:2px;
    }
    .risk-panel {
        background:#111827;
        border:1px solid #2d3748;
        border-radius:8px;
        padding:12px 14px;
        margin:12px 0;
    }
    .risk-header {
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap:10px;
        margin-bottom:10px;
    }
    .risk-title {
        font-size:0.74em;
        color:#a0aec0;
        text-transform:uppercase;
        letter-spacing:0.06em;
        font-weight:700;
    }
    .risk-grade {
        font-size:0.78em;
        font-weight:700;
        padding:3px 9px;
        border-radius:12px;
        background:rgba(0,0,0,0.22);
    }
    .risk-grid {
        display:grid;
        grid-template-columns:repeat(4, minmax(0, 1fr));
        gap:8px;
    }
    .risk-metric {
        background:#1e2130;
        border-radius:8px;
        padding:8px 10px;
        min-width:0;
    }
    .risk-label {
        font-size:0.64em;
        text-transform:uppercase;
        letter-spacing:0.05em;
        color:#718096;
        margin-bottom:2px;
    }
    .risk-value {
        font-size:1.05em;
        color:#e2e8f0;
        font-weight:700;
    }
    .risk-sub {
        font-size:0.7em;
        color:#718096;
        margin-top:2px;
    }
    .risk-checks {
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:4px 10px;
        margin-top:10px;
        color:#a0aec0;
        font-size:0.78em;
    }

    /* Color helpers */
    .green { color:#00d4aa; } .red { color:#ff4b4b; }
    .gold  { color:#f6ad55; } .gray { color:#a0aec0; }

    /* Hide sidebar globally — controls are inline */
    section[data-testid="stSidebar"],
    [data-testid="collapsedControl"] { display:none !important; }

    /* Mover section header */
    .mover-section-hdr {
        display:flex; align-items:center; gap:8px;
        margin:4px 0 10px;
    }
    .mover-section-title { font-size:0.95em; font-weight:700; color:#e2e8f0; }
    .mover-section-count {
        font-size:0.72em; color:#718096; background:#1e2130;
        padding:2px 9px; border-radius:10px;
    }

    /* Mover cards */
    .mover-card {
        display:flex; align-items:center; gap:12px;
        background:#161b2e; border:1px solid #2d3250; border-left:3px solid #4a5568;
        border-radius:10px; padding:10px 14px; margin-bottom:8px;
    }
    .mover-card.up   { border-left-color:#00d4aa; }
    .mover-card.down { border-left-color:#ff4b4b; }
    .mover-rank {
        width:22px; height:22px; border-radius:50%;
        background:#1e2130; color:#718096;
        font-size:0.68em; font-weight:700;
        display:flex; align-items:center; justify-content:center; flex-shrink:0;
    }
    .mover-main { flex:0 0 auto; min-width:120px; }
    .mover-ticker-line { display:flex; align-items:baseline; gap:8px; }
    .mover-ticker { font-weight:700; font-size:1.05em; color:#e2e8f0; }
    .mover-change { display:inline-flex; align-items:center; gap:3px; font-weight:700; font-size:0.9em; }
    .mover-change.up   { color:#00d4aa; }
    .mover-change.down { color:#ff4b4b; }
    .mover-change svg  { width:10px; height:10px; }
    .mover-price { color:#718096; font-size:0.8em; margin-top:1px; }
    .mover-catalyst {
        flex:1; min-width:0;
        display:flex; align-items:center; gap:6px;
        font-size:0.85em; color:#a0aec0;
        border-left:1px solid #2d3250; padding-left:12px;
    }
    .mover-catalyst a { color:#a0aec0; text-decoration:none; }
    .mover-catalyst a:hover { color:#e2e8f0; text-decoration:underline; }
    .mover-catalyst-text { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; min-width:0; }
    .mover-catalyst-text.mover-none { color:#4a5568; font-style:italic; }
    .mover-flash { color:#f6ad55; flex-shrink:0; }
    .mover-flash svg { width:12px; height:12px; }
    .mover-time { color:#4a5568; font-size:0.82em; white-space:nowrap; flex-shrink:0;
                  display:inline-flex; align-items:center; gap:3px; }
    .mover-time svg { width:10px; height:10px; }

    /* MACD verdict pill */
    .macd-verdict {
        border-radius:6px; padding:10px 14px;
        margin:8px 0; font-size:0.9em; line-height:1.5;
        border-left-width:3px; border-left-style:solid;
    }

    /* Mobile */
    @media (max-width: 768px) {
        .main .block-container { padding:0.75rem 0.75rem 2rem !important; max-width:100vw !important; }
        h1 { font-size:1.5rem !important; }
        h2, [data-testid="stHeading"] { font-size:1.1rem !important; }
        .analysis-card  { padding:12px 14px !important; }
        .catalyst-box   { padding:8px 10px !important; }
        .opts-metric-box { padding:8px 10px !important; }
        .opts-metric-value { font-size:1.05em !important; }
        .level-box { min-width:calc(50% - 8px) !important; }
        .tech-box  { min-width:calc(50% - 6px) !important; }
        [data-testid="stMetricValue"] { font-size:1.1em !important; }
        [data-testid="stMetricLabel"] { font-size:0.62em !important; }
        [data-testid="stDataFrame"]   { overflow-x:auto !important; }
        .mover-card { flex-wrap:wrap; }
        .mover-catalyst {
            border-left:none !important; padding-left:0 !important;
            margin-top:6px; width:100%; padding-top:6px;
            border-top:1px solid #1e2535;
        }
        .mover-catalyst-text { white-space:normal !important; }
    }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults (replaces sidebar) ─────────────────────────────────

if "_auto_refresh" not in st.session_state:
    st.session_state["_auto_refresh"] = False
if "_days_filter" not in st.session_state:
    st.session_state["_days_filter"] = 7
auto_refresh = st.session_state["_auto_refresh"]
days_filter  = st.session_state["_days_filter"]

# ── Data loaders ──────────────────────────────────────────────────────────────

# Self-contained inline icons (no external font/CDN — this app has hit enough
# network filtering issues on real user networks that a webfont dependency
# for pure decoration isn't worth the risk).
_ICON_UP = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" '
            'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<polyline points="3,17 9,11 13,15 21,7"/><polyline points="14,7 21,7 21,14"/></svg>')
_ICON_DOWN = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" '
              'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
              '<polyline points="3,7 9,13 13,9 21,17"/><polyline points="21,10 21,17 14,17"/></svg>')
_ICON_CLOCK = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
               'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
               '<circle cx="12" cy="12" r="9"/><polyline points="12,7 12,12 16,14"/></svg>')
_ICON_FLASH = ('<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
               '<path d="M13 2L3 14h7l-1 8 11-14h-7l1-6z"/></svg>')


def _relative_age(dt) -> str:
    delta = datetime.now(dt.tzinfo) - dt
    hrs = delta.total_seconds() / 3600
    if hrs < 1:
        return f"{max(1, int(delta.total_seconds() / 60))}m ago"
    if hrs < 24:
        return f"{int(hrs)}h ago"
    return f"{int(hrs / 24)}d ago"


def _attach_headline(m: dict):
    """Alpaca news first, yfinance fallback. Mutates m in place."""
    import yfinance as yf

    try:
        import research as _res
        nd = _res.get_news(m["ticker"], hours=16)
        items = nd.get("items", [])
        if items:
            item = items[0]
            m["headline"] = item.get("headline", "")
            m["headline_url"] = item.get("url", "")
            created = item.get("created_at", "")
            if created:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                m["headline_age"] = _relative_age(dt)
            return
    except Exception:
        pass

    try:
        news = yf.Ticker(m["ticker"]).news or []
        if news:
            content = news[0].get("content", news[0])
            m["headline"] = content.get("title", "")
            m["headline_url"] = (content.get("canonicalUrl") or {}).get("url", "")
            pub = content.get("pubDate")
            if pub:
                dt = datetime.fromisoformat(str(pub).replace("Z", "+00:00"))
                m["headline_age"] = _relative_age(dt)
    except Exception:
        pass


@st.cache_data(ttl=300)
def load_premarket_movers(min_pct: float = 0.8):
    """Two sections: your watchlist (always checked) + real market-wide top movers."""
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor

    wl_path = Path(__file__).parent / "data" / "watchlist.json"
    try:
        wl = json.loads(wl_path.read_text())
    except Exception:
        wl = {}

    priority = [t["symbol"] for t in wl.get("priority_tickers", [])]
    filters = wl.get("screener_filters", {})
    min_price = filters.get("min_price", 5.0)
    max_price = filters.get("max_price", 500.0)
    excluded_etfs = set(filters.get("exclude_leveraged_etfs", []))

    # ── Your Watchlist: always checked regardless of market-wide ranking ──
    def _fetch_watchlist(tkr):
        try:
            fi = yf.Ticker(tkr).fast_info
            last, prev = fi.last_price, fi.previous_close
            if not last or not prev or prev == 0:
                return None
            chg = round((last - prev) / prev * 100, 2)
            return {"ticker": tkr, "price": round(last, 2), "prev_close": round(prev, 2),
                    "change_pct": chg, "headline": "", "headline_url": "", "headline_age": ""}
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=12) as ex:
        raw_wl = list(ex.map(_fetch_watchlist, priority))
    watchlist_movers = sorted(
        [r for r in raw_wl if r and abs(r["change_pct"]) >= min_pct],
        key=lambda x: abs(x["change_pct"]), reverse=True)

    # ── Market-Wide: real top gainers/losers, filtered for quality ──
    market_movers = []
    try:
        import research as _res
        wl_set = set(priority)

        def _is_warrant_like(sym: str) -> bool:
            return "." in sym or sym.endswith(("W", "WS", "WW"))

        def _valid(m):
            return (m["symbol"] not in wl_set
                    and m["symbol"] not in excluded_etfs
                    and not _is_warrant_like(m["symbol"])
                    and min_price <= m["price"] <= max_price)

        raw_mkt = _res.get_market_movers(top=50)  # API max — most raw movers are sub-$5 penny stocks
        gainers = [m for m in raw_mkt["gainers"] if _valid(m)][:10]
        losers = [m for m in raw_mkt["losers"] if _valid(m)][:10]

        for m in gainers + losers:
            market_movers.append({
                "ticker": m["symbol"], "price": round(m["price"], 2),
                "prev_close": round(m["price"] - m["change"], 2),
                "change_pct": round(m["percent_change"], 2),
                "headline": "", "headline_url": "", "headline_age": "",
            })
    except Exception:
        pass

    # ── Headlines for both sections in parallel ──
    all_items = watchlist_movers + market_movers
    if all_items:
        with ThreadPoolExecutor(max_workers=12) as ex:
            list(ex.map(_attach_headline, all_items))

    return {"watchlist": watchlist_movers, "market": market_movers}


@st.cache_data(ttl=60)
def load_alpaca_data():
    try:
        import research
        account = research.get_account()
        market  = research.is_market_open()
        orders  = research.get_trading_client().get_orders()
        open_orders = [{
            "symbol": o.symbol,
            "side": o.side.value,
            "qty": float(o.qty) if o.qty else 0,
            "limit_price": float(o.limit_price) if o.limit_price else None,
            "status": o.status.value,
            "submitted_at": o.submitted_at.strftime("%H:%M") if o.submitted_at else "",
        } for o in orders]
        return account, market, open_orders
    except Exception as e:
        return None, {"error": str(e)}, []


@st.cache_data(ttl=30)
def load_journal_entries(days: int = 7):
    cutoff = datetime.now() - timedelta(days=days)
    files = sorted(glob.glob(str(JOURNAL_DIR / "*.json")), reverse=True)
    entries = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            written_str = data.get("_written_at", "")
            written = datetime.fromisoformat(written_str) if written_str else datetime.min
            if written >= cutoff:
                data["_filename"] = Path(f).name
                entries.append(data)
        except Exception:
            continue
    return entries


@st.cache_data(ttl=60)
def _load_intraday(sym):
    try:
        import research as _r
        result = _r.get_intraday_bars(sym, minutes=5, lookback_hours=8)
        bars   = result.get("bars", [])
        source = result.get("source", "")
        if len(bars) >= 35:
            macd = _r.calc_macd(bars)
        else:
            macd = {}
        return bars, macd, source
    except Exception as e:
        return [], {}, f"error: {e}"


BACKTEST_FILE = Path(__file__).parent / "data" / "backtest_runs.json"


@st.cache_data(ttl=10)
def load_backtest_runs():
    if not BACKTEST_FILE.exists():
        return []
    try:
        return json.loads(BACKTEST_FILE.read_text())
    except Exception:
        return []


def get_stop_target_from_journal(entries: list) -> dict:
    """
    For each symbol, find the most recent BUY with execution_status==SUBMITTED.
    Returns {symbol: {stop_loss, take_profit, entry_limit}}.
    """
    levels = {}
    for e in entries:
        for d in e.get("decisions", []):
            sym = d.get("ticker", "")
            if (d.get("action") == "BUY"
                    and d.get("execution_status") == "SUBMITTED"
                    and sym not in levels):
                levels[sym] = {
                    "stop":   d.get("stop_loss"),
                    "target": d.get("take_profit"),
                    "entry":  d.get("entry_limit"),
                }
    return levels


def load_active_stop_levels(entries: list) -> dict:
    """Load active stops from state.json first, then fill gaps from journal entries."""
    levels = {}
    state_path = Path(__file__).parent / "data" / "state.json"
    try:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        for sym, lvl in state.get("active_stops", {}).items():
            levels[sym] = {
                "stop": lvl.get("stop_loss"),
                "target": lvl.get("take_profit"),
                "entry": lvl.get("entry_limit"),
            }
    except Exception:
        pass

    for sym, lvl in get_stop_target_from_journal(entries).items():
        levels.setdefault(sym, lvl)
    return levels


def extract_trades(entries: list) -> list:
    trades = []
    for entry in entries:
        ts = entry.get("_written_at", "")[:16].replace("T", " ")
        dry = entry.get("dry_run", False)
        for d in entry.get("decisions", []):
            action = d.get("action", "")
            if action not in ("BUY", "SELL"):
                continue
            ex_status = d.get("execution_status", "DRY_RUN" if dry else "UNKNOWN")
            trades.append({
                "Time": ts,
                "Action": action,
                "Ticker": d.get("ticker", ""),
                "Qty": d.get("qty", ""),
                "Entry": d.get("entry_limit", d.get("entry", "")),
                "Stop": d.get("stop_loss", ""),
                "Target": d.get("take_profit", ""),
                "Conf": d.get("confidence", ""),
                "Status": ex_status,
                "Detail": d.get("execution_detail", d.get("order_id", "")),
                "Thesis": (d.get("thesis", d.get("reason", "")) or "")[:80],
            })
    return trades


def compute_stats(entries: list) -> dict:
    total_cycles = len(entries)
    submitted = rejected = dry_run = no_trade = hold = 0

    for e in entries:
        for d in e.get("decisions", []):
            a  = d.get("action", "")
            ex = d.get("execution_status", "")
            if a in ("BUY", "SELL"):
                if ex == "SUBMITTED":
                    submitted += 1
                elif ex in ("REJECTED", "ERROR", "DRY_RUN_REJECTED"):
                    rejected += 1
                elif ex.startswith("DRY_RUN"):
                    dry_run += 1
                else:
                    dry_run += 1
            elif a == "NO TRADE":
                no_trade += 1
            elif a == "HOLD":
                hold += 1

    no_trade_cycles = sum(
        1 for e in entries
        if not any(d.get("execution_status") == "SUBMITTED"
                   for d in e.get("decisions", []))
    )

    return {
        "total_cycles": total_cycles,
        "submitted": submitted,
        "rejected":  rejected,
        "dry_run":   dry_run,
        "no_trade":  no_trade,
        "hold":      hold,
        "no_trade_cycles": no_trade_cycles,
        "no_trade_rate": round(no_trade_cycles / total_cycles * 100, 1) if total_cycles else 0,
    }


def _to_float(value, default=None):
    try:
        if value in (None, "", "—"):
            return default
        return float(str(value).replace("$", "").replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return default


def estimate_qty(entry: float, account_value: float | None = None) -> float:
    """Match the agent's ~$100 max position sizing for proposed trades."""
    if not entry or entry <= 0:
        return 0.0
    max_dollars = min(100.0, (account_value or 1000.0) * 0.10)
    return math.floor((max_dollars / entry) * 100) / 100


def compute_trade_quality(entry, stop, target, qty, account_value=None,
                          current=None, confidence="", market_data=None,
                          tech=None, earnings=None, data_meta=None) -> dict | None:
    entry = _to_float(entry)
    stop = _to_float(stop)
    target = _to_float(target)
    qty = _to_float(qty, 0)
    current = _to_float(current)
    account_value = _to_float(account_value)
    tech = tech or {}
    earnings = earnings or {}
    data_meta = data_meta or {}

    if not entry or not stop or not target or not qty or entry <= 0 or qty <= 0:
        return None

    risk_per_share = entry - stop
    reward_per_share = target - entry
    if risk_per_share <= 0 or reward_per_share <= 0:
        return None

    position_value = qty * entry
    dollar_risk = qty * risk_per_share
    dollar_reward = qty * reward_per_share
    rr = reward_per_share / risk_per_share
    stop_pct = risk_per_share / entry * 100
    target_pct = reward_per_share / entry * 100
    account_risk_pct = (dollar_risk / account_value * 100) if account_value else None
    breakeven_win_rate = 100 / (1 + rr)
    current_r = ((current - entry) / risk_per_share) if current else None

    market_rsi = None
    if isinstance(market_data, dict):
        market_rsi = market_data.get("indices", {}).get("SPY", {}).get("rsi14")
    rsi = tech.get("rsi14")
    days_until = earnings.get("days_until")
    is_stale = data_meta.get("is_stale", False)

    checks = [
        ("R/R >= 1.5", rr >= 1.5),
        ("Stop 3-10%", 3 <= stop_pct <= 10),
        ("Account risk <= 1%", account_risk_pct is None or account_risk_pct <= 1),
        ("Position <= 10%", account_value is None or position_value <= account_value * 0.1005),
        ("RSI not overbought", not isinstance(rsi, (int, float)) or rsi <= 65),
        ("Earnings clear", not isinstance(days_until, int) or days_until > 3),
        ("Market not stretched", not isinstance(market_rsi, (int, float)) or market_rsi <= 75),
        ("Fresh price data", not is_stale),
    ]

    score = 0
    score += min(30, max(0, (rr - 1.0) / 2.0 * 30))
    score += 20 if 4 <= stop_pct <= 6 else (12 if 3 <= stop_pct <= 10 else 0)
    score += 15 if account_risk_pct is None or account_risk_pct <= 0.75 else (8 if account_risk_pct <= 1.0 else 0)
    score += {"HIGH": 15, "MEDIUM": 10, "LOW": 5}.get(str(confidence).upper(), 7)
    score += 10 if not isinstance(market_rsi, (int, float)) or market_rsi <= 70 else (4 if market_rsi <= 80 else 0)
    score += 10 if not isinstance(days_until, int) or days_until > 7 else (4 if days_until > 3 else 0)
    score = round(min(100, score))

    if score >= 80:
        grade, color = "Strong", "#00d4aa"
    elif score >= 65:
        grade, color = "Acceptable", "#f6ad55"
    elif score >= 50:
        grade, color = "Marginal", "#f6ad55"
    else:
        grade, color = "Poor", "#ff4b4b"

    return {
        "position_value": position_value,
        "dollar_risk": dollar_risk,
        "dollar_reward": dollar_reward,
        "rr": rr,
        "stop_pct": stop_pct,
        "target_pct": target_pct,
        "account_risk_pct": account_risk_pct,
        "breakeven_win_rate": breakeven_win_rate,
        "current_r": current_r,
        "checks": checks,
        "score": score,
        "grade": grade,
        "color": color,
    }


def render_trade_quality(q: dict | None) -> str:
    if not q:
        return ""
    account_risk = "—" if q["account_risk_pct"] is None else f"{q['account_risk_pct']:.2f}%"
    current_r = ""
    if q["current_r"] is not None:
        r_color = "green" if q["current_r"] >= 0 else "red"
        current_r = f"""<div class="risk-metric">
      <div class="risk-label">Open R</div>
      <div class="risk-value {r_color}">{q['current_r']:+.2f}R</div>
      <div class="risk-sub">vs entry risk</div>
    </div>"""
    checks = "".join(
        f"<div>{'OK' if passed else 'WARN'} · {label}</div>"
        for label, passed in q["checks"]
    )
    return f"""
  <div class="risk-panel">
    <div class="risk-header">
      <div class="risk-title">Risk / Reward Quality</div>
      <div class="risk-grade" style="color:{q['color']}">{q['grade']} · {q['score']}/100</div>
    </div>
    <div class="risk-grid">
      <div class="risk-metric">
        <div class="risk-label">Risk</div>
        <div class="risk-value red">${q['dollar_risk']:.2f}</div>
        <div class="risk-sub">{account_risk} account</div>
      </div>
      <div class="risk-metric">
        <div class="risk-label">Reward</div>
        <div class="risk-value green">${q['dollar_reward']:.2f}</div>
        <div class="risk-sub">target +{q['target_pct']:.1f}%</div>
      </div>
      <div class="risk-metric">
        <div class="risk-label">R/R</div>
        <div class="risk-value">{q['rr']:.2f}:1</div>
        <div class="risk-sub">need {q['breakeven_win_rate']:.1f}% wins</div>
      </div>
      <div class="risk-metric">
        <div class="risk-label">Position</div>
        <div class="risk-value">${q['position_value']:.2f}</div>
        <div class="risk-sub">stop -{q['stop_pct']:.1f}%</div>
      </div>
      {current_r}
    </div>
    <div class="risk-checks">{checks}</div>
  </div>"""


def build_equity_curve(entries: list) -> pd.DataFrame:
    rows = []
    for e in sorted(entries, key=lambda x: x.get("_written_at", "")):
        ts_str = e.get("_written_at", e.get("cycle_timestamp", ""))
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            continue
        acct = e.get("account", {})
        value = acct.get("account_value") or acct.get("equity")
        cash  = acct.get("cash")
        if value:
            rows.append({"time": ts, "equity": float(value), "cash": float(cash or 0)})
    return pd.DataFrame(rows)


# ── Load data ─────────────────────────────────────────────────────────────────

account_data, market_data, open_orders = load_alpaca_data()
journal_entries = load_journal_entries(days_filter)
stop_levels = load_active_stop_levels(
    load_journal_entries(30)  # look back 30 days for stop levels regardless of filter
)

# ── Main layout ───────────────────────────────────────────────────────────────

st.title("📈 Trading Agent")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🌅 Pre-Market", "🎯 Options", "🔍 Analysis", "📊 Portfolio", "🧪 Backtest"])

with tab1:
    # ── Pre-Market Movers ─────────────────────────────────────────────────────

    _pm_hdr, _pm_btn = st.columns([6, 1])
    _pm_hdr.subheader("🌅 Pre-Market Movers")
    if _pm_btn.button("🔄 Refresh", key="pm_refresh", width='stretch'):
        load_premarket_movers.clear()
        st.rerun()

    _pm_data = load_premarket_movers()
    _watchlist_movers = _pm_data.get("watchlist", [])
    _market_movers = _pm_data.get("market", [])

    def _mover_card_html(m: dict, rank: int) -> str:
        is_up = m["change_pct"] > 0
        direction = "up" if is_up else "down"
        icon = _ICON_UP if is_up else _ICON_DOWN
        sign = "+" if is_up else ""

        if m.get("headline"):
            headline_inner = (f'<a href="{m["headline_url"]}" target="_blank">{m["headline"]}</a>'
                              if m.get("headline_url") else m["headline"])
            catalyst_html = (f'<span class="mover-flash">{_ICON_FLASH}</span>'
                             f'<span class="mover-catalyst-text">{headline_inner}</span>')
            if m.get("headline_age"):
                catalyst_html += f'<span class="mover-time">{_ICON_CLOCK}{m["headline_age"]}</span>'
        else:
            catalyst_html = '<span class="mover-catalyst-text mover-none">No recent headline</span>'

        return (
            f'<div class="mover-card {direction}">'
            f'<div class="mover-rank">{rank}</div>'
            f'<div class="mover-main">'
            f'<div class="mover-ticker-line">'
            f'<span class="mover-ticker">{m["ticker"]}</span>'
            f'<span class="mover-change {direction}">{icon}{sign}{m["change_pct"]:.1f}%</span>'
            f'</div>'
            f'<div class="mover-price">${m["price"]:.2f}</div>'
            f'</div>'
            f'<div class="mover-catalyst">{catalyst_html}</div>'
            f'</div>'
        )

    def _render_mover_list(movers: list, empty_msg: str, limit: int = 5):
        if not movers:
            st.caption(empty_msg)
            return

        # Rank by magnitude, with a bonus for genuine catalyst-backed moves so a
        # smaller but news-driven move can outrank a bigger move with no story behind it.
        CATALYST_BONUS = 3.0
        ranked = sorted(
            movers,
            key=lambda m: abs(m["change_pct"]) + (CATALYST_BONUS if m.get("headline") else 0),
            reverse=True,
        )
        top, rest = ranked[:limit], ranked[limit:]

        st.markdown("".join(_mover_card_html(m, i + 1) for i, m in enumerate(top)),
                    unsafe_allow_html=True)

        if rest:
            with st.expander(f"Show {len(rest)} more"):
                st.markdown(
                    "".join(_mover_card_html(m, i + limit + 1) for i, m in enumerate(rest)),
                    unsafe_allow_html=True,
                )

    st.markdown(
        '<div class="mover-section-hdr">'
        '<span class="mover-section-title">Your Watchlist</span>'
        f'<span class="mover-section-count">{len(_watchlist_movers)} tracked</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    _render_mover_list(_watchlist_movers,
                       "No watchlist tickers moved >0.8% yet — check back closer to market open.")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="mover-section-hdr">'
        '<span class="mover-section-title">Market-Wide Movers</span>'
        f'<span class="mover-section-count">{len(_market_movers)} found</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    _render_mover_list(_market_movers, "No market-wide movers available right now.")

    # ── 5-min Chart + MACD ───────────────────────────────────────────────────
    _all_movers = _watchlist_movers + _market_movers
    if _all_movers:
        from plotly.subplots import make_subplots

        _ticker_options = [m["ticker"] for m in _all_movers]
        _selected = st.selectbox("📈 5-min chart + MACD", _ticker_options,
                                 key="intraday_select")

        _bars, _macd, _src = _load_intraday(_selected)

        if _bars:
            _ts    = [b["timestamp"] for b in _bars]
            _opens = [b["open"]  for b in _bars]
            _highs = [b["high"]  for b in _bars]
            _lows  = [b["low"]   for b in _bars]
            _cls   = [b["close"] for b in _bars]

            _has_macd = "macd" in _macd and len(_macd["macd"]) == len(_bars)

            _rows = 2 if _has_macd else 1
            _fig_intra = make_subplots(
                rows=_rows, cols=1, shared_xaxes=True,
                row_heights=[0.65, 0.35] if _has_macd else [1],
                vertical_spacing=0.04,
            )
            _fig_intra.add_trace(go.Candlestick(
                x=_ts, open=_opens, high=_highs, low=_lows, close=_cls,
                increasing_line_color="#48bb78", decreasing_line_color="#fc8181",
                increasing_fillcolor="#48bb78", decreasing_fillcolor="#fc8181",
                name="Price", showlegend=False,
            ), row=1, col=1)
            if _has_macd:
                _hvals = _macd["histogram"]
                _fig_intra.add_trace(go.Bar(
                    x=_ts, y=_hvals,
                    marker_color=["#48bb78" if v >= 0 else "#fc8181" for v in _hvals],
                    name="Histogram", showlegend=False,
                ), row=2, col=1)
                _fig_intra.add_trace(go.Scatter(
                    x=_ts, y=_macd["macd"],
                    line=dict(color="#90cdf4", width=1.5),
                    name="MACD", showlegend=False,
                ), row=2, col=1)
                _fig_intra.add_trace(go.Scatter(
                    x=_ts, y=_macd["signal"],
                    line=dict(color="#f6ad55", width=1.5, dash="dot"),
                    name="Signal", showlegend=False,
                ), row=2, col=1)
            _fig_intra.update_layout(
                height=420 if _has_macd else 280,
                margin=dict(l=0, r=0, t=8, b=4),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", size=11),
                xaxis_rangeslider_visible=False,
            )
            _fig_intra.update_xaxes(gridcolor="#2d3748", showgrid=True)
            _fig_intra.update_yaxes(gridcolor="#2d3748", showgrid=True)
            st.plotly_chart(_fig_intra, width='stretch',
                            config={"staticPlot": True})

            if _has_macd:
                _sig = _macd.get("signal_str", "")
                _verdicts = {
                    "bullish_crossover": ("🟢", "Bullish crossover — momentum just turned up",  "#1a3a2a", "#48bb78"),
                    "bearish_crossover": ("🔴", "Bearish crossover — momentum just turned down", "#3a1a1a", "#fc8181"),
                    "bullish_trend":     ("🔵", "Above signal line — uptrend in progress",       "#1a2a3a", "#90cdf4"),
                    "bearish_trend":     ("⚪", "Below signal line — downtrend in progress",     "#1e2130", "#718096"),
                }
                _icon, _text, _bg, _border = _verdicts.get(_sig, ("⚪", _sig, "#1e2130", "#718096"))
                _zero = "MACD above zero ✓" if _macd.get("above_zero") else "MACD below zero"
                st.markdown(
                    f'<div class="macd-verdict" style="background:{_bg};border-left-color:{_border}">'
                    f'{_icon} <strong>{_selected}</strong> — {_text} &nbsp;·&nbsp; {_zero}</div>',
                    unsafe_allow_html=True,
                )
                if _src == "yfinance_delayed":
                    st.caption("⚠️ Alpaca returned no IEX data — using yfinance (15-min delay)")
        else:
            st.caption(f"No intraday data available for {_selected} yet.")

with tab2:
    # ── Daily Options Play ────────────────────────────────────────────────────

    st.subheader("🎯 Today's Options Play")
    st.caption("Catalyst-first daily scan — identifies the sector with the strongest news catalyst and generates a directional options play.")

    if "daily_play" not in st.session_state:
        st.session_state.daily_play = None
    if "daily_play_loading" not in st.session_state:
        st.session_state.daily_play_loading = False

    scan_col, clear_col = st.columns([3, 1])
    run_scan = scan_col.button("🔍 Scan for Today's Play", width='stretch', type="primary")
    if clear_col.button("Clear", width='stretch'):
        st.session_state.daily_play = None
        st.rerun()

    if run_scan:
        with st.spinner("Scanning sectors + today's news for catalyst-driven play… (30–60s)"):
            cmd = [PYTHON, str(SCRIPTS_DIR / "agent.py"), "--daily-options"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0 and res.stdout.strip():
            try:
                st.session_state.daily_play = json.loads(res.stdout)
            except json.JSONDecodeError:
                st.error(f"Could not parse scan response: {res.stdout[:300]}")
        else:
            st.error(f"Scan failed: {(res.stderr or res.stdout)[-400:]}")

    dp = st.session_state.daily_play
    if dp:
        if "error" in dp:
            st.error(dp["error"])
        else:
            direction    = dp.get("direction", "").lower()
            dir_color    = "#00d4aa" if "bull" in direction else "#ff4b4b"
            dir_arrow    = "▲" if "bull" in direction else "▼"
            strategy     = dp.get("strategy", "—")
            sector_name  = dp.get("sector_name", dp.get("sector_etf", "—"))
            ticker       = dp.get("ticker", "—")
            catalyst     = dp.get("catalyst_headline", dp.get("catalyst", "—"))
            hist_react   = dp.get("historical_reaction", "")
            structure    = dp.get("structure", "—")
            expiry_why   = dp.get("expiry_rationale", "")
            why          = dp.get("why", "")
            max_loss     = dp.get("max_loss", "—")
            max_gain     = dp.get("max_gain", "—")
            ideal        = dp.get("ideal_outcome", "")
            risk         = dp.get("risk", "")
            conf         = dp.get("confidence", "")
            conf_color   = {"HIGH": "#00d4aa", "MEDIUM": "#f6ad55", "LOW": "#a0aec0"}.get(conf, "#a0aec0")
            gen_at       = dp.get("generated_at", "")
            try:
                gen_label = datetime.fromisoformat(gen_at).strftime("Generated %b %d at %I:%M %p")
            except Exception:
                gen_label = ""

            ticker_tech  = dp.get("ticker_technicals", {})
            last_price   = dp.get("ticker_price") or ticker_tech.get("last_trade") or ticker_tech.get("current_price")
            dp_hist_vol  = dp.get("hist_vol_pct") or ticker_tech.get("hist_vol_30d")
            dp_hist_vol_note = f"Black-Scholes · {dp_hist_vol:.0f}% hist vol · actual IV may differ" if dp_hist_vol else ""

            # Helper: build one play section for daily card
            def _daily_play_html(play, direction, label, icon, accent):
                cost_c      = play.get("cost_per_contract")
                max_gain_c  = play.get("max_gain_contract")
                breakeven   = play.get("breakeven")
                be_pct      = play.get("breakeven_pct")
                return_pct  = play.get("return_pct")
                is_credit   = play.get("is_credit", False)
                stype       = play.get("strategy_type", "")
                profit_zone = play.get("profit_zone", "")
                lower_be    = play.get("lower_breakeven")
                upper_be    = play.get("upper_breakeven")
                plain_eng   = play.get("plain_english", "")
                _names      = {
                    "bull_call_spread": "Bull Call Spread",
                    "bear_put_spread":  "Bear Put Spread",
                    "bull_put_spread":  "Bull Put Spread",
                    "bear_call_spread": "Bear Call Spread",
                    "iron_condor":      "Iron Condor",
                }
                badge = _names.get(stype, stype.replace("_", " ").title() if stype else "")
                metrics = ""
                if cost_c is not None:
                    if stype == "iron_condor":
                        be_label = "Safe Zone"
                        be_val   = profit_zone or (f"${lower_be} – ${upper_be}" if lower_be and upper_be else "—")
                        be_sub   = "stay in range = profit"
                        lbl1, v1, s1, c1 = "You Collect", f"~${max_gain_c:.0f}", "cash upfront",        "green"
                        lbl3, v3, s3, c3 = "Max Loss",    f"~${cost_c:.0f}",     "if stock breaks out", "red"
                    elif is_credit:
                        be_label = "Break-Even"
                        be_val   = f"${breakeven}" if breakeven else "—"
                        be_sub   = f"{be_pct:.1f}% cushion" if be_pct else ""
                        lbl1, v1, s1, c1 = "You Collect", f"~${max_gain_c:.0f}", "cash upfront",   "green"
                        lbl3, v3, s3, c3 = "Max Loss",    f"~${cost_c:.0f}",     "if fully wrong", "red"
                    else:
                        be_label = "Break-Even"
                        be_val   = f"${breakeven}" if breakeven else "—"
                        be_dir   = "+" if "bull" in direction.lower() else "-"
                        be_sub   = f"stock needs {be_dir}{be_pct:.1f}%" if be_pct else ""
                        lbl1, v1, s1, c1 = "You Pay", f"~${cost_c:.0f}",     "upfront, max loss", "red"
                        lbl3, v3, s3, c3 = "Max Win",  f"~${max_gain_c:.0f}", "if right",         "green"
                    rr_str = f"{return_pct}%" if return_pct is not None else "—"
                    metrics = f"""
      <div class="opts-metrics">
        <div class="opts-metric-box">
          <div class="opts-metric-label">{lbl1}</div>
          <div class="opts-metric-value {c1}">{v1}</div>
          <div class="opts-metric-sub">{s1}</div>
        </div>
        <div class="opts-metric-box">
          <div class="opts-metric-label">{be_label}</div>
          <div class="opts-metric-value">{be_val}</div>
          <div class="opts-metric-sub">{be_sub}</div>
        </div>
        <div class="opts-metric-box">
          <div class="opts-metric-label">{lbl3}</div>
          <div class="opts-metric-value {c3}">{v3}</div>
          <div class="opts-metric-sub">{s3}</div>
        </div>
        <div class="opts-metric-box">
          <div class="opts-metric-label">Return if Right</div>
          <div class="opts-metric-value green">{rr_str}</div>
          <div class="opts-metric-sub">on risk</div>
        </div>
      </div>"""
                structure   = play.get("structure", "")
                pe_html     = f'<p style="font-size:0.88em;color:#e2e8f0;margin:0 0 10px;line-height:1.5">{plain_eng}</p>' if plain_eng else ""
                badge_html  = f'<span style="font-size:0.72em;color:#718096;background:#1a2535;padding:2px 8px;border-radius:10px">{badge}</span>' if badge else ""
                struct_html = (
                    f'<div style="font-size:0.82em;color:#90cdf4;background:#0d1a2d;padding:7px 12px;'
                    f'border-radius:6px;border:1px solid #2a4a7f;margin:0 0 10px;font-family:monospace">'
                    f'📋 {structure}</div>'
                ) if structure else ""
                return f"""
      <div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
          <span style="font-size:0.92em;font-weight:700;color:{accent}">{icon} {label}</span>
          {badge_html}
        </div>
        {struct_html}
        {pe_html}
        {metrics}
      </div>"""

            dp_dir_play = dp.get("directional_play", {})
            dp_tht_play = dp.get("theta_play", {})
            dp_section  = _daily_play_html(dp_dir_play, direction, "Bet on the Move", "🎯", "#f6ad55")
            tp_section  = _daily_play_html(dp_tht_play, direction, "Collect &amp; Wait", "💰", "#68d391")

            st.markdown(f"""
    <div class="analysis-card daily">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
        <div>
          <span class="sector-badge">📊 {sector_name}</span>
          <span style="font-size:1.2em;font-weight:700;color:#e2e8f0">{ticker}</span>
          {f'<span style="color:#a0aec0;font-size:0.88em;margin-left:8px">${last_price}</span>' if last_price else ''}
        </div>
        <div style="text-align:right">
          <span style="color:{dir_color};font-size:1em;font-weight:700">{dir_arrow} {direction.capitalize()}</span>
          <span style="font-size:0.8em;padding:3px 10px;border-radius:12px;
                       background:rgba(0,0,0,0.2);color:{conf_color};font-weight:600;margin-left:8px">{conf}</span>
        </div>
      </div>

      <div class="catalyst-box">
        <div class="catalyst-label">📰 Catalyst</div>
        <div style="color:#e2e8f0;font-size:0.88em">{catalyst}</div>
      </div>

    {dp_section}
      <hr style="border:0;border-top:1px solid #2d3748;margin:14px 0">
    {tp_section}

      {f'<p style="font-size:0.8em;color:#ff4b4b;margin:10px 0 0"><strong>Risk:</strong> {risk}</p>' if risk else ''}
      {f'<p style="font-size:0.68em;color:#4a5568;margin:8px 0 0">{gen_label}</p>' if gen_label else ''}
    </div>
    """, unsafe_allow_html=True)

            with st.expander("🔬 Technical details"):
                dir_legs = dp_dir_play.get("legs_note", "")
                tht_legs = dp_tht_play.get("legs_note", "")
                st.markdown(f"""
    <div style="font-size:0.82em;color:#a0aec0;line-height:1.7">
      {f"<p><strong>Why this direction:</strong> {why}</p>" if why else ""}
      {f'<p><strong>Bet on Move legs:</strong><br><code style="color:#90cdf4;font-size:0.95em">{dir_legs}</code></p>' if dir_legs else ""}
      {f'<p><strong>Collect &amp; Wait legs:</strong><br><code style="color:#90cdf4;font-size:0.95em">{tht_legs}</code></p>' if tht_legs else ""}
      {f'<p style="margin-top:8px;color:#4a5568">{dp_hist_vol_note}</p>' if dp_hist_vol_note else ""}
    </div>
    """, unsafe_allow_html=True)

            # Sector performance snapshot
            sectors = dp.get("sector_snapshot", {})
            if sectors and isinstance(sectors, dict):
                with st.expander("📊 Sector ETF Performance (used in scan)"):
                    rows = []
                    for etf, s in sectors.items():
                        if isinstance(s, dict) and "name" in s:
                            rows.append({
                                "ETF": etf,
                                "Sector": s["name"],
                                "5d %": f"{s.get('5d_change_pct',0):+.1f}%",
                                "20d %": f"{s.get('20d_change_pct',0):+.1f}%",
                                "RSI": s.get("rsi14", "—"),
                                "Above MA20": "✅" if s.get("above_ma20") else "❌",
                                "Vol Ratio": f"{s.get('vol_ratio','—')}x",
                            })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

            # Supporting news
            news_hl = dp.get("news_headlines", [])
            if news_hl:
                with st.expander(f"📰 News used in scan ({len(news_hl)} headlines)"):
                    for h in news_hl:
                        st.markdown(f"<span style='color:#a0aec0;font-size:0.9em'>• {h}</span>",
                                    unsafe_allow_html=True)


    # ── Options Outcome Tracker ───────────────────────────────────────────────────

    st.divider()
    st.subheader("📊 Options Track Record")

    if not _OUTCOMES_AVAILABLE:
        st.caption("outcomes.py not loaded — run from trading-agent directory.")
    else:
        _oc_hdr, _oc_btn = st.columns([5, 1])
        _oc_hdr.caption("Track whether AI options picks played out.")
        if _oc_btn.button("🔄 Check Expired", width='stretch', key="check_exp_tab2"):
            _updated = _outcomes_mod.update_expired_trades()
            if _updated:
                st.success(f"Updated {len(_updated)} expired trade(s).")
            else:
                st.info("No newly expired trades.")

        _perf = _outcomes_mod.get_performance_stats()
        _m1, _m2, _m3, _m4, _m5 = st.columns(5)
        _m1.metric("Completed",  _perf["total"])
        _m2.metric("Open",       _perf["open"])
        _m3.metric("Win Rate",   f"{_perf['win_rate']}%" if _perf["total"] else "—")
        _m4.metric("Avg Return", f"{_perf['avg_return']:+.1f}%" if _perf["total"] else "—")
        _m5.metric("Total P&L",  f"${_perf['total_pnl']:+.0f}" if _perf["total"] else "—")

        if _perf["all_trades"]:
            _rows_ot = []
            _oicons = {"win": "✅", "loss": "❌", "breakeven": "➖", "pending": "⏳"}
            for _t in _perf["all_trades"][:20]:
                _oc  = _t.get("outcome") or ""
                _pnl = _t.get("profit_loss")
                _ret = _t.get("return_pct")
                _rows_ot.append({
                    "": _oicons.get(_oc, "🔵"),
                    "Date":     _t.get("date", ""),
                    "Ticker":   _t.get("ticker", ""),
                    "Strategy": _t.get("strategy", "").replace("_", " ").title(),
                    "Conf":     _t.get("confidence", ""),
                    "Expiry":   _t.get("expiration", ""),
                    "P&L":      f"${_pnl:+.0f}" if _pnl is not None else "—",
                    "Return":   f"{_ret:+.1f}%" if _ret is not None else "—",
                })
            st.dataframe(pd.DataFrame(_rows_ot), width='stretch', hide_index=True)
        elif _perf["open"] == 0:
            st.info("No options trades recorded yet. Run Today's Options Play to start tracking.")

with tab3:
    st.subheader("🔍 Symbol Analysis")
    st.caption("Search any ticker for an AI trade setup and options strategy.")

    col_sym, col_btn = st.columns([4, 1])
    symbol_input = col_sym.text_input(
        "ticker", placeholder="e.g. AAPL  ·  NVDA  ·  MSFT  ·  SPY",
        label_visibility="collapsed"
    ).strip().upper()
    analyze_clicked = col_btn.button("Analyze ▶", width='stretch', type="primary")

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "analysis_symbol" not in st.session_state:
        st.session_state.analysis_symbol = ""

    if analyze_clicked and symbol_input:
        with st.spinner(f"Fetching data and analyzing {symbol_input}…"):
            cmd = [PYTHON, str(SCRIPTS_DIR / "agent.py"), "--analyze", symbol_input]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode == 0 and res.stdout.strip():
            try:
                st.session_state.analysis_result = json.loads(res.stdout)
                st.session_state.analysis_symbol = symbol_input
            except json.JSONDecodeError:
                st.error("Could not parse analysis response.")
                st.session_state.analysis_result = None
        else:
            st.error(f"Analysis failed: {(res.stderr or res.stdout)[-400:]}")
            st.session_state.analysis_result = None
    elif analyze_clicked and not symbol_input:
        st.warning("Enter a ticker symbol first.")

    data = st.session_state.analysis_result
    if data:
        if "error" in data:
            st.error(data["error"])
        else:
            tech      = data.get("technicals", {})
            trade     = data.get("stock_trade", {})
            opt       = data.get("options_play", {})
            earn      = data.get("earnings", {})
            headlines = data.get("headlines", [])
            price     = data.get("current_price", "—")
            verdict   = data.get("verdict", "")
            action    = trade.get("action", "SKIP")

            # Verdict header
            bull_kw = any(w in verdict.lower() for w in ("bull", "upside", "positive", "strong", "buy"))
            bear_kw = any(w in verdict.lower() for w in ("bear", "downside", "negative", "weak", "skip"))
            pill_cls = "verdict-bull" if bull_kw else ("verdict-bear" if bear_kw else "verdict-neut")
            pill_lbl = "Bullish" if bull_kw else ("Bearish" if bear_kw else "Neutral")

            meta = data.get("data_meta", {})
            canonical_price = meta.get("canonical_price") or price
            price_source    = meta.get("price_source", "")
            is_stale        = meta.get("is_stale", False)
            data_age        = meta.get("data_age_minutes")
            quote_ts        = meta.get("quote_timestamp", "")
            warnings        = data.get("price_warnings", [])

            # Format data freshness label
            if is_stale:
                age_str = f"{data_age:.0f}m old" if data_age else "age unknown"
                qt_label = f"Last trade · {age_str} · market closed"
                qt_color = "#f6ad55"
            elif quote_ts:
                try:
                    qt = datetime.fromisoformat(quote_ts)
                    qt_label = f"Last trade: {qt.strftime('%I:%M %p ET')}"
                    qt_color = "#00d4aa"
                except Exception:
                    qt_label = "Live data · time unknown"
                    qt_color = "#a0aec0"
            else:
                qt_label = "No live data — using yesterday's close"
                qt_color = "#f6ad55"

            h_col, meta_col, earn_col = st.columns([3, 2, 1])
            h_col.markdown(
                f"<h3 style='margin:0'>{data['symbol']} "
                f"<span style='color:#a0aec0;font-weight:400'>${canonical_price}</span></h3>",
                unsafe_allow_html=True
            )
            meta_col.markdown(
                f"<div style='padding-top:8px;font-size:0.8em;color:{qt_color}'>"
                f"🕐 {qt_label}</div>",
                unsafe_allow_html=True
            )
            days_until = earn.get("days_until")
            if isinstance(days_until, int):
                earn_color = "#ff4b4b" if days_until <= 3 else ("#f6ad55" if days_until <= 7 else "#a0aec0")
                earn_col.markdown(
                    f"<div style='text-align:right;padding-top:6px'>"
                    f"<span style='color:{earn_color};font-size:0.85em'>📅 Earnings in {days_until}d</span></div>",
                    unsafe_allow_html=True
                )

            # Price accuracy warnings
            for w in warnings:
                st.warning(f"⚠️ **Price check:** {w}")

            st.markdown(
                f"<span class='verdict-pill {pill_cls}'>{pill_lbl}</span> "
                f"<span style='color:#cbd5e0;font-size:0.95em'>{verdict}</span>",
                unsafe_allow_html=True
            )

            # Technicals strip
            rsi = tech.get("rsi14", "—")
            rsi_color = "#ff4b4b" if isinstance(rsi, (int,float)) and rsi > 70 else (
                        "#00d4aa" if isinstance(rsi, (int,float)) and rsi < 35 else "#e2e8f0")
            chg5 = tech.get("5d_change_pct", "—")
            chg5_color = "#00d4aa" if isinstance(chg5, (int,float)) and chg5 > 0 else "#ff4b4b"
            st.markdown(f"""
    <div class="tech-strip">
      <div class="tech-box"><div class="tech-label">RSI 14</div>
        <div class="tech-value" style="color:{rsi_color}">{rsi}</div></div>
      <div class="tech-box"><div class="tech-label">MA 20</div>
        <div class="tech-value">${tech.get('ma20','—')}</div></div>
      <div class="tech-box"><div class="tech-label">MA 50</div>
        <div class="tech-value">${tech.get('ma50','—')}</div></div>
      <div class="tech-box"><div class="tech-label">Vol Ratio</div>
        <div class="tech-value">{tech.get('vol_ratio','—')}x</div></div>
      <div class="tech-box"><div class="tech-label">5d Change</div>
        <div class="tech-value" style="color:{chg5_color}">{chg5}%</div></div>
      <div class="tech-box"><div class="tech-label">Above MA20</div>
        <div class="tech-value">{'✅' if tech.get('above_ma20') else '❌'}</div></div>
    </div>
    """, unsafe_allow_html=True)

            # Trade + Options side by side
            left, right = st.columns(2)

            # ── Stock Trade card ──
            with left:
                if action == "BUY":
                    conf = trade.get("confidence","")
                    conf_color = {"HIGH":"#00d4aa","MEDIUM":"#f6ad55","LOW":"#a0aec0"}.get(conf,"#a0aec0")
                    entry  = trade.get("entry","—")
                    stop   = trade.get("stop","—")
                    target = trade.get("target","—")
                    stop_pct = trade.get("stop_pct","")
                    rr     = trade.get("reward_risk","")
                    acct_value = account_data.get("account_value") if account_data else 1000.0
                    proposed_qty = trade.get("qty") or estimate_qty(_to_float(entry), acct_value)
                    quality = compute_trade_quality(
                        entry, stop, target, proposed_qty,
                        account_value=acct_value,
                        current=canonical_price,
                        confidence=conf,
                        market_data=market_data,
                        tech=tech,
                        earnings=earn,
                        data_meta=meta,
                    )
                    quality_html = render_trade_quality(quality)
                    st.markdown(f"""
    <div class="analysis-card buy">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
        <span style="font-size:1.1em;font-weight:700;color:#00d4aa">📈 BUY</span>
        <span style="font-size:0.8em;padding:3px 10px;border-radius:12px;
                     background:rgba(0,0,0,0.2);
                     color:{conf_color};font-weight:600">{conf}</span>
      </div>
      <div class="level-grid">
        <div class="level-box level-entry">
          <div class="level-label">Entry</div>
          <div class="level-value">${entry}</div>
        </div>
        <div class="level-box level-stop">
          <div class="level-label">Stop</div>
          <div class="level-value">${stop}</div>
          <div class="level-sub">−{stop_pct}%</div>
        </div>
        <div class="level-box level-target">
          <div class="level-label">Target</div>
          <div class="level-value">${target}</div>
          <div class="level-sub">R/R {rr}:1</div>
        </div>
      </div>
      {quality_html}
      <p style="font-size:0.9em;color:#cbd5e0;margin:10px 0 6px"><strong>Thesis:</strong> {trade.get('thesis','')}</p>
      <p style="font-size:0.82em;color:#718096;margin:0"><strong>Bear case:</strong> {trade.get('risk','')}</p>
    </div>
    """, unsafe_allow_html=True)
                else:
                    skip_reason = data.get("skip_reason") or trade.get("thesis","No compelling setup today.")
                    st.markdown(f"""
    <div class="analysis-card skip">
      <div style="font-size:1.1em;font-weight:700;color:#ff4b4b;margin-bottom:10px">⛔ SKIP</div>
      <p style="font-size:0.9em;color:#cbd5e0;margin:0">{skip_reason}</p>
    </div>
    """, unsafe_allow_html=True)

            # ── Options Play card ──
            with right:
                strategy = opt.get("strategy","—")
                if strategy.lower() in ("skip","none","n/a","—","no play"):
                    st.markdown("""
    <div class="analysis-card opts">
      <div style="font-size:1.1em;font-weight:700;color:#f6ad55;margin-bottom:10px">🎯 Options Plays</div>
      <p style="color:#a0aec0;font-size:0.9em">No favorable options setup at this time.</p>
    </div>
    """, unsafe_allow_html=True)
                else:
                    opt_dir      = opt.get("direction", "")
                    opt_catalyst = opt.get("catalyst", "")
                    opt_risk     = opt.get("risk", "")
                    dir_color    = "#00d4aa" if "bull" in opt_dir.lower() else ("#ff4b4b" if "bear" in opt_dir.lower() else "#f6ad55")
                    opt_arrow    = "▲" if "bull" in opt_dir.lower() else ("▼" if "bear" in opt_dir.lower() else "→")
                    hist_vol     = opt.get("hist_vol_pct") or tech.get("hist_vol_30d")

                    dir_play = opt.get("directional_play", {})
                    tht_play = opt.get("theta_play", {})

                    def _build_play_html(play, direction, label, icon, accent):
                        cost_c      = play.get("cost_per_contract")
                        max_gain_c  = play.get("max_gain_contract")
                        breakeven   = play.get("breakeven")
                        be_pct      = play.get("breakeven_pct")
                        return_pct  = play.get("return_pct")
                        is_credit   = play.get("is_credit", False)
                        stype       = play.get("strategy_type", "")
                        profit_zone = play.get("profit_zone", "")
                        lower_be    = play.get("lower_breakeven")
                        upper_be    = play.get("upper_breakeven")
                        plain_eng   = play.get("plain_english", "")
                        _names      = {
                            "bull_call_spread": "Bull Call Spread",
                            "bear_put_spread":  "Bear Put Spread",
                            "bull_put_spread":  "Bull Put Spread",
                            "bear_call_spread": "Bear Call Spread",
                            "iron_condor":      "Iron Condor",
                        }
                        badge = _names.get(stype, stype.replace("_", " ").title() if stype else "")
                        metrics = ""
                        if cost_c is not None:
                            if stype == "iron_condor":
                                be_label = "Safe Zone"
                                be_val   = profit_zone or (f"${lower_be} – ${upper_be}" if lower_be and upper_be else "—")
                                be_sub   = "stay in range = profit"
                                lbl1, v1, s1, c1 = "You Collect", f"~${max_gain_c:.0f}", "cash upfront",        "green"
                                lbl3, v3, s3, c3 = "Max Loss",    f"~${cost_c:.0f}",     "if stock breaks out", "red"
                            elif is_credit:
                                be_label = "Break-Even"
                                be_val   = f"${breakeven}" if breakeven else "—"
                                be_sub   = f"{be_pct:.1f}% cushion" if be_pct else ""
                                lbl1, v1, s1, c1 = "You Collect", f"~${max_gain_c:.0f}", "cash upfront",   "green"
                                lbl3, v3, s3, c3 = "Max Loss",    f"~${cost_c:.0f}",     "if fully wrong", "red"
                            else:
                                be_label = "Break-Even"
                                be_val   = f"${breakeven}" if breakeven else "—"
                                be_dir   = "+" if "bull" in direction.lower() else "-"
                                be_sub   = f"stock needs {be_dir}{be_pct:.1f}%" if be_pct else ""
                                lbl1, v1, s1, c1 = "You Pay", f"~${cost_c:.0f}",     "upfront, max loss", "red"
                                lbl3, v3, s3, c3 = "Max Win",  f"~${max_gain_c:.0f}", "if right",         "green"
                            rr_str = f"{return_pct}%" if return_pct is not None else "—"
                            metrics = f"""
      <div class="opts-metrics">
        <div class="opts-metric-box">
          <div class="opts-metric-label">{lbl1}</div>
          <div class="opts-metric-value {c1}">{v1}</div>
          <div class="opts-metric-sub">{s1}</div>
        </div>
        <div class="opts-metric-box">
          <div class="opts-metric-label">{be_label}</div>
          <div class="opts-metric-value">{be_val}</div>
          <div class="opts-metric-sub">{be_sub}</div>
        </div>
        <div class="opts-metric-box">
          <div class="opts-metric-label">{lbl3}</div>
          <div class="opts-metric-value {c3}">{v3}</div>
          <div class="opts-metric-sub">{s3}</div>
        </div>
        <div class="opts-metric-box">
          <div class="opts-metric-label">Return if Right</div>
          <div class="opts-metric-value green">{rr_str}</div>
          <div class="opts-metric-sub">on risk</div>
        </div>
      </div>"""
                        structure  = play.get("structure", "")
                        pe_html    = f'<p style="font-size:0.88em;color:#e2e8f0;margin:0 0 10px;line-height:1.5">{plain_eng}</p>' if plain_eng else ""
                        badge_html = f'<span style="font-size:0.72em;color:#718096;background:#1a2535;padding:2px 8px;border-radius:10px">{badge}</span>' if badge else ""
                        struct_html = (
                            f'<div style="font-size:0.82em;color:#90cdf4;background:#0d1a2d;padding:7px 12px;'
                            f'border-radius:6px;border:1px solid #2a4a7f;margin:0 0 10px;font-family:monospace">'
                            f'📋 {structure}</div>'
                        ) if structure else ""
                        return f"""
      <div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
          <span style="font-size:0.92em;font-weight:700;color:{accent}">{icon} {label}</span>
          {badge_html}
        </div>
        {struct_html}
        {pe_html}
        {metrics}
      </div>"""

                    cat_html = f'<div class="catalyst-box"><div class="catalyst-label">📰 Catalyst</div><div style="color:#e2e8f0;font-size:0.88em">{opt_catalyst}</div></div>' if opt_catalyst else ""
                    dp_section = _build_play_html(dir_play, opt_dir, "Bet on the Move", "🎯", "#f6ad55")
                    tp_section = _build_play_html(tht_play, opt_dir, "Collect &amp; Wait", "💰", "#68d391")

                    st.markdown(f"""
    <div class="analysis-card opts">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <span style="font-size:1.1em;font-weight:700;color:#f6ad55">🎯 Options Plays</span>
        <span style="color:{dir_color};font-size:0.85em;font-weight:700">{opt_arrow} {opt_dir.upper() if opt_dir else ""}</span>
      </div>
      {cat_html}
    {dp_section}
      <hr style="border:0;border-top:1px solid #2d3748;margin:14px 0">
    {tp_section}
      {f'<p style="font-size:0.8em;color:#ff4b4b;margin:10px 0 0"><strong>Risk:</strong> {opt_risk}</p>' if opt_risk else ""}
    </div>
    """, unsafe_allow_html=True)

                    with st.expander("🔬 Technical details"):
                        vol_note  = f"Black-Scholes · {hist_vol:.0f}% hist vol · actual IV may differ" if hist_vol else ""
                        rationale = opt.get("rationale", opt.get("why", ""))
                        dir_legs  = dir_play.get("legs_note", "")
                        tht_legs  = tht_play.get("legs_note", "")
                        st.markdown(f"""
    <div style="font-size:0.82em;color:#a0aec0;line-height:1.7">
      {f"<p><strong>Why this direction:</strong> {rationale}</p>" if rationale else ""}
      {f'<p><strong>Bet on Move legs:</strong><br><code style="color:#90cdf4;font-size:0.95em">{dir_legs}</code></p>' if dir_legs else ""}
      {f'<p><strong>Collect &amp; Wait legs:</strong><br><code style="color:#90cdf4;font-size:0.95em">{tht_legs}</code></p>' if tht_legs else ""}
      {f'<p style="margin-top:8px;color:#4a5568">{vol_note}</p>' if vol_note else ""}
    </div>
    """, unsafe_allow_html=True)

            # ── Day Trade Card ──────────────────────────────────────────────
            dt = data.get("day_trade", {})
            if dt.get("available"):
                dt_entry   = dt.get("entry")
                dt_stop    = dt.get("stop")
                dt_target  = dt.get("target")
                dt_stop_pct = dt.get("stop_pct")
                dt_rr      = dt.get("rr")
                dt_shares  = dt.get("shares")
                dt_risk    = dt.get("risk_dollars")
                dt_reward  = dt.get("reward_dollars")
                dt_setup   = dt.get("setup_type", "")
                dt_why     = dt.get("why", "")
                dt_risk_txt = dt.get("risk", "")
                dt_exit    = dt.get("exit_time", "3:45 PM ET")
                dt_atr_pct = tech.get("atr14_pct", "")

                st.markdown(f"""
    <div class="analysis-card" style="border-left:4px solid #63b3ed;background:#0f1a2e;
         border-color:#2a4a7f;margin-top:12px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <span style="font-size:1.1em;font-weight:700;color:#63b3ed">⚡ Day Trade</span>
        <div>
          <span style="font-size:0.8em;padding:3px 10px;border-radius:12px;
                       background:rgba(99,179,237,0.15);color:#63b3ed;font-weight:600">{dt_setup}</span>
          <span style="font-size:0.72em;color:#718096;margin-left:10px">Exit by {dt_exit}</span>
        </div>
      </div>

      <div class="level-grid">
        <div class="level-box level-entry">
          <div class="level-label">Entry</div>
          <div class="level-value">${dt_entry}</div>
          <div class="level-sub">limit order</div>
        </div>
        <div class="level-box level-stop">
          <div class="level-label">Stop</div>
          <div class="level-value">${dt_stop}</div>
          <div class="level-sub">−{dt_stop_pct}%</div>
        </div>
        <div class="level-box level-target">
          <div class="level-label">Target</div>
          <div class="level-value">${dt_target}</div>
          <div class="level-sub">R/R {dt_rr}:1</div>
        </div>
        <div class="level-box">
          <div class="level-label">Size</div>
          <div class="level-value">{dt_shares} sh</div>
          <div class="level-sub">≤$100 max</div>
        </div>
      </div>

      <div style="display:flex;gap:16px;margin:10px 0">
        <div style="flex:1;background:#1a2535;border-radius:8px;padding:10px 14px;text-align:center">
          <div style="font-size:0.68em;text-transform:uppercase;letter-spacing:0.06em;color:#718096">Risk</div>
          <div style="font-size:1.15em;font-weight:700;color:#ff4b4b;margin-top:3px">~${dt_risk}</div>
          <div style="font-size:0.72em;color:#718096">if stop hit</div>
        </div>
        <div style="flex:1;background:#1a2535;border-radius:8px;padding:10px 14px;text-align:center">
          <div style="font-size:0.68em;text-transform:uppercase;letter-spacing:0.06em;color:#718096">Reward</div>
          <div style="font-size:1.15em;font-weight:700;color:#00d4aa;margin-top:3px">~${dt_reward}</div>
          <div style="font-size:0.72em;color:#718096">if target hit</div>
        </div>
        <div style="flex:1;background:#1a2535;border-radius:8px;padding:10px 14px;text-align:center">
          <div style="font-size:0.68em;text-transform:uppercase;letter-spacing:0.06em;color:#718096">R/R Ratio</div>
          <div style="font-size:1.15em;font-weight:700;color:#63b3ed;margin-top:3px">{dt_rr}:1</div>
          <div style="font-size:0.72em;color:#718096">reward per $ risk</div>
        </div>
      </div>

      <p style="font-size:0.85em;color:#cbd5e0;margin:8px 0 4px"><strong>Why:</strong> {dt_why}</p>
      {f'<p style="font-size:0.8em;color:#ff4b4b;margin:4px 0 0"><strong>Risk:</strong> {dt_risk_txt}</p>' if dt_risk_txt else ''}
      {f'<p style="font-size:0.7em;color:#4a5568;margin:6px 0 0">📐 ATR14: {dt_atr_pct}% of price · stop sized at 0.75× ATR</p>' if dt_atr_pct else ''}
    </div>
    """, unsafe_allow_html=True)
            else:
                st.markdown("""
    <div class="analysis-card" style="border-left:4px solid #4a5568;margin-top:12px">
      <span style="font-size:1.05em;font-weight:700;color:#718096">⚡ Day Trade</span>
      <p style="color:#4a5568;font-size:0.88em;margin:8px 0 0">No bullish setup — day trade skipped.</p>
    </div>
    """, unsafe_allow_html=True)

            # Headlines
            if headlines:
                with st.expander(f"📰 Recent news — {data['symbol']} ({len(headlines)} headlines)"):
                    for h in headlines:
                        st.markdown(f"<span style='color:#a0aec0;font-size:0.9em'>•</span> {h}",
                                    unsafe_allow_html=True)


with tab4:
    # ── Inline agent controls ────────────────────────────────────────────────────

    _ca, _cb, _cc, _cd, _ce = st.columns([2, 2, 3, 1, 1])
    _run_live = _ca.button("▶ Run Live",  width='stretch', type="primary")
    _run_dry  = _cb.button("◎ Dry Run",   width='stretch')
    _cc.toggle("Auto-refresh (60s)", key="_auto_refresh")
    _days_filter_val = _cd.number_input("Days", 1, 30,
                                        st.session_state.get("_days_filter", 7),
                                        key="_days_filter", label_visibility="visible",
                                        help="Journal lookback days")
    days_filter = int(_days_filter_val)
    if _ce.button("🔄", width='stretch', help="Refresh all data"):
        st.cache_data.clear()
        st.rerun()

    if _run_live or _run_dry:
        _flag = "" if _run_live else "--dry-run"
        _cmd  = [PYTHON, str(SCRIPTS_DIR / "agent.py")] + ([_flag] if _flag else [])
        with st.spinner("Running cycle… (up to 3 min)"):
            _res = subprocess.run(_cmd, capture_output=True, text=True, timeout=300)
        if _res.returncode == 0:
            st.success("Cycle complete — refresh to see latest positions.")
        else:
            _err = (_res.stderr or _res.stdout)[-800:]
            st.error(f"Cycle failed: {_err}")
        st.cache_data.clear()

    st.divider()

    # ── Header ────────────────────────────────────────────────────────────────────

    # Last cycle summary
    if journal_entries:
        last = journal_entries[0]
        last_ts = last.get("_written_at", last.get("cycle_timestamp", ""))[:16].replace("T", " ")
        last_dry = " (dry run)" if last.get("dry_run") else ""
        submitted_in_last = [d for d in last.get("decisions", []) if d.get("execution_status") == "SUBMITTED"]
        rejected_in_last  = [d for d in last.get("decisions", []) if d.get("execution_status") in ("REJECTED","ERROR")]
        if submitted_in_last:
            tickers = ", ".join(d.get("ticker","") for d in submitted_in_last)
            st.success(f"Last cycle {last_ts}{last_dry} — **{len(submitted_in_last)} order(s) submitted**: {tickers}")
        elif rejected_in_last:
            st.warning(f"Last cycle {last_ts}{last_dry} — {len(rejected_in_last)} order(s) rejected. No fills.")
        else:
            st.info(f"Last cycle {last_ts}{last_dry} — No trades placed.")
    else:
        st.info("No journal entries yet. Run a cycle to start.")

    st.divider()

    # ── Account Overview ──────────────────────────────────────────────────────────

    st.subheader("Account Overview")

    if account_data:
        equity   = account_data.get("account_value", 0)
        cash     = account_data.get("cash", 0)
        n_pos    = account_data.get("positions_count", 0)
        cash_pct  = (cash / equity * 100) if equity else 0
        pnl       = equity - 1000.0
        market_open = market_data.get("is_open", False) if isinstance(market_data, dict) else False

        _r1c1, _r1c2, _r1c3 = st.columns(3)
        _r2c1, _r2c2, _     = st.columns(3)
        _r1c1.metric("Portfolio Value", f"${equity:,.2f}",
                     delta=f"{pnl:+.2f} vs start",
                     delta_color="normal" if pnl >= 0 else "inverse")
        _r1c2.metric("Cash", f"${cash:,.2f}", f"{cash_pct:.0f}% of equity",
                     delta_color="off")
        _r1c3.metric("Positions", f"{n_pos}")
        _r2c1.metric("Market", "🟢 Open" if market_open else "🔴 Closed",
                     delta=market_data.get("next_close" if market_open else "next_open", "") if isinstance(market_data, dict) else "")
        _r2c2.metric("Open Orders", len(open_orders))

        if cash_pct < 25:
            st.warning(f"⚠️ Cash at {cash_pct:.1f}% — below 25% reserve. New buys blocked until cash is restored.")
    else:
        st.error("Cannot connect to Alpaca. Check `.env` credentials.")

    st.divider()

    # ── Open Positions ────────────────────────────────────────────────────────────

    st.subheader("Open Positions")

    if account_data and account_data.get("positions"):
        positions = account_data["positions"]
        rows = []
        for p in positions:
            sym   = p["symbol"]
            qty   = float(p["qty"])
            entry = float(p["avg_entry_price"])
            curr  = float(p["current_price"]) if p.get("current_price") else entry
            pl    = float(p.get("unrealized_pl", 0))
            pl_pct = float(p.get("unrealized_plpc", 0)) * 100
            mv    = float(p.get("market_value", qty * curr))

            lvl   = stop_levels.get(sym, {})
            stop  = lvl.get("stop")
            target = lvl.get("target")
            quality = compute_trade_quality(
                entry, stop, target, qty,
                account_value=account_data.get("account_value"),
                current=curr,
                market_data=market_data,
            )

            # Distance to stop/target
            stop_dist  = f"-{((curr-stop)/curr*100):.1f}%" if stop else "—"
            tgt_dist   = f"+{((target-curr)/curr*100):.1f}%" if target else "—"
            near_stop  = stop and curr <= stop * 1.02  # within 2% of stop
            risk_dollars = f"${quality['dollar_risk']:.2f}" if quality else "—"
            reward_dollars = f"${quality['dollar_reward']:.2f}" if quality else "—"
            rr_label = f"{quality['rr']:.2f}:1" if quality else "—"
            open_r = f"{quality['current_r']:+.2f}R" if quality and quality["current_r"] is not None else "—"
            acct_risk = f"{quality['account_risk_pct']:.2f}%" if quality and quality["account_risk_pct"] is not None else "—"
            trade_quality = f"{quality['grade']} ({quality['score']})" if quality else "Missing levels"

            rows.append({
                "Symbol":   sym,
                "Qty":      qty,
                "Avg Entry": f"${entry:.2f}",
                "Current":  f"${curr:.2f}",
                "Mkt Value": f"${mv:.2f}",
                "Stop":     f"${stop:.2f}" if stop else "—",
                "→ Stop":   stop_dist,
                "Target":   f"${target:.2f}" if target else "—",
                "→ Target": tgt_dist,
                "Risk $":   risk_dollars,
                "Reward $": reward_dollars,
                "R/R":      rr_label,
                "Open R":   open_r,
                "Acct Risk": acct_risk,
                "Quality":  trade_quality,
                "P&L":      f"${pl:+.2f}",
                "P&L %":    f"{pl_pct:+.2f}%",
                "_near_stop": near_stop,
            })

        df = pd.DataFrame(rows)

        # Warn about positions near their stop
        for row in rows:
            if row["_near_stop"]:
                st.warning(f"⚠️ {row['Symbol']} is within 2% of its stop loss ({row['Stop']})")

        display_df = df.drop(columns=["_near_stop"])

        def color_pl(val):
            s = str(val)
            if s.startswith("+") or (s.startswith("$+") ): return "color:#00d4aa"
            if "-" in s: return "color:#ff4b4b"
            return ""

        st.dataframe(
            display_df.style.map(color_pl, subset=["P&L", "P&L %", "→ Stop", "→ Target"]),
            width='stretch',
            hide_index=True
        )

        if len(positions) > 0:
            syms   = [p["symbol"] for p in positions]
            pl_vals = [float(p.get("unrealized_pl", 0)) for p in positions]
            colors  = ["#00d4aa" if v >= 0 else "#ff4b4b" for v in pl_vals]
            fig = go.Figure(go.Bar(
                x=syms, y=pl_vals, marker_color=colors,
                text=[f"${v:+.2f}" for v in pl_vals], textposition="outside"
            ))
            fig.update_layout(title="Unrealized P&L by Position", yaxis_title="USD",
                              height=260, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font_color="#fafafa", margin=dict(t=40,b=20))
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("No open positions.")

    # ── Open Orders ────────────────────────────────────────────────────────────────

    if open_orders:
        st.subheader("Open Orders (Alpaca)")
        st.dataframe(pd.DataFrame(open_orders), width='stretch', hide_index=True)

    st.divider()

    # ── Equity Curve ──────────────────────────────────────────────────────────────

    eq_df = build_equity_curve(load_journal_entries(30))
    if not eq_df.empty and len(eq_df) > 1:
        st.subheader("Equity Curve — Last 30 Days")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_df["time"], y=eq_df["equity"],
            mode="lines+markers", name="Equity",
            line=dict(color="#00d4aa", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.08)"
        ))
        fig.add_hline(y=1000, line_dash="dash", line_color="#a0aec0",
                      annotation_text="Start $1,000", annotation_position="bottom right")
        fig.update_layout(
            yaxis_title="USD", height=280,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#fafafa", margin=dict(t=20,b=20),
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig, width='stretch')
        st.divider()

    # ── Cycle Stats ───────────────────────────────────────────────────────────────

    st.subheader(f"Cycle Stats — Last {days_filter} Days")
    stats = compute_stats(journal_entries)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cycles Run",     stats["total_cycles"])
    c2.metric("Orders Submitted",  stats["submitted"],
              help="BUY/SELL decisions with execution_status=SUBMITTED")
    c3.metric("Rejected",       stats["rejected"],
              delta="⚠️ Review logs" if stats["rejected"] > 5 else None,
              delta_color="inverse")
    c4.metric("No-Trade Rate",  f"{stats['no_trade_rate']}%",
              help="Cycles with no submitted orders. Target: >50%")
    c5.metric("Dry-Run Orders", stats["dry_run"],
              help="Orders generated in dry-run mode (not submitted)")

    # Decision breakdown pie
    if stats["total_cycles"] > 0:
        decision_counts = {
            "Submitted": stats["submitted"],
            "Rejected":  stats["rejected"],
            "Dry-Run":   stats["dry_run"],
            "NO TRADE":  stats["no_trade"],
            "HOLD":      stats["hold"],
        }
        decision_counts = {k: v for k, v in decision_counts.items() if v > 0}
        if decision_counts:
            fig = px.pie(
                values=list(decision_counts.values()),
                names=list(decision_counts.keys()),
                color_discrete_map={
                    "Submitted": "#00d4aa", "Rejected": "#ff4b4b",
                    "Dry-Run": "#718096", "NO TRADE": "#a0aec0", "HOLD": "#f6ad55"
                },
                hole=0.45
            )
            fig.update_layout(height=260, paper_bgcolor="#0e1117",
                              font_color="#fafafa", margin=dict(t=20,b=20),
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, width='stretch')

with tab5:
    # ── Backtest ──────────────────────────────────────────────────────────────

    st.subheader("🧪 Backtest")
    st.caption(
        "Simulates the agent's BUY setups against historical price data so you can rack up "
        "practice reps any time — including when the market is closed. **Limitation:** only "
        "setups A (MA20 pullback) and B (oversold bounce) are tested — these are pure "
        "price-action rules with no news requirement. Setup C and the SHORT setups D/E need a "
        "real catalyst headline, and historical news isn't reliably available, so they're left "
        "out rather than faked."
    )

    _bt_c1, _bt_c2, _bt_c3, _bt_c4 = st.columns([2, 2, 2, 2])
    _bt_days = _bt_c1.number_input("Days back", min_value=20, max_value=500, value=180, step=10)
    _bt_lookahead = _bt_c2.number_input("Lookahead days", min_value=3, max_value=30, value=10)
    _bt_symbols = _bt_c3.text_input("Symbols (optional)", placeholder="e.g. NVDA,AMD,META")
    _bt_run = _bt_c4.button("▶ Run Backtest", width='stretch', type="primary")

    if _bt_run:
        _bt_cmd = [PYTHON, str(SCRIPTS_DIR / "backtest.py"),
                   "--days", str(int(_bt_days)), "--lookahead", str(int(_bt_lookahead))]
        if _bt_symbols.strip():
            _bt_cmd += ["--symbols", _bt_symbols.strip()]
        with st.spinner("Simulating historical trades… (10-60s depending on universe size)"):
            _bt_res = subprocess.run(_bt_cmd, capture_output=True, text=True, timeout=180)
        if _bt_res.returncode == 0:
            st.success("Backtest complete.")
        else:
            st.error(f"Backtest failed: {(_bt_res.stderr or _bt_res.stdout)[-800:]}")
        load_backtest_runs.clear()

    st.divider()

    _bt_runs = load_backtest_runs()
    if not _bt_runs:
        st.info("No backtest runs yet. Set your parameters above and click Run Backtest.")
    else:
        _bt_latest = _bt_runs[0]
        _bt_stats = _bt_latest["stats"]

        st.caption(
            f"Latest run: {_bt_latest['start']} → {_bt_latest['end']}  ·  "
            f"{_bt_latest['universe_size']} symbols  ·  {_bt_latest['lookahead_days']}d lookahead  ·  "
            f"run at {_bt_latest['run_at'][:16].replace('T', ' ')}"
        )

        _s1, _s2, _s3 = st.columns(3)
        _s4, _s5, _ = st.columns(3)
        _s1.metric("Simulated Trades", _bt_stats["total"])
        _s2.metric("Win Rate", f"{_bt_stats['win_rate']}%" if _bt_stats["total"] else "—")
        _s3.metric("Wins / Losses", f"{_bt_stats['wins']} / {_bt_stats['losses']}")
        _s4.metric("Avg Return", f"{_bt_stats['avg_return_pct']:+.2f}%" if _bt_stats["total"] else "—")
        _s5.metric("Total P&L", f"${_bt_stats['total_pnl']:+.2f}" if _bt_stats["total"] else "—")

        if _bt_latest["trades"]:
            _bt_rows = []
            _bt_icons = {"win": "✅", "loss": "❌", "breakeven": "➖"}
            for _t in reversed(_bt_latest["trades"][-50:]):
                _bt_rows.append({
                    "": _bt_icons.get(_t["outcome"], ""),
                    "Date": _t["date"],
                    "Ticker": _t["ticker"],
                    "Setup": _t["setup"],
                    "Entry": f"${_t['entry']:.2f}",
                    "Stop": f"${_t['stop']:.2f}",
                    "Target": f"${_t['target']:.2f}",
                    "Exit": _t["exit_reason"].replace("_", " "),
                    "Exit Date": _t["exit_date"],
                    "P&L": f"${_t['pnl']:+.2f}",
                    "Return": f"{_t['return_pct']:+.1f}%",
                })
            st.dataframe(pd.DataFrame(_bt_rows), width='stretch', hide_index=True)

        if len(_bt_runs) > 1:
            with st.expander(f"Previous runs ({len(_bt_runs) - 1})"):
                for _r in _bt_runs[1:6]:
                    _rs = _r["stats"]
                    st.caption(
                        f"{_r['run_at'][:16].replace('T',' ')} — {_r['start']} to {_r['end']} — "
                        f"{_rs['total']} trades, {_rs['win_rate']}% win rate, "
                        f"${_rs['total_pnl']:+.2f} P&L"
                    )

# ── Auto-refresh ──────────────────────────────────────────────────────────────

if st.session_state.get("_auto_refresh", False):
    import time
    time.sleep(60)
    st.rerun()

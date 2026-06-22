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

JOURNAL_DIR = Path(__file__).parent / "journal"
SCRIPTS_DIR = Path(__file__).parent / "scripts"
PYTHON = str(Path(__file__).parent / ".venv" / "bin" / "python")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Trading Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
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
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Controls")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    auto_refresh = st.toggle("Auto-refresh (60s)", value=False)

    st.divider()
    st.markdown("**Run Agent**")

    col_a, col_b = st.columns(2)
    run_live = col_a.button("▶ Live", use_container_width=True, help="Run a full trading cycle")
    run_dry  = col_b.button("🔍 Dry Run", use_container_width=True, help="Analyze without placing orders")

    if run_live or run_dry:
        flag = "" if run_live else "--dry-run"
        cmd = [PYTHON, str(SCRIPTS_DIR / "agent.py")] + ([flag] if flag else [])
        with st.spinner("Running cycle… (up to 3 min)"):
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            st.success("Cycle complete")
            st.text_area("Output", result.stdout[-3000:], height=200)
        else:
            st.error("Cycle failed")
            st.text_area("Error", (result.stderr or result.stdout)[-2000:], height=200)
        st.cache_data.clear()

    st.divider()
    days_filter = st.slider("Journal lookback (days)", 1, 30, 7)
    st.divider()
    st.caption("**Paper Trading Only** — simulated capital")

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_premarket_movers(min_pct: float = 0.8):
    """Fetch price change vs previous close for watchlist + common large-caps."""
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor

    wl_path = Path(__file__).parent / "data" / "watchlist.json"
    try:
        wl = json.loads(wl_path.read_text())
    except Exception:
        wl = {}

    priority = [t["symbol"] for t in wl.get("priority_tickers", [])]
    extra = ["AAPL", "AMZN", "NFLX", "MRVL", "ARM", "QCOM", "CRM", "PANW",
             "SNOW", "UBER", "SMCI", "ASML", "SHOP", "MSTR", "HOOD"]
    tickers = list(dict.fromkeys(priority + extra))[:25]

    def _fetch(tkr):
        try:
            fi = yf.Ticker(tkr).fast_info
            last = fi.last_price
            prev = fi.previous_close
            if not last or not prev or prev == 0:
                return None
            chg = round((last - prev) / prev * 100, 2)
            if abs(chg) < min_pct:
                return None
            return {"ticker": tkr, "price": round(last, 2),
                    "prev_close": round(prev, 2), "change_pct": chg, "headline": ""}
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=12) as ex:
        raw = list(ex.map(_fetch, tickers))

    movers = sorted([r for r in raw if r],
                    key=lambda x: abs(x["change_pct"]), reverse=True)[:12]

    # News for top movers via Alpaca
    try:
        import research as _res
        for m in movers[:8]:
            try:
                nd = _res.get_news(m["ticker"], hours=16)
                items = nd.get("items", [])
                if items:
                    m["headline"] = items[0].get("headline", "")
            except Exception:
                pass
    except Exception:
        pass

    return movers


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

# ── Symbol Analysis (top of page) ────────────────────────────────────────────

st.title("📈 Trading Agent")

# ── Pre-Market Movers ─────────────────────────────────────────────────────────

_pm_hdr, _pm_btn = st.columns([6, 1])
_pm_hdr.subheader("🌅 Pre-Market Movers")
if _pm_btn.button("🔄 Refresh", key="pm_refresh", use_container_width=True):
    load_premarket_movers.clear()
    st.rerun()

_pm_data = load_premarket_movers()

if _pm_data:
    _colors = ["#48bb78" if m["change_pct"] > 0 else "#fc8181" for m in _pm_data]
    _fig_pm  = go.Figure(go.Bar(
        y=[m["ticker"] for m in _pm_data],
        x=[m["change_pct"] for m in _pm_data],
        orientation="h",
        marker_color=_colors,
        text=[f"{m['change_pct']:+.1f}%" for m in _pm_data],
        textposition="outside",
        customdata=[[m["price"], m["prev_close"]] for m in _pm_data],
        hovertemplate="<b>%{y}</b><br>Price: $%{customdata[0]}<br>"
                      "Prev close: $%{customdata[1]}<br>Change: %{x:+.2f}%<extra></extra>",
    ))
    _fig_pm.update_layout(
        height=max(220, len(_pm_data) * 34),
        margin=dict(l=0, r=70, t=4, b=4),
        xaxis=dict(title="% vs prev close", zeroline=True,
                   zerolinecolor="#4a5568", zerolinewidth=1, ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", size=12),
        showlegend=False,
    )
    st.plotly_chart(_fig_pm, use_container_width=True,
                    config={"staticPlot": True})

    # Catalyst headlines
    _headlined = [m for m in _pm_data if m.get("headline")]
    if _headlined:
        st.caption("**Catalysts**")
        for m in _headlined:
            _clr = "green" if m["change_pct"] > 0 else "red"
            _tc, _pc, _hc = st.columns([1, 1, 9])
            _tc.markdown(f"**{m['ticker']}**")
            _pc.markdown(f":{_clr}[{m['change_pct']:+.1f}%]")
            _hc.caption(m["headline"])
    else:
        st.caption("No news catalysts found via Alpaca for current movers.")
else:
    st.caption("No significant movers (>0.8%) right now — check back closer to market open.")

# ── 5-min Chart + MACD ────────────────────────────────────────────────────────

if _pm_data:
    from plotly.subplots import make_subplots

    _ticker_options = [m["ticker"] for m in _pm_data]
    _selected = st.selectbox("📈 5-min chart + MACD", _ticker_options,
                             key="intraday_select")

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

        st.plotly_chart(_fig_intra, use_container_width=True,
                        config={"staticPlot": True})

        # MACD verdict
        if _has_macd:
            _sig = _macd.get("signal_str", "")
            _verdicts = {
                "bullish_crossover": ("🟢", "MACD bullish crossover — momentum just turned up"),
                "bearish_crossover": ("🔴", "MACD bearish crossover — momentum just turned down"),
                "bullish_trend":     ("🔵", "MACD above signal — uptrend in progress"),
                "bearish_trend":     ("⚪", "MACD below signal — downtrend in progress"),
            }
            _icon, _text = _verdicts.get(_sig, ("⚪", _sig))
            _zero = "above zero" if _macd.get("above_zero") else "below zero"
            st.caption(f"{_icon} **{_selected}** — {_text} · MACD {_zero}")
            if _src == "yfinance_delayed":
                st.caption("⚠️ Using yfinance fallback (15-min delay) — Alpaca returned no IEX data")
    else:
        st.caption(f"No intraday data available for {_selected} yet.")

st.markdown("---")

# ── Daily Options Play ────────────────────────────────────────────────────────

st.subheader("🎯 Today's Options Play")
st.caption("Catalyst-first daily scan — identifies the sector with the strongest news catalyst and generates a directional options play.")

if "daily_play" not in st.session_state:
    st.session_state.daily_play = None
if "daily_play_loading" not in st.session_state:
    st.session_state.daily_play_loading = False

scan_col, clear_col = st.columns([3, 1])
run_scan = scan_col.button("🔍 Scan for Today's Play", use_container_width=True, type="primary")
if clear_col.button("Clear", use_container_width=True):
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
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Supporting news
        news_hl = dp.get("news_headlines", [])
        if news_hl:
            with st.expander(f"📰 News used in scan ({len(news_hl)} headlines)"):
                for h in news_hl:
                    st.markdown(f"<span style='color:#a0aec0;font-size:0.9em'>• {h}</span>",
                                unsafe_allow_html=True)

st.markdown("---")
st.subheader("🔍 Symbol Analysis")
st.caption("Search any ticker for an AI trade setup and options strategy.")

col_sym, col_btn = st.columns([4, 1])
symbol_input = col_sym.text_input(
    "ticker", placeholder="e.g. AAPL  ·  NVDA  ·  MSFT  ·  SPY",
    label_visibility="collapsed"
).strip().upper()
analyze_clicked = col_btn.button("Analyze ▶", use_container_width=True, type="primary")

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

st.markdown("---")

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
    daytrades = account_data.get("daytrade_count", 0)
    cash_pct  = (cash / equity * 100) if equity else 0
    pnl       = equity - 1000.0
    market_open = market_data.get("is_open", False) if isinstance(market_data, dict) else False

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Portfolio Value", f"${equity:,.2f}",
                delta=f"{pnl:+.2f} vs start",
                delta_color="normal" if pnl >= 0 else "inverse")
    col2.metric("Cash", f"${cash:,.2f}", f"{cash_pct:.0f}% of equity",
                delta_color="off")
    col3.metric("Positions", f"{n_pos} / 5")
    col4.metric("Day Trades (5d)", f"{daytrades} / 3",
                delta="⚠️ Near PDT limit" if daytrades >= 2 else None,
                delta_color="inverse" if daytrades >= 2 else "normal")
    col5.metric("Market", "🟢 Open" if market_open else "🔴 Closed",
                delta=market_data.get("next_close" if market_open else "next_open", "") if isinstance(market_data, dict) else "")
    col6.metric("Open Orders", len(open_orders))

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
        display_df.style.applymap(color_pl, subset=["P&L", "P&L %", "→ Stop", "→ Target"]),
        use_container_width=True,
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
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No open positions.")

# ── Open Orders ────────────────────────────────────────────────────────────────

if open_orders:
    st.subheader("Open Orders (Alpaca)")
    st.dataframe(pd.DataFrame(open_orders), use_container_width=True, hide_index=True)

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
    st.plotly_chart(fig, use_container_width=True)
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
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Trade History ─────────────────────────────────────────────────────────────

st.subheader("Trade History")

trades = extract_trades(journal_entries)
if trades:
    trade_df = pd.DataFrame(trades)

    def color_status(val):
        mapping = {
            "SUBMITTED":     "color:#00d4aa;font-weight:600",
            "REJECTED":      "color:#ff4b4b",
            "ERROR":         "color:#ff4b4b",
            "DRY_RUN":       "color:#718096",
            "DRY_RUN_REJECTED": "color:#ff4b4b",
            "NOT_SUBMITTED": "color:#718096",
        }
        return mapping.get(str(val), "")

    def color_action(val):
        return "color:#00d4aa;font-weight:600" if val == "BUY" else "color:#ff4b4b;font-weight:600"

    styled = trade_df.style \
        .applymap(color_status, subset=["Status"]) \
        .applymap(color_action, subset=["Action"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Rejection detail expander
    rejected = [t for t in trades if t["Status"] in ("REJECTED", "ERROR")]
    if rejected:
        with st.expander(f"⛔ {len(rejected)} rejection(s) — expand for details"):
            for r in rejected:
                st.markdown(f"**{r['Time']} — {r['Action']} {r['Ticker']}**: {r['Detail'] or 'no detail'}")
else:
    st.info("No trades in the selected period.")

st.divider()

# ── Journal Viewer ────────────────────────────────────────────────────────────

st.subheader("Journal Entries")

if journal_entries:
    for entry in journal_entries[:10]:
        ts       = entry.get("_written_at", entry.get("cycle_timestamp", ""))[:16].replace("T", " ")
        dry_tag  = " 🔍" if entry.get("dry_run") else ""
        decisions = entry.get("decisions", [])
        submitted_d = [d for d in decisions if d.get("execution_status") == "SUBMITTED"]
        rejected_d  = [d for d in decisions if d.get("execution_status") in ("REJECTED","ERROR")]
        no_trade_d  = [d for d in decisions if d.get("action") == "NO TRADE"]

        summary = f"{len(submitted_d)} submitted · {len(rejected_d)} rejected · {len(no_trade_d)} no-trade"
        label   = f"📋 {ts}{dry_tag}  —  {summary}"

        with st.expander(label, expanded=False):
            # Market context — parse the nested structure from get_market_snapshot()
            ctx = entry.get("market_context", {})
            indices = ctx.get("indices", {})
            spy = indices.get("SPY", {})
            qqq = indices.get("QQQ", {})
            iwm = indices.get("IWM", {})
            if spy or qqq:
                cols = st.columns(4)
                cols[0].metric("SPY 5d", f"{spy.get('5d_change_pct', 0):+.2f}%" if spy.get('5d_change_pct') is not None else "—")
                cols[1].metric("QQQ 5d", f"{qqq.get('5d_change_pct', 0):+.2f}%" if qqq.get('5d_change_pct') is not None else "—")
                cols[2].metric("IWM 5d", f"{iwm.get('5d_change_pct', 0):+.2f}%" if iwm.get('5d_change_pct') is not None else "—")
                market_status = ctx.get("market_status", {})
                cols[3].metric("Market", "🟢 Open" if market_status.get("is_open") else "🔴 Closed")

            # Decisions
            if decisions:
                st.markdown("**Decisions**")
                for d in decisions:
                    action = d.get("action", "")
                    ticker = d.get("ticker", "")
                    ex     = d.get("execution_status", "")
                    icon   = {"BUY":"🟢","SELL":"🔴","HOLD":"🟡","NO TRADE":"⬜"}.get(action, "")
                    status_badge = {
                        "SUBMITTED":     "✅",
                        "REJECTED":      "❌",
                        "ERROR":         "❌",
                        "DRY_RUN":       "🔍",
                        "DRY_RUN_REJECTED": "❌",
                        "NOT_SUBMITTED": "⏭",
                        "NO_TRADE":      "",
                        "HOLD":          "",
                    }.get(ex, "")

                    line = f"{icon} **{action}** {ticker} {status_badge}"
                    if action == "BUY":
                        line += (f" @ ${d.get('entry_limit','')} "
                                 f"| stop ${d.get('stop_loss','')} "
                                 f"| target ${d.get('take_profit','')} "
                                 f"| {d.get('confidence','')}")
                        if d.get("execution_detail"):
                            line += f"  ⚠️ _{d['execution_detail']}_"
                    elif action in ("SELL", "NO TRADE"):
                        line += f" — {d.get('reason', '')}"
                    st.markdown(line)
                    if d.get("thesis"):
                        st.caption(d["thesis"])

            # Reflection
            if entry.get("reflection"):
                st.markdown("**Reflection**")
                st.caption(entry["reflection"])

            with st.expander("Raw JSON"):
                st.json(entry)
else:
    st.info(f"No journal entries in the last {days_filter} days.")

# ── Options Outcome Tracker ───────────────────────────────────────────────────

st.markdown("---")
st.subheader("📊 Options Outcome Tracker")

if not _OUTCOMES_AVAILABLE:
    st.warning("outcomes.py could not be loaded — run from the trading-agent directory.")
else:
    col_ref, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button("🔄 Check Expired", use_container_width=True):
            updated = _outcomes_mod.update_expired_trades()
            if updated:
                st.success(f"Updated {len(updated)} expired trade(s).")
            else:
                st.info("No newly expired trades.")

    perf = _outcomes_mod.get_performance_stats()
    total      = perf["total"]
    open_cnt   = perf["open"]
    win_rate   = perf["win_rate"]
    avg_ret    = perf["avg_return"]
    total_pnl  = perf["total_pnl"]
    all_trades = perf["all_trades"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Completed", total)
    m2.metric("Open", open_cnt)
    m3.metric("Win Rate", f"{win_rate}%" if total else "—")
    delta_color_ret = "normal" if avg_ret >= 0 else "inverse"
    m4.metric("Avg Return", f"{avg_ret:+.1f}%" if total else "—")
    delta_color_pnl = "normal" if total_pnl >= 0 else "inverse"
    m5.metric("Total P&L", f"${total_pnl:+.0f}" if total else "—")

    if all_trades:
        rows = []
        outcome_icons = {"win": "✅", "loss": "❌", "breakeven": "➖", "pending": "⏳"}
        status_icons  = {"open": "🔵", "expired": ""}
        for t in all_trades[:20]:
            outcome = t.get("outcome") or ""
            status  = t.get("status", "open")
            icon    = outcome_icons.get(outcome, status_icons.get(status, "🔵"))
            pnl     = t.get("profit_loss")
            ret     = t.get("return_pct")
            rows.append({
                "":          icon,
                "Date":      t.get("date", ""),
                "Ticker":    t.get("ticker", ""),
                "Strategy":  t.get("strategy", "").replace("_", " ").title(),
                "Dir":       t.get("direction", "").capitalize(),
                "Conf":      t.get("confidence", ""),
                "Entry $":   f"${t['entry_cost']:.0f}" if t.get("entry_cost") else "—",
                "Expiry":    t.get("expiration", ""),
                "P&L":       f"${pnl:+.0f}" if pnl is not None else "—",
                "Return":    f"{ret:+.1f}%" if ret is not None else "—",
                "Source":    t.get("source", "").replace("_", " "),
            })
        df_trades = pd.DataFrame(rows)
        st.dataframe(df_trades, use_container_width=True, hide_index=True)
    elif open_cnt == 0:
        st.info("No trades recorded yet. Analyze a symbol to start tracking.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────

if auto_refresh:
    import time
    time.sleep(60)
    st.rerun()

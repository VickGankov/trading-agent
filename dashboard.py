#!/usr/bin/env python3
"""
dashboard.py - Streamlit trading dashboard.

Usage:
    streamlit run dashboard.py
"""

import os
import sys
import json
import glob
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

load_dotenv()

JOURNAL_DIR = Path(__file__).parent / "journal"

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Trading Agent Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2d3250;
    }
    .positive { color: #00d4aa; }
    .negative { color: #ff4b4b; }
    .neutral  { color: #a0aec0; }
    [data-testid="stMetricDelta"] { font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Controls")
    auto_refresh = st.toggle("Auto-refresh (60s)", value=False)
    if auto_refresh:
        st.caption("Page refreshes every 60 seconds")

    st.divider()
    st.caption("**Paper Trading Only**")
    st.caption("This account uses simulated capital.")
    st.divider()

    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    days_filter = st.slider("Journal lookback (days)", 1, 30, 7)

# ── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_alpaca_data():
    """Load account + positions from Alpaca. Returns None if unavailable."""
    try:
        import research
        account = research.get_account()
        market = research.is_market_open()
        return account, market
    except Exception as e:
        return None, {"error": str(e)}


@st.cache_data(ttl=30)
def load_journal_entries(days: int = 7):
    """Read journal JSON files from the last N days."""
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


def extract_trades_from_journal(entries):
    """Pull all BUY/SELL decisions from journal entries into a flat list."""
    trades = []
    for entry in entries:
        ts = entry.get("_written_at", "")
        for d in entry.get("decisions", []):
            action = d.get("action", "")
            if action in ("BUY", "SELL"):
                trades.append({
                    "timestamp": ts[:16].replace("T", " "),
                    "action": action,
                    "ticker": d.get("ticker", ""),
                    "qty": d.get("qty", ""),
                    "entry": d.get("entry_limit", d.get("entry", "")),
                    "stop": d.get("stop_loss", ""),
                    "target": d.get("take_profit", ""),
                    "confidence": d.get("confidence", ""),
                    "status": d.get("status", "SUBMITTED"),
                    "thesis": d.get("thesis", d.get("reason", ""))
                })
    return trades


def compute_journal_stats(entries):
    """Aggregate stats across journal entries."""
    total_cycles = len(entries)
    buy_count = sell_count = hold_count = no_trade_count = rejection_count = 0

    for e in entries:
        for d in e.get("decisions", []):
            a = d.get("action", "NO TRADE")
            if a == "BUY":
                buy_count += 1
            elif a == "SELL":
                sell_count += 1
            elif a == "HOLD":
                hold_count += 1
            else:
                no_trade_count += 1
        rejection_count += len(e.get("rejections", []))

    no_trade_cycles = sum(
        1 for e in entries
        if not any(d.get("action") in ("BUY", "SELL") for d in e.get("decisions", []))
    )

    return {
        "total_cycles": total_cycles,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "no_trade_count": no_trade_count,
        "no_trade_cycles": no_trade_cycles,
        "rejection_count": rejection_count,
        "no_trade_rate": round(no_trade_cycles / total_cycles * 100, 1) if total_cycles else 0
    }


# ── Main layout ───────────────────────────────────────────────────────────────

st.title("📈 Trading Agent Dashboard")
st.caption(f"Paper trading account · Last updated {datetime.now().strftime('%H:%M:%S')}")

account_data, market_data = load_alpaca_data()
journal_entries = load_journal_entries(days_filter)

# ── Row 1: Account metrics ────────────────────────────────────────────────────

st.subheader("Account Overview")
col1, col2, col3, col4, col5 = st.columns(5)

if account_data:
    equity = account_data.get("account_value", 0)
    cash = account_data.get("cash", 0)
    positions_count = account_data.get("positions_count", 0)
    daytrades = account_data.get("daytrade_count", 0)
    cash_pct = (cash / equity * 100) if equity else 0

    col1.metric("Portfolio Value", f"${equity:,.2f}")
    col2.metric("Cash Available", f"${cash:,.2f}", f"{cash_pct:.1f}% of equity")
    col3.metric("Open Positions", f"{positions_count} / 5")
    col4.metric("Day Trades (5d)", f"{daytrades} / 3",
                delta="⚠️ PDT limit near" if daytrades >= 2 else None,
                delta_color="inverse" if daytrades >= 2 else "normal")

    market_open = market_data.get("is_open", False) if isinstance(market_data, dict) else False
    col5.metric(
        "Market Status",
        "🟢 Open" if market_open else "🔴 Closed",
        delta=market_data.get("next_close", market_data.get("next_open", "")) if isinstance(market_data, dict) else ""
    )

    # Cash reserve warning
    if cash_pct < 25:
        st.warning(f"⚠️ Cash reserve at {cash_pct:.1f}% — minimum required is 25%. No new buys until cash is restored.")
else:
    for col in [col1, col2, col3, col4, col5]:
        col.metric("—", "N/A")
    st.error("Could not connect to Alpaca. Check your `.env` credentials and that alpaca-py is installed.")

st.divider()

# ── Row 2: Open Positions ─────────────────────────────────────────────────────

st.subheader("Open Positions")

if account_data and account_data.get("positions"):
    positions = account_data["positions"]
    rows = []
    for p in positions:
        pl = p.get("unrealized_pl", 0)
        pl_pct = p.get("unrealized_plpc", 0) * 100
        rows.append({
            "Symbol": p["symbol"],
            "Qty": int(p["qty"]),
            "Avg Entry": f"${p['avg_entry_price']:.2f}",
            "Current": f"${p['current_price']:.2f}" if p.get("current_price") else "—",
            "Market Value": f"${p.get('market_value', 0):.2f}",
            "Unrealized P&L": f"${pl:+.2f}",
            "P&L %": f"{pl_pct:+.2f}%",
        })

    df = pd.DataFrame(rows)

    def color_pl(val):
        if "+" in str(val):
            return "color: #00d4aa"
        elif "-" in str(val):
            return "color: #ff4b4b"
        return ""

    st.dataframe(
        df.style.applymap(color_pl, subset=["Unrealized P&L", "P&L %"]),
        use_container_width=True,
        hide_index=True
    )

    # Position P&L bar chart
    if len(positions) > 0:
        symbols = [p["symbol"] for p in positions]
        pl_values = [p.get("unrealized_pl", 0) for p in positions]
        colors = ["#00d4aa" if v >= 0 else "#ff4b4b" for v in pl_values]
        fig = go.Figure(go.Bar(x=symbols, y=pl_values, marker_color=colors, text=[f"${v:+.2f}" for v in pl_values], textposition="outside"))
        fig.update_layout(title="Unrealized P&L by Position", yaxis_title="USD", height=280,
                          plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#fafafa",
                          margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No open positions.")

st.divider()

# ── Row 3: Activity Stats ─────────────────────────────────────────────────────

st.subheader(f"Cycle Stats — Last {days_filter} Days")

stats = compute_journal_stats(journal_entries)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Cycles Run", stats["total_cycles"])
c2.metric("Buys Placed", stats["buy_count"])
c3.metric("Sells Placed", stats["sell_count"])
c4.metric("No-Trade Rate", f"{stats['no_trade_rate']}%",
          help="Target: >50%. High rate = disciplined.")
c5.metric("Validation Rejections", stats["rejection_count"],
          delta="⚠️ Review CLAUDE.md" if stats["rejection_count"] > 5 else None,
          delta_color="inverse")

if stats["total_cycles"] > 0:
    decision_data = {
        "BUY": stats["buy_count"],
        "SELL": stats["sell_count"],
        "HOLD": stats["hold_count"],
        "NO TRADE": stats["no_trade_count"]
    }
    fig = px.pie(
        values=list(decision_data.values()),
        names=list(decision_data.keys()),
        color_discrete_map={"BUY": "#00d4aa", "SELL": "#ff4b4b", "HOLD": "#f6ad55", "NO TRADE": "#a0aec0"},
        hole=0.4
    )
    fig.update_layout(height=260, paper_bgcolor="#0e1117", font_color="#fafafa",
                      margin=dict(t=20, b=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Row 4: Trade History ──────────────────────────────────────────────────────

st.subheader("Trade History")

trades = extract_trades_from_journal(journal_entries)
if trades:
    trade_df = pd.DataFrame(trades)
    st.dataframe(trade_df, use_container_width=True, hide_index=True)
else:
    st.info("No trades in journal yet. Trades appear here after the agent places orders.")

st.divider()

# ── Row 5: Journal Viewer ─────────────────────────────────────────────────────

st.subheader("Journal Entries")

if journal_entries:
    for entry in journal_entries[:10]:
        ts = entry.get("_written_at", entry.get("cycle_timestamp", ""))[:16].replace("T", " ")
        filename = entry.get("_filename", "entry")

        decisions = entry.get("decisions", [])
        actions = [d.get("action", "") for d in decisions]
        summary = f"{actions.count('BUY')} buys · {actions.count('SELL')} sells · {actions.count('NO TRADE')} no-trades"

        with st.expander(f"📋 {ts}  —  {summary}", expanded=False):
            # Market context block
            ctx = entry.get("market_context", {})
            if ctx:
                cols = st.columns(3)
                cols[0].metric("SPY", f"{ctx.get('spy_change_pct', 0):+.2f}%")
                cols[1].metric("QQQ", f"{ctx.get('qqq_change_pct', 0):+.2f}%")
                cols[2].metric("VIX", ctx.get("vix", "—"))

            # Decisions
            if decisions:
                st.markdown("**Decisions:**")
                for d in decisions:
                    action = d.get("action", "")
                    ticker = d.get("ticker", "")
                    icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "NO TRADE": "⬜"}.get(action, "")
                    label = f"{icon} **{action}** {ticker}"
                    if action == "BUY":
                        label += f" @ ${d.get('entry_limit', '')} | stop ${d.get('stop_loss', '')} | target ${d.get('take_profit', '')} | {d.get('confidence', '')}"
                    elif action == "SELL":
                        label += f" — {d.get('reason', '')}"
                    elif action == "NO TRADE":
                        label += f" — {d.get('reason', '')}"
                    st.markdown(label)
                    if d.get("thesis"):
                        st.caption(d["thesis"])

            # Reflection
            reflection = entry.get("reflection", "")
            if reflection:
                st.markdown("**Reflection:**")
                st.caption(reflection)

            # Raw JSON toggle
            with st.expander("Raw JSON"):
                st.json(entry)
else:
    st.info(f"No journal entries in the last {days_filter} days. Run the agent to generate entries.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────

if auto_refresh:
    import time
    time.sleep(60)
    st.rerun()

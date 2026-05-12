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
    .metric-card { background:#1e2130; border-radius:8px; padding:16px; border:1px solid #2d3250; }
    .positive { color:#00d4aa; font-weight:600; }
    .negative { color:#ff4b4b; font-weight:600; }
    .neutral  { color:#a0aec0; }
    .status-submitted { color:#00d4aa; font-weight:600; }
    .status-rejected  { color:#ff4b4b; }
    .status-dry-run   { color:#a0aec0; }
    [data-testid="stMetricDelta"] { font-size:0.85em; }
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

@st.cache_data(ttl=60)
def load_alpaca_data():
    try:
        import research
        account = research.get_account()
        market  = research.is_market_open()
        orders  = research.trading.get_orders()
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
                elif ex in ("REJECTED", "ERROR"):
                    rejected += 1
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
stop_levels = get_stop_target_from_journal(
    load_journal_entries(30)  # look back 30 days for stop levels regardless of filter
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("📈 Trading Agent")

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

        # Distance to stop/target
        stop_dist  = f"-{((curr-stop)/curr*100):.1f}%" if stop else "—"
        tgt_dist   = f"+{((target-curr)/curr*100):.1f}%" if target else "—"
        near_stop  = stop and curr <= stop * 1.02  # within 2% of stop

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
c2.metric("Orders Filled",  stats["submitted"],
          help="BUY/SELL decisions with execution_status=SUBMITTED")
c3.metric("Rejected",       stats["rejected"],
          delta="⚠️ Review logs" if stats["rejected"] > 5 else None,
          delta_color="inverse")
c4.metric("No-Trade Rate",  f"{stats['no_trade_rate']}%",
          help="Cycles with no submitted orders. Target: >50%")
c5.metric("Dry-Run Cycles", stats["dry_run"],
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

# ── Auto-refresh ──────────────────────────────────────────────────────────────

if auto_refresh:
    import time
    time.sleep(60)
    st.rerun()

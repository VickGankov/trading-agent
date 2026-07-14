#!/usr/bin/env python3
"""
backtest.py - simulate the agent's BUY logic against historical price data.

Runs anytime, including when the market is closed, since it only needs
historical bars (yfinance), not live quotes.

IMPORTANT LIMITATION: only simulates setups A (MA20 pullback) and B (oversold
bounce) — pure price-action setups with no news requirement. Setup C (news
catalyst) and the SHORT setups D/E (which require a bearish catalyst) are
NOT simulated here, because there's no reliable historical headline feed to
gate them on. Running them without the catalyst gate would make the backtest
diverge from how the live agent actually decides — it would fire on setups
that never would have triggered live. Better to under-simulate than to
fabricate a result.

Each simulated BUY uses the exact same entry/stop/target formula as the live
Groq cycle (scripts/agent.py classify_setup + _run_groq_cycle): entry =
price + 0.1%, stop = entry - 5%, target = entry + 10%, qty sized to $100 max
(10% of the $1000 paper account, matching trade.py's MAX_POSITION_PCT).

Usage:
    python scripts/backtest.py --days 60
    python scripts/backtest.py --start 2025-01-01 --end 2025-06-01
    python scripts/backtest.py --days 90 --lookahead 12 --symbols NVDA,AMD,META
"""

import argparse
import json
import math
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_FILE = DATA_DIR / "backtest_runs.json"
MAX_STORED_RUNS = 20

# Mirrors research.screener_movers() universe + watchlist priority tickers
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "AVGO",
    "PLTR", "COIN", "SMCI", "CRWD", "ZS", "PANW", "FTNT",
    "JPM", "BAC", "GS", "MS", "C",
    "XOM", "CVX", "OXY",
    "CEG", "VST", "NEE",
    "DIS", "NFLX", "UBER", "SHOP", "XYZ",
    "BA", "LMT", "RTX", "GE",
    "WMT", "COST", "TGT", "HD",
    "MCD", "SBUX",
    "JNJ", "PFE", "LLY", "UNH",
    "KO", "PEP", "PG",
    "TSM",
]

# Sync with trade.py's live sizing (capped bankroll × max position %) so
# simulated stats stay comparable to what the live agent would actually do.
try:
    from trade import ACCOUNT_CAP_USD as ACCOUNT_VALUE, MAX_POSITION_PCT
    MAX_POSITION_USD = ACCOUNT_VALUE * (MAX_POSITION_PCT / 100.0)
except Exception:
    ACCOUNT_VALUE = 10000.0
    MAX_POSITION_USD = 2000.0


def classify_setup(price: float, ma20, ma50, rsi) -> str:
    """A/B only — mirrors agent.py's classify_setup() minus the news-gated setup C."""
    if rsi is None or ma50 is None:
        return "NO_SETUP"
    if rsi > 65:
        return "REJECT:RSI>65"
    if price < ma50:
        return "REJECT:belowMA50"

    setups = []
    if ma20:
        pct_above_ma20 = (price - ma20) / ma20 * 100
        if 0 <= pct_above_ma20 <= 4 and 38 <= rsi <= 60:
            setups.append("A")
    if rsi < 42:
        setups.append("B")
    return "/".join(setups) if setups else "NO_SETUP"


def calc_technicals_asof(df: pd.DataFrame, idx: int) -> dict:
    """MA20/MA50/RSI14 using only rows up to and including idx — no lookahead."""
    window = df.iloc[max(0, idx - 89): idx + 1]
    if len(window) < 50:
        return {}
    close = window["Close"]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    return {
        "ma20": None if pd.isna(ma20) else float(ma20),
        "ma50": None if pd.isna(ma50) else float(ma50),
        "rsi14": None if pd.isna(rsi) else float(rsi),
    }


def simulate_outcome(df: pd.DataFrame, entry_idx: int, entry: float, stop: float,
                      target: float, qty: float, lookahead: int):
    """Walk forward day-by-day; whichever of stop/target is breached first wins.
    Returns None if there isn't enough forward data yet (trade too recent)."""
    future = df.iloc[entry_idx + 1: entry_idx + 1 + lookahead]
    if len(future) == 0:
        return None

    for date, row in future.iterrows():
        if row["Low"] <= stop:
            pnl = (stop - entry) * qty
            return {"outcome": "loss", "exit_reason": "stop_hit",
                    "exit_date": date.strftime("%Y-%m-%d"), "exit_price": round(stop, 2),
                    "pnl": round(pnl, 2), "return_pct": round(pnl / (entry * qty) * 100, 2)}
        if row["High"] >= target:
            pnl = (target - entry) * qty
            return {"outcome": "win", "exit_reason": "target_hit",
                    "exit_date": date.strftime("%Y-%m-%d"), "exit_price": round(target, 2),
                    "pnl": round(pnl, 2), "return_pct": round(pnl / (entry * qty) * 100, 2)}

    last_close = float(future.iloc[-1]["Close"])
    pnl = (last_close - entry) * qty
    outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")
    return {"outcome": outcome, "exit_reason": "timeout",
            "exit_date": future.index[-1].strftime("%Y-%m-%d"), "exit_price": round(last_close, 2),
            "pnl": round(pnl, 2), "return_pct": round(pnl / (entry * qty) * 100, 2)}


def run_backtest(start: str, end: str, lookahead: int, symbols: list) -> dict:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    fetch_start = (start_dt - timedelta(days=130)).strftime("%Y-%m-%d")
    fetch_end = (end_dt + timedelta(days=lookahead * 2 + 5)).strftime("%Y-%m-%d")

    print(f"Fetching {len(symbols)} symbols, {fetch_start} -> {fetch_end} ...")
    raw = yf.download(symbols, start=fetch_start, end=fetch_end, group_by="ticker",
                       progress=False, threads=True, auto_adjust=True)

    trades = []
    for sym in symbols:
        try:
            df = raw[sym].dropna(how="all")
        except Exception:
            continue
        if df.empty or len(df) < 60:
            continue

        mask = (df.index >= start_dt) & (df.index <= end_dt)
        blocked_until = None  # mirrors live agent: never buy a symbol already held
        for sim_date in df.index[mask]:
            if blocked_until is not None and sim_date <= blocked_until:
                continue
            idx = df.index.get_loc(sim_date)
            if idx < 50:
                continue
            price = float(df.iloc[idx]["Close"])
            tech = calc_technicals_asof(df, idx)
            if not tech:
                continue
            setup = classify_setup(price, tech.get("ma20"), tech.get("ma50"), tech.get("rsi14"))
            if setup == "NO_SETUP" or setup.startswith("REJECT"):
                continue

            entry = round(price * 1.001, 2)
            stop = round(entry * 0.95, 2)
            target = round(entry * 1.10, 2)
            qty = math.floor((MAX_POSITION_USD / entry) * 100) / 100
            if qty <= 0:
                continue

            result = simulate_outcome(df, idx, entry, stop, target, qty, lookahead)
            if result is None:
                continue

            blocked_until = pd.Timestamp(result["exit_date"])
            trades.append({
                "date": sim_date.strftime("%Y-%m-%d"),
                "ticker": sym,
                "setup": setup,
                "entry": entry, "stop": stop, "target": target, "qty": qty,
                **result,
            })

    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    decided = wins + losses
    win_rate = round(len(wins) / len(decided) * 100, 1) if decided else 0.0
    avg_return = round(sum(t["return_pct"] for t in trades) / len(trades), 2) if trades else 0.0
    total_pnl = round(sum(t["pnl"] for t in trades), 2)

    return {
        "run_id": str(uuid.uuid4())[:8],
        "run_at": datetime.now().isoformat(),
        "start": start, "end": end, "lookahead_days": lookahead,
        "universe_size": len(symbols),
        "stats": {
            "total": len(trades), "wins": len(wins), "losses": len(losses),
            "timeouts": len([t for t in trades if t["exit_reason"] == "timeout"]),
            "win_rate": win_rate, "avg_return_pct": avg_return, "total_pnl": total_pnl,
        },
        "trades": sorted(trades, key=lambda t: t["date"]),
    }


def save_run(run: dict):
    DATA_DIR.mkdir(exist_ok=True)
    runs = []
    if RESULTS_FILE.exists():
        try:
            runs = json.loads(RESULTS_FILE.read_text())
        except Exception:
            runs = []
    runs.insert(0, run)
    RESULTS_FILE.write_text(json.dumps(runs[:MAX_STORED_RUNS], indent=2))


def main():
    ap = argparse.ArgumentParser(description="Backtest BUY setups A/B against historical data")
    ap.add_argument("--days", type=int, default=60, help="trading days to simulate, ending yesterday")
    ap.add_argument("--start", type=str, help="YYYY-MM-DD (overrides --days)")
    ap.add_argument("--end", type=str, help="YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--lookahead", type=int, default=10, help="trading days to check for stop/target hit")
    ap.add_argument("--symbols", type=str, help="comma-separated tickers (default: built-in universe)")
    args = ap.parse_args()

    end_dt = datetime.strptime(args.end, "%Y-%m-%d") if args.end else (datetime.now() - timedelta(days=1))
    start_dt = (datetime.strptime(args.start, "%Y-%m-%d") if args.start
                else end_dt - timedelta(days=int(args.days * 1.45)))

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else UNIVERSE

    run = run_backtest(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"),
                        args.lookahead, symbols)
    save_run(run)

    s = run["stats"]
    print(f"\n{'=' * 50}")
    print(f"Backtest {run['start']} -> {run['end']}  ({len(symbols)} symbols, {args.lookahead}d lookahead)")
    print(f"{'=' * 50}")
    print(f"Trades: {s['total']}  |  Wins: {s['wins']}  Losses: {s['losses']}  Timeouts: {s['timeouts']}")
    print(f"Win rate: {s['win_rate']}%  |  Avg return: {s['avg_return_pct']:+.2f}%  |  Total P&L: ${s['total_pnl']:+.2f}")


if __name__ == "__main__":
    main()

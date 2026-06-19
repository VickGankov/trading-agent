#!/usr/bin/env python3
"""
outcomes.py - Options recommendation outcome tracker.

Every recommendation is saved to data/outcomes.json.
Expired trades are updated with actual P&L once the expiration date passes.

Usage:
    python scripts/outcomes.py          # update expired trades, print summary
    python scripts/outcomes.py --stats  # print performance stats only
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "outcomes.json"


# ── File I/O ──────────────────────────────────────────────────────────────────

def _load() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return {"trades": []}


def _save(data: dict):
    DATA_FILE.parent.mkdir(exist_ok=True)
    DATA_FILE.write_text(json.dumps(data, indent=2))


# ── Core helpers ──────────────────────────────────────────────────────────────

def _confidence_num(conf: str) -> float:
    return {"HIGH": 8.0, "MEDIUM": 6.0, "LOW": 4.0}.get(str(conf).upper(), 5.0)


def _spread_value_at_expiry(strategy: str, atm: float, otm: float, stock: float) -> float:
    """
    Intrinsic value of the spread × 100 at options expiration.

    For debit spreads this is what you receive (exit value).
    For credit spreads this is the cost to close (what you owe).

    Bull Call Spread: long atm call, short otm call (atm < otm)
    Bear Put Spread:  long atm put,  short otm put  (atm > otm)
    Bull Put Spread:  short atm put, long otm put   (atm > otm, credit)
    Bear Call Spread: short atm call, long otm call (atm < otm, credit)
    """
    if strategy == "bull_call_spread":
        return max(0.0, min(stock - atm, otm - atm)) * 100
    if strategy == "bear_put_spread":
        return max(0.0, min(atm - stock, atm - otm)) * 100
    if strategy == "bull_put_spread":
        return max(0.0, min(atm - stock, atm - otm)) * 100
    if strategy == "bear_call_spread":
        return max(0.0, min(stock - atm, otm - atm)) * 100
    return 0.0   # iron_condor: complex — P&L left as None until implemented


# ── Public API ────────────────────────────────────────────────────────────────

def save_recommendation(ticker: str, play: dict,
                        catalyst: str = "", confidence: str = "",
                        source: str = "symbol_analysis"):
    """
    Save one options play to outcomes.json.
    Deduplicates: same ticker + date + strategy + strikes won't be saved twice.
    Call after _build_options_structure() returns a valid play dict.
    """
    stype     = play.get("strategy_type", "")
    atm       = play.get("atm_strike")
    otm       = play.get("otm_strike")
    exp_date  = play.get("expiry_date", "")
    is_credit = play.get("is_credit", False)

    if not all([stype, atm, otm, exp_date, ticker]):
        return  # Incomplete play — skip silently

    today    = date.today().isoformat()
    trade_id = f"{ticker}-{today}-{stype}-{atm}-{otm}-{exp_date}"

    data = _load()
    if any(t["id"] == trade_id for t in data["trades"]):
        return  # Already recorded today

    credit = play.get("credit_per_contract") if is_credit else None

    record = {
        "id":                trade_id,
        "ticker":            ticker.upper(),
        "date":              today,
        "catalyst":          catalyst,
        "strategy":          stype,
        "direction":         ("bullish" if "bull" in stype
                              else ("bearish" if "bear" in stype else "neutral")),
        "atm_strike":        atm,
        "otm_strike":        otm,
        "expiration":        exp_date,
        "is_credit":         is_credit,
        "entry_cost":        play.get("cost_per_contract"),
        "credit_received":   credit,
        "confidence":        confidence,
        "confidence_score":  _confidence_num(confidence),
        "source":            source,
        "status":            "open",
        "exit_value":        None,
        "profit_loss":       None,
        "return_pct":        None,
        "outcome":           None,   # "win" | "loss" | "breakeven"
        "exit_date":         None,
        "stock_price_at_exit": None,
    }
    data["trades"].append(record)
    _save(data)


def update_expired_trades() -> list:
    """
    Find open trades past their expiration, fetch closing prices via yfinance,
    compute P&L, and write results back to outcomes.json.
    Returns list of updated trade dicts.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed — cannot update expired trades", file=sys.stderr)
        return []

    data    = _load()
    today   = date.today()
    updated = []

    for trade in data["trades"]:
        if trade.get("status") != "open":
            continue

        try:
            exp_date = date.fromisoformat(trade["expiration"])
        except (KeyError, ValueError):
            continue

        if exp_date > today:
            continue  # Not expired yet

        ticker = trade.get("ticker", "")
        try:
            end_str = (exp_date + timedelta(days=4)).isoformat()
            hist    = yf.Ticker(ticker).history(
                start=trade["expiration"], end=end_str, interval="1d"
            )
            if hist.empty:
                continue
            stock_price = float(hist["Close"].iloc[0])
        except Exception as e:
            print(f"Price fetch failed for {ticker} ({trade['expiration']}): {e}", file=sys.stderr)
            continue

        stype     = trade.get("strategy", "")
        atm       = trade.get("atm_strike") or 0.0
        otm       = trade.get("otm_strike") or 0.0
        is_credit = trade.get("is_credit", False)
        sv        = _spread_value_at_expiry(stype, atm, otm, stock_price)

        if stype == "iron_condor":
            # Iron condor P&L requires both spreads — defer
            trade.update({"status": "expired", "exit_date": trade["expiration"],
                          "stock_price_at_exit": round(stock_price, 2),
                          "exit_value": None, "profit_loss": None,
                          "return_pct": None, "outcome": "pending"})
        elif is_credit:
            credit = trade.get("credit_received") or 0.0
            pnl    = round(credit - sv, 2)
            basis  = credit or 1.0
            ret    = round(pnl / basis * 100, 1)
            trade.update({
                "status":              "expired",
                "exit_value":          round(sv, 2),
                "profit_loss":         pnl,
                "return_pct":          ret,
                "outcome":             "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven"),
                "exit_date":           trade["expiration"],
                "stock_price_at_exit": round(stock_price, 2),
            })
        else:
            cost  = trade.get("entry_cost") or 1.0
            pnl   = round(sv - cost, 2)
            ret   = round(pnl / cost * 100, 1)
            trade.update({
                "status":              "expired",
                "exit_value":          round(sv, 2),
                "profit_loss":         pnl,
                "return_pct":          ret,
                "outcome":             "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven"),
                "exit_date":           trade["expiration"],
                "stock_price_at_exit": round(stock_price, 2),
            })

        updated.append(trade)

    if updated:
        _save(data)

    return updated


def get_performance_stats() -> dict:
    """Return summary stats and full trade list for the Performance dashboard."""
    data       = _load()
    all_trades = data.get("trades", [])
    completed  = [t for t in all_trades if t.get("outcome") in ("win", "loss", "breakeven")]
    open_trades = [t for t in all_trades if t.get("status") == "open"]
    wins        = [t for t in completed if t["outcome"] == "win"]

    base = {
        "total":      len(completed),
        "open":       len(open_trades),
        "wins":       len(wins),
        "losses":     len(completed) - len(wins),
        "win_rate":   0.0,
        "avg_return": 0.0,
        "total_pnl":  0.0,
        "all_trades": sorted(all_trades, key=lambda t: t.get("date", ""), reverse=True),
    }

    if completed:
        base["win_rate"]   = round(len(wins) / len(completed) * 100, 1)
        base["avg_return"] = round(
            sum(t.get("return_pct", 0) or 0 for t in completed) / len(completed), 1
        )
        base["total_pnl"] = round(
            sum(t.get("profit_loss", 0) or 0 for t in completed), 2
        )

    return base


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--stats" not in sys.argv:
        print("Checking for expired trades…")
        updated = update_expired_trades()
        if updated:
            print(f"Updated {len(updated)} expired trade(s).")
        else:
            print("No expired trades to update.")

    stats = get_performance_stats()
    print(f"\n{'─'*40}")
    print(f"Total completed : {stats['total']}")
    print(f"Open positions  : {stats['open']}")
    print(f"Win / Loss      : {stats['wins']} / {stats['losses']}")
    print(f"Win rate        : {stats['win_rate']}%")
    print(f"Avg return      : {stats['avg_return']}%")
    print(f"Total P&L       : ${stats['total_pnl']:+.2f}")

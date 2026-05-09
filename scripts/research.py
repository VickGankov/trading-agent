#!/usr/bin/env python3
"""
research.py - Market data fetcher for the trading agent.

Usage:
    python scripts/research.py account                    # Account snapshot
    python scripts/research.py market                     # Index/macro snapshot
    python scripts/research.py bars NVDA 60               # 60 days of bars
    python scripts/research.py quote NVDA                 # Current quote
    python scripts/research.py news NVDA                  # Recent news
    python scripts/research.py screener movers            # Top movers
    python scripts/research.py technicals NVDA            # MA, RSI, volume
    python scripts/research.py calendar AAPL,NVDA,GOOGL   # Earnings calendar check
    python scripts/research.py daytrade_count             # PDT compliance check

All output is JSON to stdout for the agent to consume.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Optional

try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, NewsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.requests import GetCalendarRequest
    from alpaca.trading.enums import OrderStatus
except ImportError:
    print(json.dumps({"error": "alpaca-py not installed. Run: pip install alpaca-py"}), file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print(json.dumps({"error": "pandas/numpy not installed. Run: pip install pandas numpy"}), file=sys.stderr)
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

# Per-process cache so one cycle doesn't re-fetch the same symbol's bars/technicals
_tech_cache: dict = {}

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER = os.getenv("PAPER", "True").lower() == "true"

if not API_KEY or not SECRET_KEY:
    print(json.dumps({"error": "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"}), file=sys.stderr)
    sys.exit(1)

trading = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
data = StockHistoricalDataClient(API_KEY, SECRET_KEY)
news = NewsClient(API_KEY, SECRET_KEY)


def get_account():
    """Account snapshot: cash, equity, buying power, positions."""
    acct = trading.get_account()
    positions = trading.get_all_positions()
    
    pos_data = []
    for p in positions:
        pos_data.append({
            "symbol": p.symbol,
            "qty": float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price) if p.current_price else None,
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc),
            "side": p.side.value
        })
    
    return {
        "account_value": float(acct.equity),
        "cash": float(acct.cash),
        "buying_power": float(acct.buying_power),
        "daytrade_count": int(acct.daytrade_count),
        "pattern_day_trader": acct.pattern_day_trader,
        "trading_blocked": acct.trading_blocked,
        "account_blocked": acct.account_blocked,
        "positions_count": len(pos_data),
        "positions": pos_data,
        "is_paper": PAPER
    }


def is_market_open():
    """Check if market is currently open and get next open/close times."""
    clock = trading.get_clock()
    return {
        "is_open": clock.is_open,
        "next_open": clock.next_open.isoformat() if clock.next_open else None,
        "next_close": clock.next_close.isoformat() if clock.next_close else None,
        "current_time": clock.timestamp.isoformat() if clock.timestamp else None
    }


def get_bars(symbol: str, days: int = 60):
    """Get daily bars for technical analysis."""
    end = datetime.now() - timedelta(minutes=20)  # IEX free feed has 15min delay
    start = end - timedelta(days=days * 2)  # extra buffer for weekends
    
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        limit=days
    )
    bars = data.get_stock_bars(req)
    
    if symbol not in bars.data:
        return {"symbol": symbol, "bars": [], "error": "no data"}
    
    bar_list = [
        {
            "date": b.timestamp.strftime("%Y-%m-%d"),
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": int(b.volume)
        }
        for b in bars.data[symbol]
    ]
    
    return {"symbol": symbol, "bars": bar_list[-days:]}


def get_quote(symbol: str):
    """Latest quote (bid/ask)."""
    req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    quote = data.get_stock_latest_quote(req)
    
    if symbol not in quote:
        return {"symbol": symbol, "error": "no quote"}
    
    q = quote[symbol]
    return {
        "symbol": symbol,
        "bid": float(q.bid_price),
        "ask": float(q.ask_price),
        "bid_size": int(q.bid_size),
        "ask_size": int(q.ask_size),
        "spread_pct": ((float(q.ask_price) - float(q.bid_price)) / float(q.ask_price) * 100) if q.ask_price else None,
        "timestamp": q.timestamp.isoformat()
    }


def get_news(symbol: Optional[str] = None, hours: int = 24):
    """Recent news for a symbol or general market news."""
    start = datetime.now() - timedelta(hours=hours)
    req = NewsRequest(
        symbols=symbol if symbol else None,
        start=start,
        limit=20
    )
    news_data = news.get_news(req)
    
    items = []
    for n in news_data.data.get("news", []):
        items.append({
            "headline": n.headline,
            "summary": n.summary[:200] if n.summary else "",
            "source": n.source,
            "symbols": n.symbols,
            "url": n.url,
            "created_at": n.created_at.isoformat()
        })
    
    return {"symbol": symbol or "market", "news_count": len(items), "items": items}


def calc_technicals(symbol: str):
    """Calculate MA20, MA50, RSI14, volume vs average. Cached per process."""
    if symbol in _tech_cache:
        return _tech_cache[symbol]
    bars_data = get_bars(symbol, days=60)
    if "error" in bars_data or len(bars_data["bars"]) < 20:
        return {"symbol": symbol, "error": "insufficient data"}
    
    df = pd.DataFrame(bars_data["bars"])
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    
    # Moving averages
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    
    # RSI 14
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # Volume ratio
    df["vol_avg20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_avg20"]
    
    last = df.iloc[-1]
    result = {
        "symbol": symbol,
        "current_price": round(float(last["close"]), 2),
        "ma20": round(float(last["ma20"]), 2) if not pd.isna(last["ma20"]) else None,
        "ma50": round(float(last["ma50"]), 2) if not pd.isna(last["ma50"]) else None,
        "rsi14": round(float(last["rsi"]), 1) if not pd.isna(last["rsi"]) else None,
        "volume": int(last["volume"]),
        "vol_avg20": int(last["vol_avg20"]) if not pd.isna(last["vol_avg20"]) else None,
        "vol_ratio": round(float(last["vol_ratio"]), 2) if not pd.isna(last["vol_ratio"]) else None,
        "above_ma20": bool(last["close"] > last["ma20"]) if not pd.isna(last["ma20"]) else None,
        "above_ma50": bool(last["close"] > last["ma50"]) if not pd.isna(last["ma50"]) else None,
        "5d_change_pct": round(float((last["close"] / df["close"].iloc[-6] - 1) * 100), 2) if len(df) >= 6 else None,
        "20d_change_pct": round(float((last["close"] / df["close"].iloc[-21] - 1) * 100), 2) if len(df) >= 21 else None
    }
    _tech_cache[symbol] = result
    return result


def get_market_snapshot():
    """SPY, QQQ, IWM and VIX proxy current state."""
    indices = ["SPY", "QQQ", "IWM"]
    snapshot = {}
    for idx in indices:
        tech = calc_technicals(idx)
        if "error" not in tech:
            snapshot[idx] = {
                "price": tech["current_price"],
                "5d_change_pct": tech["5d_change_pct"],
                "20d_change_pct": tech["20d_change_pct"],
                "above_ma20": tech["above_ma20"],
                "rsi14": tech["rsi14"]
            }
    return {
        "indices": snapshot,
        "market_status": is_market_open()
    }


def screener_movers():
    """
    Get most active stocks. Note: Alpaca doesn't have a built-in 'top movers' endpoint
    on the free tier — we use the Most Active list as a proxy.
    For richer screening, integrate a free API like Finnhub or polygon (not included here).
    """
    # Use a static list of high-volume names + the watchlist universe as starting universe
    universe = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "AVGO",
        "PLTR", "COIN", "MSTR", "SMCI", "CRWD", "ZS", "PANW", "FTNT",
        "JPM", "BAC", "GS", "MS", "C",
        "XOM", "CVX", "OXY",
        "CEG", "VST", "NEE",
        "SPY", "QQQ", "IWM", "SMH", "XLK", "XLF", "XLE",
        "DIS", "NFLX", "UBER", "SHOP", "SQ",
        "BA", "LMT", "RTX", "GE",
        "WMT", "COST", "TGT", "HD", "LOW",
        "MCD", "SBUX", "CMG",
        "JNJ", "PFE", "MRK", "LLY", "UNH",
        "KO", "PEP", "PG"
    ]
    
    movers = []
    for symbol in universe[:40]:  # cap to control API calls
        try:
            tech = calc_technicals(symbol)
            if "error" not in tech and tech.get("vol_ratio") and tech.get("5d_change_pct") is not None:
                movers.append({
                    "symbol": symbol,
                    "price": tech["current_price"],
                    "5d_change_pct": tech["5d_change_pct"],
                    "vol_ratio": tech["vol_ratio"],
                    "rsi14": tech["rsi14"]
                })
        except Exception:
            continue
    
    # Sort by absolute 5d change × volume ratio
    movers.sort(key=lambda x: abs(x["5d_change_pct"] or 0) * (x["vol_ratio"] or 0), reverse=True)
    return {"movers": movers[:20]}


def check_earnings_calendar(symbols: str):
    """
    Check upcoming earnings. Alpaca doesn't have a built-in earnings calendar in free tier.
    For production, integrate a free source like Yahoo Finance via yfinance, Finnhub, or NASDAQ.
    This is a placeholder that returns warnings for known dates.
    """
    known_earnings = {
        "NVDA": "2026-05-20",
        "CEG": "2026-05-11",
        "VST": "2026-05-07",
        "PLTR": "2026-08-03",
        "MAIN": "2026-05-08"
    }
    
    today = datetime.now().date()
    results = []
    for sym in symbols.split(","):
        sym = sym.strip().upper()
        if sym in known_earnings:
            earn_date = datetime.strptime(known_earnings[sym], "%Y-%m-%d").date()
            days_until = (earn_date - today).days
            results.append({
                "symbol": sym,
                "earnings_date": known_earnings[sym],
                "days_until": days_until,
                "avoid_new_positions": -1 <= days_until <= 3
            })
        else:
            results.append({
                "symbol": sym,
                "earnings_date": "unknown",
                "note": "Integrate yfinance or Finnhub for live calendar"
            })
    return {"earnings_check": results}


def daytrade_count():
    """PDT compliance check."""
    acct = trading.get_account()
    return {
        "daytrade_count_5days": int(acct.daytrade_count),
        "pdt_status": acct.pattern_day_trader,
        "remaining_daytrades": max(0, 3 - int(acct.daytrade_count)),
        "warning": "PDT enforced if account < $25k and 4+ day trades in 5 days"
    }


# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[
        "account", "market", "bars", "quote", "news", "technicals",
        "screener", "calendar", "daytrade_count", "is_open"
    ])
    parser.add_argument("arg1", nargs="?", default=None)
    parser.add_argument("arg2", nargs="?", default=None)
    args = parser.parse_args()
    
    try:
        if args.command == "account":
            print(json.dumps(get_account(), indent=2))
        elif args.command == "market":
            print(json.dumps(get_market_snapshot(), indent=2))
        elif args.command == "is_open":
            print(json.dumps(is_market_open(), indent=2))
        elif args.command == "bars":
            days = int(args.arg2) if args.arg2 else 60
            print(json.dumps(get_bars(args.arg1, days), indent=2))
        elif args.command == "quote":
            print(json.dumps(get_quote(args.arg1), indent=2))
        elif args.command == "news":
            print(json.dumps(get_news(args.arg1), indent=2))
        elif args.command == "technicals":
            print(json.dumps(calc_technicals(args.arg1), indent=2))
        elif args.command == "screener":
            print(json.dumps(screener_movers(), indent=2))
        elif args.command == "calendar":
            print(json.dumps(check_earnings_calendar(args.arg1), indent=2))
        elif args.command == "daytrade_count":
            print(json.dumps(daytrade_count(), indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e), "type": type(e).__name__}), file=sys.stderr)
        sys.exit(1)

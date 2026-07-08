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
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, StockLatestTradeRequest, NewsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
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

_trading_client = None
_data_client = None
_news_client = None


def _require_alpaca_credentials():
    if not API_KEY or not SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")


def get_trading_client():
    global _trading_client
    _require_alpaca_credentials()
    if _trading_client is None:
        _trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
    return _trading_client


def get_data_client():
    global _data_client
    _require_alpaca_credentials()
    if _data_client is None:
        _data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    return _data_client


def get_news_client():
    global _news_client
    _require_alpaca_credentials()
    if _news_client is None:
        _news_client = NewsClient(API_KEY, SECRET_KEY)
    return _news_client


def get_account():
    """Account snapshot: cash, equity, buying power, positions."""
    trading = get_trading_client()
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
        "daytrade_count": int(acct.daytrade_count) if acct.daytrade_count is not None else 0,
        "pattern_day_trader": acct.pattern_day_trader,
        "trading_blocked": acct.trading_blocked,
        "account_blocked": acct.account_blocked,
        "positions_count": len(pos_data),
        "positions": pos_data,
        "is_paper": PAPER
    }


def is_market_open():
    """Check if market is currently open and get next open/close times."""
    trading = get_trading_client()
    clock = trading.get_clock()
    return {
        "is_open": clock.is_open,
        "next_open": clock.next_open.isoformat() if clock.next_open else None,
        "next_close": clock.next_close.isoformat() if clock.next_close else None,
        "current_time": clock.timestamp.isoformat() if clock.timestamp else None
    }


def get_bars(symbol: str, days: int = 60):
    """Get daily bars for technical analysis."""
    data = get_data_client()
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


def get_intraday_bars(symbol: str, minutes: int = 5, lookback_hours: int = 8) -> dict:
    """
    Real-time intraday bars via Alpaca IEX feed (no delay during market hours).
    Falls back to yfinance if Alpaca returns nothing (pre-market / no IEX trades).
    """
    data = get_data_client()
    start = datetime.now() - timedelta(hours=lookback_hours)

    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
            start=start,
        )
        bars = data.get_stock_bars(req)
        if symbol in bars.data and bars.data[symbol]:
            return {
                "symbol": symbol,
                "source": "alpaca_iex",
                "bars": [
                    {
                        "timestamp": b.timestamp.isoformat(),
                        "open":   float(b.open),
                        "high":   float(b.high),
                        "low":    float(b.low),
                        "close":  float(b.close),
                        "volume": int(b.volume),
                    }
                    for b in bars.data[symbol]
                ],
            }
    except Exception:
        pass

    # Fallback: yfinance (covers pre-market and after-hours)
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period="1d", interval=f"{minutes}m", prepost=False)
        if not hist.empty:
            return {
                "symbol": symbol,
                "source": "yfinance_delayed",
                "bars": [
                    {
                        "timestamp": str(ts),
                        "open":   round(float(row["Open"]), 4),
                        "high":   round(float(row["High"]), 4),
                        "low":    round(float(row["Low"]), 4),
                        "close":  round(float(row["Close"]), 4),
                        "volume": int(row["Volume"]),
                    }
                    for ts, row in hist.iterrows()
                ],
            }
    except Exception:
        pass

    return {"symbol": symbol, "source": "none", "bars": [], "error": "no data"}


def calc_macd(bars: list, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """
    Calculate MACD on a list of bar dicts (each with a 'close' key).
    Returns macd/signal/histogram series plus a plain-English signal string.
    Needs at least slow+signal bars (35) to be meaningful.
    """
    if len(bars) < slow + signal:
        return {"error": f"need {slow+signal} bars, got {len(bars)}"}

    closes = pd.Series([b["close"] for b in bars])
    timestamps = [b["timestamp"] for b in bars]

    ema_fast   = closes.ewm(span=fast,   adjust=False).mean()
    ema_slow   = closes.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    sig_line   = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - sig_line

    last_m, last_s = macd_line.iloc[-1], sig_line.iloc[-1]
    prev_m, prev_s = macd_line.iloc[-2], sig_line.iloc[-2]

    if prev_m <= prev_s and last_m > last_s:
        signal_str = "bullish_crossover"
    elif prev_m >= prev_s and last_m < last_s:
        signal_str = "bearish_crossover"
    elif last_m > last_s:
        signal_str = "bullish_trend"
    else:
        signal_str = "bearish_trend"

    return {
        "timestamps":       timestamps,
        "macd":             [round(v, 4) for v in macd_line.tolist()],
        "signal":           [round(v, 4) for v in sig_line.tolist()],
        "histogram":        [round(v, 4) for v in histogram.tolist()],
        "signal_str":       signal_str,
        "macd_value":       round(float(last_m), 4),
        "signal_value":     round(float(last_s), 4),
        "histogram_value":  round(float(histogram.iloc[-1]), 4),
        "above_zero":       bool(last_m > 0),
    }


def get_quote(symbol: str):
    """Latest quote (bid/ask) + latest trade price. Detects stale quotes."""
    from datetime import timezone as _tz

    data = get_data_client()
    result: dict = {"symbol": symbol}

    # --- latest NBBO quote ---
    try:
        q = data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol)).get(symbol)
    except Exception:
        q = None

    quote_ts = None
    if q:
        ask = float(q.ask_price) if q.ask_price else None
        bid = float(q.bid_price) if q.bid_price else None
        quote_ts = q.timestamp
        result.update({
            "bid": bid,
            "ask": ask,
            "bid_size": int(q.bid_size) if q.bid_size else None,
            "ask_size": int(q.ask_size) if q.ask_size else None,
            "spread_pct": ((ask - bid) / ask * 100) if ask and bid else None,
            "timestamp": quote_ts.isoformat(),
        })

    # --- latest trade (more reliable intraday price) ---
    trade_price = None
    trade_ts = None
    try:
        t = data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol)).get(symbol)
        if t:
            trade_price = float(t.price)
            trade_ts = t.timestamp
            result["last_trade_price"] = trade_price
            result["last_trade_timestamp"] = trade_ts.isoformat()
    except Exception:
        pass

    # --- staleness check ---
    # Mark the quote stale if it's older than 20 minutes relative to now
    now_utc = datetime.now(_tz.utc)
    ref_ts = trade_ts or quote_ts
    if ref_ts:
        ts_utc = ref_ts.astimezone(_tz.utc) if ref_ts.tzinfo else ref_ts.replace(tzinfo=_tz.utc)
        age_minutes = (now_utc - ts_utc).total_seconds() / 60
        result["data_age_minutes"] = round(age_minutes, 1)
        result["is_stale"] = age_minutes > 20
    else:
        result["is_stale"] = True

    # Use last trade price as the canonical "ask" if the NBBO ask is absent or stale
    if trade_price and (not result.get("ask") or result.get("is_stale")):
        result["ask"] = trade_price

    return result


def get_news(symbol: Optional[str] = None, hours: int = 24):
    """Recent news for a symbol or general market news."""
    news = get_news_client()
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

    # 30-day annualized historical volatility (log-return std × √252)
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    hist_vol_30d = float(log_ret.iloc[-30:].std() * np.sqrt(252)) if len(log_ret) >= 10 else 0.30

    # ATR14 — for day trade stop sizing
    df["high"] = df["high"].astype(float)
    df["low"]  = df["low"].astype(float)
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["prev_close"]).abs(),
            (df["low"]  - df["prev_close"]).abs()
        )
    )
    atr14 = float(df["tr"].rolling(14).mean().iloc[-1]) if len(df) >= 14 else float(last["close"] * 0.015)
    atr14_pct = round(atr14 / float(last["close"]) * 100, 2)

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
        "5d_change_pct":  round(float((last["close"] / df["close"].iloc[-6]  - 1) * 100), 2) if len(df) >= 6  else None,
        "20d_change_pct": round(float((last["close"] / df["close"].iloc[-21] - 1) * 100), 2) if len(df) >= 21 else None,
        "hist_vol_30d": round(hist_vol_30d * 100, 1),   # annualized %, e.g. 37.2
        "atr14":        round(atr14, 2),                 # raw $ ATR
        "atr14_pct":    atr14_pct,                       # ATR as % of price
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


SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLE":  "Energy",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XBI":  "Biotech",
    "XLI":  "Industrials",
    "XLC":  "Communications",
    "XLY":  "Consumer Discretionary",
    "GLD":  "Gold",
    "USO":  "Oil",
}

# Representative high-liquidity stocks per sector for options plays
SECTOR_LEADERS = {
    "XLK":  ["NVDA", "MSFT", "AAPL", "AMD", "AVGO"],
    "XLE":  ["XOM",  "CVX",  "SLB",  "EOG",  "MPC"],
    "XLF":  ["JPM",  "GS",   "BAC",  "MS",   "V"],
    "XLV":  ["UNH",  "LLY",  "JNJ",  "ABBV", "MRK"],
    "XBI":  ["BIIB", "MRNA", "GILD", "REGN", "SGEN"],
    "XLI":  ["CAT",  "DE",   "HON",  "GE",   "RTX"],
    "XLC":  ["META", "GOOGL","NFLX", "DIS",  "SNAP"],
    "XLY":  ["AMZN", "TSLA", "NKE",  "MCD",  "HD"],
    "GLD":  ["GLD",  "GOLD", "NEM"],
    "USO":  ["XOM",  "CVX",  "USO"],
}


def get_sector_snapshot():
    """Performance snapshot for major sector ETFs."""
    snapshot = {}
    for etf, name in SECTOR_ETFS.items():
        try:
            tech = calc_technicals(etf)
            if "error" not in tech:
                snapshot[etf] = {
                    "name":           name,
                    "price":          tech["current_price"],
                    "1d_change_pct":  tech.get("5d_change_pct"),   # proxy — daily not available on free tier
                    "5d_change_pct":  tech["5d_change_pct"],
                    "20d_change_pct": tech["20d_change_pct"],
                    "rsi14":          tech["rsi14"],
                    "above_ma20":     tech["above_ma20"],
                    "above_ma50":     tech["above_ma50"],
                    "vol_ratio":      tech["vol_ratio"],
                    "leaders":        SECTOR_LEADERS.get(etf, [])[:3],
                }
        except Exception:
            pass
    return snapshot


def _setup_score(tech: dict) -> float:
    """
    Score a stock on setup quality for entry, NOT recent price momentum.
    Looks for pullbacks to MA support, oversold bounces, and volume breakouts
    from consolidation. Returns 0 for explicitly disqualified setups (overbought,
    huge gap-and-run). Higher score = better candidate for deep analysis.
    """
    rsi = tech.get("rsi14") or 50.0
    price = tech.get("current_price") or 0.0
    ma20 = tech.get("ma20") or price
    ma50 = tech.get("ma50") or price
    vol_ratio = tech.get("vol_ratio") or 1.0
    change_5d = tech.get("5d_change_pct") or 0.0
    above_ma20 = tech.get("above_ma20")

    # Hard disqualifiers — skip these entirely
    if rsi > 65:
        return 0.0
    if abs(change_5d) > 20:  # massive gap move; thesis already played out
        return 0.0

    score = 0.0
    above_ma50 = price > ma50 if ma50 else False

    # Setup 1: pullback to MA20 in an uptrend
    # Best when price is 0-3% above MA20, RSI has cooled to 38-58
    if above_ma20 and ma20:
        pct_above_ma20 = ((price - ma20) / ma20) * 100
        if 0 <= pct_above_ma20 <= 4 and 38 <= rsi <= 58:
            score += 10.0 - (pct_above_ma20 * 1.5)  # tighter to MA = better

    # Setup 2: oversold with uptrend structure intact (MA50 support)
    if rsi < 42 and above_ma50:
        score += 7.0 + (42 - rsi) * 0.25  # more oversold but still above MA50 = better

    # Setup 3: volume breakout from consolidation (high vol, small recent price move)
    # Distinguishes early-stage breakouts from already-extended momentum plays
    if vol_ratio >= 2.0 and abs(change_5d) < 5:
        score += vol_ratio * 2.5

    # Minor bonus for stocks in established uptrend (above MA50)
    if above_ma50:
        score += 1.0

    # Decay score for stocks that are already extended — less favorable entry
    if abs(change_5d) > 7:
        decay = min(0.8, (abs(change_5d) - 7) * 0.07)
        score *= (1.0 - decay)

    return round(score, 2)


def screener_movers():
    """
    Screen for stocks with actionable setups: pullbacks to MA, oversold bounces,
    volume breakouts from consolidation. Ranked by setup quality, NOT raw momentum.
    """
    universe = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "AVGO",
        "PLTR", "COIN", "SMCI", "CRWD", "ZS", "PANW", "FTNT",
        "JPM", "BAC", "GS", "MS", "C",
        "XOM", "CVX", "OXY",
        "CEG", "VST", "NEE",
        "SMH", "XLK", "XLF", "XLE",
        "DIS", "NFLX", "UBER", "SHOP", "SQ",
        "BA", "LMT", "RTX", "GE",
        "WMT", "COST", "TGT", "HD",
        "MCD", "SBUX",
        "JNJ", "PFE", "LLY", "UNH",
        "KO", "PEP", "PG"
    ]

    candidates = []
    for symbol in universe[:50]:
        try:
            tech = calc_technicals(symbol)
            if "error" in tech or tech.get("vol_ratio") is None:
                continue
            candidates.append({
                "symbol": symbol,
                "price": tech["current_price"],
                "5d_change_pct": tech["5d_change_pct"],
                "vol_ratio": tech["vol_ratio"],
                "rsi14": tech["rsi14"],
                "above_ma20": tech["above_ma20"],
                "setup_score": _setup_score(tech),
            })
        except Exception:
            continue

    candidates.sort(key=lambda x: x["setup_score"], reverse=True)
    return {"movers": candidates[:20]}


def check_earnings_calendar(symbols: str):
    """
    Check upcoming earnings via yfinance. Returns avoid_new_positions=True
    if earnings are within the next 3 trading days.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "yfinance not installed — run: pip install yfinance"}

    today = datetime.now().date()
    results = []
    for sym in symbols.split(","):
        sym = sym.strip().upper()
        if not sym:
            continue
        try:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ticker = yf.Ticker(sym)
                cal = ticker.calendar  # dict with keys like 'Earnings Date'
            earn_date = None

            if isinstance(cal, dict):
                raw = cal.get("Earnings Date")
                if raw is not None:
                    # yfinance returns a list of timestamps or a single timestamp
                    if hasattr(raw, "__iter__") and not isinstance(raw, str):
                        dates = [d.date() if hasattr(d, "date") else d for d in raw]
                        future = [d for d in dates if d >= today]
                        earn_date = min(future) if future else (max(dates) if dates else None)
                    elif hasattr(raw, "date"):
                        earn_date = raw.date()

            if earn_date:
                days_until = (earn_date - today).days
                results.append({
                    "symbol": sym,
                    "earnings_date": earn_date.isoformat(),
                    "days_until": days_until,
                    "avoid_new_positions": -1 <= days_until <= 3
                })
            else:
                results.append({"symbol": sym, "earnings_date": "unknown", "avoid_new_positions": False})
        except Exception as e:
            results.append({"symbol": sym, "earnings_date": "unknown", "avoid_new_positions": False, "note": str(e)})

    return {"earnings_check": results}


def daytrade_count():
    """PDT compliance check."""
    trading = get_trading_client()
    acct = trading.get_account()
    dt_count = int(acct.daytrade_count) if acct.daytrade_count is not None else 0
    return {
        "daytrade_count_5days": dt_count,
        "pdt_status": acct.pattern_day_trader,
        "remaining_daytrades": max(0, 3 - dt_count),
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

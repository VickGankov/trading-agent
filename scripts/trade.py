#!/usr/bin/env python3
"""
trade.py - Order placement with HARD-CODED guardrails.

The validate_order() function enforces risk rules that the agent CANNOT bypass.
Even if the LLM generates an invalid order, this function rejects it.

Usage from agent:
    python scripts/trade.py buy NVDA 1 197.50 --stop 187.00 --target 215.00 --reason "thesis"
    python scripts/trade.py sell NVDA 1 --reason "stop hit"
    python scripts/trade.py close_all                  # emergency
    python scripts/trade.py list_orders                # open orders
    python scripts/trade.py cancel <order_id>

Returns JSON to stdout for the agent to parse.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest, StopLossRequest, TakeProfitRequest,
        GetOrdersRequest
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
except ImportError:
    print(json.dumps({"error": "alpaca-py not installed. Run: pip install alpaca-py"}), file=sys.stderr)
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER = os.getenv("PAPER", "True").lower() == "true"

# ===================================================================
# HARD RULES - These are the guardrails. Do not loosen without review.
# ===================================================================

MAX_POSITION_PCT = 10.0         # Max 10% of account per position
MIN_POSITION_USD = 50.0         # Min $50 per position
MAX_CONCURRENT_POSITIONS = 5    # Max 5 open at once
MIN_CASH_RESERVE_PCT = 25.0     # Always keep 25% cash
MIN_STOP_PCT = 3.0              # Stop at least 3% below entry
MAX_STOP_PCT = 10.0             # Stop at most 10% below entry (caps risk)
MIN_RR_RATIO = 1.5              # Take-profit at least 1.5x the risk
DAILY_LOSS_LIMIT_PCT = 3.0      # Halt if account down 3% in a day
MIN_PRICE = 5.0                 # No penny stocks
MAX_PRICE = 1500.0              # Cap — fractional shares supported

WEEKLY_LOSS_LIMIT_PCT = 8.0         # Halt week if account drops 8%
STATE_FILE = Path(__file__).parent.parent / "data" / "state.json"

LEVERAGED_ETFS = {
    "TQQQ", "SQQQ", "SOXL", "SOXS", "TNA", "TZA", "UPRO", "SPXU",
    "TMF", "TMV", "FAS", "FAZ", "LABU", "LABD", "BOIL", "KOLD",
    "UVXY", "SVXY", "VXX", "TVIX"
}

if not API_KEY or not SECRET_KEY:
    print(json.dumps({"error": "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"}), file=sys.stderr)
    sys.exit(1)

trading = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    except Exception:
        return {}


def _save_state(state: dict):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def save_stop_level(symbol: str, stop_loss: float, take_profit: float, entry_limit: float):
    """Persist stop/target into state.json so remote routines can monitor positions."""
    state = _load_state()
    state.setdefault("active_stops", {})[symbol.upper()] = {
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "entry_limit": entry_limit,
    }
    _save_state(state)


def remove_stop_level(symbol: str):
    """Remove stop level from state.json when a position is closed."""
    state = _load_state()
    state.get("active_stops", {}).pop(symbol.upper(), None)
    _save_state(state)


def check_weekly_circuit_breaker(current_equity: float) -> tuple:
    """
    Returns (tripped: bool, message: str).
    Resets week_start_equity each Monday. Halts buys if down 8% from week open.
    """
    state = _load_state()
    today = datetime.now()
    iso_week = today.strftime("%G-W%V")  # e.g. "2026-W20"

    if state.get("week_key") != iso_week:
        # New week — record starting equity
        state = {"week_key": iso_week, "week_start_equity": current_equity}
        _save_state(state)
        return False, "OK"

    week_start = state.get("week_start_equity", current_equity)
    if week_start <= 0:
        return False, "OK"

    weekly_change_pct = ((current_equity - week_start) / week_start) * 100
    if weekly_change_pct <= -WEEKLY_LOSS_LIMIT_PCT:
        return True, (
            f"Weekly loss circuit breaker: account down {weekly_change_pct:.2f}% "
            f"from ${week_start:.2f} → ${current_equity:.2f}. No new buys this week."
        )
    return False, "OK"


def get_account_state():
    """Snapshot for validation."""
    acct = trading.get_account()
    positions = trading.get_all_positions()
    return {
        "equity": float(acct.equity),
        "cash": float(acct.cash),
        "last_equity": float(acct.last_equity),
        "buying_power": float(acct.buying_power),
        "daytrade_count": int(acct.daytrade_count),
        "pattern_day_trader": acct.pattern_day_trader,
        "trading_blocked": acct.trading_blocked,
        "positions": [{"symbol": p.symbol, "qty": float(p.qty), "side": p.side.value} for p in positions],
        "positions_count": len(positions)
    }


def validate_order(symbol: str, side: str, qty: float, limit_price: Optional[float],
                   stop_price: Optional[float], target_price: Optional[float]) -> tuple:
    """
    Pre-flight validation. Returns (is_valid: bool, message: str).
    EVERY ORDER must pass this before submission.
    """
    symbol = symbol.upper()
    side = side.lower()
    
    # 1. Account state checks
    state = get_account_state()
    
    if state["trading_blocked"]:
        return False, "Account trading is BLOCKED"
    
    # 2. Daily loss circuit breaker
    if state["last_equity"] > 0:
        day_change_pct = ((state["equity"] - state["last_equity"]) / state["last_equity"]) * 100
        if day_change_pct <= -DAILY_LOSS_LIMIT_PCT and side == "buy":
            return False, f"Daily loss circuit breaker tripped ({day_change_pct:.2f}%). No new buys today."

    # 2b. Weekly loss circuit breaker
    if side == "buy":
        tripped, msg = check_weekly_circuit_breaker(state["equity"])
        if tripped:
            return False, msg

    # 3. Excluded instruments
    if symbol in LEVERAGED_ETFS:
        return False, f"{symbol} is a leveraged ETF — forbidden by hard rules"
    
    # 4. Side-specific checks
    if side == "buy":
        if limit_price is None:
            return False, "BUY orders must specify a limit price (no market orders)"
        if stop_price is None or target_price is None:
            return False, "BUY orders must include both stop_loss and take_profit"
        
        # Price range
        if limit_price < MIN_PRICE or limit_price > MAX_PRICE:
            return False, f"Price {limit_price} outside allowed range ${MIN_PRICE}-${MAX_PRICE}"
        
        # Stop placement
        stop_pct = ((limit_price - stop_price) / limit_price) * 100
        if stop_pct < MIN_STOP_PCT:
            return False, f"Stop too tight: {stop_pct:.2f}% (min {MIN_STOP_PCT}%)"
        if stop_pct > MAX_STOP_PCT:
            return False, f"Stop too wide: {stop_pct:.2f}% (max {MAX_STOP_PCT}%)"
        
        # Risk/reward
        risk = limit_price - stop_price
        reward = target_price - limit_price
        if reward / risk < MIN_RR_RATIO:
            return False, f"R:R too low: {reward/risk:.2f} (min {MIN_RR_RATIO})"
        
        # Position size
        order_value = qty * limit_price
        position_pct = (order_value / state["equity"]) * 100
        if order_value < MIN_POSITION_USD:
            return False, f"Position too small: ${order_value:.2f} (min ${MIN_POSITION_USD})"
        if position_pct > MAX_POSITION_PCT:
            return False, f"Position too large: {position_pct:.2f}% (max {MAX_POSITION_PCT}%)"
        
        # Concurrent positions
        existing_symbols = {p["symbol"] for p in state["positions"]}
        if symbol not in existing_symbols and len(state["positions"]) >= MAX_CONCURRENT_POSITIONS:
            return False, f"Max {MAX_CONCURRENT_POSITIONS} concurrent positions reached"
        
        # Cash reserve
        cash_after = state["cash"] - order_value
        cash_reserve_pct = (cash_after / state["equity"]) * 100
        if cash_reserve_pct < MIN_CASH_RESERVE_PCT:
            return False, f"Would breach cash reserve: {cash_reserve_pct:.2f}% (min {MIN_CASH_RESERVE_PCT}%)"
    
    elif side == "sell":
        # Must have an existing position
        existing = {p["symbol"]: float(p["qty"]) for p in state["positions"]}
        if symbol not in existing:
            return False, f"Cannot sell {symbol} — no open position"
        if qty > existing[symbol]:
            return False, f"Cannot sell {qty} {symbol} — only {existing[symbol]} held"
    
    else:
        return False, f"Invalid side: {side}"
    
    return True, "OK"


def place_buy(symbol: str, qty: float, limit_price: float, stop_price: float, target_price: float, reason: str):
    """
    Place a BUY order. Whole shares get a bracket order (entry + stop + target atomic).
    Fractional shares get a simple limit order — Alpaca doesn't support bracket on fractions.
    """
    valid, msg = validate_order(symbol, "buy", qty, limit_price, stop_price, target_price)
    if not valid:
        return {"status": "REJECTED", "reason": msg, "symbol": symbol, "side": "buy", "qty": qty}

    is_fractional = (qty % 1) != 0

    try:
        if is_fractional:
            order = trading.submit_order(LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
            ))
        else:
            order = trading.submit_order(LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
                take_profit=TakeProfitRequest(limit_price=round(target_price, 2))
            ))
        return {
            "status": "SUBMITTED",
            "order_id": str(order.id),
            "symbol": symbol,
            "side": "buy",
            "qty": qty,
            "limit_price": limit_price,
            "stop_loss": stop_price,
            "take_profit": target_price,
            "order_type": "simple_limit" if is_fractional else "bracket",
            "note": "Manual stop/target required — Alpaca bracket orders don't support fractional qty" if is_fractional else None,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e), "symbol": symbol}


def place_sell(symbol: str, qty: float, reason: str):
    """Sell an existing position."""
    valid, msg = validate_order(symbol, "sell", qty, None, None, None)
    if not valid:
        return {"status": "REJECTED", "reason": msg, "symbol": symbol, "side": "sell", "qty": qty}
    
    try:
        order = trading.submit_order(MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        ))
        return {
            "status": "SUBMITTED",
            "order_id": str(order.id),
            "symbol": symbol,
            "side": "sell",
            "qty": qty,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e), "symbol": symbol}


def list_orders(status: str = "open"):
    """List open or recent orders."""
    s = QueryOrderStatus.OPEN if status == "open" else QueryOrderStatus.ALL
    req = GetOrdersRequest(status=s, limit=20)
    orders = trading.get_orders(filter=req)
    return [{
        "id": str(o.id),
        "symbol": o.symbol,
        "side": o.side.value,
        "qty": float(o.qty) if o.qty else None,
        "type": o.order_type.value,
        "limit_price": float(o.limit_price) if o.limit_price else None,
        "stop_price": float(o.stop_price) if o.stop_price else None,
        "status": o.status.value,
        "submitted_at": o.submitted_at.isoformat() if o.submitted_at else None
    } for o in orders]


def cancel_order(order_id: str):
    """Cancel an open order."""
    try:
        trading.cancel_order_by_id(order_id)
        return {"status": "CANCELED", "order_id": order_id}
    except Exception as e:
        return {"status": "ERROR", "error": str(e), "order_id": order_id}


def close_all_positions(reason: str = "manual"):
    """EMERGENCY: Liquidate everything."""
    try:
        trading.close_all_positions(cancel_orders=True)
        return {"status": "ALL_CLOSED", "reason": reason, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    
    pb = sub.add_parser("buy")
    pb.add_argument("symbol")
    pb.add_argument("qty", type=int)
    pb.add_argument("limit_price", type=float)
    pb.add_argument("--stop", type=float, required=True)
    pb.add_argument("--target", type=float, required=True)
    pb.add_argument("--reason", default="agent decision")
    
    ps = sub.add_parser("sell")
    ps.add_argument("symbol")
    ps.add_argument("qty", type=int)
    ps.add_argument("--reason", default="agent decision")
    
    sub.add_parser("list_orders").add_argument("--status", default="open")
    
    pc = sub.add_parser("cancel")
    pc.add_argument("order_id")
    
    sub.add_parser("close_all").add_argument("--reason", default="manual emergency")
    
    sub.add_parser("validate")  # placeholder for testing
    
    args = parser.parse_args()
    
    if args.cmd == "buy":
        result = place_buy(args.symbol, args.qty, args.limit_price, args.stop, args.target, args.reason)
    elif args.cmd == "sell":
        result = place_sell(args.symbol, args.qty, args.reason)
    elif args.cmd == "list_orders":
        result = list_orders(args.status)
    elif args.cmd == "cancel":
        result = cancel_order(args.order_id)
    elif args.cmd == "close_all":
        result = close_all_positions(args.reason)
    else:
        parser.print_help()
        sys.exit(1)
    
    print(json.dumps(result, indent=2))

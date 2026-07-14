#!/usr/bin/env python3
"""
agent.py - Autonomous trading cycle orchestrator.

Gathers market data, calls an LLM to make decisions,
executes orders through validate_order(), and writes journal.

Supports two LLM backends (set LLM_PROVIDER in .env):
  - "groq"      Free tier. Uses llama-3.3-70b-versatile. Good for testing.
  - "anthropic"  Paid. Uses claude-sonnet-4-6. Better for production.

Usage:
    python scripts/agent.py              # Full cycle (research + decide + trade)
    python scripts/agent.py --dry-run    # Decide but don't submit orders
    python scripts/agent.py --premarket  # Pre-market scan only (no orders)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# Add parent dir to path so we can import sibling scripts
sys.path.insert(0, str(Path(__file__).parent))

import research
import trade as trade_module
import outcomes
import journal as journal_module

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

MODEL = "claude-sonnet-4-6" if LLM_PROVIDER == "anthropic" else "llama-3.3-70b-versatile"
_anthropic_client = None
_groq_client = None


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY must be set in .env")
        import anthropic as _anthropic
        _anthropic_client = _anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY must be set in .env (get free key at console.groq.com)")
        from openai import OpenAI as _OpenAI
        import httpx as _httpx
        cert_bundle = str(Path(__file__).parent.parent / "corporate_certs.pem")
        verify = cert_bundle if Path(cert_bundle).exists() else True
        _groq_client = _OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            http_client=_httpx.Client(verify=verify)
        )
    return _groq_client

CLAUDE_MD = Path(__file__).parent.parent / "CLAUDE.md"
WATCHLIST_PATH = Path(__file__).parent.parent / "data" / "watchlist.json"

# Condensed system prompt for Groq (stays under 12K TPM free tier limit)
GROQ_SYSTEM = """Disciplined paper trading agent. $1000 paper account.
Rules: max $100/position (10%), min $50 order, max 5 open, keep $250+ cash. No options, no crypto, no leveraged ETFs.

BUY entry_limit = current_price + 0.3% (round to 2 decimals). SHORT entry_limit = current_price - 0.3%.
Fractional qty: floor(100 / entry_limit * 100) / 100. Verify qty × entry_limit ≥ $50.

OUTPUT BUY when one setup applies and no rejection fires:
  A. MA20 PULLBACK — price within 4% above MA20, RSI 38-60, above MA50.
  B. OVERSOLD BOUNCE — RSI < 42, above MA50.
  C. NEWS CATALYST — analyst upgrade/PT raise, earnings beat, product launch + above MA50, RSI ≤ 65.

OUTPUT SHORT when one setup applies and no rejection fires:
  D. BREAKDOWN — below MA50 AND below MA20, RSI > 55 (not yet oversold), bearish catalyst.
  E. OVERBOUGHT FADE — RSI > 70, below MA50, analyst downgrade or guidance cut.

REJECT for BUY: RSI > 65 | earnings ≤ 3 days | below MA50.
REJECT for SHORT: RSI < 45 (oversold, covering risk) | earnings ≤ 3 days | above MA50.

BUY bracket: stop 4-8% BELOW entry, target 8-12% ABOVE entry (min 1.5:1 R:R).
SHORT bracket: stop 4-8% ABOVE entry, target 8-12% BELOW entry (min 1.5:1 R:R).

One JSON per candidate:
{"action":"BUY","ticker":"X","qty":0.00,"entry_limit":0.00,"stop_loss":0.00,"take_profit":0.00,"confidence":"MEDIUM","thesis":"Setup A/B/C: <one sentence>."}
{"action":"SHORT","ticker":"X","qty":0.00,"entry_limit":0.00,"stop_loss":0.00,"take_profit":0.00,"confidence":"MEDIUM","thesis":"Setup D/E: <one sentence>."}
{"action":"NO TRADE","ticker":"X","reason":"<exact rejection rule>"}"""


def load_system_prompt():
    return CLAUDE_MD.read_text() if CLAUDE_MD.exists() else ""


def load_watchlist():
    if WATCHLIST_PATH.exists():
        return json.loads(WATCHLIST_PATH.read_text())
    return {"priority_tickers": []}


# ── Tool definitions exposed to Claude ──────────────────────────────────────

TOOLS = [
    {
        "name": "get_account",
        "description": "Get current account state: equity, cash, open positions, PDT count.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "is_market_open",
        "description": "Check if the market is currently open and get next open/close times.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_market_snapshot",
        "description": "Get SPY, QQQ, IWM technicals and market status as macro context.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_bars",
        "description": "Get daily OHLCV bars for a symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "days": {"type": "integer", "default": 60}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_quote",
        "description": "Get current bid/ask quote for a symbol.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"]
        }
    },
    {
        "name": "get_news",
        "description": "Get recent news for a symbol (last 24h).",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "hours": {"type": "integer", "default": 24}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "calc_technicals",
        "description": "Calculate MA20, MA50, RSI14, volume ratio for a symbol.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"]
        }
    },
    {
        "name": "screener_movers",
        "description": "Scan the universe for top movers by volume × price change.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "check_earnings_calendar",
        "description": "Check if any symbols have earnings in the next 3 trading days.",
        "input_schema": {
            "type": "object",
            "properties": {"symbols": {"type": "string", "description": "Comma-separated list"}},
            "required": ["symbols"]
        }
    },
    {
        "name": "daytrade_count",
        "description": "Check PDT compliance: how many day trades used in last 5 days.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "place_buy",
        "description": "Place a bracket BUY order. Always validated first — invalid orders are rejected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "qty": {"type": "number", "description": "Fractional shares allowed, e.g. 0.28"},
                "limit_price": {"type": "number"},
                "stop_price": {"type": "number"},
                "target_price": {"type": "number"},
                "reason": {"type": "string"}
            },
            "required": ["symbol", "qty", "limit_price", "stop_price", "target_price", "reason"]
        }
    },
    {
        "name": "place_sell",
        "description": "Sell an existing position (market order).",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "qty": {"type": "integer"},
                "reason": {"type": "string"}
            },
            "required": ["symbol", "qty", "reason"]
        }
    },
    {
        "name": "write_journal",
        "description": "Write a structured journal entry for this cycle.",
        "input_schema": {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "description": "The full cycle journal entry as a JSON object"
                }
            },
            "required": ["payload"]
        }
    }
]


def _trim(result: dict, name: str) -> dict:
    """Trim large tool results to stay within Groq free-tier token limits."""
    if name == "screener_movers":
        result = dict(result)
        result["movers"] = result.get("movers", [])[:8]
    elif name == "get_bars":
        result = dict(result)
        result["bars"] = result.get("bars", [])[-20:]  # last 20 days is enough
    elif name == "get_news":
        result = dict(result)
        result["items"] = result.get("items", [])[:5]
        for item in result["items"]:
            item.pop("summary", None)  # drop summaries to save tokens
    elif name == "get_market_snapshot":
        # Strip bars data from indices, keep just the summary fields
        result = dict(result)
    return result


def dispatch_tool(name: str, inputs: dict, dry_run: bool = False) -> dict:
    """Route a tool call to the appropriate function."""
    try:
        if name == "get_account":
            result = research.get_account()
        elif name == "is_market_open":
            result = research.is_market_open()
        elif name == "get_market_snapshot":
            result = research.get_market_snapshot()
        elif name == "get_bars":
            result = research.get_bars(inputs["symbol"], inputs.get("days", 30))
        elif name == "get_quote":
            result = research.get_quote(inputs["symbol"])
        elif name == "get_news":
            result = research.get_news(inputs["symbol"], inputs.get("hours", 24))
        elif name == "calc_technicals":
            result = research.calc_technicals(inputs["symbol"])
        elif name == "screener_movers":
            result = research.screener_movers()
        elif name == "check_earnings_calendar":
            result = research.check_earnings_calendar(inputs["symbols"])
        elif name == "daytrade_count":
            result = research.daytrade_count()
        elif name == "place_buy":
            if dry_run:
                valid, msg = trade_module.validate_order(
                    inputs["symbol"], "buy", inputs["qty"],
                    inputs["limit_price"], inputs["stop_price"], inputs["target_price"]
                )
                return {"status": "DRY_RUN", "would_be_valid": valid, "validation_msg": msg, **inputs}
            return trade_module.place_buy(
                inputs["symbol"], round(float(inputs["qty"]), 2), inputs["limit_price"],
                inputs["stop_price"], inputs["target_price"], inputs["reason"]
            )
        elif name == "place_sell":
            if dry_run:
                valid, msg = trade_module.validate_order(
                    inputs["symbol"], "sell", inputs["qty"], None, None, None
                )
                return {"status": "DRY_RUN", "would_be_valid": valid, "validation_msg": msg, **inputs}
            return trade_module.place_sell(inputs["symbol"], inputs["qty"], inputs["reason"])
        elif name == "write_journal":
            return journal_module.write_entry(inputs["payload"])
        else:
            return {"error": f"Unknown tool: {name}"}

        return _trim(result, name) if LLM_PROVIDER != "anthropic" else result
    except Exception as e:
        return {"error": str(e), "tool": name}


def _tools_for_provider():
    """Convert tool definitions to the format expected by each provider."""
    if LLM_PROVIDER == "anthropic":
        return TOOLS  # Anthropic format (input_schema)
    # OpenAI/Groq format (parameters instead of input_schema)
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"]
            }
        }
        for t in TOOLS
    ]


def _call_llm(system: str, messages: list) -> tuple[str, list]:
    """
    Call the configured LLM provider.
    Returns (stop_reason, content_blocks).
    Content blocks are dicts with keys: type, text (optional), name, input, id.
    For Groq, uses GROQ_SYSTEM (condensed) instead of full CLAUDE.md.
    """
    effective_system = system if LLM_PROVIDER == "anthropic" else GROQ_SYSTEM

    if LLM_PROVIDER == "anthropic":
        resp = get_anthropic_client().messages.create(
            model=MODEL,
            max_tokens=8192,
            system=effective_system,
            tools=_tools_for_provider(),
            messages=messages,
        )
        blocks = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        return resp.stop_reason, blocks

    else:  # groq / openai-compatible
        # Convert messages: tool_result -> tool role
        oai_messages = [{"role": "system", "content": effective_system}]
        for m in messages:
            if m["role"] == "user" and isinstance(m["content"], list):
                # tool results
                for tr in m["content"]:
                    oai_messages.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_use_id"],
                        "content": tr["content"]
                    })
            elif m["role"] == "assistant" and isinstance(m["content"], list):
                tool_calls = []
                text_parts = []
                for b in m["content"]:
                    if b["type"] == "text":
                        text_parts.append(b["text"])
                    elif b["type"] == "tool_use":
                        tool_calls.append({
                            "id": b["id"],
                            "type": "function",
                            "function": {"name": b["name"], "arguments": json.dumps(b["input"])}
                        })
                msg = {"role": "assistant", "content": " ".join(text_parts) or None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                oai_messages.append(msg)
            else:
                oai_messages.append({"role": m["role"], "content": m["content"]})

        # Single-shot Groq path uses no tools — plain text completion only
        resp = get_groq_client().chat.completions.create(
            model=MODEL,
            max_tokens=8192,
            messages=oai_messages
        )
        choice = resp.choices[0]
        msg = choice.message
        blocks = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        if msg.tool_calls:
            for tc in msg.tool_calls:
                blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments)
                })
        stop = "tool_use" if msg.tool_calls else "end_turn"
        return stop, blocks


def _collect_market_data(top_n: int = 5) -> dict:
    """
    Pre-collect all market data in Python. Used by Groq path to avoid
    accumulating context across multiple tool calls (free tier token limits).
    """
    print("  Fetching account...")
    account = research.get_account()
    pdt = research.daytrade_count()

    print("  Fetching market snapshot...")
    market = research.get_market_snapshot()
    clock = research.is_market_open()

    print("  Running screener (top 20)...")
    movers = research.screener_movers()
    all_movers = movers.get("movers", [])[:20]  # full screener list

    watchlist = load_watchlist()
    priority_syms = [t["symbol"] for t in watchlist.get("priority_tickers", [])]
    screener_syms = [m["symbol"] for m in all_movers]

    # Exclude symbols with open positions — never average down or buy more of a held stock
    open_symbols = {p["symbol"] for p in account.get("positions", [])}

    # Merge priority + screener, dedupe, exclude held, cap at 20
    universe = list(dict.fromkeys(
        s for s in (priority_syms + screener_syms) if s not in open_symbols
    ))[:20]
    print(f"  Universe ({len(universe)} candidates, excluding held: {open_symbols or 'none'}): {universe}")

    # Pre-filter to top_n for deep analysis using setup quality scores.
    # setup_score ranks pullbacks/oversold/breakouts — not raw momentum chasers.
    screener_scores = {m["symbol"]: m.get("setup_score", 0)
                       for m in all_movers}
    def score(sym):
        base = screener_scores.get(sym, 0)
        return base * 1.5 if sym in priority_syms else base  # priority boost

    deep_candidates = sorted(universe, key=score, reverse=True)[:top_n]
    print(f"  Deep analysis on top {top_n}: {deep_candidates}")

    technicals = {}
    news = {}
    for sym in deep_candidates:
        try:
            technicals[sym] = research.calc_technicals(sym)
        except Exception as e:
            technicals[sym] = {"error": str(e)}
        try:
            n = research.get_news(sym, hours=24)
            news[sym] = [i["headline"] for i in n.get("items", [])[:2]]
        except Exception:
            news[sym] = []

    earnings = {}
    try:
        cal = research.check_earnings_calendar(",".join(deep_candidates))
        for item in cal.get("earnings_check", []):
            earnings[item["symbol"]] = item
    except Exception:
        pass

    print("  Fetching live quotes for deep candidates...")
    quotes = {}
    for sym in deep_candidates:
        try:
            q = research.get_quote(sym)
            if "error" not in q:
                quotes[sym] = q
        except Exception:
            pass

    return {
        "account": account,
        "pdt": pdt,
        "market": market,
        "clock": clock,
        "all_movers": all_movers,          # full 20 for display
        "deep_candidates": deep_candidates, # top 7 for LLM analysis
        "technicals": technicals,
        "news": news,
        "earnings": earnings,
        "quotes": quotes,
    }


def _run_groq_cycle(dry_run: bool, premarket: bool, system: str):
    """
    Single-shot Groq cycle: collect data in Python, make one LLM call,
    parse decisions, execute orders. Stays within free-tier token limits.
    """
    print("\nCollecting market data...")
    data = _collect_market_data(top_n=7)  # screener gets 20, deep analysis on top 7

    mode_note = "DRY RUN - analyze only, no orders." if dry_run else (
                "PRE-MARKET - plan only, no orders." if premarket else "")

    acct = data["account"]
    mkt = data["market"]["indices"]
    spy = mkt.get("SPY", {})
    qqq = mkt.get("QQQ", {})

    # Compact screener summary — show RSI so LLM can immediately apply RSI>65 filter
    screener_summary = ", ".join(
        f"{m['symbol']}(RSI:{m['rsi14']},{m['5d_change_pct']:+.1f}%,vol:{m['vol_ratio']:.1f}x)"
        for m in data["all_movers"]
        if m.get("rsi14") is not None
    )

    # Pre-classify each deep candidate's setup in Python using live price.
    # Uses live ask so MA/RSI checks reflect today's market, not yesterday's close.
    # Pre-computes entry/stop/target so the LLM just copies numbers, no arithmetic needed.
    import math as _math

    def classify_setup(live_price: float, t: dict, news_headlines: list) -> str:
        rsi  = t.get("rsi14") or 50.0
        ma20 = t.get("ma20") or live_price
        ma50 = t.get("ma50") or live_price

        if rsi > 65:
            return "REJECT:RSI>65"
        if live_price < ma50:
            return "REJECT:belowMA50"

        pct_above_ma20 = ((live_price - ma20) / ma20 * 100) if ma20 else 0
        has_strong_news = any(
            kw in h.lower() for h in news_headlines
            for kw in ("upgrade", "raises price target", "beat", "launch", "contract", "raises pt")
        )

        setups = []
        if 0 <= pct_above_ma20 <= 4 and 38 <= rsi <= 60:
            setups.append(f"A({pct_above_ma20:+.1f}%abvMA20)")
        if rsi < 42:
            setups.append("B(oversold)")
        if has_strong_news:
            setups.append("C(news)")

        return "/".join(setups) if setups else "NO_SETUP"

    candidate_rows = []
    for sym in data["deep_candidates"]:
        t = data["technicals"].get(sym, {})
        if "error" in t:
            continue
        quote = data.get("quotes", {}).get(sym, {})
        ask   = quote.get("ask") if quote.get("ask", 0) > 0 else None
        # live_price: use ask if available, fall back to technicals close
        live_price = ask if ask else (t.get("current_price") or 0.0)
        if live_price <= 0:
            continue

        earn      = data["earnings"].get(sym, {})
        earn_days = earn.get("days_until")
        earn_flag = f"⚠EARNINGS{earn_days}d" if isinstance(earn_days, int) and earn_days <= 3 else ""

        setup = classify_setup(live_price, t, data["news"].get(sym, []))

        # ── Check Mark wiring into BUY decisions ─────────────────
        # If the strict Check Mark long pattern is detected from 15m/5m candles,
        # override setup + the exact entry/stop/target so the LLM copies them.
        try:
            intraday_15m = research.get_intraday_bars(sym, minutes=15, lookback_hours=10)
            intraday_5m  = research.get_intraday_bars(sym, minutes=5,  lookback_hours=10)
            bars15 = intraday_15m.get("bars", [])
            bars5  = intraday_5m.get("bars", [])

            prev_day_high = None
            prev_day_low = None
            daily = research.get_bars(sym, days=3).get("bars", [])
            if len(daily) >= 2:
                prev_day = daily[-2]
                prev_day_high = float(prev_day.get("high"))
                prev_day_low  = float(prev_day.get("low"))

            checkmark_tech = {}
            if prev_day_high is not None and prev_day_low is not None and bars15 and bars5:
                checkmark_tech.update(
                    _compute_check_mark_long_features(
                        bars15=bars15,
                        bars5=bars5,
                        prev_day_high=prev_day_high,
                        prev_day_low=prev_day_low,
                    )
                )

            if checkmark_tech.get("checkmark_long_ready"):
                atr_pct_local = t.get("atr14_pct") or 1.5
                levels = _compute_day_trade_levels(live_price, atr_pct_local, tech=checkmark_tech)

                if levels.get("entry") and levels.get("stop") and levels.get("target"):
                    setup = "Check Mark Long"
                    entry  = float(levels["entry"])
                    stop   = float(levels["stop"])
                    target = float(levels["target"])
        except Exception:
            pass

        # Pre-compute order parameters — LLM copies these exactly, no arithmetic
        # (default levels; may be overridden above)
        if "entry" not in locals():
            entry  = round(live_price * 1.001, 2)      # ask + 0.1% to cross spread
        if "stop" not in locals():
            stop   = round(entry * 0.950, 2)          # 5% stop
        if "target" not in locals():
            target = round(entry * 1.100, 2)         # 10% target
        qty    = _math.floor((100.0 / entry) * 100) / 100  # max $100 position, floored

        action_hint = "→ BUY" if (setup not in ("NO_SETUP",) and not setup.startswith("REJECT") and not earn_flag) else "→ NO TRADE"

        row = (
            f"{sym} [{action_hint}] SETUP={setup}"
            f" | MA20=${t.get('ma20')} MA50=${t.get('ma50')} RSI={t.get('rsi14')} 5d={t.get('5d_change_pct',0):+.1f}%"
            f" | USE: entry={entry} stop={stop} target={target} qty={qty}"
        )
        if earn_flag:
            row += f" | {earn_flag}"
        headlines = data["news"].get(sym, [])
        if headlines:
            row += f" | news: {'; '.join(headlines[:2])}"
        candidate_rows.append(row)

    candidates_block = "\n".join(candidate_rows)

    prompt = f"""Date: {datetime.now().strftime('%Y-%m-%d %H:%M ET')} {mode_note}
Account: ${acct['account_value']:.0f} total, ${acct['cash']:.0f} cash, {acct['positions_count']}/5 positions, {data['pdt']['daytrade_count_5days']}/3 day trades
Market: open={data['clock']['is_open']} | SPY 5d:{spy.get('5d_change_pct',0):+.1f}% RSI:{spy.get('rsi14','?')} | QQQ 5d:{qqq.get('5d_change_pct',0):+.1f}% RSI:{qqq.get('rsi14','?')}

SCREENER (top 20): {screener_summary}

DEEP CANDIDATES (entry/stop/target/qty pre-calculated from live price — copy exactly):
{candidates_block}

Instructions:
- "→ BUY": copy entry/stop/target/qty exactly into the JSON. Do not recalculate.
- "→ NO TRADE": output NO TRADE with the SETUP or EARNINGS reason shown.
- If ⚠EARNINGS within 3 days: NO TRADE regardless of setup.
Output one JSON per candidate, then one-sentence reflection."""

    print("\nAsking LLM for decisions...")
    messages = [{"role": "user", "content": prompt}]
    _, blocks = _call_llm(system, messages)

    response_text = ""
    for b in blocks:
        if b["type"] == "text":
            print(b["text"])
            response_text += b["text"]

    # Execute orders and capture results for journal annotation
    execution_results = {}
    if dry_run:
        execution_results = _validate_from_text(response_text)
    elif not premarket:
        execution_results = _execute_from_text(response_text, data["account"])

    # Write journal with decisions annotated with execution status
    try:
        decisions = _parse_decisions_from_text(response_text)

        for d in decisions:
            action = d.get("action", "")
            ticker = d.get("ticker", "")
            if action in ("BUY", "SELL"):
                if dry_run:
                    res = execution_results.get(ticker, {})
                    d["execution_status"] = "DRY_RUN" if res.get("would_be_valid") else "DRY_RUN_REJECTED"
                    d["would_be_valid"] = res.get("would_be_valid", False)
                    d["validation_msg"] = res.get("validation_msg", "No validation result")
                elif premarket:
                    d["execution_status"] = "NOT_SUBMITTED"
                elif ticker in execution_results:
                    res = execution_results[ticker]
                    d["execution_status"] = res.get("status", "ERROR")
                    if res.get("status") == "SUBMITTED":
                        d["order_id"] = res.get("order_id")
                    elif res.get("status") in ("REJECTED", "ERROR"):
                        d["execution_detail"] = res.get("reason") or res.get("error")
                else:
                    d["execution_status"] = "NOT_SUBMITTED"
            elif action == "NO TRADE":
                d["execution_status"] = "NO_TRADE"
            elif action == "HOLD":
                d["execution_status"] = "HOLD"

        import re as _re
        reflection = _re.sub(r'\{[^{}]*"action"[^{}]*\}', '', response_text, flags=_re.DOTALL).strip()
        journal_module.write_entry({
            "cycle_timestamp": datetime.now().isoformat(),
            "provider": f"{LLM_PROVIDER}/{MODEL}",
            "account": data["account"],
            "market_context": data["market"],
            "screener_universe": [m["symbol"] for m in data["all_movers"]],
            "deep_candidates": data["deep_candidates"],
            "decisions": decisions,
            "reflection": reflection[:500],
            "dry_run": dry_run,
        })
        print("\n[Journal entry written]")
    except Exception as e:
        print(f"\n[Journal write failed: {e}]")


def _parse_decisions_from_text(text: str) -> list:
    """Extract all JSON decision blocks from LLM response text."""
    import re
    decisions = []
    # Match any top-level JSON object containing an "action" key
    for block in re.findall(r'\{[^{}]*"action"\s*:\s*"[^"]*"[^{}]*\}', text, re.DOTALL):
        try:
            decisions.append(json.loads(block))
        except Exception:
            continue
    return decisions


def _execute_from_text(text: str, account: dict) -> dict:
    """Parse and execute BUY/SELL decisions. Returns {ticker: result} for journal annotation."""
    import math
    results = {}
    for decision in _parse_decisions_from_text(text):
        action = decision.get("action", "")
        ticker = decision.get("ticker", "")
        if action == "BUY" and ticker:
            raw_qty = float(decision.get("qty", 0))
            entry = float(decision.get("entry_limit", decision.get("entry", 0)) or 0)
            # Floor, then hard-cap at floor(100/entry*100)/100 — LLMs sometimes round up
            qty = math.floor(raw_qty * 100) / 100
            if entry > 0:
                qty = min(qty, math.floor((100.0 / entry) * 100) / 100)
            stop  = float(decision.get("stop_loss",  decision.get("stop",   0)))
            tgt   = float(decision.get("take_profit", decision.get("target", 0)))
            entry = float(decision.get("entry_limit", decision.get("entry",  0)))
            result = trade_module.place_buy(ticker, qty, entry, stop, tgt,
                                            decision.get("thesis", "agent decision"))
            print(f"\n→ ORDER: {json.dumps(result)}")
            results[ticker] = result
            # Persist stop level to state.json so remote routines can monitor it
            if result.get("status") == "SUBMITTED" and stop and tgt:
                trade_module.save_stop_level(ticker, stop, tgt, entry)
        elif action == "SHORT" and ticker:
            import math
            raw_qty = float(decision.get("qty", 0))
            entry = float(decision.get("entry_limit", decision.get("entry", 0)) or 0)
            qty = math.floor(raw_qty * 100) / 100
            if entry > 0:
                qty = min(qty, math.floor((100.0 / entry) * 100) / 100)
            stop = float(decision.get("stop_loss", decision.get("stop", 0)))
            tgt  = float(decision.get("take_profit", decision.get("target", 0)))
            entry = float(decision.get("entry_limit", decision.get("entry", 0)))
            result = trade_module.place_short(ticker, qty, entry, stop, tgt,
                                              decision.get("thesis", "agent decision"))
            print(f"\n→ ORDER: {json.dumps(result)}")
            results[ticker] = result
            if result.get("status") == "SUBMITTED" and stop and tgt:
                trade_module.save_stop_level(ticker, stop, tgt, entry)
        elif action == "SELL" and ticker:
            result = trade_module.place_sell(
                ticker,
                round(float(decision.get("qty", 0)), 2),
                decision.get("reason", "agent decision")
            )
            print(f"\n→ ORDER: {json.dumps(result)}")
            results[ticker] = result
            if result.get("status") == "SUBMITTED":
                trade_module.remove_stop_level(ticker)
        elif action == "COVER" and ticker:
            result = trade_module.place_cover(
                ticker,
                round(float(decision.get("qty", 0)), 2),
                decision.get("reason", "agent decision")
            )
            print(f"\n→ ORDER: {json.dumps(result)}")
            results[ticker] = result
            if result.get("status") == "SUBMITTED":
                trade_module.remove_stop_level(ticker)
    return results


def _validate_from_text(text: str) -> dict:
    """Validate parsed BUY/SELL decisions without submitting orders."""
    import math
    results = {}
    for decision in _parse_decisions_from_text(text):
        action = decision.get("action", "")
        ticker = decision.get("ticker", "")
        if action == "BUY" and ticker:
            raw_qty = float(decision.get("qty", 0))
            entry = float(decision.get("entry_limit", decision.get("entry", 0)) or 0)
            qty = math.floor(raw_qty * 100) / 100
            if entry > 0:
                qty = min(qty, math.floor((100.0 / entry) * 100) / 100)
            stop = float(decision.get("stop_loss", decision.get("stop", 0)))
            tgt = float(decision.get("take_profit", decision.get("target", 0)))
            valid, msg = trade_module.validate_order(ticker, "buy", qty, entry, stop, tgt)
            results[ticker] = {
                "status": "DRY_RUN",
                "would_be_valid": valid,
                "validation_msg": msg,
                "symbol": ticker,
                "side": "buy",
                "qty": qty,
            }
            print(f"\n→ DRY RUN VALIDATION: {ticker} buy qty={qty} @ {entry}: {msg}")
        elif action == "SHORT" and ticker:
            raw_qty = float(decision.get("qty", 0))
            entry = float(decision.get("entry_limit", decision.get("entry", 0)) or 0)
            qty = math.floor(raw_qty * 100) / 100
            if entry > 0:
                qty = min(qty, math.floor((100.0 / entry) * 100) / 100)
            stop = float(decision.get("stop_loss", decision.get("stop", 0)))
            tgt = float(decision.get("take_profit", decision.get("target", 0)))
            valid, msg = trade_module.validate_order(ticker, "short", qty, entry, stop, tgt)
            results[ticker] = {
                "status": "DRY_RUN",
                "would_be_valid": valid,
                "validation_msg": msg,
                "symbol": ticker,
                "side": "short",
                "qty": qty,
            }
            print(f"\n→ DRY RUN VALIDATION: {ticker} short qty={qty} @ {entry}: {msg}")
        elif action == "SELL" and ticker:
            qty = round(float(decision.get("qty", 0)), 2)
            valid, msg = trade_module.validate_order(ticker, "sell", qty, None, None, None)
            results[ticker] = {
                "status": "DRY_RUN",
                "would_be_valid": valid,
                "validation_msg": msg,
                "symbol": ticker,
                "side": "sell",
                "qty": qty,
            }
            print(f"\n→ DRY RUN VALIDATION: {ticker} sell qty={qty}: {msg}")
        elif action == "COVER" and ticker:
            qty = round(float(decision.get("qty", 0)), 2)
            valid, msg = trade_module.validate_order(ticker, "cover", qty, None, None, None)
            results[ticker] = {
                "status": "DRY_RUN",
                "would_be_valid": valid,
                "validation_msg": msg,
                "symbol": ticker,
                "side": "cover",
                "qty": qty,
            }
            print(f"\n→ DRY RUN VALIDATION: {ticker} cover qty={qty}: {msg}")
    return results


def _load_stop_levels() -> dict:
    """
    Load stop/target levels for all open positions.
    Primary source: data/state.json (committed to git — visible to remote routines).
    Fallback: local journal files (for levels not yet written to state.json).
    Returns {symbol: {"stop_loss": float, "take_profit": float, "entry_limit": float}}.
    """
    stops = {}

    # Primary: state.json — survives git clone, works in remote routines
    state_file = Path(__file__).parent.parent / "data" / "state.json"
    try:
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        for sym, levels in state.get("active_stops", {}).items():
            stops[sym] = levels
    except Exception:
        pass

    # Fallback: local journal files (supplement, don't overwrite state.json entries)
    import glob
    journal_dir = Path(__file__).parent.parent / "journal"
    for f in sorted(glob.glob(str(journal_dir / "*.json")), reverse=True)[:30]:
        try:
            with open(f) as fh:
                entry = json.load(fh)
            for d in entry.get("decisions", []):
                sym = d.get("ticker", "")
                if (d.get("action") == "BUY" and d.get("execution_status") == "SUBMITTED"
                        and sym not in stops):
                    stops[sym] = {
                        "stop_loss":  d.get("stop_loss"),
                        "take_profit": d.get("take_profit"),
                        "entry_limit": d.get("entry_limit"),
                    }
        except Exception:
            continue
    return stops


def _check_fractional_stops(dry_run: bool = False):
    """
    For each open fractional position, compare current price to stop/target.
    Exits the position if the stop is breached. Fractional orders have no
    automatic stop orders in Alpaca, so this is manual enforcement.
    """
    try:
        account = research.get_account()
        positions = account.get("positions", [])
        fractional = [p for p in positions if (float(p["qty"]) % 1) != 0]
        if not fractional:
            return

        stop_levels = _load_stop_levels()
        print(f"\n[Stop check] {len(fractional)} fractional position(s):")

        for p in fractional:
            sym = p["symbol"]
            current = p.get("current_price") or p.get("avg_entry_price")
            levels = stop_levels.get(sym, {})
            stop = levels.get("stop_loss")
            target = levels.get("take_profit")

            stop_hit = stop and current and current <= stop
            target_hit = target and current and current >= target

            if stop_hit or target_hit:
                reason = f"stop hit @ ${current} (stop ${stop})" if stop_hit else f"target hit @ ${current} (target ${target})"
                print(f"  {sym}: {reason} — {'DRY RUN skip' if dry_run else 'SELLING'}")
                if not dry_run:
                    result = trade_module.place_sell(sym, float(p["qty"]), reason)
                    print(f"  → {json.dumps(result)}")
                    if result.get("status") == "SUBMITTED":
                        trade_module.remove_stop_level(sym)
            else:
                stop_str = f"stop ${stop}" if stop else "no stop recorded"
                target_str = f"target ${target}" if target else "no target recorded"
                print(f"  {sym}: ${current} — {stop_str} | {target_str} — HOLD")
    except Exception as e:
        print(f"[Stop check failed: {e}]")


def run_cycle(dry_run: bool = False, premarket: bool = False):
    """Run one full agent cycle."""
    system = load_system_prompt()

    print(f"\n{'='*60}")
    print(f"Trading Agent Cycle — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Provider: {LLM_PROVIDER.upper()} / {MODEL}")
    print(f"Mode: {'DRY RUN' if dry_run else 'PREMARKET' if premarket else 'LIVE'}")
    print(f"{'='*60}\n")

    if not premarket:
        _check_fractional_stops(dry_run=dry_run)

    # Groq free tier: single-shot approach (no accumulating context)
    if LLM_PROVIDER != "anthropic":
        _run_groq_cycle(dry_run, premarket, system)
        print("\n[Agent cycle complete]")
        return

    # Anthropic: full agentic tool-use loop
    watchlist = load_watchlist()
    priority = [t["symbol"] for t in watchlist.get("priority_tickers", [])]

    mode_note = ""
    if dry_run:
        mode_note = "\n\nDRY RUN MODE: Research and decide as normal, but do NOT call place_buy or place_sell. Only call write_journal at the end."
    if premarket:
        mode_note = "\n\nPRE-MARKET MODE: Market is not yet open. Run your analysis and form a watchlist for when it opens. Do NOT place orders. Write a pre-market journal entry."

    user_msg = f"""Run a full trading cycle now. Today is {datetime.now().strftime('%A, %B %d %Y %H:%M ET')}.

Priority watchlist tickers: {', '.join(priority)}

Follow your decision framework from CLAUDE.md exactly:
1. Account health check (get_account, daytrade_count)
2. Market context (is_market_open, get_market_snapshot)
3. Screener scan (screener_movers) + watchlist
4. Deep analysis on top 5 candidates (calc_technicals, get_news, check_earnings_calendar)
5. Generate decisions (BUY / SELL / HOLD / NO TRADE) with full thesis
6. Execute any qualifying orders through place_buy / place_sell
7. Write journal entry (write_journal)

Be disciplined. Default to NO TRADE. Every BUY must have a stop and target.{mode_note}"""

    messages = [{"role": "user", "content": user_msg}]

    while True:
        stop_reason, blocks = _call_llm(system, messages)

        for b in blocks:
            if b["type"] == "text" and b.get("text"):
                print(b["text"])

        if stop_reason == "end_turn":
            print("\n[Agent cycle complete]")
            break

        if stop_reason != "tool_use":
            print(f"\n[Unexpected stop reason: {stop_reason}]")
            break

        tool_results = []
        for b in blocks:
            if b["type"] != "tool_use":
                continue
            print(f"\n→ Tool: {b['name']}({json.dumps(b['input'], separators=(',', ':'))})")
            result = dispatch_tool(b["name"], b["input"], dry_run=dry_run)
            result_str = json.dumps(result)
            print(f"  ← {result_str[:300]}{'...' if len(result_str) > 300 else ''}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": b["id"],
                "content": result_str
            })

        messages.append({"role": "assistant", "content": blocks})
        messages.append({"role": "user", "content": tool_results})


def print_status():
    """Quick portfolio snapshot — no LLM call, no cycle. Used by Dispatch routine."""
    import glob
    journal_dir = Path(__file__).parent.parent / "journal"
    journals = sorted(glob.glob(str(journal_dir / "*.json")), reverse=True)
    last_cycle = "never"
    if journals:
        try:
            with open(journals[0]) as f:
                j = json.load(f)
                last_cycle = j.get("cycle_timestamp", "unknown")
        except Exception:
            pass

    try:
        account = research.get_account()
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return

    positions = account.get("positions", [])
    pos_summary = [
        {
            "symbol": p.get("symbol"),
            "qty": p.get("qty"),
            "entry_price": p.get("avg_entry_price"),
            "current_price": p.get("current_price"),
            "unrealized_pl": p.get("unrealized_pl"),
            "unrealized_plpc": round(p.get("unrealized_plpc", 0) * 100, 2),
        }
        for p in positions
    ]

    print(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "account_value": account["account_value"],
        "cash": account["cash"],
        "buying_power": account["buying_power"],
        "positions_count": account["positions_count"],
        "positions": pos_summary,
        "last_cycle": last_cycle,
        "is_paper": account.get("is_paper", True),
    }, indent=2))


def analyze_symbol(symbol: str) -> dict:
    """
    On-demand deep analysis for a single symbol.
    Returns a structured dict with:
      - technicals, news, earnings
      - stock_trade: BUY setup or SKIP
      - options_play: specific strategy with strikes/expiry concept
      - verdict: one-line summary
      - raw_analysis: full LLM text
    """
    symbol = symbol.strip().upper()

    # Gather data
    try:
        tech = research.calc_technicals(symbol)
    except Exception as e:
        return {"error": f"Could not fetch data for {symbol}: {e}"}

    # Always fetch a live quote — this is more current than the bar close
    try:
        quote = research.get_quote(symbol)
    except Exception:
        quote = {}

    # Price resolution: prefer last trade > NBBO ask > bar close
    # get_quote now returns last_trade_price when available
    trade_price = quote.get("last_trade_price")
    live_ask    = quote.get("ask")
    live_bid    = quote.get("bid")
    bar_close   = tech.get("current_price")
    is_stale    = quote.get("is_stale", True)
    data_age    = quote.get("data_age_minutes")

    # Use the freshest price available
    canonical_price = trade_price or (live_ask if not is_stale else None) or bar_close or 0
    price = canonical_price

    # Pick the best timestamp to display
    quote_ts = quote.get("last_trade_timestamp") or quote.get("timestamp", "")

    # Warn if canonical price diverges significantly from bar close (>2%)
    price_divergence = None
    if canonical_price and bar_close and bar_close > 0:
        price_divergence = round((canonical_price - bar_close) / bar_close * 100, 2)

    try:
        news_data = research.get_news(symbol, hours=48)
        headlines = [i["headline"] for i in news_data.get("items", [])[:5]]
    except Exception:
        headlines = []

    try:
        cal = research.check_earnings_calendar(symbol)
        earnings_info = cal.get("earnings_check", [{}])[0]
    except Exception:
        earnings_info = {}

    system = """You are a professional buy-side analyst and options strategist.
Given market data for a stock, provide:
1. A stock trade recommendation (BUY setup with entry/stop/target, or SKIP)
2. A catalyst-driven options play (call or put, specific strikes, expiry tied to a REAL upcoming event)
3. A one-line verdict

CRITICAL PRICE RULE: Base ALL price levels (entry, stop, target, strike prices) on the CURRENT PRICE provided,
not the MA or bar close. Entry for a BUY should be at or just above the current price.

CRITICAL OPTIONS RULE: The options play must be tied to a SPECIFIC catalyst:
- If there is recent news → explain how that news historically moves this stock/sector, then size the play
- If earnings are approaching → time the expiry past the earnings date
- If no clear catalyst → recommend Skip for options, not a generic spread
- Strike prices must be realistic: ATM or slightly OTM relative to the current price
- Prefer bull call spread (bullish) or bear put spread (bearish) over naked options
- Always state: what event makes this profitable, and when it should happen

Output valid JSON only — no markdown, no extra text."""

    price_source = "last trade" if trade_price else ("NBBO ask" if live_ask and not is_stale else "bar close (stale)")
    age_str = f"{data_age:.0f}m ago" if data_age is not None else "unknown age"

    today_label = datetime.now().strftime("%A, %B %d, %Y")

    prompt = f"""Analyze {symbol} for today ({today_label}).

CURRENT PRICE (use this as your anchor for ALL price levels):
- Price: ${canonical_price} [{price_source}, {age_str}] ← USE THIS for entry, stop, target, strikes
- Bid/Ask: ${live_bid or '?'} / ${live_ask or '?'} | Spread: {quote.get('spread_pct', '?')}%
- Quote time: {quote_ts[:19].replace('T',' ') if quote_ts else 'unknown'}
- Data fresh: {'YES' if not is_stale else 'NO — market closed, using last trade price'}

TECHNICALS (based on yesterday's close = ${bar_close}):
- MA20: ${tech.get('ma20', '?')} | MA50: ${tech.get('ma50', '?')}
- RSI14: {tech.get('rsi14', '?')} | Above MA20: {tech.get('above_ma20', '?')} | Above MA50: {tech.get('above_ma50', '?')}
- Volume ratio vs 20d avg: {tech.get('vol_ratio', '?')}x
- 5d change: {tech.get('5d_change_pct', '?')}% | 20d change: {tech.get('20d_change_pct', '?')}%

EARNINGS: {earnings_info.get('earnings_date', 'unknown')} ({earnings_info.get('days_until', '?')} days away)

RECENT NEWS:
{chr(10).join(f'- {h}' for h in headlines) if headlines else '- No recent news'}

Return ONLY this JSON (fill in all fields):
{{
  "symbol": "{symbol}",
  "current_price": {price},
  "verdict": "one sentence: bullish/bearish/neutral and why",
  "stock_trade": {{
    "action": "BUY or SKIP",
    "entry": null,
    "stop": null,
    "target": null,
    "stop_pct": null,
    "reward_risk": null,
    "confidence": "HIGH/MEDIUM/LOW",
    "thesis": "2-3 sentences",
    "risk": "main bear case"
  }},
  "options_play": {{
    "direction": "bullish / bearish / neutral  (neutral → iron condor will be built; use skip field instead if truly no play)",
    "catalyst": "specific news or upcoming event driving this — or NONE if no clear catalyst",
    "historical_reaction": "how this type of news typically moves this stock (from training knowledge)",
    "why": "2-3 sentence rationale for the options direction",
    "ideal_outcome": "what needs to happen for this to win",
    "risk": "what would invalidate this",
    "skip": false
  }},
  "skip_reason": null
}}

IMPORTANT for options_play:
- Do NOT output strike prices, expiry dates, structure strings, or any dollar amounts — Python will compute those.
- Set skip=true only if there is genuinely no catalyst and no directional edge.
- direction must be exactly "bullish", "bearish", or "neutral"."""

    try:
        if LLM_PROVIDER == "anthropic":
            resp = get_anthropic_client().messages.create(
                model=MODEL, max_tokens=1500,
                system=system,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = resp.content[0].text
        else:
            resp = get_groq_client().chat.completions.create(
                model=MODEL, max_tokens=1500,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
            )
            raw = resp.choices[0].message.content

        # Strip markdown code fences if present
        import re as _re
        raw = _re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=_re.MULTILINE)
        raw = _re.sub(r'\s*```$', '', raw.strip(), flags=_re.MULTILINE)

        result = json.loads(raw)
        result["raw_analysis"] = raw
        result["technicals"] = tech
        result["headlines"] = headlines
        result["earnings"] = earnings_info

        # ── Python computes ALL options structure numbers ──────────────
        opt_llm  = result.get("options_play", {})
        opt_dir  = opt_llm.get("direction", "neutral").lower()
        opt_skip = opt_llm.get("skip", False)

        if not opt_skip and canonical_price and canonical_price > 0:
            hist_vol   = tech.get("hist_vol_30d", 35.0)
            confidence = result.get("stock_trade", {}).get("confidence", "MEDIUM")
            computed   = _build_options_structure(canonical_price, opt_dir,
                                                  target_weeks=4, hist_vol_pct=hist_vol,
                                                  confidence=confidence)
            # Always compute BOTH plays so dashboard can show both tabs
            _dir_type  = {"bullish": "bull_call_spread", "bearish": "bear_put_spread"}.get(opt_dir)
            _crd_type  = {"bullish": "bull_put_spread",  "bearish": "bear_call_spread"}.get(opt_dir, "iron_condor")
            directional_play = (_build_options_structure(canonical_price, opt_dir,
                                                          target_weeks=4, hist_vol_pct=hist_vol,
                                                          force_type=_dir_type)
                                 if _dir_type else None)
            theta_play = _build_options_structure(canonical_price, opt_dir,
                                                   target_weeks=4, hist_vol_pct=hist_vol,
                                                   force_type=_crd_type)
            expiry_note = (
                f"Expiry {computed['expiry_date']} gives {computed['expiry_weeks']} weeks "
                f"for the catalyst to resolve."
            )
            result["options_play"] = {
                **opt_llm,
                "strategy":          computed["strategy"],
                "strategy_type":     computed.get("strategy_type", ""),
                "is_credit":         computed.get("is_credit", False),
                "structure":         computed["structure"],
                "legs_note":         computed.get("legs_note", ""),
                "atm_strike":        computed["atm_strike"],
                "otm_strike":        computed["otm_strike"],
                "spread_width":      computed["spread_width"],
                "expiry_date":       computed["expiry_date"],
                "expiry_weeks":      computed["expiry_weeks"],
                "expiry_rationale":  expiry_note,
                "hist_vol_pct":      computed["hist_vol_pct"],
                "long_leg_bs":       computed["long_leg_bs"],
                "short_leg_bs":      computed["short_leg_bs"],
                "spread_cost_share": computed["spread_cost_share"],
                "net_credit":        computed.get("net_credit"),
                "credit_per_contract": computed.get("credit_per_contract"),
                "cost_per_contract": computed["cost_per_contract"],
                "max_gain_contract": computed["max_gain_contract"],
                "breakeven":         computed["breakeven"],
                "breakeven_pct":     computed["breakeven_pct"],
                "lower_breakeven":   computed.get("lower_breakeven"),
                "upper_breakeven":   computed.get("upper_breakeven"),
                "profit_zone":       computed.get("profit_zone"),
                "return_pct":        computed["return_pct"],
                "max_loss":          computed["max_loss"],
                "max_gain":          computed["max_gain"],
                "directional_play":  directional_play,
                "theta_play":        theta_play,
            }
            _cat = opt_llm.get("catalyst", "")
            if directional_play:
                outcomes.save_recommendation(symbol, directional_play,
                    catalyst=_cat, confidence=confidence, source="symbol_analysis")
            if theta_play:
                outcomes.save_recommendation(symbol, theta_play,
                    catalyst=_cat, confidence=confidence, source="symbol_analysis")
        else:
            result["options_play"] = {**opt_llm, "strategy": "Skip", "skip": True}

        # ── Day trade signal (Python-computed, no extra LLM call) ──────
        stock_trade = result.get("stock_trade", {})
        day_trade_direction = stock_trade.get("action", "SKIP")
        if day_trade_direction == "BUY" and canonical_price and canonical_price > 0:
            atr_pct  = tech.get("atr14_pct") or 1.5

            # 15-minute + 5-minute chart features for day-trade setup classification
            # (fallback to existing daily-tech logic if intraday data is missing)
            try:
                intraday_15m = research.get_intraday_bars(symbol, minutes=15, lookback_hours=10)
                intraday_5m  = research.get_intraday_bars(symbol, minutes=5,  lookback_hours=10)
                bars15 = intraday_15m.get("bars", [])
                bars5  = intraday_5m.get("bars", [])

                # Previous day OHLC for “blow-off” validation
                prev_day_high = None
                prev_day_low = None
                try:
                    daily = research.get_bars(symbol, days=3).get("bars", [])
                    if len(daily) >= 2:
                        prev_day = daily[-2]
                        prev_day_high = float(prev_day.get("high"))
                        prev_day_low  = float(prev_day.get("low"))
                except Exception:
                    pass

                # Full Check Mark features when we can
                if (
                    prev_day_high is not None
                    and prev_day_low is not None
                    and bars15
                    and len(bars15) >= 10
                    and bars5
                    and len(bars5) >= 20
                ):
                    tech.update(
                        _compute_check_mark_long_features(
                            bars15=bars15,
                            bars5=bars5,
                            prev_day_high=prev_day_high,
                            prev_day_low=prev_day_low,
                        )
                    )

                # Always compute simpler 15m fallback features too
                if bars15 and len(bars15) >= 10:
                    tech.update(_compute_15m_day_trade_features(bars15))
            except Exception:
                pass

            setup    = _detect_day_trade_setup(tech)
            levels   = _compute_day_trade_levels(canonical_price, atr_pct, tech)
            result["day_trade"] = {
                "available":     True,
                "setup_type":    setup,
                "catalyst":      opt_llm.get("catalyst") or stock_trade.get("thesis", "")[:80],
                "why":           stock_trade.get("thesis", ""),
                "risk":          stock_trade.get("risk", ""),
                "exit_rule":     "Exit by 3:45 PM ET — no overnight holds on day trades",
                **levels,
            }
        else:
            result["day_trade"] = {
                "available": False,
                "reason":    "No bullish setup detected — day trade skipped",
            }

        # ── Data freshness & price accuracy metadata ──────────────────
        result["data_meta"] = {
            "canonical_price":      canonical_price,
            "price_source":         price_source,
            "last_trade_price":     trade_price,
            "live_ask":             live_ask,
            "live_bid":             live_bid,
            "bar_close":            bar_close,
            "quote_timestamp":      quote_ts,
            "data_age_minutes":     data_age,
            "is_stale":             is_stale,
            "price_divergence_pct": price_divergence,
            "fetched_at":           datetime.now().isoformat(),
        }

        # ── Validate LLM entry against canonical price ─────────────────
        warnings = []
        trade = result.get("stock_trade", {})
        llm_entry = trade.get("entry")
        if llm_entry and canonical_price and canonical_price > 0:
            entry_diff_pct = (llm_entry - canonical_price) / canonical_price * 100
            if abs(entry_diff_pct) > 5:
                warnings.append(
                    f"Entry ${llm_entry} is {entry_diff_pct:+.1f}% from current price ${canonical_price:.2f} "
                    f"({price_source}). Consider using ${round(canonical_price * 1.002, 2)} as entry instead."
                )
        if price_divergence and abs(price_divergence) > 2:
            warnings.append(
                f"Current price (${canonical_price:.2f}) differs {price_divergence:+.1f}% from yesterday's bar close "
                f"(${bar_close:.2f}). Technicals (MA, RSI) reflect yesterday's close."
            )
        if is_stale:
            age_label = f"{data_age:.0f} min" if data_age else "unknown"
            warnings.append(
                f"Market data is stale ({age_label} old) — market may be closed. "
                f"Price shown is the last known trade (${canonical_price:.2f})."
            )
        elif not quote_ts:
            warnings.append("No live quote available — all prices are based on yesterday's closing bar.")

        result["price_warnings"] = warnings
        return result

    except json.JSONDecodeError:
        return {
            "symbol": symbol,
            "error": "LLM returned non-JSON response",
            "raw_analysis": raw if "raw" in dir() else "",
            "technicals": tech,
            "headlines": headlines,
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


def _detect_day_trade_setup(tech: dict) -> str:
    """Backwards-compatible setup classifier (no veto reasons).


    Pass 1 adds veto-capable variants, but this function stays stable so
    existing detectors/sweep behavior doesn't change until Pass 2.
    """
    # ── Check Mark (strict long variant) ────────────────────────────────
    if tech.get("checkmark_long_ready"):
        return "Check Mark Long"

    # ── 15-minute chart features (preferred) ─────────────────────────
    if tech.get("reclaimed_15m_high"):
        return "Reclaim-15m High"
    if tech.get("broke_15m_range_high"):
        return "15m Range Breakout"
    if tech.get("bounced_from_15m_low"):
        return "15m Low Bounce"

    # ── Fallback: existing daily-tech logic ───────────────────────────
    rsi = tech.get("rsi14") or 50
    above_ma20 = tech.get("above_ma20", False)
    vol_ratio = tech.get("vol_ratio") or 1.0
    chg5 = tech.get("5d_change_pct") or 0
    atr_pct = tech.get("atr14_pct") or 1.5

    if rsi < 35 and not above_ma20:
        return "Oversold Bounce"
    if chg5 > 3 and vol_ratio > 2.0:
        return "Gap & Go"
    if above_ma20 and vol_ratio > 1.5 and chg5 > 1.5:
        return "Momentum Continuation"
    if chg5 < -1.5 and above_ma20 and rsi > 40:
        return "Pullback-to-MA Bounce"
    if above_ma20 and abs(chg5) < 1 and 45 < rsi < 60:
        return "Tight Consolidation Breakout"
    return "Technical Bounce"


def _detect_day_trade_setup_veto(tech: dict) -> tuple[str | None, str | None]:
    """Veto-capable setup detector.

    Returns: (setup_name, veto_reason)
      - setup_name is None when a setup is vetoed or when no setup matches.
      - veto_reason is set when a *specific* setup was considered and vetoed.

    IMPORTANT: Pass 1 keeps existing behavior unchanged elsewhere by not
    switching callers yet.
    """

    enforce_pdh = bool(tech.get("enforce_pdh_proximity"))

    # New setup: PDL Sweep Reclaim (long)
    if tech.get("pdl_sweep_reclaim_ready"):
        # If we computed this flag, we already encoded sweep freshness + depth.
        return "PDL Sweep Reclaim", None

    # Check Mark (kept high priority)
    if tech.get("checkmark_long_ready"):
        return "Check Mark Long", None

    # Reclaim-15m High with PDH proximity veto (optional gating)
    if tech.get("reclaimed_15m_high"):
        # Shadow mode: PDH proximity used as a label, not a block.
        # We still return the setup name so the sweep can execute,
        # but we attach veto_reason so we can compare cohorts later.
        veto_reason: str | None = None
        if enforce_pdh:
            dist_pct = tech.get("distance_to_pdh_pct")
            if dist_pct is None:
                veto_reason = "PDH proximity veto: missing prior day high context"
            elif dist_pct > 0.5:
                veto_reason = (
                    f"PDH proximity veto: distance_to_pdh_pct={dist_pct:.3f}% > 0.5%"
                )
        return "Reclaim-15m High", veto_reason

    # Plain 15m features
    if tech.get("broke_15m_range_high"):
        return "15m Range Breakout", None
    if tech.get("bounced_from_15m_low"):
        return "15m Low Bounce", None

    # No match
    return None, "no setup matched"


def _compute_prior_day_context(daily_bars: list[dict]) -> dict:
    """Compute prior-day context (PDH/PDL/ATR14%) used by Pass 2 features.

    Expects daily_bars as: [{date, open, high, low, close, volume}, ...]
    and assumes the last element may be the current (incomplete) day.
    """
    if not daily_bars or len(daily_bars) < 5:
        return {}

    # Drop the last daily bar to avoid partial current-day contamination
    usable = daily_bars[:-1]
    if len(usable) < 5:
        return {}

    prior = usable[-1]
    prior_high = float(prior.get("high"))
    prior_low = float(prior.get("low"))
    prior_close = float(prior.get("close"))

    # ATR14 via True Range over the last 14 usable days (excluding the very first TR seed day).
    trs = []
    for i in range(1, len(usable)):
        cur = usable[i]
        prev = usable[i - 1]
        h = float(cur.get("high"))
        l = float(cur.get("low"))
        pc = float(prev.get("close"))
        tr = max(h - l, abs(h - pc), abs(l - pc))
        if tr > 0:
            trs.append(tr)

    if not trs:
        return {}

    atr14 = sum(trs[-14:]) / max(1, min(14, len(trs[-14:])))
    atr14_pct = (atr14 / prior_close) * 100.0 if prior_close > 0 else None

    return {
        "prior_day_high": prior_high,
        "prior_day_low": prior_low,
        "prior_day_close": prior_close,
        "atr14": atr14,
        "atr14_pct": atr14_pct,
        "prior_day_range": prior_high - prior_low,
        "prior_day_range_pct": (prior_high - prior_low) / prior_close * 100.0 if prior_close > 0 else None,
    }


def _compute_15m_day_trade_features(bars15: list[dict], ctx: dict | None = None) -> dict:
    """Compute 15m intraday swing/breakout features.

    Pass-1 behavior:
      - When ctx is None: preserves current feature set (no session slicing).
      - When ctx is provided (session_aware=true): derives session-aware
        bars15_today and adds range/high/low + distance-to-PDH.

    NOTE: This is intentionally conservative to avoid breaking existing
    callers in Pass 1.
    """

    if not bars15 or len(bars15) < 10:
        return {}

    # ---- Optional session-aware slicing (ctx-gated) --------------
    bars_use = bars15
    if ctx is not None and ctx.get("session_aware", True):
        try:
            et = ZoneInfo("America/New_York")

            def _to_et_iso(b: dict) -> str | None:
                ts = b.get("timestamp")
                if not ts:
                    return None
                try:
                    # Alpaca timestamps are usually ISO with tz (UTC). If naive,
                    # treat as UTC.
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                    return dt.astimezone(et).isoformat()
                except Exception:
                    return None

            today_et_date = None
            # Determine "today" from last bar (DST-safe)
            last_et = _to_et_iso(bars15[-1])
            if last_et:
                today_et_date = last_et.split("T")[0]

            if today_et_date:
                bars_today = []
                for b in bars15:
                    et_ts = _to_et_iso(b)
                    if not et_ts:
                        continue
                    if et_ts.split("T")[0] == today_et_date:
                        bars_today.append(b)

                # Require enough bars for stable features
                if len(bars_today) >= 10:
                    bars_use = bars_today
        except Exception:
            # If anything fails, keep original bars15 (don't block Pass 1).
            bars_use = bars15

    # ---- Feature computation (from existing logic) -------------
    highs = [float(b.get("high")) for b in bars_use if b.get("high") is not None]
    lows = [float(b.get("low")) for b in bars_use if b.get("low") is not None]
    closes = [float(b.get("close")) for b in bars_use if b.get("close") is not None]

    if len(closes) < 10 or len(highs) < 10 or len(lows) < 10:
        return {}

    n = min(len(highs), len(lows), len(closes))
    highs, lows, closes = highs[-n:], lows[-n:], closes[-n:]

    last_close = closes[-1]
    prev_close = closes[-2]

    range_window = min(10, len(closes) - 2)
    if range_window <= 1:
        return {}

    prev_slice = slice(-(range_window + 1), -1)
    range_high = max(highs[prev_slice])
    range_low = min(lows[prev_slice])

    broke_15m_range_high = bool(prev_close <= range_high and last_close > range_high)

    w = 2
    pivot_highs = []
    pivot_lows = []
    for i in range(w, len(closes) - w):
        if highs[i] == max(highs[i - w:i + w + 1]):
            pivot_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - w:i + w + 1]):
            pivot_lows.append((i, lows[i]))

    last_pivot_high = pivot_highs[-1][1] if pivot_highs else None
    last_pivot_low = pivot_lows[-1][1] if pivot_lows else None

    buffer_pct = 0.002
    reclaimed_15m_high = (
        bool(last_pivot_high is not None
             and prev_close <= last_pivot_high * (1 + buffer_pct)
             and last_close > last_pivot_high * (1 + buffer_pct))
    )

    bounced_from_15m_low = (
        bool(last_pivot_low is not None
             and prev_close >= last_pivot_low * (1 - buffer_pct)
             and last_close > last_pivot_low * (1 - buffer_pct)
             and min(closes[-5:]) <= last_pivot_low * (1 + buffer_pct))
    )

    tech = {
        "reclaimed_15m_high": reclaimed_15m_high,
        "broke_15m_range_high": broke_15m_range_high,
        "bounced_from_15m_low": bounced_from_15m_low,
        "last_pivot_high": last_pivot_high,
        "last_pivot_low": last_pivot_low,
    }

    # --- Spec E: PDH-proximity gating inputs (ctx-gated) ---------
    if ctx is not None:
        pdh = ctx.get("prior_day_high")
        if pdh and pdh > 0:
            tech["distance_to_pdh_pct"] = ((last_close - float(pdh)) / float(pdh)) * 100.0
            tech["enforce_pdh_proximity"] = True

        # --- Spec D: PDL Sweep Reclaim (ctx-gated) --------------
        # Detect a session-low sweep below prior day low, then require
        # a reclaim to near/above the sweep_low within a freshness window.
        pdl = ctx.get("prior_day_low")
        atr14_pct = ctx.get("atr14_pct")

        if pdl and pdl > 0:
            pdl = float(pdl)
            # Sweep threshold: scale by ATR% when available; fall back to a small fixed band.
            if atr14_pct and atr14_pct > 0:
                # ~0.1% of PDL adjusted by ATR% magnitude
                sweep_thresh = max(pdl * 0.001, pdl * (atr14_pct / 100.0) * 0.05)
            else:
                sweep_thresh = max(pdl * 0.001, 0.01)

            # Session high/low for TP structure
            session_high = float(max(highs))
            session_low = float(min(lows))
            tech["session_high"] = session_high
            tech["session_low"] = session_low

            sweep_idx = None
            sweep_low = None
            for i in range(len(bars_use)):
                lo = bars_use[i].get("low")
                if lo is None:
                    continue
                lo = float(lo)
                if sweep_idx is None:
                    if lo <= pdl - sweep_thresh:
                        sweep_idx = i
                        sweep_low = lo
                else:
                    sweep_low = min(sweep_low, lo)

            # Reclaim rule: last_close reclaims above sweep_low by buffer
            reclaim_buffer = max(0.002 * float(sweep_low), 0.01) if sweep_low else 0.0
            freshness_bars = int(ctx.get("reclaim_freshness_bars", 5)) if ctx else 5

            pdl_sweep_reclaim_ready = False
            if sweep_idx is not None and sweep_low is not None:
                bars_since_sweep = (len(bars_use) - 1) - sweep_idx
                if bars_since_sweep <= max(0, freshness_bars):
                    if last_close >= (sweep_low + reclaim_buffer):
                        # Additional depth guard: ensure sweep wasn't just a tiny poke.
                        sweep_depth_pct = (pdl - sweep_low) / pdl * 100.0
                        if sweep_depth_pct >= ctx.get("min_sweep_depth_pct", 0.3):
                            pdl_sweep_reclaim_ready = True
                            tech["sweep_depth_pct"] = sweep_depth_pct

            tech["pdl_sweep_low"] = float(sweep_low) if sweep_low is not None else None
            tech["pdl_sweep_reclaim_ready"] = bool(pdl_sweep_reclaim_ready)
            tech["pdl_sweep_freshness_bars"] = int(bars_since_sweep) if sweep_idx is not None else None
            # Target structure selector for Pass 2 levels
            if pdl_sweep_reclaim_ready:
                tech["target_ref"] = "range_mid"

    return tech


def _compute_check_mark_long_features(
    bars15: list[dict],
    bars5: list[dict],
    prev_day_high: float,
    prev_day_low: float,
) -> dict:
    """Strict long “Check Mark” pattern detector.

    Returns a dict that _detect_day_trade_setup() / _compute_day_trade_levels()
    can consume, or {} if conditions aren't met.

    Notes/approximations (due to available bar data in this codebase):
    - Uses the FIRST available 15m candle as the “opening range” check.
    - Treats a long blow-off as the opening candle taking out the PRIOR DAY low.
    """
    try:
        if not bars15 or not bars5:
            return {}
        if len(bars15) < 15 or len(bars5) < 20:
            return {}

        highs15 = [float(b.get("high")) for b in bars15 if b.get("high") is not None]
        lows15 = [float(b.get("low")) for b in bars15 if b.get("low") is not None]
        closes15 = [float(b.get("close")) for b in bars15 if b.get("close") is not None]
        vols15 = [float(b.get("volume") or 0.0) for b in bars15 if b.get("volume") is not None]

        # Volume participation gate:
        # Require the latest 15m bar to have meaningfully elevated volume vs the prior bars.
        # This reduces false positives in the Check Mark strict pattern detector.
        if len(vols15) >= 5:
            last_vol = vols15[-1]
            prev_avg_vol = sum(vols15[:-1]) / max(1, len(vols15[:-1]))
            if prev_avg_vol > 0:
                vol_ratio = last_vol / prev_avg_vol
                if vol_ratio < 1.2:
                    return {}

        if len(highs15) < 15 or len(lows15) < 15 or len(closes15) < 15:
            return {}

        # Day-so-far range from available 15m bars (proxy)
        day_high = max(highs15)
        day_low = min(lows15)

        opening15 = bars15[0]
        op_high = float(opening15.get("high"))
        op_low = float(opening15.get("low"))
        op_close = float(opening15.get("close"))
        candle_range = op_high - op_low

        # ATR(15m) approximation via TR mean
        trs = []
        for i in range(1, len(bars15)):
            h = float(bars15[i].get("high"))
            l = float(bars15[i].get("low"))
            pc = float(bars15[i - 1].get("close"))
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(float(tr))
        if len(trs) < 2:
            return {}
        atr15 = sum(trs[-14:]) / min(14, len(trs))
        if atr15 <= 0:
            return {}

        # Check: manipulation candle if range > 20% of ATR
        is_manipulation = (candle_range / atr15) >= 0.20
        if not is_manipulation:
            return {}

        # Blow-off (LONG): opening candle takes out prior day low
        # small tolerance to avoid floating noise
        tol_prev = max(prev_day_low * 0.0005, 0.01)
        is_blowoff_low = op_low <= (prev_day_low - tol_prev)
        if not is_blowoff_low:
            return {}

        blowoff_low = op_low

        # Pivot/double-test on 5m: two touches near blow-off low
        highs5 = [float(b.get("high")) for b in bars5 if b.get("high") is not None]
        lows5 = [float(b.get("low")) for b in bars5 if b.get("low") is not None]
        closes5 = [float(b.get("close")) for b in bars5 if b.get("close") is not None]
        opens5 = [float(b.get("open")) for b in bars5 if b.get("open") is not None]
        if len(highs5) < 30 or len(lows5) < 30 or len(closes5) < 30 or len(opens5) < 30:
            return {}

        last_close_5m = closes5[-1]
        tol = max(blowoff_low * 0.002, 0.01)

        # Touch = low tags blowoff + close shows acceptance
        touch_idxs = []
        for i in range(len(lows5)):
            if lows5[i] <= blowoff_low + tol and closes5[i] >= blowoff_low - tol:
                touch_idxs.append(i)

        if len(touch_idxs) < 2:
            return {}

        # pick two touches separated by >=2 bars
        touch1 = touch_idxs[0]
        touch2 = None
        for j in touch_idxs[1:]:
            if j - touch1 >= 2:
                touch2 = j
                break
        if touch2 is None:
            return {}

        # Ensure price has started to accept/reverse after second touch
        if not (last_close_5m > blowoff_low + tol * 0.1):
            return {}

        # Entry rule: close above high of previous RED candle
        # Search after touch2
        red_high = None
        for i in range(touch2 + 1, len(closes5) - 1):
            if closes5[i] < opens5[i]:
                red_high = highs5[i]

        if red_high is None:
            return {}

        entry_ready = last_close_5m > red_high
        if not entry_ready:
            return {}

        # Stop just outside wick (below blow-off low)
        stop_ref = blowoff_low - tol * 0.3

        # Target proxy: median of day range
        target_ref = (day_high + day_low) / 2.0

        # Exit guard: require sensible ordering
        # If the target proxy collapses below the entry ref candle, bump it slightly
        # so we can form a valid R:R on realistic intraday ranges.
        if target_ref <= red_high:
            target_ref = red_high * 1.01

        if not (stop_ref < red_high < target_ref):
            return {}

        return {
            "checkmark_long_ready": True,
            "checkmark_stop_ref_price": float(stop_ref),
            "checkmark_entry_ref_price": float(red_high),
            "checkmark_target_ref_price": float(target_ref),
        }

    except Exception:
        return {}


def _compute_day_trade_levels(price: float, atr_pct: float, tech: dict | None = None) -> dict:
    """Pure-Python day trade levels.

    Pass-1: keeps existing Check Mark + ATR fallback.
    Pass-1 also adds a structural target path for:
      - target_ref == "range_mid" (PDL Sweep Reclaim)

    Note: In Pass 2, the sweep script will compute shares using the
    structural TP distance; this function only sets entry/stop/target.
    """

    tech = tech or {}

    # Structural sizing/TP logic expects a stable entry basis.
    entry = round(float(price) * 1.001, 2)

    # ── PDL Sweep Reclaim (range_mid target path) ───────────────────────
    if tech.get("target_ref") == "range_mid":
        try:
            sweep_low = tech.get("pdl_sweep_low")
            session_high = tech.get("session_high")
            if sweep_low is not None and session_high is not None:
                sweep_low = float(sweep_low)
                session_high = float(session_high)

                # Spec F:
                #   stop = sweep_low * 0.998
                #   tp   = (session_high + stop) / 2
                stop = round(sweep_low * 0.998, 2)
                target = round((session_high + stop) / 2.0, 2)

                if stop <= 0 or target <= 0:
                    return {}
                if stop >= entry or target <= entry:
                    return {}

                risk = entry - stop
                reward = target - entry
                rr = round(reward / risk, 2) if risk > 0 else 0

                # RR-fail => reject (do not TP-widen)
                from scripts.trade import MIN_RR_RATIO
                if rr < MIN_RR_RATIO:
                    return {}

                stop_pct = round(((entry - stop) / entry) * 100.0, 3) if entry > 0 else None
                reward_pct = round((reward / entry) * 100.0, 3) if entry > 0 else None

                return {
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "stop_pct": stop_pct,
                    "reward_pct": reward_pct,
                    "rr": rr,
                    "risk_dollars": None,
                    "reward_dollars": None,
                    "shares": None,
                    "exit_time": "3:45 PM ET",
                }
        except Exception:
            return {}

    # ── Check Mark overrides ────────────────────────────────────────────
    if tech.get("checkmark_long_ready"):

        entry_ref = float(tech.get("checkmark_entry_ref_price") or 0)
        stop_ref = float(tech.get("checkmark_stop_ref_price") or 0)
        target_ref = float(tech.get("checkmark_target_ref_price") or 0)
        if entry_ref > 0 and stop_ref > 0 and target_ref > 0:
            # limit premium for entry to cross spread
            entry = round(entry_ref * 1.001, 2)

            # Clamp stop % into allowed day-trade band [0.5%, 2.0%]
            raw_stop_pct = ((entry - stop_ref) / entry) * 100
            min_stop_pct = 0.5
            max_stop_pct = 2.0
            stop_pct = min(max(round(raw_stop_pct, 2), min_stop_pct), max_stop_pct)
            stop = round(entry * (1 - stop_pct / 100), 2)

            risk = entry - stop
            if risk > 0:
                # RR: target must be >= 1.5x risk above entry (validate_order uses MIN_RR=1.5)
                min_target = entry + 1.5 * risk
                target = round(max(target_ref, min_target), 2)
            else:
                target = round(entry * 1.03, 2)

            rr = round((target - entry) / risk, 1) if risk > 0 else 0

            max_shares = max(1, int(100 / entry))
            risk_dollars = round(max_shares * risk, 2)
            reward_dollars = round(max_shares * (target - entry), 2)
            reward_pct = round((target - entry) / entry * 100, 2) if entry > 0 else 0

            return {
                "entry": entry,
                "stop": stop,
                "target": target,
                "stop_pct": stop_pct,
                "reward_pct": reward_pct,
                "rr": rr,
                "shares": max_shares,
                "risk_dollars": risk_dollars,
                "reward_dollars": reward_dollars,
                "exit_time": "3:45 PM ET",
            }

    # ── Fallback: existing ATR-based levels ─────────────────────────────
    stop_pct   = min(max(round(atr_pct * 0.75, 2), 2), 2.0)
    entry      = round(price * 1.001, 2)          # slight limit premium
    stop       = round(entry * (1 - stop_pct / 100), 2)
    risk       = entry - stop
    target     = round(entry + 3 * risk, 2)
    reward     = target - entry
    rr         = round(reward / risk, 1) if risk > 0 else 0

    max_shares    = max(1, int(100 / entry))
    risk_dollars  = round(max_shares * risk, 2)
    reward_dollars = round(max_shares * reward, 2)
    reward_pct    = round(reward / entry * 100, 2)

    return {
        "entry":          entry,
        "stop":           stop,
        "target":         target,
        "stop_pct":       stop_pct,
        "reward_pct":     reward_pct,
        "rr":             rr,
        "shares":         max_shares,
        "risk_dollars":   risk_dollars,
        "reward_dollars": reward_dollars,
        "exit_time":      "3:45 PM ET",
    }

    """
    Pure-Python day trade levels using ATR-based stop.
    Stop = 0.75× ATR below entry, clamped to 0.5–2%.
    Target = 3:1 R/R from stop.
    Position size capped at $100 (10% of $1k account).
    """
    stop_pct   = min(max(round(atr_pct * 0.75, 2), 0.5), 2.0)
    entry      = round(price * 1.001, 2)          # slight limit premium
    stop       = round(entry * (1 - stop_pct / 100), 2)
    risk       = entry - stop
    target     = round(entry + 3 * risk, 2)
    reward     = target - entry
    rr         = round(reward / risk, 1) if risk > 0 else 0

    max_shares    = max(1, int(100 / entry))
    risk_dollars  = round(max_shares * risk, 2)
    reward_dollars = round(max_shares * reward, 2)
    reward_pct    = round(reward / entry * 100, 2)

    return {
        "entry":          entry,
        "stop":           stop,
        "target":         target,
        "stop_pct":       stop_pct,
        "reward_pct":     reward_pct,
        "rr":             rr,
        "shares":         max_shares,
        "risk_dollars":   risk_dollars,
        "reward_dollars": reward_dollars,
        "exit_time":      "3:45 PM ET",
    }


def _next_monthly_expiries(n: int = 4, min_days: int = 7) -> list[dict]:
    """
    Return the next n standard monthly options expiry dates (3rd Friday of each month)
    that are at least min_days away. Each entry: {date_str, weeks_out, date_obj}.
    No LLM involvement — pure calendar arithmetic.
    """
    from datetime import date, timedelta
    results = []
    today = date.today()
    year, month = today.year, today.month
    while len(results) < n:
        first_day   = date(year, month, 1)
        first_fri   = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_fri   = first_fri + timedelta(weeks=2)
        days_out    = (third_fri - today).days
        if days_out >= min_days:
            results.append({
                "date_obj":  third_fri,
                "date_str":  third_fri.strftime("%b %d, %Y"),
                "days_out":  days_out,
                "weeks_out": days_out // 7,
            })
        month += 1
        if month > 12:
            month = 1
            year  += 1
    return results


def _bs_price(S: float, K: float, T_days: float, sigma: float,
              r: float = 0.05, option_type: str = "call") -> float:
    """
    Black-Scholes option price. Pure Python stdlib — no scipy needed.
    S=spot, K=strike, T_days=calendar days to expiry, sigma=annualized vol (decimal),
    r=risk-free rate (decimal, default 5%).
    """
    import math
    if T_days <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    T = T_days / 365.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        N  = lambda x: (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
        if option_type == "call":
            return S * N(d1) - K * math.exp(-r * T) * N(d2)
        else:
            return K * math.exp(-r * T) * N(-d2) - S * N(-d1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _select_options_strategy(direction: str, confidence: str) -> str:
    """
    Pick strategy type from direction + confidence.
    HIGH confidence + strong direction → debit (bet on the move, unlimited upside within spread).
    MEDIUM / LOW                       → credit (collect theta, win if flat or slightly wrong).
    Neutral                            → iron condor (collect premium from both sides).
    """
    d = direction.lower()
    c = (confidence or "MEDIUM").upper()
    if "neutral" in d or not d:
        return "iron_condor"
    if "bull" in d:
        return "bull_call_spread" if c == "HIGH" else "bull_put_spread"
    if "bear" in d:
        return "bear_put_spread" if c == "HIGH" else "bear_call_spread"
    return "skip"


def _options_plain_english(s: dict) -> str:
    """One plain-English sentence describing what happens if you enter this trade."""
    st = s.get("strategy_type", "")
    exp = s.get("expiry_date", "expiry")
    be  = s.get("breakeven")
    lbe = s.get("lower_breakeven")
    ube = s.get("upper_breakeven")
    gc  = s.get("max_gain_contract", 0)
    cc  = s.get("cost_per_contract", 0)
    if st == "bull_call_spread":
        return f"You pay ~${cc:.0f}. You profit if the stock rises above ${be} by {exp}. Max win: ~${gc:.0f}."
    if st == "bear_put_spread":
        return f"You pay ~${cc:.0f}. You profit if the stock falls below ${be} by {exp}. Max win: ~${gc:.0f}."
    if st == "bull_put_spread":
        return f"You collect ~${gc:.0f} now. You keep it if the stock stays above ${be} through {exp}. You risk ~${cc:.0f} if wrong."
    if st == "bear_call_spread":
        return f"You collect ~${gc:.0f} now. You keep it if the stock stays below ${be} through {exp}. You risk ~${cc:.0f} if wrong."
    if st == "iron_condor":
        return f"You collect ~${gc:.0f} now. You keep it if the stock stays between ${lbe} and ${ube} through {exp}. You risk ~${cc:.0f} if it breaks out."
    return ""


def _build_options_structure(price: float, direction: str,
                              target_weeks: int = 4,
                              hist_vol_pct: float = 35.0,
                              confidence: str = "MEDIUM",
                              force_type: str = None) -> dict:
    """
    Build the concrete options structure — ALL arithmetic here, no LLM.
    Strategy selection via _select_options_strategy():

      bullish  HIGH   → Bull Call Spread  (debit  — pay premium, bet on move)
      bullish  MED/LO → Bull Put Spread   (credit — collect theta, win if flat/up)
      bearish  HIGH   → Bear Put Spread   (debit  — pay premium, bet on move)
      bearish  MED/LO → Bear Call Spread  (credit — collect theta, win if flat/down)
      neutral         → Iron Condor       (credit — win if stock stays in range)

    Credit-spread key metrics:
      cost_per_contract  = max loss at risk (spread_width - credit) × 100
      max_gain_contract  = credit received × 100
      credit_per_contract = same as max_gain_contract (explicit)
      return_pct         = credit / max_risk × 100
    """
    if not price or price <= 0:
        return {"error": "no valid price"}

    if   price <  20: incr = 0.50
    elif price <  50: incr = 1.00
    elif price < 100: incr = 2.50
    else:             incr = 5.00

    def snap(p): return round(round(p / incr) * incr, 2)

    st = force_type or _select_options_strategy(direction, confidence)
    sigma = hist_vol_pct / 100.0

    expiries = _next_monthly_expiries(n=6)
    min_d    = max(7, (target_weeks - 1) * 7)
    chosen   = next((e for e in expiries if e["days_out"] >= min_d), expiries[0])
    T        = chosen["days_out"]
    exp_str  = chosen["date_str"]
    exp_wks  = chosen["weeks_out"]

    def _base():
        return {"expiry_date": exp_str, "expiry_weeks": exp_wks, "expiry_days": T,
                "hist_vol_pct": round(hist_vol_pct, 1), "strategy_type": st}

    # ── Bull Call Spread (debit, bullish HIGH) ────────────────────────────
    if st == "bull_call_spread":
        atm = snap(price);  otm = snap(price * 1.07)
        lb  = round(_bs_price(price, atm, T, sigma, option_type="call"), 2)
        sb  = round(_bs_price(price, otm, T, sigma, option_type="call"), 2)
        dc  = max(lb - sb, 0.01);  sw = round(otm - atm, 2)
        mg  = round(sw - dc, 2)
        cc  = round(dc * 100, 2);  gc = round(mg * 100, 2)
        be  = round(atm + dc, 2);  bp = round((be - price) / price * 100, 1)
        rp  = round(gc / cc * 100, 1) if cc > 0 else 0
        r = {**_base(), "strategy": "Bull Call Spread", "is_credit": False,
             "structure": f"Buy ${atm} call · Sell ${otm} call · Exp {exp_str} ({exp_wks} wks)",
             "legs_note": f"Long ${atm} call ~${lb:.2f}/sh · Short ${otm} call ~${sb:.2f}/sh · Net debit ~${dc:.2f}/sh",
             "atm_strike": atm, "otm_strike": otm, "spread_width": sw,
             "long_leg_bs": lb, "short_leg_bs": sb, "spread_cost_share": round(dc, 2),
             "cost_per_contract": cc, "max_gain_contract": gc,
             "breakeven": be, "breakeven_pct": bp, "return_pct": rp,
             "max_loss": f"~${cc:.0f}", "max_gain": f"~${gc:.0f}"}
        r["plain_english"] = _options_plain_english(r); return r

    # ── Bear Put Spread (debit, bearish HIGH) ─────────────────────────────
    if st == "bear_put_spread":
        atm = snap(price);  otm = snap(price * 0.93)
        lb  = round(_bs_price(price, atm, T, sigma, option_type="put"), 2)
        sb  = round(_bs_price(price, otm, T, sigma, option_type="put"), 2)
        dc  = max(lb - sb, 0.01);  sw = round(atm - otm, 2)
        mg  = round(sw - dc, 2)
        cc  = round(dc * 100, 2);  gc = round(mg * 100, 2)
        be  = round(atm - dc, 2);  bp = round((price - be) / price * 100, 1)
        rp  = round(gc / cc * 100, 1) if cc > 0 else 0
        r = {**_base(), "strategy": "Bear Put Spread", "is_credit": False,
             "structure": f"Buy ${atm} put · Sell ${otm} put · Exp {exp_str} ({exp_wks} wks)",
             "legs_note": f"Long ${atm} put ~${lb:.2f}/sh · Short ${otm} put ~${sb:.2f}/sh · Net debit ~${dc:.2f}/sh",
             "atm_strike": atm, "otm_strike": otm, "spread_width": sw,
             "long_leg_bs": lb, "short_leg_bs": sb, "spread_cost_share": round(dc, 2),
             "cost_per_contract": cc, "max_gain_contract": gc,
             "breakeven": be, "breakeven_pct": bp, "return_pct": rp,
             "max_loss": f"~${cc:.0f}", "max_gain": f"~${gc:.0f}"}
        r["plain_english"] = _options_plain_english(r); return r

    # ── Bull Put Spread (credit, bullish MED/LOW) ─────────────────────────
    if st == "bull_put_spread":
        sp = snap(price * 0.93);  lp = snap(price * 0.86)
        sb = round(_bs_price(price, sp, T, sigma, option_type="put"), 2)
        lb = round(_bs_price(price, lp, T, sigma, option_type="put"), 2)
        nc = max(sb - lb, 0.01);  sw = round(sp - lp, 2)
        ml = round(sw - nc, 2)
        cc = round(ml * 100, 2);  gc = round(nc * 100, 2)
        be = round(sp - nc, 2);   bp = round((price - be) / price * 100, 1)
        rp = round(gc / cc * 100, 1) if cc > 0 else 0
        r = {**_base(), "strategy": "Bull Put Spread", "is_credit": True,
             "structure": f"Sell ${sp} put · Buy ${lp} put · Exp {exp_str} ({exp_wks} wks) · Credit ~${nc:.2f}/sh",
             "legs_note": f"Short ${sp} put ~${sb:.2f}/sh · Long ${lp} put ~${lb:.2f}/sh · Net credit ~${nc:.2f}/sh",
             "atm_strike": sp, "otm_strike": lp, "spread_width": sw,
             "long_leg_bs": lb, "short_leg_bs": sb, "spread_cost_share": round(nc, 2),
             "net_credit": round(nc, 2), "credit_per_contract": gc,
             "cost_per_contract": cc, "max_gain_contract": gc,
             "breakeven": be, "lower_breakeven": be, "breakeven_pct": bp, "return_pct": rp,
             "max_loss": f"~${cc:.0f}", "max_gain": f"~${gc:.0f}"}
        r["plain_english"] = _options_plain_english(r); return r

    # ── Bear Call Spread (credit, bearish MED/LOW) ────────────────────────
    if st == "bear_call_spread":
        sc = snap(price * 1.07);  lc = snap(price * 1.14)
        sb = round(_bs_price(price, sc, T, sigma, option_type="call"), 2)
        lb = round(_bs_price(price, lc, T, sigma, option_type="call"), 2)
        nc = max(sb - lb, 0.01);  sw = round(lc - sc, 2)
        ml = round(sw - nc, 2)
        cc = round(ml * 100, 2);  gc = round(nc * 100, 2)
        be = round(sc + nc, 2);   bp = round((be - price) / price * 100, 1)
        rp = round(gc / cc * 100, 1) if cc > 0 else 0
        r = {**_base(), "strategy": "Bear Call Spread", "is_credit": True,
             "structure": f"Sell ${sc} call · Buy ${lc} call · Exp {exp_str} ({exp_wks} wks) · Credit ~${nc:.2f}/sh",
             "legs_note": f"Short ${sc} call ~${sb:.2f}/sh · Long ${lc} call ~${lb:.2f}/sh · Net credit ~${nc:.2f}/sh",
             "atm_strike": sc, "otm_strike": lc, "spread_width": sw,
             "long_leg_bs": lb, "short_leg_bs": sb, "spread_cost_share": round(nc, 2),
             "net_credit": round(nc, 2), "credit_per_contract": gc,
             "cost_per_contract": cc, "max_gain_contract": gc,
             "breakeven": be, "upper_breakeven": be, "breakeven_pct": bp, "return_pct": rp,
             "max_loss": f"~${cc:.0f}", "max_gain": f"~${gc:.0f}"}
        r["plain_english"] = _options_plain_english(r); return r

    # ── Iron Condor (credit, neutral) ────────────────────────────────────
    if st == "iron_condor":
        sc = snap(price * 1.07);  lc = snap(price * 1.14)
        sp = snap(price * 0.93);  lp = snap(price * 0.86)
        sc_bs = round(_bs_price(price, sc, T, sigma, option_type="call"), 2)
        lc_bs = round(_bs_price(price, lc, T, sigma, option_type="call"), 2)
        sp_bs = round(_bs_price(price, sp, T, sigma, option_type="put"), 2)
        lp_bs = round(_bs_price(price, lp, T, sigma, option_type="put"), 2)
        nc  = max((sc_bs - lc_bs) + (sp_bs - lp_bs), 0.01)
        sw  = round(lc - sc, 2)           # wing width (lc > sc, always positive)
        ml  = round(sw - nc, 2)
        cc  = round(ml * 100, 2);  gc = round(nc * 100, 2)
        ube = round(sc + nc, 2);   lbe = round(sp - nc, 2)
        bpw = round((ube - lbe) / price * 100, 1)   # profit-zone width as % of price
        rp  = round(gc / cc * 100, 1) if cc > 0 else 0
        r = {**_base(), "strategy": "Iron Condor", "is_credit": True,
             "structure": f"Sell ${sp}p/${sc}c · Buy ${lp}p/${lc}c · Exp {exp_str} ({exp_wks} wks) · Credit ~${nc:.2f}/sh",
             "legs_note": f"Puts: short ${sp} ~${sp_bs:.2f} / long ${lp} ~${lp_bs:.2f} · Calls: short ${sc} ~${sc_bs:.2f} / long ${lc} ~${lc_bs:.2f}",
             "atm_strike": sp, "otm_strike": sc, "spread_width": sw,
             "long_leg_bs": lp_bs, "short_leg_bs": sp_bs, "spread_cost_share": round(nc, 2),
             "net_credit": round(nc, 2), "credit_per_contract": gc,
             "cost_per_contract": cc, "max_gain_contract": gc,
             "breakeven": lbe, "lower_breakeven": lbe, "upper_breakeven": ube,
             "profit_zone": f"${lbe} – ${ube}",
             "breakeven_pct": bpw, "return_pct": rp,
             "max_loss": f"~${cc:.0f}", "max_gain": f"~${gc:.0f}",
             "ic_short_call": sc, "ic_long_call": lc, "ic_short_put": sp, "ic_long_put": lp}
        r["plain_english"] = _options_plain_english(r); return r

    return {"strategy": "Skip", "skip": True}


def analyze_daily_options_play() -> dict:
    """
    Catalyst-first daily options play scanner.

    LLM does ONLY: sector selection, ticker choice, direction, catalyst/thesis (qualitative).
    Python does ALL math: strike prices, expiry date, max loss/gain, spread width.
    """
    import re as _re
    import research

    # ── Step 1: Sector snapshot ────────────────────────────────────────────
    try:
        sectors = research.get_sector_snapshot()
    except Exception as e:
        sectors = {}

    sector_lines = []
    for etf, s in sectors.items():
        if isinstance(s, dict) and "name" in s:
            arrow = "↑" if s.get("above_ma20") else "↓"
            sector_lines.append(
                f"  {etf} ({s['name']}): 5d={s.get('5d_change_pct',0):+.1f}%  "
                f"20d={s.get('20d_change_pct',0):+.1f}%  RSI={s.get('rsi14','?')}  "
                f"{arrow}MA20"
            )

    # ── Step 2: Market-wide news ───────────────────────────────────────────
    try:
        news_data = research.get_news(None, hours=48)
        headlines = [
            f"  [{i['source']}] {i['headline']}"
            for i in news_data.get("items", [])[:15]
        ]
    except Exception:
        news_data = {}
        headlines = ["  (news unavailable)"]

    # ── Step 3: LLM — qualitative judgment only, zero math ────────────────
    today_str = datetime.now().strftime("%A, %B %d, %Y")

    system_pick = """You are a professional options strategist doing a daily market scan.
Given sector ETF performance and today's market news, identify the ONE best
catalyst-driven options opportunity for the next 2-4 weeks.

Your job is ONLY to identify:
- Which sector has the clearest catalyst today
- Which specific liquid stock in that sector to play
- Bullish or bearish direction
- Why (catalyst + historical pattern)

Do NOT include any numbers, prices, strikes, dates, or calculations — those are handled separately.

Output ONLY valid JSON with these exact fields:
{
  "sector_etf": "XLK",
  "sector_name": "Technology",
  "sector_trend": "bullish",
  "catalyst_headline": "exact headline or news item driving this",
  "catalyst_type": "earnings/macro/product/regulatory/sentiment",
  "historical_reaction": "how this type of news typically moves this sector/stock (1-2 sentences from training knowledge)",
  "direction": "bullish",
  "ticker": "NVDA",
  "why_ticker": "why this stock specifically over other sector leaders",
  "why": "2-3 sentence thesis connecting catalyst to directional play",
  "ideal_outcome": "what needs to happen over the next few weeks for this to work",
  "risk": "what would invalidate this thesis",
  "confidence": "HIGH/MEDIUM/LOW"
}"""

    prompt_pick = f"""Daily options scan — {today_str}.

SECTOR ETF PERFORMANCE (5d / 20d change, RSI, vs MA20):
{chr(10).join(sector_lines) if sector_lines else '  (unavailable)'}

TODAY'S MARKET NEWS (last 48 hours):
{chr(10).join(headlines)}

Identify the sector with the strongest catalyst and the best stock to play. Output JSON only."""

    try:
        if LLM_PROVIDER == "anthropic":
            r = get_anthropic_client().messages.create(
                model=MODEL, max_tokens=800,
                system=system_pick,
                messages=[{"role": "user", "content": prompt_pick}]
            )
            raw = r.content[0].text
        else:
            r = get_groq_client().chat.completions.create(
                model=MODEL, max_tokens=800,
                messages=[{"role": "system", "content": system_pick},
                          {"role": "user",   "content": prompt_pick}]
            )
            raw = r.choices[0].message.content or ""

        raw = raw.strip()
        raw = _re.sub(r'^```(?:json)?\s*', '', raw)
        raw = _re.sub(r'\s*```$', '', raw.strip())
        pick = json.loads(raw.strip())
    except Exception as e:
        return {"error": f"LLM failed: {e}", "raw": raw if "raw" in dir() else ""}

    # ── Step 4: Fetch live price — Python only ─────────────────────────────
    ticker = pick.get("ticker", "")
    ticker_data = {}
    last_price = None
    if ticker:
        try:
            ticker_data = research.calc_technicals(ticker)
            quote       = research.get_quote(ticker)
            last_price  = (quote.get("last_trade_price")
                           or quote.get("ask")
                           or ticker_data.get("current_price"))
            ticker_data["last_trade"] = last_price
            ticker_data["is_stale"]   = quote.get("is_stale", True)
        except Exception:
            pass

    # ── Step 5: Python builds the entire options structure ─────────────────
    direction  = pick.get("direction", "bullish").lower()
    confidence = pick.get("confidence", "MEDIUM")
    hist_vol   = ticker_data.get("hist_vol_30d", 35.0)
    opt_struct = {}
    if last_price:
        opt_struct   = _build_options_structure(last_price, direction,
                                                target_weeks=4, hist_vol_pct=hist_vol,
                                                confidence=confidence)
        _dir_type    = {"bullish": "bull_call_spread", "bearish": "bear_put_spread"}.get(direction)
        _crd_type    = {"bullish": "bull_put_spread",  "bearish": "bear_call_spread"}.get(direction, "iron_condor")
        dir_play     = (_build_options_structure(last_price, direction,
                                                  target_weeks=4, hist_vol_pct=hist_vol,
                                                  force_type=_dir_type) if _dir_type else None)
        theta_play   = _build_options_structure(last_price, direction,
                                                 target_weeks=4, hist_vol_pct=hist_vol,
                                                 force_type=_crd_type)
    else:
        opt_struct = {"error": "no live price — cannot compute strikes"}
        dir_play = theta_play = None

    # Build expiry rationale from computed data (no LLM needed)
    expiry_rationale = ""
    if opt_struct.get("expiry_weeks"):
        expiry_rationale = (
            f"Expiry {opt_struct['expiry_date']} gives {opt_struct['expiry_weeks']} weeks "
            f"for the catalyst to play out — standard 4-week window for news-driven moves."
        )

    # ── Assemble final result ──────────────────────────────────────────────
    play = {
        **pick,
        "strategy":            opt_struct.get("strategy", "—"),
        "strategy_type":       opt_struct.get("strategy_type", ""),
        "is_credit":           opt_struct.get("is_credit", False),
        "structure":           opt_struct.get("structure", "—"),
        "legs_note":           opt_struct.get("legs_note", ""),
        "atm_strike":          opt_struct.get("atm_strike"),
        "otm_strike":          opt_struct.get("otm_strike"),
        "spread_width":        opt_struct.get("spread_width"),
        "expiry_date":         opt_struct.get("expiry_date"),
        "expiry_weeks":        opt_struct.get("expiry_weeks"),
        "expiry_days":         opt_struct.get("expiry_days"),
        "expiry_rationale":    expiry_rationale,
        "hist_vol_pct":        opt_struct.get("hist_vol_pct"),
        "long_leg_bs":         opt_struct.get("long_leg_bs"),
        "short_leg_bs":        opt_struct.get("short_leg_bs"),
        "spread_cost_share":   opt_struct.get("spread_cost_share"),
        "net_credit":          opt_struct.get("net_credit"),
        "credit_per_contract": opt_struct.get("credit_per_contract"),
        "cost_per_contract":   opt_struct.get("cost_per_contract"),
        "max_gain_contract":   opt_struct.get("max_gain_contract"),
        "breakeven":           opt_struct.get("breakeven"),
        "breakeven_pct":       opt_struct.get("breakeven_pct"),
        "lower_breakeven":     opt_struct.get("lower_breakeven"),
        "upper_breakeven":     opt_struct.get("upper_breakeven"),
        "profit_zone":         opt_struct.get("profit_zone"),
        "return_pct":          opt_struct.get("return_pct"),
        "max_loss":            opt_struct.get("max_loss", "—"),
        "max_gain":            opt_struct.get("max_gain", "—"),
        "directional_play":    dir_play,
        "theta_play":          theta_play,
        "ticker_price":        last_price,
        "ticker_technicals":   ticker_data,
        "sector_snapshot":     sectors,
        "news_headlines":      [i["headline"] for i in news_data.get("items", [])[:8]],
        "generated_at":        datetime.now().isoformat(),
    }
    _dp_ticker = pick.get("ticker", "")
    _dp_cat    = pick.get("catalyst_headline", pick.get("catalyst", ""))
    _dp_conf   = pick.get("confidence", "MEDIUM")
    if _dp_ticker:
        if dir_play:
            outcomes.save_recommendation(_dp_ticker, dir_play,
                catalyst=_dp_cat, confidence=_dp_conf, source="daily_options")
        if theta_play:
            outcomes.save_recommendation(_dp_ticker, theta_play,
                catalyst=_dp_cat, confidence=_dp_conf, source="daily_options")
    return play


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading agent cycle runner")
    parser.add_argument("--dry-run", action="store_true", help="Research and decide, but skip order submission")
    parser.add_argument("--premarket", action="store_true", help="Pre-market scan only")
    parser.add_argument("--status", action="store_true", help="Quick portfolio snapshot (no LLM call)")
    parser.add_argument("--analyze", metavar="SYMBOL", help="On-demand analysis for a single symbol")
    parser.add_argument("--daily-options", action="store_true", help="Run catalyst-first daily options play scan")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.analyze:
        result = analyze_symbol(args.analyze)
        print(json.dumps(result, indent=2))
    elif args.daily_options:
        result = analyze_daily_options_play()
        print(json.dumps(result, indent=2))
    else:
        run_cycle(dry_run=args.dry_run, premarket=args.premarket)

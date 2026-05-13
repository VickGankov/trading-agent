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
from dotenv import load_dotenv

# Add parent dir to path so we can import sibling scripts
sys.path.insert(0, str(Path(__file__).parent))

import research
import trade as trade_module
import journal as journal_module

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

if LLM_PROVIDER == "anthropic":
    import anthropic as _anthropic
    _api_key = os.getenv("ANTHROPIC_API_KEY")
    if not _api_key:
        print(json.dumps({"error": "ANTHROPIC_API_KEY must be set in .env"}), file=sys.stderr)
        sys.exit(1)
    _anthropic_client = _anthropic.Anthropic(api_key=_api_key)
    MODEL = "claude-sonnet-4-6"
else:
    from openai import OpenAI as _OpenAI
    _api_key = os.getenv("GROQ_API_KEY")
    if not _api_key:
        print(json.dumps({"error": "GROQ_API_KEY must be set in .env (get free key at console.groq.com)"}), file=sys.stderr)
        sys.exit(1)
    import httpx as _httpx
    # Use the combined cert bundle (macOS system keychain + corporate CA)
    _CERT_BUNDLE = str(Path(__file__).parent.parent / "corporate_certs.pem")
    _verify = _CERT_BUNDLE if Path(_CERT_BUNDLE).exists() else True
    _groq_client = _OpenAI(
        api_key=_api_key,
        base_url="https://api.groq.com/openai/v1",
        http_client=_httpx.Client(verify=_verify)
    )
    MODEL = "llama-3.3-70b-versatile"

CLAUDE_MD = Path(__file__).parent.parent / "CLAUDE.md"
WATCHLIST_PATH = Path(__file__).parent.parent / "data" / "watchlist.json"

# Condensed system prompt for Groq (stays under 12K TPM free tier limit)
GROQ_SYSTEM = """Disciplined paper trading agent. $1000 account.
Rules: long-only, max $100/position (10%), min $50 order, max 5 open, keep $250+ cash.
Every BUY: stop 4-6% below entry_limit, target 8-12% above entry_limit. No leveraged ETFs.

entry_limit MUST be current_price + 0.3% (round to 2 decimals). Never set below current_price — it will expire unfilled.
Fractional qty: floor(100 / entry_limit * 100) / 100. Verify qty × entry_limit ≥ $50.

OUTPUT BUY when one setup applies and no rejection fires:
  A. MA20 PULLBACK — price within 4% above MA20, RSI 38-60, above MA50. No news required.
  B. OVERSOLD BOUNCE — RSI < 42, above MA50. No news required.
  C. NEWS CATALYST — analyst upgrade/PT raise, earnings beat, product launch + above MA50, any RSI ≤ 65.

REJECT (these override any setup):
  - RSI > 65 (overbought)
  - Earnings ≤ 3 days away
  - Below MA50 (downtrend — no catching falling knives)

If a setup matches and no rejection fires → lean BUY. Do not invent extra reasons to pass.

One JSON per candidate, then 1-sentence reflection:
{"action":"BUY","ticker":"X","qty":0.00,"entry_limit":0.00,"stop_loss":0.00,"take_profit":0.00,"confidence":"MEDIUM","thesis":"Setup [A/B/C]: <one sentence why>."}
{"action":"NO TRADE","ticker":"X","reason":"<exact rejection rule: RSI 67 / earnings 2d / below MA50>"}"""


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
        resp = _anthropic_client.messages.create(
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
        resp = _groq_client.chat.completions.create(
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

    # Merge priority + screener, dedupe, cap at 20
    universe = list(dict.fromkeys(priority_syms + screener_syms))[:20]
    print(f"  Universe ({len(universe)} candidates): {universe}")

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

    # Pre-classify each deep candidate's setup in Python — eliminates LLM arithmetic errors
    # and ensures consistent setup identification regardless of model reasoning quality.
    def classify_setup(sym: str, t: dict, news_headlines: list) -> str:
        rsi = t.get("rsi14") or 50.0
        price = t.get("current_price") or 0.0
        ma20 = t.get("ma20") or price
        ma50 = t.get("ma50") or price
        above_ma50 = t.get("above_ma50", price > ma50)

        # Hard rejection checks
        if rsi > 65:
            return "REJECT: RSI>65 overbought"
        if not above_ma50:
            return "REJECT: below MA50 (downtrend)"

        pct_above_ma20 = ((price - ma20) / ma20 * 100) if ma20 else 0
        has_news = bool(news_headlines)

        setups = []
        if 0 <= pct_above_ma20 <= 4 and 38 <= rsi <= 60:
            setups.append(f"A-MA20pull({pct_above_ma20:+.1f}%aboveMA20)")
        if rsi < 42:
            setups.append("B-oversold")
        if has_news and rsi <= 65:
            setups.append("C-newscatalyst")

        return "/".join(setups) if setups else "NO_SETUP"

    candidate_rows = []
    for sym in data["deep_candidates"]:
        t = data["technicals"].get(sym, {})
        if "error" in t:
            continue
        earn = data["earnings"].get(sym, {})
        earn_days = earn.get("days_until")
        earn_str = f"EARNINGS {earn_days}d" if isinstance(earn_days, int) and earn_days <= 7 else ""
        setup = classify_setup(sym, t, data["news"].get(sym, []))
        price = t.get("current_price", 0)
        # Use live ask if available — guarantees same-day fill at or near current market
        quote = data.get("quotes", {}).get(sym, {})
        ask = quote.get("ask") if quote.get("ask") and quote["ask"] > 0 else None
        entry_suggest = round(ask * 1.001, 2) if ask else round(price * 1.003, 2)
        row = (
            f"{sym}: ${price} MA20=${t.get('ma20')} MA50=${t.get('ma50')} "
            f"RSI={t.get('rsi14')} 5d={t.get('5d_change_pct'):+.1f}% "
            f"| SETUP={setup} | entry_suggest=${entry_suggest}"
        )
        if earn_str:
            row += f" | {earn_str}"
        if data["news"].get(sym):
            row += f" | NEWS: {'; '.join(data['news'][sym][:2])}"
        candidate_rows.append(row)

    candidates_block = "\n".join(candidate_rows)

    prompt = f"""Date: {datetime.now().strftime('%Y-%m-%d %H:%M ET')} {mode_note}
Account: ${acct['account_value']:.0f} total, ${acct['cash']:.0f} cash, {acct['positions_count']}/5 positions, {data['pdt']['daytrade_count_5days']}/3 day trades
Market: open={data['clock']['is_open']} | SPY 5d:{spy.get('5d_change_pct',0):+.1f}% RSI:{spy.get('rsi14','?')} | QQQ 5d:{qqq.get('5d_change_pct',0):+.1f}% RSI:{qqq.get('rsi14','?')}

SCREENER (top 20): {screener_summary}

DEEP CANDIDATES — setup pre-classified, entry_suggest already +0.3%:
{candidates_block}

Rules: If SETUP is A/B/C and no REJECT/EARNINGS block → output BUY using entry_suggest as entry_limit.
If SETUP=REJECT or NO_SETUP → output NO TRADE with the rejection reason.
For each candidate output one JSON decision block, then 1-sentence reflection."""

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
    if not dry_run and not premarket:
        execution_results = _execute_from_text(response_text, data["account"])

    # Write journal with decisions annotated with execution status
    try:
        decisions = _parse_decisions_from_text(response_text)

        for d in decisions:
            action = d.get("action", "")
            ticker = d.get("ticker", "")
            if action in ("BUY", "SELL"):
                if dry_run:
                    d["execution_status"] = "DRY_RUN"
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
            result = trade_module.place_buy(
                ticker,
                qty,
                float(decision.get("entry_limit", decision.get("entry", 0))),
                float(decision.get("stop_loss", decision.get("stop", 0))),
                float(decision.get("take_profit", decision.get("target", 0))),
                decision.get("thesis", "agent decision")
            )
            print(f"\n→ ORDER: {json.dumps(result)}")
            results[ticker] = result
        elif action == "SELL" and ticker:
            result = trade_module.place_sell(
                ticker,
                round(float(decision.get("qty", 0)), 2),
                decision.get("reason", "agent decision")
            )
            print(f"\n→ ORDER: {json.dumps(result)}")
            results[ticker] = result
    return results


def _load_stop_levels() -> dict:
    """
    Read the most recent BUY decision for each symbol from journal history.
    Returns {symbol: {"stop_loss": float, "take_profit": float, "entry": float}}.
    Used to monitor fractional positions that have no automatic stop orders.
    """
    import glob
    journal_dir = Path(__file__).parent.parent / "journal"
    files = sorted(glob.glob(str(journal_dir / "*.json")), reverse=True)
    stops = {}
    for f in files[:30]:  # look back at most 30 entries
        try:
            with open(f) as fh:
                entry = json.load(fh)
            for d in entry.get("decisions", []):
                sym = d.get("ticker", "")
                if d.get("action") == "BUY" and d.get("execution_status") == "SUBMITTED" and sym not in stops:
                    stops[sym] = {
                        "stop_loss": d.get("stop_loss"),
                        "take_profit": d.get("take_profit"),
                        "entry": d.get("entry_limit"),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading agent cycle runner")
    parser.add_argument("--dry-run", action="store_true", help="Research and decide, but skip order submission")
    parser.add_argument("--premarket", action="store_true", help="Pre-market scan only")
    parser.add_argument("--status", action="store_true", help="Quick portfolio snapshot (no LLM call)")
    args = parser.parse_args()

    if args.status:
        print_status()
    else:
        run_cycle(dry_run=args.dry_run, premarket=args.premarket)

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
GROQ_SYSTEM = """Disciplined paper trading agent. $1000 account. Rules:
- Long-only. Max $100/position (10%). Min $50. Max 5 open. Keep $250+ cash.
- Every BUY needs stop 3-10% below entry, take-profit ≥1.5× risk. No leveraged ETFs. No earnings within 3 days.
- qty=floor(100/entry_limit). If qty=0 → NO TRADE. Default to NO TRADE.

One JSON block per candidate (on its own line):
{"action":"BUY","ticker":"X","qty":1,"entry_limit":0.00,"stop_loss":0.00,"take_profit":0.00,"confidence":"MEDIUM","thesis":"..."}
{"action":"NO TRADE","ticker":"X","reason":"..."}
Then a brief reflection."""


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
                "qty": {"type": "integer"},
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
                inputs["symbol"], inputs["qty"], inputs["limit_price"],
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

    # Pre-filter to top_n for deep analysis using screener scores.
    # Score = abs(5d_change_pct) * vol_ratio. Priority tickers get a boost.
    screener_scores = {m["symbol"]: abs(m.get("5d_change_pct", 0)) * m.get("vol_ratio", 1)
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

    return {
        "account": account,
        "pdt": pdt,
        "market": market,
        "clock": clock,
        "all_movers": all_movers,          # full 20 for display
        "deep_candidates": deep_candidates, # top 5 for LLM analysis
        "technicals": technicals,
        "news": news,
        "earnings": earnings,
    }


def _run_groq_cycle(dry_run: bool, premarket: bool, system: str):
    """
    Single-shot Groq cycle: collect data in Python, make one LLM call,
    parse decisions, execute orders. Stays within free-tier token limits.
    """
    print("\nCollecting market data...")
    data = _collect_market_data(top_n=5)  # screener gets 20, deep analysis on top 5

    mode_note = "DRY RUN - analyze only, no orders." if dry_run else (
                "PRE-MARKET - plan only, no orders." if premarket else "")

    acct = data["account"]
    mkt = data["market"]["indices"]
    spy = mkt.get("SPY", {})
    qqq = mkt.get("QQQ", {})

    # Compact screener summary for all 20 (one line each)
    screener_summary = ", ".join(
        f"{m['symbol']}({m['5d_change_pct']:+.1f}%,vol:{m['vol_ratio']:.1f}x)"
        for m in data["all_movers"]
    )

    # Compact technicals — omit fields already in screener summary (5d_change_pct, vol_ratio)
    # and derivable ones (above_ma20 = price > ma20)
    compact_tech = {
        sym: {k: v for k, v in t.items() if k in
              ("current_price", "ma20", "ma50", "rsi14")}
        for sym, t in data["technicals"].items()
        if "error" not in t
    }

    prompt = f"""Date: {datetime.now().strftime('%Y-%m-%d %H:%M ET')} {mode_note}
Account: ${acct['account_value']:.0f} total, ${acct['cash']:.0f} cash, {acct['positions_count']}/5 positions, {data['pdt']['daytrade_count_5days']}/3 day trades
Market: open={data['clock']['is_open']} | SPY 5d:{spy.get('5d_change_pct',0):+.1f}% RSI:{spy.get('rsi14','?')} | QQQ 5d:{qqq.get('5d_change_pct',0):+.1f}% RSI:{qqq.get('rsi14','?')}

SCREENER (top 20 by momentum×volume): {screener_summary}

DEEP ANALYSIS — top 5 candidates selected for full review:
Technicals: {json.dumps(compact_tech, separators=(',', ':'))}
News: {json.dumps({s: h for s, h in data['news'].items() if h}, separators=(',', ':'))}
Earnings risk (days until): {', '.join(f"{s}:{v.get('days_until','?')}d" for s,v in data['earnings'].items() if isinstance(v.get('days_until'), int)) or 'none known'}

For each of the 5 deep candidates output a JSON decision block, then a brief reflection."""

    print("\nAsking LLM for decisions...")
    messages = [{"role": "user", "content": prompt}]
    _, blocks = _call_llm(system, messages)

    response_text = ""
    for b in blocks:
        if b["type"] == "text":
            print(b["text"])
            response_text += b["text"]

    # Try to extract and execute any BUY/SELL decisions from the response
    if not dry_run and not premarket:
        _execute_from_text(response_text, data["account"])

    # Write journal with structured decisions so the dashboard can parse them
    try:
        decisions = _parse_decisions_from_text(response_text)
        # Extract reflection (non-JSON text after the last decision block)
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


def _execute_from_text(text: str, account: dict):
    """Parse and execute BUY/SELL decisions from LLM response text."""
    for decision in _parse_decisions_from_text(text):
        action = decision.get("action", "")
        ticker = decision.get("ticker", "")
        if action == "BUY" and ticker:
            result = trade_module.place_buy(
                ticker,
                int(decision.get("qty", 1)),
                float(decision.get("entry_limit", decision.get("entry", 0))),
                float(decision.get("stop_loss", decision.get("stop", 0))),
                float(decision.get("take_profit", decision.get("target", 0))),
                decision.get("thesis", "agent decision")
            )
            print(f"\n→ ORDER: {json.dumps(result)}")
        elif action == "SELL" and ticker:
            result = trade_module.place_sell(
                ticker,
                int(decision.get("qty", 1)),
                decision.get("reason", "agent decision")
            )
            print(f"\n→ ORDER: {json.dumps(result)}")


def run_cycle(dry_run: bool = False, premarket: bool = False):
    """Run one full agent cycle."""
    system = load_system_prompt()

    print(f"\n{'='*60}")
    print(f"Trading Agent Cycle — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Provider: {LLM_PROVIDER.upper()} / {MODEL}")
    print(f"Mode: {'DRY RUN' if dry_run else 'PREMARKET' if premarket else 'LIVE'}")
    print(f"{'='*60}\n")

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

#!/usr/bin/env python3
"""Detector-only 15m day-trade sweep that submits bracket orders.

Designed for cron/auto-run.
- Uses existing day-trade detectors in scripts/agent.py
- Uses existing hard guardrails in scripts/trade.py (validate_order + place_buy)
- Targets bracket take-profit so filled P&L is approximately a desired $ profit.

SAFETY:
- Refuses to run outside US market hours (ET), weekends.
- Never bypasses validate_order.
"""

from __future__ import annotations

import argparse
from datetime import datetime, time
from zoneinfo import ZoneInfo

import sys
import json
import math

# Local imports from repo — derive from this file's location so the script
# runs on the VPS and locally without editing paths.
from pathlib import Path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import research
import agent as a
import trade as t
import journal as journal_module


def _shadow_tags_from_veto(veto_reason: str | None) -> list[str]:
    if not veto_reason:
        return []
    tags: list[str] = []
    # Specific cohort tags for later cohort testing
    if veto_reason.startswith("PDH proximity veto"):
        tags.append("shadow:pdh_proximity")
        # Distinguish missing context vs too-far distance if present
        if "missing prior day high context" in veto_reason:
            tags.append("shadow:pdh_missing")
    else:
        tags.append("shadow:veto")
    return tags


def _execution_journal_payload(
    *,
    action: str,
    execution_status: str,
    su: str,
    setup: str | None,
    qty: int | float | None,
    entry_px: float | None,
    stop_px: float | None,
    target_px: float | None,
    veto_reason: str | None,
    shadow_tags: list[str],
    validate_msg: str | None,
    order_id: str | None,
    execution_detail: str | None,
    account_state: dict,
    now_et: datetime,
    args,
) -> dict:
    """Create a minimal journal payload that journal.py can summarize.

    IMPORTANT: journal.py summary expects decisions[i].action in {BUY, SELL, HOLD, NO TRADE}
    and uses execution_status to count validation_rejections and trades_placed.
    """

    return {
        "cycle_timestamp": now_et.isoformat(),
        "provider": "auto_trade_sweep_15m",
        "account": account_state,
        "market_context": {},
        "screener_universe": [su],
        "deep_candidates": [],
        "decisions": [
            {
                "action": action,
                "ticker": su,
                "qty": qty,
                "entry_limit": entry_px,
                "stop_loss": stop_px,
                "take_profit": target_px,
                "execution_status": execution_status,
                "order_id": order_id,
                "validate_msg": validate_msg,
                "setup": setup,
                "veto_reason": veto_reason,
                "shadow_tags": shadow_tags,
                "execution_detail": execution_detail,
            }
        ],
        "reflection": "",
        "dry_run": bool(getattr(args, "dry_run", False)),
        "args": {
            "min_notional": getattr(args, "min_notional", None),
            "target_profit_cash": getattr(args, "target_profit_cash", None),
            "profit_max_cash": getattr(args, "profit_max_cash", None),
            "max_submissions": getattr(args, "max_submissions", None),
            "top": getattr(args, "top", None),
            "lookback_hours": getattr(args, "lookback_hours", None),
        },
    }


def _write_journal(payload: dict):
    # Never let journal failure block trading.
    try:
        journal_module.write_entry(payload)
    except Exception:
        pass



ET = ZoneInfo("America/New_York")


def is_market_open(now_et: datetime) -> bool:
    # Weekdays: Mon-Fri
    if now_et.weekday() >= 5:
        return False

    open_t = time(9, 30)
    close_t = time(16, 0)

    # Conservative window to avoid last-minute slippage.
    # Market might still be open right at 16:00; bracket fills won't happen after.
    if now_et.time() < open_t:
        return False
    # Stop submitting new orders after 15:15 ET (day-trade hard exit target)
    if now_et.time() > time(15, 15):
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-profit-cash", type=float, default=80.0)
    parser.add_argument("--profit-max-cash", type=float, default=100.0)
    # Pass-2 decision: min_notional in the sweep should NOT hard-lock
    # structure-based setups near $2k. validate_order has the real $50 floor.
    parser.add_argument("--min-notional", type=float, default=300.0)
    parser.add_argument("--max-submissions", type=int, default=10)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--lookback-hours", type=int, default=10)

    parser.add_argument("--paper-only", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true", default=False)

    args = parser.parse_args()

    now_et = datetime.now(ET)
    if not is_market_open(now_et):
        print({"status": "SKIP_MARKET_CLOSED", "now_et": now_et.isoformat()})
        return

    if args.dry_run:
        print({"status": "DRY_RUN", "now_et": now_et.isoformat()})

    # Pull movers
    movers = research.get_market_movers(top=args.top)

    # Add fixed watchlist universe alongside movers (lever 2 tier-1)
    fixed_universe = [
        "NVDA",
        "TSLA",
        "AMD",
        "PLTR",
        "SPY",
        "QQQ",
        "IWM",
    ]

    raw_syms = [g.get("symbol") for g in movers.get("gainers", [])]
    raw_syms.extend(fixed_universe)

    candidates: list[str] = []
    for sym in raw_syms:
        if not sym or "." in sym:
            continue
        # avoid warrants/units-ish tickers
        if sym.endswith("W") or sym.endswith("T"):
            continue

        # price filter (try to get price from mover dict)
        try:
            price = float(next(g.get("price") for g in movers.get("gainers", []) if g.get("symbol") == sym))
        except Exception:
            continue

        if not (5 <= price <= 500):
            continue
        candidates.append(sym)

    # de-dupe, preserve order
    candidates = list(dict.fromkeys(candidates))[:25]

    st = t.get_account_state()
    existing_positions = {p["symbol"].upper() for p in st.get("positions", [])}

    open_orders = t.list_orders("open")
    open_buy_syms = {
        o["symbol"].upper()
        for o in open_orders
        if o.get("side") == "buy" and o.get("status") in ("new", "submitted")
    }

    allowed_setups = {
        "Check Mark Long",
        "Reclaim-15m High",
        "15m Range Breakout",
        "15m Low Bounce",
        "PDL Sweep Reclaim",
    }

    submitted = []

    for sym in candidates:
        if len(submitted) >= args.max_submissions:
            break

        su = sym.upper()
        if su in existing_positions:
            continue
        if su in open_buy_syms:
            continue

        try:
            intr15 = research.get_intraday_bars(su, minutes=15, lookback_hours=args.lookback_hours)
            bars15 = intr15.get("bars", [])
            if not bars15 or len(bars15) < 10:
                continue

            # Pass-2: fetch daily bars for prior-day context and
            # compute session-aware intraday features.
            daily = research.get_bars(su, days=30).get("bars", [])
            prior_ctx = a._compute_prior_day_context(daily)

            tech15 = a._compute_15m_day_trade_features(
                bars15,
                ctx={
                    **prior_ctx,
                    "session_aware": True,
                    "reclaim_freshness_bars": 5,
                    "min_sweep_depth_pct": 0.3,
                },
            )
            if not tech15:
                continue

            # Veto-capable detector (Pass-2)
            setup_name, veto_reason = a._detect_day_trade_setup_veto(tech15)

            if setup_name is None:
                # No setup matched => true no-trade (not shadow-tagging)
                submitted.append(
                    {
                        "symbol": su,
                        "setup": None,
                        "qty": None,
                        "entry": None,
                        "stop": None,
                        "target": None,
                        "profit_cash": None,
                        "order_id": None,
                        "status": "NO_SETUP",
                        "validate_msg": None,
                        "veto_reason": veto_reason,
                    }
                )
                continue

            setup = setup_name
            # Shadow mode: veto_reason is a label, not a block.
            setup_reason = veto_reason
            if setup not in allowed_setups:
                continue

            # Optional: tag cohort even if label is present, but do not block submission.

            q = research.get_quote(su)
            entry_raw = float(q.get("ask") or q.get("last_trade_price") or 0)
            if entry_raw <= 0:
                continue

            # Pass-2: levels now include structural TP when target_ref == range_mid
            levels = a._compute_day_trade_levels(entry_raw, atr_pct=1.5, tech=tech15)
            # Fix rounding so realized stop % never exceeds 2.0% after rounding.
            # We choose stop_px by snapping to 0.01 increments but ensuring stop_pct <= 2.0%.
            entry_px = round(float(levels["entry"]), 2)
            target_px = round(float(levels["target"]), 2)

            raw_stop = float(levels["stop"])
            # start from 2 decimal rounding
            stop_px = round(raw_stop, 2)

            # For day-trades we enforce tighter stop band (<=2.0%).
            # If rounding pushes stop_pct above 2.0%, move stop down slightly.
            stop_pct = ((entry_px - stop_px) / entry_px) * 100.0
            if stop_pct > 2.0:
                # Maximum allowable stop distance in dollars
                max_dist = entry_px * (2.0 / 100.0)
                # stop_px should be entry_px - max_dist, rounded down to cents to be safe
                stop_px = math.floor((entry_px - max_dist) * 100.0) / 100.0

            if stop_px >= entry_px:
                continue

            risk = entry_px - stop_px
            if risk <= 0:
                continue

            # Pass-2: TP is structural (from agent levels). We size shares
            # using shares = floor(target_profit_cash / (tp-entry)),
            # then clamp by max notional from the capped bankroll.

            target_distance = target_px - entry_px
            if target_distance <= 0:
                continue

            qty_base = int((args.target_profit_cash) // target_distance)
            if qty_base < 1:
                qty_base = 1

            # Effective bankroll cap (paper account may report much larger equity)
            max_notional = t.max_position_usd(float(st["equity"]))
            qty_max = int(max_notional // entry_px)
            if qty_max < 1:
                continue

            qty = min(qty_base, qty_max)

            # Loosen expected profit cap to allow more entries through (still constrained by validate_order)
            expected_profit = target_distance * qty
            if expected_profit > args.profit_max_cash:
                qty = int((args.profit_max_cash) // target_distance)
                qty = max(1, min(qty, qty_max))

            order_value = qty * entry_px
            if order_value < args.min_notional:
                continue

            # Hard guardrails
            ok, msg = t.validate_order(
                su,
                "buy",
                qty,
                entry_px,
                stop_px,
                target_px,
                is_day_trade=True,
            )
            if not ok:
                # Still record as a cohort-tagged attempt (shadow label)
                shadow_tags = _shadow_tags_from_veto(setup_reason)
                submitted.append(
                    {
                        "symbol": su,
                        "setup": setup,
                        "qty": qty,
                        "entry": entry_px,
                        "stop": stop_px,
                        "target": target_px,
                        "profit_cash": round((target_px - entry_px) * qty, 2),
                        "order_id": None,
                        "status": "VALIDATION_REJECTED",
                        "validate_msg": msg,
                        "veto_reason": setup_reason,
                        "shadow_tags": shadow_tags,
                    }
                )

                # Write journal entry for funnel completeness
                try:
                    payload = _execution_journal_payload(
                        action="BUY",
                        execution_status="DRY_RUN_REJECTED",
                        su=su,
                        setup=setup,
                        qty=qty,
                        entry_px=entry_px,
                        stop_px=stop_px,
                        target_px=target_px,
                        veto_reason=setup_reason,
                        shadow_tags=shadow_tags,
                        validate_msg=msg,
                        order_id=None,
                        execution_detail=f"validate_order rejected: {msg}",
                        account_state=st,
                        now_et=now_et,
                        args=args,
                    )
                    _write_journal(payload)
                except Exception:
                    pass

                continue

            if not args.dry_run:
                res = t.place_buy(
                    su,
                    qty,
                    entry_px,
                    stop_px,
                    target_px,
                    reason=f"Auto 15m day-trade ({setup}); structural TP",
                    is_day_trade=True,
                )
            else:
                res = {"status": "DRY_SUBMIT", "order_id": None}

            expected_profit_cash = (target_px - entry_px) * qty
            shadow_tags = _shadow_tags_from_veto(setup_reason)

            # Write a minimal journal entry for cohort testing
            execution_status = res.get("status")
            try:
                payload = _execution_journal_payload(
                    action="BUY",
                    execution_status=execution_status,
                    su=su,
                    setup=setup,
                    qty=qty,
                    entry_px=entry_px,
                    stop_px=stop_px,
                    target_px=target_px,
                    veto_reason=setup_reason,
                    shadow_tags=shadow_tags,
                    validate_msg=msg,
                    order_id=res.get("order_id"),
                    execution_detail=res.get("reason") or res.get("error"),
                    account_state=st,
                    now_et=now_et,
                    args=args,
                )
                _write_journal(payload)
            except Exception:
                pass

            submitted.append(
                {
                    "symbol": su,
                    "setup": setup,
                    "qty": qty,
                    "entry": entry_px,
                    "stop": stop_px,
                    "target": target_px,
                    "profit_cash": round(expected_profit_cash, 2),
                    "order_id": res.get("order_id"),
                    "status": res.get("status"),
                    "validate_msg": msg,
                    "veto_reason": setup_reason,
                    "shadow_tags": shadow_tags,
                }
            )
            open_buy_syms.add(su)
            print({"status": "SUBMITTED", **submitted[-1]})
            break


        except Exception:
            # detector/screener failures are common; just skip
            continue

    print({"status": "DONE", "submitted_count": len(submitted), "submitted": submitted})


if __name__ == "__main__":
    main()

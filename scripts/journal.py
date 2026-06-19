#!/usr/bin/env python3
"""
journal.py - Append cycle entries to the structured trade journal.

Usage:
    python scripts/journal.py write '<json_payload>'
    python scripts/journal.py summary 7              # last 7 days
    python scripts/journal.py win_rate
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime, timedelta
from pathlib import Path

JOURNAL_DIR = Path(__file__).parent.parent / "journal"
JOURNAL_DIR.mkdir(exist_ok=True)


def write_entry(payload: dict):
    """Append a cycle entry. File named by timestamp."""
    ts = datetime.now()
    filename = ts.strftime("%Y-%m-%d_%H%M%S") + ".json"
    path = JOURNAL_DIR / filename
    
    payload["_written_at"] = ts.isoformat()
    
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    
    return {"status": "written", "path": str(path), "filename": filename}


def get_summary(days: int = 7):
    """Aggregate stats from recent journal entries."""
    cutoff = datetime.now() - timedelta(days=days)
    files = sorted(glob.glob(str(JOURNAL_DIR / "*.json")))
    
    entries = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            written = datetime.fromisoformat(data.get("_written_at", "1970-01-01"))
            if written >= cutoff:
                entries.append(data)
        except Exception:
            continue
    
    total_cycles = len(entries)
    trades_placed = 0
    rejections = 0
    dry_run_orders = 0
    no_trade_cycles = 0
    decisions_summary = {"BUY": 0, "SELL": 0, "HOLD": 0, "NO TRADE": 0}
    
    for e in entries:
        for d in e.get("decisions", []):
            action = d.get("action", "NO TRADE")
            status = d.get("execution_status", "")
            if action in decisions_summary:
                decisions_summary[action] += 1
            if action in ("BUY", "SELL") and status == "SUBMITTED":
                trades_placed += 1
            if status in ("REJECTED", "ERROR", "DRY_RUN_REJECTED"):
                rejections += 1
            if status.startswith("DRY_RUN"):
                dry_run_orders += 1
        if not any(d.get("execution_status") == "SUBMITTED" for d in e.get("decisions", [])):
            no_trade_cycles += 1
    
    return {
        "period_days": days,
        "total_cycles": total_cycles,
        "trades_placed": trades_placed,
        "dry_run_orders": dry_run_orders,
        "no_trade_cycles": no_trade_cycles,
        "decisions": decisions_summary,
        "validation_rejections": rejections,
        "no_trade_rate_pct": round((no_trade_cycles / total_cycles * 100), 1) if total_cycles else 0
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    
    pw = sub.add_parser("write")
    pw.add_argument("payload", help="JSON string")
    
    ps = sub.add_parser("summary")
    ps.add_argument("days", type=int, nargs="?", default=7)
    
    args = parser.parse_args()
    
    if args.cmd == "write":
        payload = json.loads(args.payload)
        print(json.dumps(write_entry(payload), indent=2))
    elif args.cmd == "summary":
        print(json.dumps(get_summary(args.days), indent=2))
    else:
        parser.print_help()

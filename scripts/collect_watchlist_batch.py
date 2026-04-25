"""Drive collect_listings.py through a watchlist batch for overnight data collection.

Used by the AC9 data-coverage recovery flow: iterates a prioritized slice of
the watchlist, invoking the single-artist collector per entry. Logs progress
to a JSONL trail so resumption after interruption is possible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_TIERS = ["stadium", "arena", "theater", "emerging"]


def load_watchlist(path: Path, tiers: list[str]) -> list[str]:
    data = json.loads(path.read_text())
    out: list[str] = []
    for t in tiers:
        out.extend(data.get(t, []))
    return out


def load_progress(trail: Path) -> set[str]:
    if not trail.exists():
        return set()
    done: set[str] = set()
    for line in trail.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if rec.get("status") == "ok" and rec.get("artist"):
                done.add(rec["artist"])
        except json.JSONDecodeError:
            continue
    return done


def run_one(artist: str, max_events: int, max_listings: int, delay: float) -> dict:
    start = time.monotonic()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/collect_listings.py",
                "--artist", artist,
                "--max-events", str(max_events),
                "--max-listings", str(max_listings),
                "--delay", str(delay),
            ],
            capture_output=True,
            text=True,
            timeout=1200,
        )
        status = "ok" if result.returncode == 0 else "failed"
        return {
            "artist": artist,
            "status": status,
            "exit_code": result.returncode,
            "elapsed_seconds": round(time.monotonic() - start, 1),
            "stderr_tail": result.stderr[-400:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "artist": artist,
            "status": "timeout",
            "elapsed_seconds": round(time.monotonic() - start, 1),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "artist": artist,
            "status": "error",
            "error": str(e),
            "elapsed_seconds": round(time.monotonic() - start, 1),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-drive collect_listings.py across a watchlist")
    parser.add_argument("--watchlist", default="data/artist_watchlist.json", type=Path)
    parser.add_argument("--trail", default=".claude/coral/experiments/collect_trail.jsonl", type=Path)
    parser.add_argument("--tiers", nargs="+", default=DEFAULT_TIERS, choices=DEFAULT_TIERS)
    parser.add_argument("--limit", type=int, default=0, help="Cap artists processed (0=no cap)")
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--max-listings", type=int, default=500)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true", help="Skip artists already ok in trail")
    args = parser.parse_args()

    args.trail.parent.mkdir(parents=True, exist_ok=True)
    artists = load_watchlist(args.watchlist, args.tiers)
    done = load_progress(args.trail) if args.resume else set()
    pending = [a for a in artists if a not in done]
    if args.limit > 0:
        pending = pending[: args.limit]

    print(f"Watchlist: {len(artists)} total | already done: {len(done)} | pending: {len(pending)}")

    with args.trail.open("a") as f:
        for i, artist in enumerate(pending, 1):
            print(f"[{i}/{len(pending)}] {artist}", flush=True)
            rec = run_one(artist, args.max_events, args.max_listings, args.delay)
            rec["ts"] = datetime.now(UTC).isoformat()
            rec["idx"] = i
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if rec["status"] != "ok":
                print(f"  -> {rec['status']} ({rec.get('exit_code', '-')})", flush=True)
            else:
                print(f"  -> ok in {rec['elapsed_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

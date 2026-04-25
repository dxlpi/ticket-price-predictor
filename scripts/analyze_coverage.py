#!/usr/bin/env python3
"""Analyze temporal split coverage to quantify unseen-event fraction.

Usage:
    python scripts/analyze_coverage.py
    python scripts/analyze_coverage.py --split-ratio 0.85 --format json
    python scripts/analyze_coverage.py --format json --out coverage.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Pure computation (importable from tests without DataLoader)
# ---------------------------------------------------------------------------


def compute_coverage(df: pd.DataFrame, train_ratio: float = 0.85) -> dict[str, object]:
    """Compute temporal split coverage metrics from a listings DataFrame.

    Args:
        df: DataFrame with columns ``event_id``, ``event_datetime``,
            ``artist_or_team``, and ``city``.
        train_ratio: Fraction of listings (by sorted event_date) assigned to
            train. Default 0.85 (85% train / 15% test).

    Returns:
        Dict with keys:
        - n_listings_total
        - n_listings_train
        - n_listings_test
        - n_events_train
        - n_events_test
        - n_events_unseen
        - unseen_event_pct_by_event
        - underrepresented_regions  (list of dicts)
        - events_per_artist_histogram  (dict bucket → count)
        - region_month_heatmap  (list of dicts)
    """
    if df.empty:
        return {
            "n_listings_total": 0,
            "n_listings_train": 0,
            "n_listings_test": 0,
            "n_events_train": 0,
            "n_events_test": 0,
            "n_events_unseen": 0,
            "unseen_event_pct_by_event": 0.0,
            "underrepresented_regions": [],
            "events_per_artist_histogram": {},
            "region_month_heatmap": [],
        }

    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_datetime"]).dt.normalize()

    # Temporal split: sort by event_date, first train_ratio% → train
    df_sorted = df.sort_values("event_date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    train_events: set[str] = set(train_df["event_id"].unique())
    test_events: set[str] = set(test_df["event_id"].unique())
    unseen_events = test_events - train_events

    n_events_test = len(test_events)
    unseen_event_pct = len(unseen_events) / n_events_test if n_events_test > 0 else 0.0

    # Underrepresented regions: city × month bucket with < 5 distinct events
    df["region_month"] = (
        df["city"].fillna("unknown")
        + " | "
        + df["event_date"].dt.tz_localize(None).dt.to_period("M").astype(str)
    )
    region_month_counts = (
        df.groupby("region_month")["event_id"].nunique().reset_index()
    )
    region_month_counts.columns = ["region_month", "n_events"]
    underrepresented = (
        region_month_counts[region_month_counts["n_events"] < 5]
        .sort_values("n_events")
        .to_dict(orient="records")
    )

    # Events-per-artist histogram
    events_per_artist = df.groupby("artist_or_team")["event_id"].nunique()

    def _bucket(n: int) -> str:
        if n == 1:
            return "1"
        if n <= 3:
            return "2-3"
        if n <= 10:
            return "4-10"
        return "11+"

    histogram: dict[str, int] = {"1": 0, "2-3": 0, "4-10": 0, "11+": 0}
    for count in events_per_artist:
        histogram[_bucket(count)] += 1

    # Region × month heatmap: all city × month combinations with event counts
    heatmap = (
        region_month_counts.sort_values("n_events", ascending=False)
        .head(50)
        .to_dict(orient="records")
    )

    return {
        "n_listings_total": len(df),
        "n_listings_train": len(train_df),
        "n_listings_test": len(test_df),
        "n_events_train": len(train_events),
        "n_events_test": n_events_test,
        "n_events_unseen": len(unseen_events),
        "unseen_event_pct_by_event": round(unseen_event_pct, 4),
        "underrepresented_regions": underrepresented,
        "events_per_artist_histogram": histogram,
        "region_month_heatmap": heatmap,
    }


# ---------------------------------------------------------------------------
# CLI rendering
# ---------------------------------------------------------------------------


def _print_text(metrics: dict[str, object]) -> None:
    print("\n=== Coverage Analysis ===\n")
    print(f"Total listings : {metrics['n_listings_total']:,}")
    print(f"Train listings : {metrics['n_listings_train']:,}")
    print(f"Test listings  : {metrics['n_listings_test']:,}")
    print()
    print(f"Train events   : {metrics['n_events_train']:,}")
    print(f"Test events    : {metrics['n_events_test']:,}")
    print(f"Unseen events  : {metrics['n_events_unseen']:,}")
    pct = cast(float, metrics["unseen_event_pct_by_event"])
    status = "OK (<= 30%)" if pct <= 0.30 else "ABOVE TARGET (> 30%)"
    print(f"unseen_event_pct_by_event: {pct:.1%}  [{status}]")

    histogram = cast(dict[str, int], metrics["events_per_artist_histogram"])
    print("\n--- Events-per-artist histogram ---")
    for bucket in ("1", "2-3", "4-10", "11+"):
        print(f"  {bucket:>5} events/artist: {histogram[bucket]:,} artists")

    print("\n--- Top underrepresented regions (< 5 events in city×month) ---")
    under = cast(list[dict[str, Any]], metrics["underrepresented_regions"])
    if not under:
        print("  (none)")
    else:
        for row in under[:20]:
            print(f"  {row['n_events']:2d} event(s)  {row['region_month']}")

    print("\n--- Top city×month buckets (by event count, top 20) ---")
    heatmap = cast(list[dict[str, Any]], metrics["region_month_heatmap"])
    for row in heatmap[:20]:
        print(f"  {row['n_events']:4d}  {row['region_month']}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze temporal split coverage (unseen-event fraction)."
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.85,
        metavar="RATIO",
        help="Fraction of listings assigned to train (default: 0.85)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        metavar="DIR",
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write JSON output to FILE (implies --format json)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    # Lazy import so tests can use compute_coverage() without DataLoader
    try:
        # Add project src to path when run as a script
        src_path = Path(__file__).resolve().parent.parent / "src"
        if src_path.exists() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from ticket_price_predictor.ml.training.data_loader import DataLoader

        loader = DataLoader(data_dir=args.data_dir)
        df = loader.load_all_listings()
    except Exception as exc:
        print(f"[ERROR] Could not load data: {exc}", file=sys.stderr)
        print(
            "Hint: run from the project root or pass --data-dir pointing to your data/ folder.",
            file=sys.stderr,
        )
        sys.exit(1)

    if df.empty:
        print("[ERROR] No listings found. Is the data directory correct?", file=sys.stderr)
        sys.exit(1)

    metrics = compute_coverage(df, train_ratio=args.split_ratio)

    if args.out is not None or args.format == "json":
        payload = json.dumps(metrics, indent=2)
        if args.out is not None:
            args.out.write_text(payload)
            print(f"Wrote coverage metrics to {args.out}")
        else:
            print(payload)
    else:
        _print_text(metrics)


if __name__ == "__main__":
    main()

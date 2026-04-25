#!/usr/bin/env python3
"""Measure whether low training-event-count predicts unseen test rows.

AC7 gate: logistic AUC between `low_training_count` and `has_unseen_test_rows`
must be ≥ 0.6 for the `low_count_upweight` sample-weight strategy to be
included in training.  If AUC < 0.6, AC7 is dropped (documented in leaderboard
notes as insufficient signal).

Usage:
    python scripts/measure_low_count_gate.py
    python scripts/measure_low_count_gate.py --threshold 0.30 --gate-auc 0.6
    python scripts/measure_low_count_gate.py --out results/ac7_gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Pure computation (importable from tests without DataLoader)
# ---------------------------------------------------------------------------


def compute_gate(
    df: pd.DataFrame,
    train_ratio: float = 0.85,
    low_count_percentile: float = 0.30,
) -> dict[str, object]:
    """Compute per-artist low-count and unseen-test-rows features.

    Args:
        df: DataFrame with columns ``event_id``, ``artist_or_team``, and a
            sortable temporal column (``event_datetime`` or ``scraped_at``).
        train_ratio: Fraction of listings assigned to train by temporal sort.
        low_count_percentile: Artists whose training event count falls below
            this percentile threshold are flagged as low-count.

    Returns:
        Dict with keys:
        - ``artist_table``: list of per-artist dicts (artist, train_events,
          low_count, has_unseen_test_rows)
        - ``auc``: logistic AUC score (float) or None if degenerate
        - ``n_artists``: total artists with training rows
        - ``low_count_threshold``: event-count threshold used
        - ``low_count_percentile``: percentile parameter used
    """
    required = {"event_id", "artist_or_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Determine temporal sort column
    sort_col: str
    if "event_datetime" in df.columns:
        sort_col = "event_datetime"
    elif "scraped_at" in df.columns:
        sort_col = "scraped_at"
    else:
        raise ValueError(
            "DataFrame must have 'event_datetime' or 'scraped_at' for temporal split"
        )

    # Temporal split: sort by sort_col, assign first train_ratio to train
    sorted_df = df.sort_values(sort_col).reset_index(drop=True)
    n_train = int(len(sorted_df) * train_ratio)
    train_df = sorted_df.iloc[:n_train]
    test_df = sorted_df.iloc[n_train:]

    # Per-artist training event counts
    artist_event_counts = (
        train_df.groupby("artist_or_team")["event_id"].nunique().rename("train_events")
    )

    # Low-count threshold: percentile across artists' training event counts
    threshold = float(np.percentile(artist_event_counts.values, low_count_percentile * 100))
    low_count_artists = set(artist_event_counts[artist_event_counts < threshold].index)

    # Per-artist: does the artist have any test row whose event_id was not in train?
    train_event_ids = set(train_df["event_id"].unique())
    test_unseen = test_df[~test_df["event_id"].isin(train_event_ids)]
    artists_with_unseen = set(test_unseen["artist_or_team"].unique())

    # Build per-artist table (only artists present in train)
    rows = []
    for artist, train_events in artist_event_counts.items():
        rows.append(
            {
                "artist": artist,
                "train_events": int(train_events),
                "low_count": artist in low_count_artists,
                "has_unseen_test_rows": artist in artists_with_unseen,
            }
        )

    artist_table = pd.DataFrame(rows)

    # Logistic regression AUC: has_unseen_test_rows ~ low_count
    auc: float | None = None
    if len(artist_table) >= 2 and artist_table["has_unseen_test_rows"].nunique() > 1:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        X = artist_table[["low_count"]].astype(float).values
        y = artist_table["has_unseen_test_rows"].astype(float).values
        clf = LogisticRegression(random_state=0, max_iter=200)
        clf.fit(X, y)
        proba = clf.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, proba))
    else:
        print(
            "WARNING: degenerate labels (only one class in has_unseen_test_rows) — "
            "AUC cannot be computed; gate FAILS by default"
        )

    return {
        "artist_table": artist_table.to_dict(orient="records"),
        "auc": auc,
        "n_artists": len(artist_table),
        "low_count_threshold": threshold,
        "low_count_percentile": low_count_percentile,
    }


def format_decision(auc: float | None, gate_auc: float) -> str:
    """Return 'IMPLEMENT' or 'DROP' based on AUC vs gate."""
    if auc is None:
        return "DROP"
    return "IMPLEMENT" if auc >= gate_auc else "DROP"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the AC7 measurement gate against real data."""
    parser = argparse.ArgumentParser(
        description="Measure AC7 low-count gate AUC using real listing data"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help="Train split ratio, same as analyze_coverage.py (default: 0.85)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Percentile for low-count threshold (default: 0.30 = 30th percentile)",
    )
    parser.add_argument(
        "--gate-auc",
        type=float,
        default=0.6,
        help="Minimum AUC required for gate to pass (default: 0.6)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON summary to this path",
    )
    args = parser.parse_args()

    # Import here so the pure compute functions above are testable without DataLoader
    from ticket_price_predictor.ml.training.data_loader import DataLoader

    print("Loading listings...")
    loader = DataLoader(args.data_dir)
    listings = loader.load_listings()

    if listings.empty:
        print("ERROR: no listings found — cannot compute gate", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(listings):,} listings")

    result = compute_gate(
        listings,
        train_ratio=args.train_ratio,
        low_count_percentile=args.threshold,
    )

    auc = cast(float | None, result["auc"])
    decision = format_decision(auc, args.gate_auc)
    auc_str = f"{auc:.4f}" if auc is not None else "N/A"

    print()
    print(f"Artists in train:       {result['n_artists']}")
    print(f"Low-count threshold:    {result['low_count_threshold']:.1f} events "
          f"(p{args.threshold * 100:.0f})")
    print(f"AUC = {auc_str}. Gate {'PASSES' if decision == 'IMPLEMENT' else 'FAILS'} "
          f"at threshold {args.gate_auc}.")
    print()
    memo = (
        f"AC7 measurement: AUC={auc_str}, threshold={args.gate_auc}, "
        f"decision={decision}"
    )
    print(memo)

    summary: dict[str, object] = {
        "auc": auc,
        "gate_auc": args.gate_auc,
        "decision": decision,
        "n_artists": result["n_artists"],
        "low_count_threshold": result["low_count_threshold"],
        "low_count_percentile": args.threshold,
        "train_ratio": args.train_ratio,
        "memo": memo,
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary written to {args.out}")

    if decision == "DROP":
        sys.exit(1)


if __name__ == "__main__":
    main()

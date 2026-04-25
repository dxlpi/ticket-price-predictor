"""Section name structural feature extraction."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor

# Named section keywords (non-numeric premium/named areas)
_NAMED_SECTION_KEYWORDS = frozenset(
    {
        "orchestra",
        "loge",
        "club",
        "terrace",
        "garden",
        "mezzanine",
        "parterre",
        "promenade",
        "pavilion",
        "lodge",
        "loggia",
        "tier",
        "dress circle",
        "grand tier",
        "founder",
        "diamond",
        "platinum",
        "gold",
        "silver",
    }
)

# GA / standing keywords
_GA_KEYWORDS = frozenset({"ga", "general admission", "standing", "standing room", "floor standing"})

# VIP / premium / box keywords
_VIP_KEYWORDS = frozenset(
    {"vip", "premium", "platinum", "suite", "box", "skybox", "sky box", "luxury", "lounge"}
)

# Floor / pit / field keywords (structural position, not necessarily GA)
_FLOOR_KEYWORDS = frozenset({"floor", "pit", "field", "courtside", "infield", "on ice", "ice"})

# Obstructed view keywords
_OBSTRUCTED_KEYWORDS = frozenset({"limited view", "obstructed", "restricted view", "partial view"})

# Section level map: number range → level float
_RANGE_LEVEL: list[tuple[int, int, float]] = [
    (100, 199, 0.2),
    (200, 299, 0.5),
    (300, 399, 0.7),
    (400, 499, 0.85),
    (500, 9999, 1.0),
]

# Named keyword → level overrides (checked after number ranges)
_KEYWORD_LEVEL: list[tuple[frozenset[str], float]] = [
    (
        frozenset(
            {"floor", "pit", "field", "courtside", "infield", "ga", "general admission", "standing"}
        ),
        0.0,
    ),
    (frozenset({"orchestra", "parterre", "lower", "dress circle"}), 0.2),
    (frozenset({"mezzanine", "loge", "club", "terrace", "promenade", "grand tier"}), 0.5),
    (frozenset({"upper", "balcony", "gallery", "nosebleed", "founder", "tier"}), 0.85),
]


def _section_level(section: str) -> float:
    """Infer vertical tier level [0.0-1.0] from section name string."""
    s = section.lower().strip()

    # Check floor/GA keywords first (lowest level)
    for keywords, level in _KEYWORD_LEVEL[:1]:
        if any(kw in s for kw in keywords):
            return level

    # Extract leading number to check range
    m = re.search(r"\b(\d{1,4})\b", s)
    if m:
        n = int(m.group(1))
        for lo, hi, level in _RANGE_LEVEL:
            if lo <= n <= hi:
                return level

    # Check remaining named keywords
    for keywords, level in _KEYWORD_LEVEL[1:]:
        if any(kw in s for kw in keywords):
            return level

    return 0.5  # default unknown


def _row_quality(row: str | None) -> float:
    """Normalize row string to quality score [0.0=best, 1.0=rear].

    Row A=0.02, B=0.04, ... Z=0.52; numeric rows mapped proportionally (cap 50=1.0).
    GA / missing → 0.5 default.
    """
    if row is None or (isinstance(row, float) and math.isnan(row)):
        return 0.5
    r = str(row).strip().upper()
    if not r or r in ("GA", "GENERAL", "STANDING", "N/A", "-"):
        return 0.5

    # Single letter
    if len(r) == 1 and r.isalpha():
        return min((ord(r) - ord("A") + 1) * 0.02, 1.0)

    # Double letter (AA, BB, ...)
    if len(r) == 2 and r.isalpha():
        idx = (ord(r[0]) - ord("A")) * 26 + (ord(r[1]) - ord("A")) + 1
        return min(idx * 0.02, 1.0)

    # Numeric
    nums = re.findall(r"\d+", r)
    if nums:
        n = int(nums[0])
        return min(n / 50.0, 1.0)

    return 0.5


def _section_hash(section: str) -> int:
    """32-bucket hash of normalized section name."""
    return int(hashlib.md5(section.lower().strip().encode()).hexdigest(), 16) % 32


def _has_number(section: str) -> bool:
    """Return True if section name contains a numeric component."""
    return bool(re.search(r"\d", section))


def _is_named(section: str) -> int:
    """Return 1 if this is a named (non-numbered) section, 0 otherwise."""
    s = section.lower()
    if any(kw in s for kw in _NAMED_SECTION_KEYWORDS):
        return 1
    # Named if no digits at all (e.g. "Orchestra", "Loge Left")
    if not _has_number(s):
        return 1
    return 0


def _check_keywords(section: str, keywords: frozenset[str]) -> int:
    s = section.lower()
    return int(any(kw in s for kw in keywords))


class SectionFeatureExtractor(FeatureExtractor):
    """Extract 12 structural features from raw section names.

    All features are derived from the section/row/venue_name string itself —
    no price data is used.
    """

    def __init__(self) -> None:
        self._venue_section_counts: dict[str, float] = {}

    @property
    def feature_names(self) -> list[str]:
        return [
            "section_number",
            "section_level",
            "row_quality",
            "is_named_section",
            "sec_is_ga",
            "sec_is_vip",
            "sec_is_floor",
            "is_obstructed",
            "section_name_hash",
            "has_row_info",
            "section_word_count",
            "venue_section_count",
        ]

    def fit(self, df: pd.DataFrame) -> SectionFeatureExtractor:
        """Learn per-venue unique section counts from training data."""
        if "venue_name" not in df.columns or "section" not in df.columns:
            return self

        counts: dict[str, float] = {}
        for venue, group in df.groupby("venue_name", observed=True):
            unique_count = group["section"].nunique()
            counts[str(venue)] = math.log1p(unique_count)

        self._venue_section_counts = counts
        return self

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract structural section features.

        Expects: section (required), row (optional), venue_name (optional).
        """
        result = pd.DataFrame(index=df.index)

        sections: pd.Series = df["section"].fillna("").astype(str)
        has_row = "row" in df.columns
        rows: pd.Series = df["row"] if has_row else pd.Series([""] * len(df), index=df.index)

        # 1. section_number
        def _extract_number(s: str) -> int:
            m = re.search(r"\b(\d{1,4})\b", s)
            if m:
                n = int(m.group(1))
                if 0 <= n <= 999:
                    return n
            return 0

        result["section_number"] = sections.apply(_extract_number)

        # 2. section_level
        result["section_level"] = sections.apply(_section_level)

        # 3. row_quality
        result["row_quality"] = rows.apply(
            lambda r: (
                0.5
                if not has_row
                else _row_quality(r if not (isinstance(r, float) and math.isnan(r)) else None)
            )
        )

        # 4. is_named_section
        result["is_named_section"] = sections.apply(_is_named)

        # 5. sec_is_ga (prefixed to avoid conflict with SeatingFeatureExtractor.is_ga)
        result["sec_is_ga"] = sections.apply(lambda s: _check_keywords(s, _GA_KEYWORDS))

        # 6. sec_is_vip (prefixed for naming consistency)
        result["sec_is_vip"] = sections.apply(lambda s: _check_keywords(s, _VIP_KEYWORDS))

        # 7. sec_is_floor (prefixed to avoid conflict with SeatingFeatureExtractor.is_floor)
        result["sec_is_floor"] = sections.apply(lambda s: _check_keywords(s, _FLOOR_KEYWORDS))

        # 8. is_obstructed
        result["is_obstructed"] = sections.apply(lambda s: _check_keywords(s, _OBSTRUCTED_KEYWORDS))

        # 9. section_name_hash
        result["section_name_hash"] = sections.apply(_section_hash)

        # 10. has_row_info
        if has_row:
            result["has_row_info"] = (
                df["row"].notna() & (df["row"].astype(str).str.strip() != "")
            ).astype(int)
        else:
            result["has_row_info"] = 0

        # 11. section_word_count
        result["section_word_count"] = sections.apply(lambda s: len(s.split()) if s.strip() else 0)

        # 12. venue_section_count
        if "venue_name" in df.columns:
            result["venue_section_count"] = df["venue_name"].apply(
                lambda v: self._venue_section_counts.get(str(v), 0.0)
            )
        else:
            result["venue_section_count"] = 0.0

        return result

    def get_params(self) -> dict[str, Any]:
        return {"venue_section_counts": self._venue_section_counts}

    def set_params(self, params: dict[str, Any]) -> None:
        if "venue_section_counts" in params:
            self._venue_section_counts = params["venue_section_counts"]

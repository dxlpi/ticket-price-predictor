"""Tests for SectionFeatureExtractor."""

from __future__ import annotations

import math

import pandas as pd

from ticket_price_predictor.ml.features.section_encoding import SectionFeatureExtractor

_FEATURE_NAMES = [
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


def _single(section: str, row: str | None = None, venue: str | None = None) -> pd.Series:
    data: dict[str, list[object]] = {"section": [section]}
    if row is not None:
        data["row"] = [row]
    if venue is not None:
        data["venue_name"] = [venue]
    df = pd.DataFrame(data)
    ext = SectionFeatureExtractor()
    ext.fit(df)
    result = ext.extract(df)
    return result.iloc[0]


def test_section_107_number_and_level() -> None:
    row = _single("Section 107")
    assert row["section_number"] == 107
    assert math.isclose(row["section_level"], 0.2)
    assert row["is_named_section"] == 0


def test_orchestra_left_named_section() -> None:
    row = _single("Orchestra Left")
    assert row["is_named_section"] == 1
    assert math.isclose(row["section_level"], 0.2)


def test_ga_floor() -> None:
    row = _single("GA Floor")
    assert row["sec_is_ga"] == 1
    assert row["sec_is_floor"] == 1
    assert math.isclose(row["section_level"], 0.0)


def test_vip_box() -> None:
    row = _single("VIP Box 3")
    assert row["sec_is_vip"] == 1


def test_upper_level_308_row_a() -> None:
    row = _single("Upper Level 308", row="A")
    assert row["section_number"] == 308
    assert math.isclose(row["section_level"], 0.7)
    assert math.isclose(row["row_quality"], 0.02)


def test_extract_returns_12_columns() -> None:
    df = pd.DataFrame(
        {
            "section": ["Section 101", "GA Pit", "Orchestra", "Upper 412"],
            "row": ["A", None, "B", "5"],
            "venue_name": ["Madison Square Garden"] * 4,
        }
    )
    ext = SectionFeatureExtractor()
    ext.fit(df)
    result = ext.extract(df)
    assert list(result.columns) == _FEATURE_NAMES
    assert len(result) == 4


def test_venue_section_count_learned_from_fit() -> None:
    train_df = pd.DataFrame(
        {
            "section": ["101", "102", "103", "201"],
            "venue_name": ["Arena A", "Arena A", "Arena A", "Arena B"],
        }
    )
    test_df = pd.DataFrame(
        {
            "section": ["101", "201"],
            "venue_name": ["Arena A", "Arena B"],
        }
    )
    ext = SectionFeatureExtractor()
    ext.fit(train_df)
    result = ext.extract(test_df)

    import math as _math

    # Arena A has 3 unique sections → log1p(3)
    assert math.isclose(result.iloc[0]["venue_section_count"], _math.log1p(3))
    # Arena B has 1 unique section → log1p(1)
    assert math.isclose(result.iloc[1]["venue_section_count"], _math.log1p(1))


def test_missing_row_column_handled() -> None:
    df = pd.DataFrame({"section": ["Section 202", "Floor Pit"]})
    ext = SectionFeatureExtractor()
    ext.fit(df)
    result = ext.extract(df)
    # has_row_info defaults to 0
    assert (result["has_row_info"] == 0).all()
    # row_quality defaults to 0.5
    assert (result["row_quality"] == 0.5).all()
    assert list(result.columns) == _FEATURE_NAMES


def test_section_number_zero_for_unknown() -> None:
    row = _single("Orchestra Left")
    assert row["section_number"] == 0


def test_is_obstructed() -> None:
    row = _single("Section 112 Limited View")
    assert row["is_obstructed"] == 1


def test_section_hash_range() -> None:
    row = _single("Section 101")
    assert 0 <= row["section_name_hash"] <= 31


def test_section_word_count() -> None:
    row = _single("Upper Level West")
    assert row["section_word_count"] == 3

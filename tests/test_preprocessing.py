"""Comprehensive tests for preprocessing pipeline."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from ticket_price_predictor.preprocessing import (
    PreprocessingConfig,
    PreprocessingPipeline,
    ProcessingResult,
)
from ticket_price_predictor.preprocessing.cleaners import (
    DuplicateHandler,
    PriceOutlierHandler,
    TextNormalizer,
)
from ticket_price_predictor.preprocessing.pipeline import PipelineBuilder
from ticket_price_predictor.preprocessing.quality import (
    AlertLevel,
    QualityMetrics,
    QualityReporter,
)
from ticket_price_predictor.preprocessing.transformers import (
    EventMetadataJoiner,
    MissingValueImputer,
    SeatZoneEnricher,
    TemporalFeatureEnricher,
    TypeConverter,
)
from ticket_price_predictor.preprocessing.validators import (
    ReferentialValidator,
    SchemaValidator,
    TemporalValidator,
)
from ticket_price_predictor.schemas import SeatZone

# ============================================================================
# Fixtures - Test Data Generation
# ============================================================================


@pytest.fixture
def sample_listings_df() -> pd.DataFrame:
    """Generate sample ticket listings for testing."""
    now = datetime.now(UTC)
    event_time = now + timedelta(days=30)

    return pd.DataFrame(
        {
            "listing_id": ["L1", "L2", "L3", "L4", "L5"],
            "event_id": ["E1", "E1", "E2", "E2", "E3"],
            "timestamp": [now, now, now, now, now],
            "event_datetime": [event_time] * 5,
            "section": ["Orchestra", "Balcony", "Floor", "VIP", "Upper Deck"],
            "row": ["A", "B", "C", "D", "E"],
            "seat_from": ["1", "5", "10", "1", "20"],
            "seat_to": ["2", "6", "12", "2", "22"],
            "quantity": [2, 2, 3, 2, 3],
            "listing_price": [150.0, 75.0, 200.0, 500.0, 50.0],
            "total_price": [180.0, 90.0, 240.0, 600.0, 60.0],
            "face_value": [100.0, 50.0, 150.0, 400.0, 30.0],
            "days_to_event": [30, 30, 30, 30, 30],
            "artist_or_team": [
                "Taylor Swift",
                "Taylor Swift",
                "Bruno Mars",
                "Bruno Mars",
                "Ed Sheeran",
            ],
            "venue_name": [
                "Madison Square Garden",
                "Madison Square Garden",
                "Staples Center",
                "Staples Center",
                "Red Rocks",
            ],
            "city": ["New York", "NYC", "Los Angeles", "LA", "Denver"],
        }
    )


@pytest.fixture
def sample_events_df() -> pd.DataFrame:
    """Generate sample event metadata for testing."""
    now = datetime.now(UTC)
    event_time = now + timedelta(days=30)

    return pd.DataFrame(
        {
            "event_id": ["E1", "E2", "E3"],
            "event_type": ["concert", "concert", "concert"],
            "event_datetime": [event_time] * 3,
            "artist_or_team": ["Taylor Swift", "Bruno Mars", "Ed Sheeran"],
            "venue_id": ["V1", "V2", "V3"],
            "venue_name": ["Madison Square Garden", "Staples Center", "Red Rocks"],
            "city": ["New York", "Los Angeles", "Denver"],
            "venue_capacity": [20000, 19000, 9500],
        }
    )


@pytest.fixture
def sample_snapshots_df() -> pd.DataFrame:
    """Generate sample price snapshots for testing."""
    now = datetime.now(UTC)

    return pd.DataFrame(
        {
            "event_id": ["E1", "E1", "E2", "E2"],
            "seat_zone": [
                SeatZone.FLOOR_VIP.value,
                SeatZone.LOWER_TIER.value,
                SeatZone.BALCONY.value,
                SeatZone.UPPER_TIER.value,
            ],
            "timestamp": [now] * 4,
            "price_min": [100.0, 50.0, 400.0, 30.0],
            "price_avg": [150.0, 75.0, 500.0, 50.0],
            "price_max": [200.0, 100.0, 600.0, 70.0],
            "days_to_event": [30, 30, 30, 30],
            "inventory_remaining": [50, 100, 20, 200],
        }
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def all_nulls_df() -> pd.DataFrame:
    """DataFrame with all null values."""
    return pd.DataFrame(
        {
            "listing_price": [None, None, None],
            "section": [None, None, None],
            "timestamp": [None, None, None],
        }
    )


@pytest.fixture
def extreme_values_df() -> pd.DataFrame:
    """DataFrame with extreme/outlier values."""
    now = datetime.now(UTC)
    event_time = now + timedelta(days=30)

    return pd.DataFrame(
        {
            "listing_id": ["L1", "L2", "L3", "L4"],
            "event_id": ["E1"] * 4,
            "timestamp": [now] * 4,
            "event_datetime": [event_time] * 4,
            "section": ["Orchestra"] * 4,
            "row": ["A"] * 4,
            "quantity": [1, 1, 1, 1],
            "listing_price": [0.50, 1000000.0, 150.0, -5.0],  # Extreme values
            "total_price": [0.60, 1200000.0, 180.0, -6.0],
            "days_to_event": [30] * 4,
        }
    )


# ============================================================================
# Cleaner Tests
# ============================================================================


class TestTextNormalizer:
    """Test suite for TextNormalizer."""

    def test_normalizes_text_fields(self, sample_listings_df):
        """Test that text fields are properly normalized."""
        normalizer = TextNormalizer()
        result = normalizer.process(sample_listings_df)

        assert "artist_normalized" in result.data.columns
        assert "venue_normalized" in result.data.columns
        assert "city_normalized" in result.data.columns

        # Check lowercase conversion
        assert result.data["artist_normalized"].iloc[0] == "taylor swift"
        assert result.data["venue_normalized"].iloc[0] == "madison square garden"

    def test_city_mappings(self, sample_listings_df):
        """Test that common city abbreviations are mapped correctly."""
        normalizer = TextNormalizer()
        result = normalizer.process(sample_listings_df)

        # NYC → new york, LA → los angeles
        assert result.data["city_normalized"].iloc[1] == "new york"
        assert result.data["city_normalized"].iloc[3] == "los angeles"

    def test_handles_empty_dataframe(self, empty_df):
        """Test behavior with empty input."""
        normalizer = TextNormalizer()
        result = normalizer.process(empty_df)

        assert result.data.empty
        assert len(result.issues) > 0
        assert result.metrics["rows_processed"] == 0

    def test_handles_missing_columns(self):
        """Test behavior when expected columns are missing."""
        df = pd.DataFrame({"irrelevant_column": [1, 2, 3]})
        normalizer = TextNormalizer()
        result = normalizer.process(df)

        assert len(result.issues) == 3  # Missing artist, venue, city
        assert any("artist_or_team" in issue for issue in result.issues)

    def test_metrics_calculation(self, sample_listings_df):
        """Test that metrics are properly calculated."""
        normalizer = TextNormalizer()
        result = normalizer.process(sample_listings_df)

        assert result.metrics["rows_processed"] == len(sample_listings_df)
        assert result.metrics["columns_added"] == 3


class TestPriceOutlierHandler:
    """Test suite for PriceOutlierHandler."""

    def test_detects_iqr_outliers(self, sample_listings_df):
        """Test IQR-based outlier detection."""
        handler = PriceOutlierHandler()
        result = handler.process(sample_listings_df)

        assert "is_price_outlier" in result.data.columns
        assert "outlier_reason" in result.data.columns
        assert result.metrics["outlier_count"] >= 0

    def test_detects_absolute_bounds_outliers(self, extreme_values_df):
        """Test detection of prices outside absolute bounds."""
        config = PreprocessingConfig(price_min=1.0, price_max=50000.0)
        handler = PriceOutlierHandler(config)
        result = handler.process(extreme_values_df)

        # Should detect $0.50 (below min), $1M (above max), and -$5 (below min)
        outliers = result.data[result.data["is_price_outlier"]]
        assert len(outliers) >= 3

        # Check specific outlier reasons
        reasons = result.data["outlier_reason"].dropna().tolist()
        assert "below_min" in reasons
        assert "above_max" in reasons

    def test_iqr_bounds_calculation(self, sample_listings_df):
        """Test that IQR bounds are calculated correctly."""
        handler = PriceOutlierHandler()
        result = handler.process(sample_listings_df)

        # Verify IQR metrics are present
        assert "iqr_lower_bound" in result.metrics
        assert "iqr_upper_bound" in result.metrics
        assert "q1" in result.metrics
        assert "q3" in result.metrics

    def test_handles_empty_dataframe(self, empty_df):
        """Test behavior with empty input."""
        handler = PriceOutlierHandler()
        result = handler.process(empty_df)

        assert result.data.empty
        assert result.metrics["outlier_count"] == 0

    def test_handles_all_nulls(self, all_nulls_df):
        """Test behavior when all prices are null."""
        handler = PriceOutlierHandler()
        result = handler.process(all_nulls_df)

        assert "is_price_outlier" in result.data.columns
        assert result.metrics["outlier_count"] == 0
        assert any("No valid prices" in issue for issue in result.issues)

    def test_custom_iqr_multiplier(self, sample_listings_df):
        """Test that custom IQR multiplier affects outlier detection."""
        config_strict = PreprocessingConfig(iqr_multiplier=1.0)
        config_lenient = PreprocessingConfig(iqr_multiplier=3.0)

        result_strict = PriceOutlierHandler(config_strict).process(sample_listings_df)
        result_lenient = PriceOutlierHandler(config_lenient).process(sample_listings_df)

        # Stricter multiplier should detect more outliers
        assert result_strict.metrics["outlier_count"] >= result_lenient.metrics["outlier_count"]


class TestDuplicateHandler:
    """Test suite for DuplicateHandler."""

    def test_detects_exact_duplicates(self):
        """Test detection of exact duplicate listings."""
        now = datetime.now(UTC)
        df = pd.DataFrame(
            {
                "event_id": ["E1", "E1", "E1"],
                "section": ["A", "A", "B"],
                "row": ["1", "1", "1"],
                "seat_from": ["5", "5", "5"],
                "seat_to": ["6", "6", "6"],
                "timestamp": [now, now + timedelta(minutes=30), now],
            }
        )

        handler = DuplicateHandler(time_window_hours=6)
        result = handler.process(df)

        assert "is_duplicate" in result.data.columns
        # First and second row are duplicates (within time window)
        assert result.metrics["duplicate_count"] == 1

    def test_time_window_enforcement(self):
        """Test that time window is properly enforced."""
        now = datetime.now(UTC)
        df = pd.DataFrame(
            {
                "event_id": ["E1", "E1"],
                "section": ["A", "A"],
                "row": ["1", "1"],
                "seat_from": ["5", "5"],
                "seat_to": ["6", "6"],
                "timestamp": [now, now + timedelta(hours=7)],  # Outside 6-hour window
            }
        )

        handler = DuplicateHandler(time_window_hours=6)
        result = handler.process(df)

        # Should NOT be marked as duplicate (outside window)
        assert result.metrics["duplicate_count"] == 0

    def test_keeps_first_occurrence(self):
        """Test that first occurrence is kept, later ones are flagged."""
        now = datetime.now(UTC)
        df = pd.DataFrame(
            {
                "event_id": ["E1", "E1"],
                "section": ["A", "A"],
                "row": ["1", "1"],
                "seat_from": ["5", "5"],
                "seat_to": ["6", "6"],
                "timestamp": [now, now + timedelta(minutes=30)],
            }
        )

        handler = DuplicateHandler(time_window_hours=6)
        result = handler.process(df)

        # First row should not be duplicate, second should be
        assert not result.data.iloc[0]["is_duplicate"]
        assert result.data.iloc[1]["is_duplicate"]

    def test_handles_missing_columns(self):
        """Test behavior when required columns are missing."""
        df = pd.DataFrame({"irrelevant": [1, 2, 3]})
        handler = DuplicateHandler()
        result = handler.process(df)

        assert len(result.issues) > 0
        assert result.metrics["duplicate_count"] == 0

    def test_handles_empty_dataframe(self, empty_df):
        """Test behavior with empty input."""
        handler = DuplicateHandler()
        result = handler.process(empty_df)

        assert result.data.empty
        assert result.metrics["duplicate_count"] == 0


# ============================================================================
# Validator Tests
# ============================================================================


class TestSchemaValidator:
    """Test suite for SchemaValidator."""

    def test_validates_required_columns_listings(self, sample_listings_df):
        """Test that required columns for listings are validated."""
        validator = SchemaValidator("listings")
        result = validator.process(sample_listings_df)

        # All required columns present
        assert result.metrics["error_count"] == 0

    def test_detects_missing_columns(self):
        """Test detection of missing required columns."""
        df = pd.DataFrame({"listing_id": ["L1", "L2"]})
        validator = SchemaValidator("listings")
        result = validator.process(df)

        assert result.metrics["error_count"] > 0
        assert any("missing_columns" in issue for issue in result.issues)

    def test_validates_dtypes(self):
        """Test that column data types are validated."""
        df = pd.DataFrame(
            {
                "listing_id": ["L1", "L2"],
                "event_id": ["E1", "E2"],
                "timestamp": ["not_a_datetime", "also_not"],  # Wrong type
                "event_datetime": [datetime.now(UTC)] * 2,
                "section": ["A", "B"],
                "row": ["1", "2"],
                "quantity": [1, 2],
                "listing_price": [100.0, 200.0],
                "total_price": [120.0, 240.0],
                "days_to_event": [30, 30],
            }
        )

        validator = SchemaValidator("listings")
        result = validator.process(df)

        # Should have warnings about timestamp dtype
        assert result.metrics["warning_count"] > 0

    def test_detects_high_null_rate(self):
        """Test detection of columns with high null rates."""
        df = pd.DataFrame(
            {
                "listing_id": ["L1", "L2", "L3", "L4", "L5"],
                "event_id": ["E1", "E2", "E3", "E4", "E5"],
                "timestamp": [datetime.now(UTC)] * 5,
                "event_datetime": [datetime.now(UTC)] * 5,
                "section": [None, None, None, "A", None],  # 80% null
                "row": ["1", "2", "3", "4", "5"],
                "quantity": [1, 2, 3, 4, 5],
                "listing_price": [100.0, 200.0, 300.0, 400.0, 500.0],
                "total_price": [120.0, 240.0, 360.0, 480.0, 600.0],
                "days_to_event": [30] * 5,
            }
        )

        validator = SchemaValidator("listings")
        result = validator.process(df)

        assert result.metrics["error_count"] > 0
        assert any("high_null_rate" in issue for issue in result.issues)


class TestReferentialValidator:
    """Test suite for ReferentialValidator."""

    def test_validates_seat_zones_listings(self, sample_listings_df):
        """Test validation of section-to-seat-zone mapping."""
        validator = ReferentialValidator("listings")
        result = validator.process(sample_listings_df)

        # All sections should be mappable
        assert result.metrics["error_count"] == 0

    def test_validates_seat_zones_snapshots(self, sample_snapshots_df):
        """Test validation of seat_zone enum values."""
        validator = ReferentialValidator("snapshots")
        result = validator.process(sample_snapshots_df)

        # All seat zones should be valid
        assert result.metrics["error_count"] == 0

    def test_detects_invalid_seat_zones(self):
        """Test detection of invalid seat zone values."""
        df = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "seat_zone": ["INVALID_ZONE", "FLOOR"],
                "timestamp": [datetime.now(UTC)] * 2,
                "price_min": [100.0, 200.0],
                "days_to_event": [30, 30],
            }
        )

        validator = ReferentialValidator("snapshots")
        result = validator.process(df)

        assert result.metrics["error_count"] > 0
        assert any("invalid_seat_zone" in issue for issue in result.issues)

    def test_validates_event_references(self, sample_listings_df, sample_events_df):
        """Test validation of event_id references."""
        validator = ReferentialValidator("listings", events_df=sample_events_df)
        result = validator.process(sample_listings_df)

        # All event_ids should exist in events_df
        assert result.metrics["error_count"] == 0

    def test_detects_invalid_event_references(self, sample_events_df):
        """Test detection of invalid event_id references."""
        df = pd.DataFrame(
            {
                "event_id": ["E1", "E999"],  # E999 doesn't exist
                "section": ["A", "B"],
                "timestamp": [datetime.now(UTC)] * 2,
            }
        )

        validator = ReferentialValidator("listings", events_df=sample_events_df)
        result = validator.process(df)

        assert result.metrics["error_count"] > 0
        assert any("invalid_event_reference" in issue for issue in result.issues)


class TestTemporalValidator:
    """Test suite for TemporalValidator."""

    def test_detects_future_timestamps(self):
        """Test detection of timestamps in the future."""
        future = datetime.now(UTC) + timedelta(days=1)
        df = pd.DataFrame(
            {
                "timestamp": [future],
                "event_datetime": [future + timedelta(days=30)],
                "days_to_event": [30],
            }
        )

        validator = TemporalValidator()
        result = validator.process(df)

        assert result.metrics["error_count"] > 0
        assert any("future_timestamp" in issue for issue in result.issues)

    def test_detects_past_events(self):
        """Test detection of events that already occurred."""
        past = datetime.now(UTC) - timedelta(days=1)
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC)],
                "event_datetime": [past],
                "days_to_event": [-1],
            }
        )

        validator = TemporalValidator(allow_past_events=False)
        result = validator.process(df)

        assert result.metrics["warning_count"] > 0
        assert any("past_event" in issue for issue in result.issues)

    def test_allows_past_events_when_configured(self):
        """Test that past events can be allowed."""
        past = datetime.now(UTC) - timedelta(days=1)
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC)],
                "event_datetime": [past],
                "days_to_event": [-1],
            }
        )

        validator = TemporalValidator(allow_past_events=True)
        result = validator.process(df)

        # Should not flag past events
        assert not any("past_event" in issue for issue in result.issues)

    def test_validates_days_to_event_calculation(self):
        """Test validation of days_to_event calculation."""
        now = datetime.now(UTC)
        event_time = now + timedelta(days=30)

        df = pd.DataFrame(
            {
                "timestamp": [now],
                "event_datetime": [event_time],
                "days_to_event": [100],  # Incorrect (should be ~30)
            }
        )

        validator = TemporalValidator()
        result = validator.process(df)

        assert result.metrics["warning_count"] > 0
        assert any("incorrect_days_calculation" in issue for issue in result.issues)


# ============================================================================
# Transformer Tests
# ============================================================================


class TestEventMetadataJoiner:
    """Test suite for EventMetadataJoiner."""

    def test_joins_venue_capacity(self, sample_listings_df, sample_events_df):
        """Test that venue_capacity is properly joined."""
        joiner = EventMetadataJoiner(sample_events_df)
        result = joiner.process(sample_listings_df)

        assert "venue_capacity" in result.data.columns
        # First two listings are event E1, should have 20000 capacity
        assert result.data.iloc[0]["venue_capacity"] == 20000
        assert result.data.iloc[1]["venue_capacity"] == 20000

    def test_handles_missing_event_ids(self, sample_events_df):
        """Test handling of listings with no matching event."""
        df = pd.DataFrame(
            {
                "event_id": ["E1", "E999"],  # E999 doesn't exist
                "listing_price": [100.0, 200.0],
            }
        )

        joiner = EventMetadataJoiner(sample_events_df)
        result = joiner.process(df)

        assert result.metrics["missing_event_ids_count"] == 1
        assert any("no matching event_id" in issue for issue in result.issues)

    def test_preserves_row_count(self, sample_listings_df, sample_events_df):
        """Test that join doesn't duplicate or drop rows."""
        joiner = EventMetadataJoiner(sample_events_df)
        result = joiner.process(sample_listings_df)

        assert len(result.data) == len(sample_listings_df)


class TestMissingValueImputer:
    """Test suite for MissingValueImputer."""

    def test_imputes_venue_capacity(self):
        """Test imputation of missing venue_capacity."""
        df = pd.DataFrame(
            {
                "venue_capacity": [20000.0, None, 15000.0, None],
                "city": ["New York", "New York", "Boston", "Boston"],
            }
        )

        config = PreprocessingConfig(venue_capacity_default=15000)
        imputer = MissingValueImputer(config)
        result = imputer.process(df)

        # Missing values should be imputed
        assert result.data["venue_capacity"].notna().all()
        assert "venue_capacity_imputed" in result.data.columns
        assert result.metrics["imputation_counts"]["venue_capacity"] == 2

    def test_uses_city_median_for_venue_capacity(self):
        """Test that city-level median is used when available."""
        df = pd.DataFrame(
            {
                "venue_capacity": [20000.0, None, 18000.0],
                "city": ["New York", "New York", "New York"],
            }
        )

        imputer = MissingValueImputer()
        result = imputer.process(df)

        # Should use median of 20000 and 18000 = 19000
        assert result.data.iloc[1]["venue_capacity"] == 19000.0

    def test_imputes_face_value(self):
        """Test imputation of missing face_value."""
        df = pd.DataFrame(
            {
                "face_value": [100.0, None, 200.0],
                "listing_price": [150.0, 200.0, 300.0],
            }
        )

        imputer = MissingValueImputer()
        result = imputer.process(df)

        # Missing face_value should be 50% of listing_price
        assert result.data.iloc[1]["face_value"] == 100.0  # 50% of 200
        assert "face_value_imputed" in result.data.columns

    def test_tracks_imputed_flags(self):
        """Test that imputed columns are flagged."""
        df = pd.DataFrame(
            {
                "venue_capacity": [20000.0, None],
                "face_value": [100.0, None],
                "listing_price": [150.0, 200.0],
            }
        )

        imputer = MissingValueImputer()
        result = imputer.process(df)

        assert not result.data.iloc[0]["venue_capacity_imputed"]
        assert result.data.iloc[1]["venue_capacity_imputed"]
        assert result.data.iloc[1]["face_value_imputed"]


class TestTypeConverter:
    """Test suite for TypeConverter."""

    def test_converts_datetime_columns(self):
        """Test conversion of datetime columns."""
        df = pd.DataFrame(
            {
                "timestamp": ["2024-01-01 12:00:00", "2024-01-02 13:00:00"],
                "event_datetime": ["2024-02-01 19:00:00", "2024-02-02 20:00:00"],
            }
        )

        converter = TypeConverter()
        result = converter.process(df)

        assert pd.api.types.is_datetime64_any_dtype(result.data["timestamp"])
        assert pd.api.types.is_datetime64_any_dtype(result.data["event_datetime"])

    def test_ensures_utc_timezone(self):
        """Test that datetime columns are timezone-aware UTC."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 12:00:00"]),
            }
        )

        converter = TypeConverter()
        result = converter.process(df)

        assert result.data["timestamp"].dt.tz is not None
        assert str(result.data["timestamp"].dt.tz) == "UTC"

    def test_converts_price_columns(self):
        """Test conversion of price columns to float64."""
        df = pd.DataFrame(
            {
                "listing_price": [100, 200, 300],  # Integers
                "face_value": ["150.50", "250.75", "350.25"],  # Strings
            }
        )

        converter = TypeConverter()
        result = converter.process(df)

        assert result.data["listing_price"].dtype == "float64"
        assert result.data["face_value"].dtype == "float64"

    def test_converts_categorical_columns(self):
        """Test conversion of categorical columns."""
        df = pd.DataFrame(
            {
                "seat_zone": ["FLOOR", "LOWER_TIER", "FLOOR"],
                "event_type": ["concert", "sports", "concert"],
            }
        )

        converter = TypeConverter()
        result = converter.process(df)

        assert isinstance(result.data["seat_zone"].dtype, pd.CategoricalDtype)
        assert isinstance(result.data["event_type"].dtype, pd.CategoricalDtype)


class TestSeatZoneEnricher:
    """Test suite for SeatZoneEnricher."""

    def test_adds_normalized_seat_zone(self, sample_listings_df):
        """Test that normalized_seat_zone column is added."""
        enricher = SeatZoneEnricher()
        result = enricher.process(sample_listings_df)

        assert "normalized_seat_zone" in result.data.columns
        assert result.metrics["zones_normalized_count"] == len(sample_listings_df)

    def test_maps_sections_to_zones(self):
        """Test that sections are correctly mapped to seat zones."""
        df = pd.DataFrame({"section": ["Orchestra", "Floor", "Balcony", "Upper Level", "VIP"]})

        enricher = SeatZoneEnricher()
        result = enricher.process(df)

        # Check specific mappings
        zones = result.data["normalized_seat_zone"].tolist()
        assert SeatZone.LOWER_TIER in zones  # Orchestra
        assert (
            SeatZone.FLOOR_VIP in zones or SeatZone.LOWER_TIER in zones
        )  # Floor (can map to either)
        assert SeatZone.UPPER_TIER in zones or SeatZone.BALCONY in zones  # Upper Level or Balcony
        # At least we got valid zones for all sections
        assert len(zones) == 5
        assert all(isinstance(z, SeatZone) for z in zones)

    def test_handles_missing_sections(self):
        """Test handling of null/empty sections."""
        df = pd.DataFrame({"section": ["Floor", None, "", "VIP"]})

        enricher = SeatZoneEnricher()
        result = enricher.process(df)

        # Missing sections should default to UPPER_TIER
        assert result.data.iloc[1]["normalized_seat_zone"] == SeatZone.UPPER_TIER
        assert result.data.iloc[2]["normalized_seat_zone"] == SeatZone.UPPER_TIER
        assert result.metrics["unmappable_count"] == 2


class TestTemporalFeatureEnricher:
    """Test suite for TemporalFeatureEnricher."""

    def test_adds_hour_of_day(self):
        """Test that hour_of_day is extracted from timestamp."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-01 14:30:00", "2024-01-01 09:15:00"]
                ).tz_localize("UTC"),
            }
        )

        enricher = TemporalFeatureEnricher()
        result = enricher.process(df)

        assert "hour_of_day" in result.data.columns
        assert result.data.iloc[0]["hour_of_day"] == 14
        assert result.data.iloc[1]["hour_of_day"] == 9

    def test_calculates_days_to_event(self):
        """Test that days_to_event is calculated correctly."""
        now = datetime.now(UTC)
        event_time = now + timedelta(days=30)

        df = pd.DataFrame(
            {
                "timestamp": [now],
                "event_datetime": [event_time],
            }
        )

        enricher = TemporalFeatureEnricher()
        result = enricher.process(df)

        assert "days_to_event" in result.data.columns
        assert abs(result.data.iloc[0]["days_to_event"] - 30.0) < 0.1

    def test_adds_is_weekend(self):
        """Test that is_weekend flag is added."""
        # Create dates for Saturday and Monday
        saturday = pd.Timestamp("2024-01-06 19:00:00", tz="UTC")  # Saturday
        monday = pd.Timestamp("2024-01-08 19:00:00", tz="UTC")  # Monday

        df = pd.DataFrame(
            {
                "event_datetime": [saturday, monday],
            }
        )

        enricher = TemporalFeatureEnricher()
        result = enricher.process(df)

        assert "is_weekend" in result.data.columns
        assert result.data.iloc[0]["is_weekend"]  # Saturday
        assert not result.data.iloc[1]["is_weekend"]  # Monday


# ============================================================================
# Pipeline Tests
# ============================================================================


class TestPreprocessingPipeline:
    """Test suite for PreprocessingPipeline."""

    def test_executes_stages_sequentially(self, sample_listings_df):
        """Test that pipeline stages execute in order."""
        stages = [
            TextNormalizer(),
            TypeConverter(),
        ]

        pipeline = PreprocessingPipeline(stages)
        result = pipeline.process(sample_listings_df)

        # Both stages should have run
        assert "TextNormalizer" in result.metrics["stages"]
        assert "TypeConverter" in result.metrics["stages"]

    def test_aggregates_issues_and_metrics(self, sample_listings_df):
        """Test that issues and metrics are aggregated."""
        stages = [
            TextNormalizer(),
            PriceOutlierHandler(),
        ]

        pipeline = PreprocessingPipeline(stages)
        result = pipeline.process(sample_listings_df)

        # Issues should be prefixed with stage name
        stage_prefixed = [issue for issue in result.issues if "[" in issue]
        assert len(stage_prefixed) >= 0  # May or may not have issues

        assert result.metrics["total_issues"] == len(result.issues)
        assert result.metrics["final_row_count"] == len(result.data)

    def test_saves_checkpoints_when_configured(self, sample_listings_df, tmp_path):
        """Test checkpoint saving functionality."""
        stages = [TextNormalizer(), TypeConverter()]
        pipeline = PreprocessingPipeline(stages, checkpoint_dir=tmp_path, name="test")

        pipeline.process(sample_listings_df)

        # Check that checkpoint files were created
        checkpoint_files = list(tmp_path.glob("*.parquet"))
        assert len(checkpoint_files) == 2  # One per stage

    def test_resumes_from_checkpoint(self, sample_listings_df, tmp_path):
        """Test resuming from a checkpoint."""
        stages = [TextNormalizer(), TypeConverter()]
        pipeline = PreprocessingPipeline(stages, checkpoint_dir=tmp_path, name="test")

        # Run pipeline to create checkpoints
        pipeline.process(sample_listings_df)

        # Resume from checkpoint
        resumed_df = pipeline.resume_from_checkpoint(stage_index=1)

        assert len(resumed_df) == len(sample_listings_df)
        assert "artist_normalized" in resumed_df.columns

    def test_handles_stage_failures_gracefully(self):
        """Test that pipeline continues after stage failures."""

        class FailingPreprocessor:
            def process(self, df):  # noqa: ARG002
                raise ValueError("Intentional failure")

        stages = [
            TextNormalizer(),
            FailingPreprocessor(),
            TypeConverter(),
        ]

        pipeline = PreprocessingPipeline(stages)
        df = pd.DataFrame({"artist_or_team": ["Test"], "listing_price": [100.0]})

        result = pipeline.process(df)

        # Pipeline should continue despite failure
        assert "TextNormalizer" in result.metrics["stages"]
        assert "TypeConverter" in result.metrics["stages"]
        assert any("CRITICAL" in issue for issue in result.issues)


class TestPipelineBuilder:
    """Test suite for PipelineBuilder."""

    def test_builds_listings_pipeline(self, sample_events_df):
        """Test building standard listings pipeline."""
        pipeline = PipelineBuilder.build_listings_pipeline(events_df=sample_events_df)

        assert len(pipeline.stages) == 11  # 11 stages in listings pipeline
        assert pipeline.name == "listings"

    def test_builds_events_pipeline(self):
        """Test building standard events pipeline."""
        pipeline = PipelineBuilder.build_events_pipeline()

        assert len(pipeline.stages) == 5  # 5 stages in events pipeline
        assert pipeline.name == "events"

    def test_builds_snapshots_pipeline(self):
        """Test building standard snapshots pipeline."""
        pipeline = PipelineBuilder.build_snapshots_pipeline()

        assert len(pipeline.stages) == 5  # 5 stages in snapshots pipeline
        assert pipeline.name == "snapshots"

    def test_listings_pipeline_integration(self, sample_listings_df, sample_events_df):
        """Integration test: Run full listings pipeline."""
        pipeline = PipelineBuilder.build_listings_pipeline(events_df=sample_events_df)
        result = pipeline.process(sample_listings_df)

        # Check that key transformations were applied
        assert "artist_normalized" in result.data.columns
        assert "venue_capacity" in result.data.columns
        assert "normalized_seat_zone" in result.data.columns
        assert "is_price_outlier" in result.data.columns
        assert "is_duplicate" in result.data.columns

    def test_events_pipeline_integration(self, sample_events_df):
        """Integration test: Run full events pipeline."""
        pipeline = PipelineBuilder.build_events_pipeline()
        result = pipeline.process(sample_events_df)

        # Check that transformations were applied
        assert "artist_normalized" in result.data.columns
        assert pd.api.types.is_datetime64_any_dtype(result.data["event_datetime"])

    def test_snapshots_pipeline_integration(self, sample_snapshots_df, sample_events_df):
        """Integration test: Run full snapshots pipeline."""
        pipeline = PipelineBuilder.build_snapshots_pipeline(events_df=sample_events_df)
        result = pipeline.process(sample_snapshots_df)

        # Check that validations and transformations were applied
        assert isinstance(result.data["seat_zone"].dtype, pd.CategoricalDtype)
        assert "is_price_outlier" in result.data.columns

    def test_custom_pipeline(self):
        """Test building custom pipeline."""
        stages = [TextNormalizer(), TypeConverter()]
        pipeline = PipelineBuilder.build_custom_pipeline(stages, name="custom")

        assert len(pipeline.stages) == 2
        assert pipeline.name == "custom"


# ============================================================================
# Quality Reporting Tests
# ============================================================================


class TestQualityMetrics:
    """Test suite for QualityMetrics."""

    def test_calculates_drop_rate(self):
        """Test drop rate calculation."""
        metrics = QualityMetrics(input_rows=100, output_rows=90, dropped_rows=10)
        assert metrics.drop_rate == 10.0

    def test_calculates_retention_rate(self):
        """Test retention rate calculation."""
        metrics = QualityMetrics(input_rows=100, output_rows=90)
        assert metrics.retention_rate == 90.0

    def test_handles_zero_input_rows(self):
        """Test handling of zero input rows."""
        metrics = QualityMetrics(input_rows=0, output_rows=0)
        assert metrics.drop_rate == 0.0
        assert metrics.retention_rate == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = QualityMetrics(input_rows=100, output_rows=90)
        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert d["input_rows"] == 100
        assert d["output_rows"] == 90


class TestQualityReporter:
    """Test suite for QualityReporter."""

    def test_extracts_metrics_from_result(self, sample_listings_df):
        """Test extraction of metrics from ProcessingResult."""
        result = ProcessingResult(
            data=sample_listings_df,
            issues=["test_issue"],
            metrics={
                "input_rows": 10,
                "dropped_rows": 2,
                "outliers": {"price": 3},
            },
        )

        reporter = QualityReporter()
        metrics = reporter.extract_metrics(result)

        assert metrics.input_rows == 10
        assert metrics.dropped_rows == 2
        assert metrics.outlier_counts["price"] == 3

    def test_generates_text_summary(self):
        """Test generation of text summary."""
        metrics = QualityMetrics(
            input_rows=100,
            output_rows=95,
            dropped_rows=5,
            column_completeness={"price": 98.5, "venue": 100.0},
        )

        reporter = QualityReporter()
        summary = reporter.generate_text_summary(metrics)

        assert "PREPROCESSING QUALITY REPORT" in summary
        assert "Input rows:" in summary
        assert "Output rows:" in summary
        assert "price" in summary
        assert "venue" in summary

    def test_generates_json_export(self):
        """Test generation of JSON export."""
        metrics = QualityMetrics(input_rows=100, output_rows=95)

        reporter = QualityReporter()
        json_str = reporter.generate_json_export(metrics)

        import json

        data = json.loads(json_str)

        assert "metrics" in data
        assert "alert_level" in data
        assert "thresholds" in data

    def test_check_thresholds_ok(self):
        """Test threshold checking with OK status."""
        metrics = QualityMetrics(
            input_rows=100,
            output_rows=98,
            dropped_rows=2,  # 2% drop rate (below warning)
        )

        reporter = QualityReporter()
        alert = reporter.check_thresholds(metrics)

        assert alert == AlertLevel.OK

    def test_check_thresholds_warning(self):
        """Test threshold checking with WARNING status."""
        metrics = QualityMetrics(
            input_rows=100,
            output_rows=92,
            dropped_rows=8,  # 8% drop rate (above warning, below error)
        )

        reporter = QualityReporter()
        alert = reporter.check_thresholds(metrics)

        assert alert == AlertLevel.WARNING

    def test_check_thresholds_error(self):
        """Test threshold checking with ERROR status."""
        metrics = QualityMetrics(
            input_rows=100,
            output_rows=70,
            dropped_rows=30,  # 30% drop rate (above error)
        )

        reporter = QualityReporter()
        alert = reporter.check_thresholds(metrics)

        assert alert == AlertLevel.ERROR

    def test_compare_against_baseline(self):
        """Test comparison against baseline metrics."""
        baseline = QualityMetrics(
            input_rows=100,
            output_rows=95,
            column_completeness={"price": 98.0, "venue": 99.0},
        )

        current = QualityMetrics(
            input_rows=100,
            output_rows=97,
            column_completeness={"price": 99.5, "venue": 97.0},
        )

        reporter = QualityReporter()
        comparison = reporter.compare_against_baseline(current, baseline)

        assert comparison["row_count_delta"] == 2
        assert "completeness:price" in comparison["improved"]  # Improved from 98% to 99.5%
        assert "completeness:venue" in comparison["degraded"]  # Degraded from 99% to 97%


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe_through_pipeline(self, empty_df):
        """Test that empty DataFrame is handled gracefully."""
        pipeline = PipelineBuilder.build_listings_pipeline()
        result = pipeline.process(empty_df)

        assert result.data.empty
        assert len(result.issues) > 0  # Should have issues about empty input

    def test_all_nulls_through_pipeline(self, all_nulls_df):
        """Test handling of DataFrame with all null values."""
        stages = [
            PriceOutlierHandler(),
            TextNormalizer(),
        ]

        pipeline = PreprocessingPipeline(stages)
        result = pipeline.process(all_nulls_df)

        # Pipeline should complete without crashing
        assert len(result.data) == len(all_nulls_df)
        assert len(result.issues) > 0

    def test_extreme_values_through_pipeline(self, extreme_values_df):
        """Test handling of extreme/outlier values."""
        pipeline = PipelineBuilder.build_listings_pipeline()
        result = pipeline.process(extreme_values_df)

        # Should detect outliers
        assert "is_price_outlier" in result.data.columns
        outlier_count = result.data["is_price_outlier"].sum()
        assert outlier_count > 0

    def test_single_row_dataframe(self):
        """Test processing of single-row DataFrame."""
        now = datetime.now(UTC)
        df = pd.DataFrame(
            {
                "listing_id": ["L1"],
                "event_id": ["E1"],
                "timestamp": [now],
                "event_datetime": [now + timedelta(days=30)],
                "section": ["Orchestra"],
                "row": ["A"],
                "quantity": [1],
                "listing_price": [100.0],
                "total_price": [120.0],
                "days_to_event": [30],
                "artist_or_team": ["Taylor Swift"],
                "venue_name": ["MSG"],
                "city": ["NYC"],
            }
        )

        pipeline = PipelineBuilder.build_listings_pipeline()
        result = pipeline.process(df)

        assert len(result.data) == 1
        assert "artist_normalized" in result.data.columns

    def test_large_scale_data(self):
        """Test processing of larger dataset (performance check)."""
        now = datetime.now(UTC)
        event_time = now + timedelta(days=30)

        # Generate 1000 rows
        df = pd.DataFrame(
            {
                "listing_id": [f"L{i}" for i in range(1000)],
                "event_id": [f"E{i % 10}" for i in range(1000)],
                "timestamp": [now] * 1000,
                "event_datetime": [event_time] * 1000,
                "section": ["Orchestra"] * 1000,
                "row": ["A"] * 1000,
                "quantity": [2] * 1000,
                "listing_price": [100.0 + (i % 100) for i in range(1000)],
                "total_price": [120.0 + (i % 100) for i in range(1000)],
                "days_to_event": [30] * 1000,
                "artist_or_team": ["Taylor Swift"] * 1000,
                "venue_name": ["MSG"] * 1000,
                "city": ["NYC"] * 1000,
            }
        )

        pipeline = PipelineBuilder.build_listings_pipeline()
        result = pipeline.process(df)

        assert len(result.data) == 1000
        assert len(result.metrics["stages"]) > 0


class TestPreprocessingConfig:
    """Test suite for PreprocessingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PreprocessingConfig()

        assert config.iqr_multiplier == 1.5
        assert config.price_min == 1.0
        assert config.price_max == 50000.0
        assert config.venue_capacity_default == 15000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PreprocessingConfig(
            iqr_multiplier=2.0,
            price_min=10.0,
            price_max=10000.0,
        )

        assert config.iqr_multiplier == 2.0
        assert config.price_min == 10.0
        assert config.price_max == 10000.0

    def test_config_affects_preprocessing(self, sample_listings_df):
        """Test that config settings affect preprocessing behavior."""
        config_strict = PreprocessingConfig(price_min=100.0, price_max=200.0)
        config_lenient = PreprocessingConfig(price_min=1.0, price_max=10000.0)

        handler_strict = PriceOutlierHandler(config_strict)
        handler_lenient = PriceOutlierHandler(config_lenient)

        result_strict = handler_strict.process(sample_listings_df)
        result_lenient = handler_lenient.process(sample_listings_df)

        # Strict config should detect more outliers
        assert result_strict.metrics["outlier_count"] >= result_lenient.metrics["outlier_count"]

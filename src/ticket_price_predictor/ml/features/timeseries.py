"""Time-series and momentum feature extraction."""

import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor


class TimeSeriesFeatureExtractor(FeatureExtractor):
    """Extract time-based and momentum features."""

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names."""
        return [
            "days_to_event",
            "days_to_event_squared",
            "days_to_event_log",
            "urgency_bucket",
            "is_last_week",
            "is_last_day",
        ]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-series features.

        Expects column: days_to_event
        """
        result = pd.DataFrame(index=df.index)

        days = df["days_to_event"].fillna(30)

        # Core time features
        result["days_to_event"] = days
        result["days_to_event_squared"] = days ** 2
        result["days_to_event_log"] = (days + 1).apply(lambda x: x ** 0.5)  # sqrt transform

        # Urgency buckets
        result["urgency_bucket"] = days.apply(self._urgency_bucket)

        # Binary urgency flags
        result["is_last_week"] = (days <= 7).astype(int)
        result["is_last_day"] = (days <= 1).astype(int)

        return result

    def _urgency_bucket(self, days: float) -> int:
        """Bucket days to event into urgency levels."""
        if days <= 1:
            return 5  # Extreme urgency
        elif days <= 7:
            return 4  # High urgency (last week)
        elif days <= 14:
            return 3  # Moderate urgency
        elif days <= 30:
            return 2  # Low urgency
        elif days <= 60:
            return 1  # Planning phase
        else:
            return 0  # Early bird


class MomentumFeatureExtractor(FeatureExtractor):
    """Extract price momentum features from historical snapshots.

    This extractor requires pre-computed momentum columns.
    Use compute_momentum_features() to prepare the data.
    """

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names."""
        return [
            "price_momentum_7d",
            "price_momentum_30d",
            "price_vs_initial",
            "price_volatility",
        ]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract momentum features.

        Expects pre-computed columns or fills with defaults.
        """
        result = pd.DataFrame(index=df.index)

        # Use pre-computed columns if available, else defaults
        result["price_momentum_7d"] = df.get("price_momentum_7d", 0.0)
        result["price_momentum_30d"] = df.get("price_momentum_30d", 0.0)
        result["price_vs_initial"] = df.get("price_vs_initial", 1.0)
        result["price_volatility"] = df.get("price_volatility", 0.0)

        return result

    @staticmethod
    def compute_momentum_features(
        df: pd.DataFrame,
        price_col: str = "listing_price",
        time_col: str = "timestamp",
        group_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute momentum features from raw price data.

        Args:
            df: DataFrame with price time series
            price_col: Column containing prices
            time_col: Column containing timestamps
            group_cols: Columns to group by (e.g., ['event_id', 'section'])

        Returns:
            DataFrame with added momentum columns
        """
        df = df.copy()

        if group_cols is None:
            group_cols = ["event_id"]

        # Sort by time within groups
        df = df.sort_values(group_cols + [time_col])

        # Compute rolling stats within groups
        for group_col in group_cols:
            grouped = df.groupby(group_col)[price_col]

            # Price changes
            df["price_change"] = grouped.diff()
            df["price_pct_change"] = grouped.pct_change()

            # Rolling momentum (7-day approximation using last N rows)
            df["price_momentum_7d"] = grouped.transform(
                lambda x: x.pct_change(periods=min(3, len(x) - 1)).fillna(0)
            )
            df["price_momentum_30d"] = grouped.transform(
                lambda x: x.pct_change(periods=min(10, len(x) - 1)).fillna(0)
            )

            # Price vs initial
            df["price_vs_initial"] = grouped.transform(
                lambda x: x / x.iloc[0] if len(x) > 0 and x.iloc[0] > 0 else 1.0
            )

            # Volatility (std of pct changes)
            df["price_volatility"] = grouped.transform(
                lambda x: x.pct_change().rolling(min(5, len(x)), min_periods=1).std().fillna(0)
            )

        return df

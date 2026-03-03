"""Listing context feature extraction."""

from typing import Any

import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor


class ListingContextFeatureExtractor(FeatureExtractor):
    """Extract listing-level context features.

    Features derived from listing metadata such as source marketplace,
    ticket quantity, and listing timestamp.
    """

    SOURCE_ENCODING = {"vividseats": 0, "stubhub": 1}

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names."""
        return [
            "source_encoded",
            "quantity",
            "is_single_ticket",
            "is_pair",
        ]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract listing context features.

        Expects columns (all optional with graceful defaults):
            source: Marketplace source (VividSeats, StubHub)
            quantity: Number of tickets in listing
        """
        result = pd.DataFrame(index=df.index)

        # Source encoding
        if "source" in df.columns:
            result["source_encoded"] = (
                df["source"].str.lower().str.strip().map(self.SOURCE_ENCODING).fillna(0).astype(int)
            )
        else:
            result["source_encoded"] = 0

        # Quantity features
        if "quantity" in df.columns:
            qty = df["quantity"].fillna(1).astype(int)
            result["quantity"] = qty
            result["is_single_ticket"] = (qty == 1).astype(int)
            result["is_pair"] = (qty == 2).astype(int)
        else:
            result["quantity"] = 1
            result["is_single_ticket"] = 1
            result["is_pair"] = 0

        return result

    def get_params(self) -> dict[str, Any]:
        """Return extractor parameters."""
        return {}

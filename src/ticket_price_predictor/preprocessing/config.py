"""Configuration for preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""

    # Outlier detection
    iqr_multiplier: float = 1.5
    price_min: float = 1.0
    price_max: float = 50000.0

    # Imputation
    venue_capacity_default: int = 15000
    imputation_strategy: str = "median"

    # Text normalization
    normalize_case: bool = True

    # Validation
    strict_mode: bool = False

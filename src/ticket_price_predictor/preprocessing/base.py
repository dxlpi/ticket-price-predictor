"""Base classes for preprocessing pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ProcessingResult:
    """Result of a preprocessing operation."""

    data: pd.DataFrame
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class Preprocessor(ABC):
    """Abstract base class for data preprocessors."""

    @abstractmethod
    def process(self, df: pd.DataFrame) -> ProcessingResult:
        """Process input DataFrame and return result.

        Args:
            df: Input DataFrame to process

        Returns:
            ProcessingResult containing processed data, issues, and metrics
        """
        pass

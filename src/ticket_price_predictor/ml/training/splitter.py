"""Time-based data splitting to avoid data leakage."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DataSplit:
    """Container for train/validation/test splits."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return len(self.X_train)

    @property
    def n_val(self) -> int:
        """Number of validation samples."""
        return len(self.X_val)

    @property
    def n_test(self) -> int:
        """Number of test samples."""
        return len(self.X_test)


class TimeBasedSplitter:
    """Split data by time to avoid data leakage."""

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        time_col: str = "event_datetime",
    ) -> None:
        """Initialize splitter.

        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            time_col: Column to use for time-based ordering
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")

        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._time_col = time_col

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        raw_df: pd.DataFrame | None = None,
    ) -> DataSplit:
        """Split data into train/validation/test sets.

        Args:
            X: Feature DataFrame
            y: Target Series
            raw_df: Original DataFrame with time column (if X doesn't have it)

        Returns:
            DataSplit with train/val/test sets
        """
        # Get time column for ordering
        if raw_df is not None and self._time_col in raw_df.columns:
            time_series = raw_df[self._time_col]
        elif self._time_col in X.columns:
            time_series = X[self._time_col]
        else:
            # Fall back to index-based split
            time_series = pd.Series(range(len(X)), index=X.index)

        # Sort by time
        sorted_indices = time_series.sort_values().index

        # Calculate split points
        n = len(sorted_indices)
        train_end = int(n * self._train_ratio)
        val_end = int(n * (self._train_ratio + self._val_ratio))

        # Split indices
        train_idx = sorted_indices[:train_end]
        val_idx = sorted_indices[train_end:val_end]
        test_idx = sorted_indices[val_end:]

        return DataSplit(
            X_train=X.loc[train_idx],
            y_train=y.loc[train_idx],
            X_val=X.loc[val_idx],
            y_val=y.loc[val_idx],
            X_test=X.loc[test_idx],
            y_test=y.loc[test_idx],
        )


class GroupKFoldSplitter:
    """K-fold cross-validation grouped by event_id."""

    def __init__(
        self,
        n_splits: int = 5,
        group_col: str = "event_id",
    ) -> None:
        """Initialize splitter.

        Args:
            n_splits: Number of folds
            group_col: Column to group by
        """
        self._n_splits = n_splits
        self._group_col = group_col

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
    ) -> Any:
        """Generate train/test indices for each fold.

        Args:
            X: Feature DataFrame
            y: Target Series
            groups: Group labels (e.g., event_id)

        Yields:
            Tuples of (train_indices, test_indices)
        """
        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=self._n_splits)

        yield from gkf.split(X, y, groups)

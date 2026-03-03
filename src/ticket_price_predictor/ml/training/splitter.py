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


@dataclass
class RawDataSplit:
    """Container for raw DataFrame splits (before feature extraction)."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return len(self.train_df)

    @property
    def n_val(self) -> int:
        """Number of validation samples."""
        return len(self.val_df)

    @property
    def n_test(self) -> int:
        """Number of test samples."""
        return len(self.test_df)


class TimeBasedSplitter:
    """Split data by time to avoid data leakage."""

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        time_col: str = "event_datetime",
        stratify_col: str | None = None,
    ) -> None:
        """Initialize splitter.

        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            time_col: Column to use for time-based ordering
            stratify_col: Column to stratify by (e.g. artist) for balanced splits
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")

        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._time_col = time_col
        self._stratify_col = stratify_col

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

    def split_raw(self, df: pd.DataFrame) -> RawDataSplit:
        """Split raw DataFrame before feature extraction.

        When stratify_col is set, groups by that column, splits each group
        temporally, and combines. Artists with <3 samples go entirely to train.

        Args:
            df: Raw DataFrame with time column

        Returns:
            RawDataSplit with train/val/test DataFrames
        """
        if self._stratify_col and self._stratify_col in df.columns:
            return self._split_stratified(df)

        return self._split_temporal(df)

    def _split_temporal(self, df: pd.DataFrame) -> RawDataSplit:
        """Simple temporal split on raw DataFrame."""
        if self._time_col in df.columns:
            time_series = df[self._time_col]
        else:
            time_series = pd.Series(range(len(df)), index=df.index)

        sorted_indices = time_series.sort_values().index

        n = len(sorted_indices)
        train_end = int(n * self._train_ratio)
        val_end = int(n * (self._train_ratio + self._val_ratio))

        train_idx = sorted_indices[:train_end]
        val_idx = sorted_indices[train_end:val_end]
        test_idx = sorted_indices[val_end:]

        return RawDataSplit(
            train_df=df.loc[train_idx].copy(),
            val_df=df.loc[val_idx].copy(),
            test_df=df.loc[test_idx].copy(),
        )

    def _split_stratified(self, df: pd.DataFrame) -> RawDataSplit:
        """Split stratified by group column, temporal within each group."""
        train_parts: list[pd.DataFrame] = []
        val_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []

        assert self._stratify_col is not None
        for _group, group_df in df.groupby(self._stratify_col):
            if len(group_df) < 10:
                # Too few samples — put entirely in train to avoid noisy splits
                train_parts.append(group_df)
                continue

            # Temporal sort within group
            if self._time_col in group_df.columns:
                group_df = group_df.sort_values(self._time_col)
            n = len(group_df)
            train_end = max(1, int(n * self._train_ratio))
            val_end = max(train_end + 1, int(n * (self._train_ratio + self._val_ratio)))

            train_parts.append(group_df.iloc[:train_end])
            val_parts.append(group_df.iloc[train_end:val_end])
            test_parts.append(group_df.iloc[val_end:])

        return RawDataSplit(
            train_df=pd.concat(train_parts, ignore_index=True).copy()
            if train_parts
            else pd.DataFrame(),
            val_df=pd.concat(val_parts, ignore_index=True).copy() if val_parts else pd.DataFrame(),
            test_df=pd.concat(test_parts, ignore_index=True).copy()
            if test_parts
            else pd.DataFrame(),
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


class TemporalGroupCV:
    """Expanding-window temporal cross-validation grouped by event.

    Each fold uses an expanding training window and a fixed-size
    validation/test window. Events are never split across folds.
    """

    def __init__(
        self,
        n_folds: int = 5,
        time_col: str = "event_datetime",
        group_col: str = "event_id",
    ) -> None:
        """Initialize temporal CV.

        Args:
            n_folds: Number of folds
            time_col: Column to sort by time
            group_col: Column to group by (prevents same event in train and val)
        """
        self._n_folds = n_folds
        self._time_col = time_col
        self._group_col = group_col

    def split(self, df: pd.DataFrame) -> list[RawDataSplit]:
        """Generate expanding-window folds from raw DataFrame.

        Args:
            df: Raw DataFrame with time and group columns

        Returns:
            List of RawDataSplit objects, one per fold
        """
        if self._time_col in df.columns:
            df = df.sort_values(self._time_col)

        # Get unique groups in temporal order
        if self._group_col in df.columns:
            groups = df.groupby(self._group_col)[self._time_col].min().sort_values()
            group_order = groups.index.tolist()
        else:
            group_order = list(range(len(df)))

        n_groups = len(group_order)
        fold_size = max(1, n_groups // (self._n_folds + 1))

        folds: list[RawDataSplit] = []

        for i in range(self._n_folds):
            # Expanding training window
            train_end = fold_size * (i + 1)
            val_end = min(train_end + fold_size, n_groups)
            test_end = min(val_end + fold_size, n_groups)

            if val_end >= n_groups:
                break

            train_groups = set(group_order[:train_end])
            val_groups = set(group_order[train_end:val_end])
            test_groups = set(group_order[val_end:test_end])

            if self._group_col in df.columns:
                train_mask = df[self._group_col].isin(train_groups)
                val_mask = df[self._group_col].isin(val_groups)
                test_mask = df[self._group_col].isin(test_groups)
            else:
                train_mask = df.index.isin(train_groups)
                val_mask = df.index.isin(val_groups)
                test_mask = df.index.isin(test_groups)

            folds.append(
                RawDataSplit(
                    train_df=df[train_mask].copy(),
                    val_df=df[val_mask].copy(),
                    test_df=df[test_mask].copy(),
                )
            )

        return folds

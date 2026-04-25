"""Target transform abstractions for the training pipeline.

Provides pluggable target transforms (log, Box-Cox, quantile) that are
fitted on training data only and can be serialized alongside the model
for consistent inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    pass


class TargetTransform(ABC):
    """Base class for target transforms."""

    @abstractmethod
    def fit(self, y: npt.NDArray[Any]) -> TargetTransform:
        """Fit the transform on training target values.

        Args:
            y: Training target array (raw price scale, strictly positive)

        Returns:
            self
        """
        ...

    @abstractmethod
    def transform(
        self,
        y: npt.NDArray[Any],
        df: pd.DataFrame | None = None,
        *,
        is_train: bool | None = None,
    ) -> npt.NDArray[Any]:
        """Transform target values.

        Args:
            y: Target array (raw price scale)
            df: Optional DataFrame for row-level context (used by RelativeResidualTransform)
            is_train: Whether the rows are training rows (used by RelativeResidualTransform)

        Returns:
            Transformed target array
        """
        ...

    @abstractmethod
    def inverse_transform(
        self,
        y: npt.NDArray[Any],
        df: pd.DataFrame | None = None,
    ) -> npt.NDArray[Any]:
        """Inverse-transform predictions back to raw price scale.

        Args:
            y: Predictions in transformed space
            df: Optional DataFrame for row-level context (used by RelativeResidualTransform)

        Returns:
            Predictions in raw price scale
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this transform."""
        ...


class LogTransform(TargetTransform):
    """Standard log1p / expm1 transform (current default)."""

    def fit(self, y: npt.NDArray[Any]) -> LogTransform:  # noqa: ARG002
        """No fitting needed for log transform."""
        return self

    def transform(
        self,
        y: npt.NDArray[Any],
        df: pd.DataFrame | None = None,  # noqa: ARG002
        *,
        is_train: bool | None = None,  # noqa: ARG002
    ) -> npt.NDArray[Any]:
        """Apply log1p transform."""
        return np.log1p(np.asarray(y, dtype=np.float64))

    def inverse_transform(
        self,
        y: npt.NDArray[Any],
        df: pd.DataFrame | None = None,  # noqa: ARG002
    ) -> npt.NDArray[Any]:
        """Apply expm1 inverse transform, clipping negatives."""
        return np.clip(np.expm1(np.asarray(y, dtype=np.float64)), 0, None)

    @property
    def name(self) -> str:
        return "log"


class BoxCoxTransform(TargetTransform):
    """Box-Cox transform with fitted lambda.

    Box-Cox adapts to the skewness of the data rather than assuming
    log is optimal. Requires strictly positive values (guaranteed by
    the $10 minimum price filter).

    Falls back to log1p if the fitted lambda is outside [0, 2] or
    if scipy is unavailable.
    """

    def __init__(self) -> None:
        self._lambda: float | None = None
        self._shift: float = 0.0  # shift to ensure positivity

    def fit(self, y: npt.NDArray[Any]) -> BoxCoxTransform:
        """Fit Box-Cox lambda on training data."""
        from scipy import stats

        y_arr = np.asarray(y, dtype=np.float64)

        # Ensure strictly positive (add small shift if needed)
        min_val = y_arr.min()
        if min_val <= 0:
            self._shift = abs(min_val) + 1.0
        else:
            self._shift = 0.0

        _, fitted_lambda = stats.boxcox(y_arr + self._shift)

        # Guard against extreme lambda values
        if fitted_lambda < -1 or fitted_lambda > 3:
            print(
                f"WARNING: Box-Cox lambda={fitted_lambda:.3f} outside safe range [-1, 3]. "
                f"Clamping to nearest bound."
            )
            fitted_lambda = max(-1.0, min(3.0, fitted_lambda))

        self._lambda = float(fitted_lambda)
        print(f"Box-Cox fitted lambda={self._lambda:.4f} (shift={self._shift:.2f})")
        return self

    def transform(
        self,
        y: npt.NDArray[Any],
        df: pd.DataFrame | None = None,  # noqa: ARG002
        *,
        is_train: bool | None = None,  # noqa: ARG002
    ) -> npt.NDArray[Any]:
        """Apply Box-Cox transform with fitted lambda."""
        if self._lambda is None:
            raise RuntimeError("BoxCoxTransform not fitted. Call fit() first.")

        y_arr = np.asarray(y, dtype=np.float64) + self._shift

        # boxcox1p(x, lmbda) computes ((1+x)^lmbda - 1) / lmbda for lmbda != 0
        # For our case, we use the standard Box-Cox: (y^lmbda - 1) / lmbda
        if abs(self._lambda) < 1e-10:
            return np.log(y_arr)  # type: ignore[no-any-return]
        return (np.power(y_arr, self._lambda) - 1.0) / self._lambda

    def inverse_transform(
        self,
        y: npt.NDArray[Any],
        df: pd.DataFrame | None = None,  # noqa: ARG002
    ) -> npt.NDArray[Any]:
        """Inverse Box-Cox transform."""
        if self._lambda is None:
            raise RuntimeError("BoxCoxTransform not fitted. Call fit() first.")

        from scipy.special import inv_boxcox

        y_arr = np.asarray(y, dtype=np.float64)

        # inv_boxcox expects the result of boxcox (y^lam - 1)/lam
        # It returns y^lam = y_arr * lam + 1, then y = (y^lam)^(1/lam)
        result = inv_boxcox(y_arr, self._lambda)

        # Remove shift and clip negatives
        result = result - self._shift
        return np.clip(np.where(np.isfinite(result), result, 0.0), 0, None)

    @property
    def name(self) -> str:
        lam = f"({self._lambda:.3f})" if self._lambda is not None else ""
        return f"boxcox{lam}"


class SqrtTransform(TargetTransform):
    """Square root transform — gentler compression than log.

    sqrt($100)=10, sqrt($1000)=31.6 — 21.6 units of difference vs log's 2.29.
    Much better separation for high-value tickets.
    """

    def fit(self, y: npt.NDArray[Any]) -> SqrtTransform:  # noqa: ARG002
        """No fitting needed for sqrt transform."""
        return self

    def transform(
        self,
        y: npt.NDArray[Any],
        df: pd.DataFrame | None = None,  # noqa: ARG002
        *,
        is_train: bool | None = None,  # noqa: ARG002
    ) -> npt.NDArray[Any]:
        """Apply sqrt transform."""
        return np.sqrt(np.clip(np.asarray(y, dtype=np.float64), 0, None))

    def inverse_transform(
        self,
        y: npt.NDArray[Any],
        df: pd.DataFrame | None = None,  # noqa: ARG002
    ) -> npt.NDArray[Any]:
        """Apply inverse sqrt (square) transform, clipping negatives."""
        y_arr = np.asarray(y, dtype=np.float64)
        return np.clip(np.square(y_arr), 0, None)

    @property
    def name(self) -> str:
        return "sqrt"


class EventBaseResolver(Protocol):
    """Resolves b_e (event log-price base) for a given row.

    Call-site dispatch: trainer.py invokes `resolve_train` for training rows
    (LOO), `resolve_inference` for val/test rows.  Row identity (row.name)
    is NEVER consulted inside the resolver — the call site is the sole gate.
    """

    def resolve_train(self, row: pd.Series) -> float:
        """Return LOO-adjusted event base for a training row."""
        ...

    def resolve_inference(self, row: pd.Series) -> float:
        """Return full-training smoothed event base for val/test/inference."""
        ...


class EventBaseResolverImpl:
    """Concrete resolver using Bayesian-smoothed LOO event log-price means.

    Uses the same integer-cents guard as EventPricingFeatureExtractor to
    protect against Parquet float round-trip artifacts (see event_pricing.py:141-143).

    Smoothing factor m=20 matches EventPricingFeatureExtractor.SMOOTHING_FACTOR.
    """

    SMOOTHING_FACTOR: int = 20

    def __init__(self) -> None:
        # Per-event sum of log1p(price) and count — for smoothed mean
        self._event_log_sums: dict[str, float] = {}
        self._event_log_counts: dict[str, int] = {}
        # Integer-cents set per event — matches event_pricing.py:141-143 pattern
        self._train_event_prices: dict[str, set[int]] = {}
        self._global_log_mean: float = 0.0
        self._fitted: bool = False

    def fit(
        self,
        train_df: pd.DataFrame,
        global_log_mean: float | None = None,
        coldstart_resolver: Any | None = None,  # reserved for AC5 Group-1 fallback
    ) -> EventBaseResolverImpl:
        """Populate internal state from training data.

        Args:
            train_df: Training split DataFrame (must contain event_id, listing_price)
            global_log_mean: Override for global log-mean (default: computed from train_df)
            coldstart_resolver: Optional callable(row) -> float for unseen-event fallback
        """
        if "event_id" not in train_df.columns or "listing_price" not in train_df.columns:
            raise ValueError(
                "EventBaseResolverImpl.fit() requires 'event_id' and 'listing_price' columns"
            )

        prices = train_df["listing_price"].dropna()
        log_prices = np.log1p(prices.astype(np.float64))
        self._global_log_mean = (
            float(log_prices.mean()) if global_log_mean is None else global_log_mean
        )
        self._coldstart_resolver = coldstart_resolver

        # Build per-event aggregates
        self._event_log_sums = {}
        self._event_log_counts = {}
        self._train_event_prices = {}

        for event_id, group in train_df.groupby("event_id"):
            grp_prices = group["listing_price"].dropna()
            n = len(grp_prices)
            if n == 0:
                continue
            event_id_str = str(event_id)
            log_vals = np.log1p(grp_prices.astype(np.float64))
            self._event_log_sums[event_id_str] = float(log_vals.sum())
            self._event_log_counts[event_id_str] = n
            # Integer-cents guard: same pattern as event_pricing.py:141-143
            self._train_event_prices[event_id_str] = {
                int(round(float(p) * 100)) for p in grp_prices.tolist()
            }

        self._fitted = True
        return self

    def _smoothed_mean(self, event_id: str) -> float:
        """Return Bayesian-smoothed log-mean for an event (full training set)."""
        n = self._event_log_counts.get(event_id, 0)
        s = self._event_log_sums.get(event_id, 0.0)
        m = self.SMOOTHING_FACTOR
        return (s + m * self._global_log_mean) / (n + m)

    def resolve_train(self, row: pd.Series) -> float:
        """LOO-adjusted event base for a training row.

        Subtracts this row's log-price contribution from the smoothed mean so
        the model cannot trivially invert its own target.  Never inspects row.name.

        Falls back to resolve_inference if the event is unknown (shouldn't happen
        during training, but guards against edge cases).
        """
        event_id = str(row.get("event_id", ""))
        price = row.get("listing_price", float("nan"))

        if event_id not in self._event_log_counts or pd.isna(price):
            return self._smoothed_mean(event_id)

        # Integer-cents check: matches event_pricing.py:343-347 pattern
        price_cents = int(round(float(price) * 100))
        if price_cents not in self._train_event_prices.get(event_id, set()):
            # Price not found in training set for this event — use full mean
            return self._smoothed_mean(event_id)

        n = self._event_log_counts[event_id]
        s = self._event_log_sums[event_id]
        m = self.SMOOTHING_FACTOR
        log_price = float(np.log1p(float(price)))

        if n > 1:
            # LOO: subtract this row's contribution
            loo_sum = s - log_price
            loo_n = n - 1
            return (loo_sum + m * self._global_log_mean) / (loo_n + m)
        else:
            # Single-listing event: fall back to global mean
            return self._global_log_mean

    def resolve_inference(self, row: pd.Series) -> float:
        """Full-training smoothed event base for val/test/inference rows.

        Never applies LOO — val/test rows are not in the training aggregate.
        Never inspects row.name.
        """
        event_id = str(row.get("event_id", ""))

        if event_id in self._event_log_counts:
            return self._smoothed_mean(event_id)

        # Unseen event: use coldstart fallback if available, else global mean
        if self._coldstart_resolver is not None:
            return float(self._coldstart_resolver(row))
        return self._global_log_mean

    @property
    def train_event_ids(self) -> frozenset[str]:
        """Set of event IDs seen during fit()."""
        return frozenset(self._event_log_counts.keys())


class RelativeResidualTransform(TargetTransform):
    """Target transform that predicts log(price / event_median) residuals.

    Decomposes log(price) into a deterministic event-level base b_e plus a
    within-event residual y' = log1p(price) - b_e.  The main learner models
    y'; the event base is added back at inference.

    Call-site dispatch (CRITICAL): trainer.py passes is_train=True for training
    rows (LOO branch) and is_train=False for val/test (full-training branch).
    Row identity (row.name) is NEVER consulted — call site is the sole gate.
    """

    def __init__(self, resolver: EventBaseResolverImpl) -> None:
        self._resolver = resolver
        self._fitted: bool = False

    def fit(self, y: npt.NDArray[Any]) -> RelativeResidualTransform:  # noqa: ARG002
        """No fitting needed — resolver is pre-fitted by caller."""
        self._fitted = True
        return self

    def transform(
        self,
        y: npt.NDArray[Any],
        df: pd.DataFrame | None = None,
        *,
        is_train: bool | None = None,
    ) -> npt.NDArray[Any]:
        """Compute y' = log1p(price) - b_e per row.

        Args:
            y: Raw price array (NOT yet log-transformed — same as LogTransform input)
            df: DataFrame rows corresponding to y (required)
            is_train: True for training rows (LOO b_e), False for val/test (full-train b_e)

        Raises:
            ValueError: If df or is_train is not provided
        """
        if df is None or is_train is None:
            raise ValueError(
                "RelativeResidualTransform.transform() requires df and is_train keyword arg"
            )

        y_arr = np.log1p(np.asarray(y, dtype=np.float64))

        b = np.array(
            [
                self._resolver.resolve_train(row)
                if is_train
                else self._resolver.resolve_inference(row)
                for _, row in df.iterrows()
            ],
            dtype=np.float64,
        )
        return y_arr - b

    def inverse_transform(
        self,
        y: npt.NDArray[Any],
        df: pd.DataFrame | None = None,
    ) -> npt.NDArray[Any]:
        """Inverse: add event base back and exponentiate.

        At prediction time (val/test/inference), uses resolve_inference for b_e.
        df is required to look up event_id for each row.

        Args:
            y: Residual predictions in transformed space
            df: DataFrame rows corresponding to y (required for event base lookup)

        Raises:
            ValueError: If df is not provided
        """
        if df is None:
            raise ValueError(
                "RelativeResidualTransform.inverse_transform() requires df for event base lookup"
            )

        b = np.array(
            [self._resolver.resolve_inference(row) for _, row in df.iterrows()],
            dtype=np.float64,
        )
        return np.expm1(np.asarray(y, dtype=np.float64) + b)  # type: ignore[no-any-return]

    @property
    def name(self) -> str:
        return "relative"


def create_target_transform(
    name: str,
    *,
    event_base_resolver: EventBaseResolverImpl | None = None,
) -> TargetTransform:
    """Factory function to create a target transform by name.

    Args:
        name: Transform name — "log", "boxcox", "sqrt", or "relative"
        event_base_resolver: Required when name="relative"; pre-fitted resolver.

    Returns:
        TargetTransform instance

    Raises:
        ValueError: If the name is not recognized or required args are missing
    """
    if name == "relative":
        if event_base_resolver is None:
            raise ValueError(
                "--target-transform relative requires a pre-fitted EventBaseResolverImpl "
                "passed as event_base_resolver="
            )
        return RelativeResidualTransform(event_base_resolver)

    simple_transforms: dict[str, type[TargetTransform]] = {
        "log": LogTransform,
        "boxcox": BoxCoxTransform,
        "sqrt": SqrtTransform,
    }

    if name not in simple_transforms:
        valid = ", ".join(sorted([*simple_transforms.keys(), "relative"]))
        raise ValueError(f"Unknown target transform: {name!r}. Valid: {valid}")

    return simple_transforms[name]()

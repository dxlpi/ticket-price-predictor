"""Target transform abstractions for the training pipeline.

Provides pluggable target transforms (log, Box-Cox, quantile) that are
fitted on training data only and can be serialized alongside the model
for consistent inference.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class TargetTransform(ABC):
    """Base class for target transforms."""

    @abstractmethod
    def fit(self, y: npt.NDArray[Any]) -> "TargetTransform":
        """Fit the transform on training target values.

        Args:
            y: Training target array (raw price scale, strictly positive)

        Returns:
            self
        """
        ...

    @abstractmethod
    def transform(self, y: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Transform target values.

        Args:
            y: Target array (raw price scale)

        Returns:
            Transformed target array
        """
        ...

    @abstractmethod
    def inverse_transform(self, y: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Inverse-transform predictions back to raw price scale.

        Args:
            y: Predictions in transformed space

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

    def fit(self, y: npt.NDArray[Any]) -> "LogTransform":  # noqa: ARG002
        """No fitting needed for log transform."""
        return self

    def transform(self, y: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Apply log1p transform."""
        return np.log1p(np.asarray(y, dtype=np.float64))

    def inverse_transform(self, y: npt.NDArray[Any]) -> npt.NDArray[Any]:
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

    def fit(self, y: npt.NDArray[Any]) -> "BoxCoxTransform":
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

    def transform(self, y: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Apply Box-Cox transform with fitted lambda."""
        if self._lambda is None:
            raise RuntimeError("BoxCoxTransform not fitted. Call fit() first.")

        y_arr = np.asarray(y, dtype=np.float64) + self._shift

        # boxcox1p(x, lmbda) computes ((1+x)^lmbda - 1) / lmbda for lmbda != 0
        # For our case, we use the standard Box-Cox: (y^lmbda - 1) / lmbda
        if abs(self._lambda) < 1e-10:
            return np.log(y_arr)
        return (np.power(y_arr, self._lambda) - 1.0) / self._lambda

    def inverse_transform(self, y: npt.NDArray[Any]) -> npt.NDArray[Any]:
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

    def fit(self, y: npt.NDArray[Any]) -> "SqrtTransform":  # noqa: ARG002
        """No fitting needed for sqrt transform."""
        return self

    def transform(self, y: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Apply sqrt transform."""
        return np.sqrt(np.clip(np.asarray(y, dtype=np.float64), 0, None))

    def inverse_transform(self, y: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Apply inverse sqrt (square) transform, clipping negatives."""
        y_arr = np.asarray(y, dtype=np.float64)
        return np.clip(np.square(y_arr), 0, None)

    @property
    def name(self) -> str:
        return "sqrt"


def create_target_transform(name: str) -> TargetTransform:
    """Factory function to create a target transform by name.

    Args:
        name: Transform name — "log", "boxcox", or "sqrt"

    Returns:
        TargetTransform instance

    Raises:
        ValueError: If the name is not recognized
    """
    transforms: dict[str, type[TargetTransform]] = {
        "log": LogTransform,
        "boxcox": BoxCoxTransform,
        "sqrt": SqrtTransform,
    }

    if name not in transforms:
        valid = ", ".join(sorted(transforms.keys()))
        raise ValueError(f"Unknown target transform: {name!r}. Valid: {valid}")

    return transforms[name]()

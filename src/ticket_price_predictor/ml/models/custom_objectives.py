"""Custom loss functions for LightGBM."""

from typing import Any

import numpy as np
import numpy.typing as npt


def asymmetric_huber_objective(
    y_pred: npt.NDArray[Any],
    dtrain: Any,
    delta: float = 0.5,
    under_penalty: float = 1.5,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Asymmetric Huber loss: penalize under-prediction more heavily.

    Under-prediction (model predicts too low) is worse for ticket pricing
    because it leads to missed revenue. This objective applies a higher
    penalty multiplier when the true price exceeds the prediction.

    Operates in log-space (since targets are log1p-transformed).

    Args:
        y_pred: Predicted values (log-space)
        dtrain: LightGBM Dataset with labels
        delta: Huber transition point (log-space units).
            Controls where loss transitions from L2 (quadratic) to L1 (linear).
        under_penalty: Multiplier for under-prediction errors (residual > 0).
            1.0 = symmetric, 1.5 = 50% more penalty for under-prediction.

    Returns:
        Tuple of (gradient, hessian) arrays
    """
    y_true = dtrain.get_label()
    residual = y_true - y_pred  # positive when under-predicting

    # Asymmetry: under-prediction (residual > 0) penalized more
    alpha = np.where(residual > 0, under_penalty, 1.0)

    abs_r = np.abs(residual)

    # Gradient: derivative of loss w.r.t. prediction
    grad = np.where(
        abs_r <= delta,
        -alpha * residual,  # L2 region
        -alpha * delta * np.sign(residual),  # L1 region
    )

    # Hessian: second derivative (must be positive for convergence)
    hess = np.where(abs_r <= delta, alpha, np.float64(0.0)) + 1e-6

    return grad.astype(np.float64), hess.astype(np.float64)


def asymmetric_huber_metric(
    y_pred: npt.NDArray[Any],
    dtrain: Any,
) -> tuple[str, float, bool]:
    """Evaluation metric for asymmetric Huber (MAE in log-space).

    LightGBM requires an eval metric when using custom objectives.
    We use standard MAE since the asymmetric loss is for training only.

    Args:
        y_pred: Predicted values
        dtrain: LightGBM Dataset with labels

    Returns:
        Tuple of (metric_name, metric_value, is_higher_better)
    """
    y_true = dtrain.get_label()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return "asymmetric_mae", mae, False


def make_asymmetric_huber(
    delta: float = 0.5,
    under_penalty: float = 1.5,
) -> tuple[Any, Any]:
    """Create asymmetric Huber objective and metric pair.

    Returns closures with the specified delta and under_penalty baked in,
    ready to pass to LightGBM's fobj and feval parameters.

    Args:
        delta: Huber transition point
        under_penalty: Multiplier for under-prediction errors

    Returns:
        Tuple of (objective_fn, metric_fn)
    """

    def objective(
        y_pred: npt.NDArray[Any], dtrain: Any
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        return asymmetric_huber_objective(y_pred, dtrain, delta=delta, under_penalty=under_penalty)

    def metric(y_pred: npt.NDArray[Any], dtrain: Any) -> tuple[str, float, bool]:
        return asymmetric_huber_metric(y_pred, dtrain)

    return objective, metric

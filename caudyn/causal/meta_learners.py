from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from xgboost import XGBRegressor

from causalml.inference.meta import BaseRRegressor, BaseTRegressor, BaseXRegressor

from .data_utils import CONTROL_LABEL, MODEL_KEY, TREATMENT_LABEL


LearnerMap = dict[str, Any]


def build_base_learner(random_state: int = 42) -> XGBRegressor:
    """Create the control-side base learner used by all meta-models."""
    return XGBRegressor(
        n_estimators=300,
        max_depth=5,
        objective="reg:squarederror",
        random_state=random_state,
    )


def build_treatment_learner(random_state: int = 42) -> XGBRegressor:
    """Create the treatment-side learner with a shallower tree depth."""
    return XGBRegressor(
        n_estimators=300,
        max_depth=3,
        objective="reg:squarederror",
        random_state=random_state,
    )


def initialize_meta_learners(
    base_learner: XGBRegressor,
    treatment_learner: XGBRegressor,
    control_name: int = CONTROL_LABEL,
) -> LearnerMap:
    """Initialize T/X/R learners with explicit control and treatment learners."""
    return {
        "T-Learner": BaseTRegressor(
            control_learner=clone(base_learner),
            treatment_learner=clone(treatment_learner),
            control_name=control_name,
        ),
        "X-Learner": BaseXRegressor(
            control_outcome_learner=clone(base_learner),
            treatment_outcome_learner=clone(treatment_learner),
            control_effect_learner=clone(base_learner),
            treatment_effect_learner=clone(treatment_learner),
            control_name=control_name,
        ),
        "R-Learner": BaseRRegressor(
            outcome_learner=clone(base_learner),
            effect_learner=clone(treatment_learner),
            control_name=control_name,
        ),
    }


def fit_meta_learners(
    learners: Mapping[str, Any],
    x: NDArray[np.float64],
    treatment: NDArray[np.str_],
    y: NDArray[np.float64],
) -> LearnerMap:
    """Fit each meta-learner on the same feature, treatment, and outcome arrays."""
    fitted: LearnerMap = dict(learners)
    for learner in fitted.values():
        learner.fit(X=x, treatment=treatment, y=y)
    return fitted


def _treatment_index(learner: Any, treatment_label: int = TREATMENT_LABEL) -> int:
    classes = getattr(learner, "_classes", {})
    if isinstance(classes, dict) and treatment_label in classes:
        return int(classes[treatment_label])
    return 0


def _vector_for_treatment(
    values: Any,
    learner: Any,
    treatment_label: int = TREATMENT_LABEL,
) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.array([float(arr)], dtype=float)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr[:, 0].astype(float)
        idx = _treatment_index(learner, treatment_label=treatment_label)
        return arr[:, idx].astype(float)
    return arr.reshape(-1).astype(float)


def _scalar_for_treatment(
    values: Any,
    learner: Any,
    treatment_label: int = TREATMENT_LABEL,
) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        if arr.size == 1:
            return float(arr[0])
        idx = _treatment_index(learner, treatment_label=treatment_label)
        return float(arr[idx])
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            idx = _treatment_index(learner, treatment_label=treatment_label)
            return float(arr[0, idx])
        return _scalar_for_treatment(arr, learner, treatment_label=treatment_label)
    return float(arr.reshape(-1)[0])


def estimate_ate_interval(
    learner: Any,
    x: NDArray[np.float64],
    treatment: NDArray[np.str_],
    y: NDArray[np.float64],
    treatment_label: int = TREATMENT_LABEL,
    bootstrap_ci: bool = False,
    n_bootstraps: int = 100,
    bootstrap_size: int = 5_000,
) -> tuple[float, float, float]:
    """
    Estimate ATE and confidence bounds for one treatment arm.

    The function first attempts pretrain mode (using already-fitted learners).
    If pretrain mode is incompatible for a learner/dataset shape, it falls back
    to fit-on-dataset mode.
    """
    kwargs: dict[str, Any] = {
        "X": x,
        "treatment": treatment,
        "y": y,
        "bootstrap_ci": bootstrap_ci,
    }
    if bootstrap_ci:
        kwargs["n_bootstraps"] = n_bootstraps
        kwargs["bootstrap_size"] = min(int(bootstrap_size), len(y))

    try:
        ate, ate_lb, ate_ub = learner.estimate_ate(pretrain=True, **kwargs)
    except Exception:
        ate, ate_lb, ate_ub = learner.estimate_ate(pretrain=False, **kwargs)

    return (
        _scalar_for_treatment(ate, learner, treatment_label=treatment_label),
        _scalar_for_treatment(ate_lb, learner, treatment_label=treatment_label),
        _scalar_for_treatment(ate_ub, learner, treatment_label=treatment_label),
    )


def add_meta_predictions(
    df: pd.DataFrame,
    learners: Mapping[str, Any],
    x: NDArray[np.float64],
    model_key: Mapping[str, str] = MODEL_KEY,
    treatment_label: int = TREATMENT_LABEL,
) -> pd.DataFrame:
    """Append CATE point predictions for each model."""
    out = df.copy()
    for model_name, learner in learners.items():
        key = model_key[model_name]
        estimate = _vector_for_treatment(learner.predict(x), learner, treatment_label=treatment_label)

        # Backward-compatible score used by decile and Qini ranking.
        out[f"pred_cate_{key}"] = estimate

        # Explicit alias for reporting tables.
        out[f"pred_cate_estimate_{key}"] = estimate

    return out

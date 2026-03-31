from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone


AVERAGE_RIDE_VALUE = 25.0
FACE_VALUE_BY_TREATMENT = {
    0: 0.0,
    1: 0.10 * AVERAGE_RIDE_VALUE,
    2: 0.20 * AVERAGE_RIDE_VALUE,
}
OPTIMIZATION_BUDGET = 5_000.0
MU_CLIP_EPS = 1e-6
OPTIMIZATION_CANDIDATE_TREATMENTS: tuple[int, ...] = (1, 2)
DEFAULT_BASE_LAMBDA = 0.1936


def _format_percent(value: object) -> str:
    if value is None or value is pd.NA:
        return "nan"
    if isinstance(value, (float, np.floating)) and np.isnan(float(value)):
        return "nan"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.2%}"
    return str(value)


def _format_table(df: pd.DataFrame, percent_columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for column in percent_columns:
        if column in out.columns:
            out[column] = out[column].map(_format_percent)
    return out


def _log_section(logger: logging.Logger, title: str) -> None:
    logger.info("\n%s\n%s", title, "-" * len(title))


def _format_int(value: int) -> str:
    return f"{value:,d}"


def _fit_r_learner_mu_model(
    *,
    r_learner: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """Fit and return a deployable mu(x) model from an R-Learner template."""
    model_mu_template = getattr(r_learner, "model_mu", None)
    if model_mu_template is None:
        raise ValueError("R-Learner does not expose an outcome learner template (`model_mu`).")

    model_mu_fitted = clone(model_mu_template)
    model_mu_fitted.fit(x_train, y_train)
    return model_mu_fitted


def _extract_r_learner_mu0(
    *,
    mu_model: Any,
    x_score: np.ndarray,
) -> np.ndarray:
    """Extract baseline conversion mu(x) from a fitted mu-model."""
    mu_0 = np.asarray(mu_model.predict(x_score), dtype=float).reshape(-1)
    return np.clip(mu_0, MU_CLIP_EPS, 1.0 - MU_CLIP_EPS)


def _extract_multi_treatment_cates(
    *,
    r_learner: Any,
    x_score: np.ndarray,
    candidate_treatments: Sequence[int],
) -> dict[int, np.ndarray]:
    """Extract treatment-specific CATE vectors from a fitted multi-treatment R-Learner."""
    cates = np.asarray(r_learner.predict(x_score), dtype=float)
    if cates.ndim == 1:
        cates = cates.reshape(-1, 1)

    classes = getattr(r_learner, "_classes", None)
    if not isinstance(classes, Mapping):
        raise ValueError("R-Learner is missing treatment-class metadata (`_classes`).")

    tau_by_treatment: dict[int, np.ndarray] = {}
    available_treatments = {int(k) for k in classes.keys()} if classes else set()
    for treatment in candidate_treatments:
        treatment_int = int(treatment)
        if treatment_int not in classes:
            available = ", ".join(str(k) for k in sorted(available_treatments))
            raise ValueError(
                f"Requested treatment {treatment_int} not available in fitted R-Learner classes: {available}."
            )

        col_idx = int(classes[treatment_int])
        if col_idx >= cates.shape[1]:
            raise ValueError(
                f"Treatment {treatment_int} mapped to invalid column index {col_idx} for CATE matrix "
                f"with shape {cates.shape}."
            )

        tau_by_treatment[treatment_int] = cates[:, col_idx].astype(float, copy=False)

    return tau_by_treatment


def _prepare_optimization_dataframe(
    *,
    df_rct: pd.DataFrame,
    x_rct: np.ndarray,
    r_learner: Any,
    mu_model: Any,
    candidate_treatments: Sequence[int] = OPTIMIZATION_CANDIDATE_TREATMENTS,
) -> pd.DataFrame:
    """Build optimization-ready data from holdout predictions."""
    mu_0 = _extract_r_learner_mu0(
        mu_model=mu_model,
        x_score=x_rct,
    )
    tau_by_treatment = _extract_multi_treatment_cates(
        r_learner=r_learner,
        x_score=x_rct,
        candidate_treatments=candidate_treatments,
    )

    base = df_rct.copy()
    base["user_id"] = base.index.astype(int)
    base["mu_0"] = mu_0

    candidate_frames: list[pd.DataFrame] = []
    for treatment in candidate_treatments:
        treatment_int = int(treatment)
        if treatment_int not in FACE_VALUE_BY_TREATMENT:
            raise ValueError(f"Treatment {treatment_int} is missing from FACE_VALUE_BY_TREATMENT mapping.")

        candidate = base.copy()
        candidate["treatment"] = treatment_int
        candidate["face_value"] = float(FACE_VALUE_BY_TREATMENT[treatment_int])
        candidate["tau_hat"] = tau_by_treatment[treatment_int]
        candidate_frames.append(candidate)

    if not candidate_frames:
        raise ValueError("No optimization candidate treatments were provided.")

    out = pd.concat(candidate_frames, ignore_index=True)
    return out


def _build_online_inference_fn(
    *,
    features: Sequence[str],
    r_learner: Any,
    mu_model: Any,
    candidate_treatments: Sequence[int] = OPTIMIZATION_CANDIDATE_TREATMENTS,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Create an online inference callable backed by fitted offline models."""
    feature_cols = [str(col) for col in features]
    treatment_set = {int(t) for t in candidate_treatments}
    required_treatments = {1, 2}
    if not required_treatments.issubset(treatment_set):
        raise ValueError(
            "Online simulation requires treatment IDs 1 and 2 in candidate_treatments. "
            f"Received: {sorted(treatment_set)}"
        )

    def _inference(riders_df: pd.DataFrame) -> pd.DataFrame:
        missing_features = [col for col in feature_cols if col not in riders_df.columns]
        if missing_features:
            raise ValueError(
                "Online inference input is missing required feature columns: "
                + ", ".join(missing_features)
            )

        scored_df = riders_df.copy()
        x_score = scored_df[feature_cols].to_numpy(dtype=float, copy=True)

        mu_0 = np.asarray(mu_model.predict(x_score), dtype=float).reshape(-1)
        scored_df["mu_0"] = np.clip(mu_0, MU_CLIP_EPS, 1.0 - MU_CLIP_EPS)

        tau_by_treatment = _extract_multi_treatment_cates(
            r_learner=r_learner,
            x_score=x_score,
            candidate_treatments=tuple(sorted(treatment_set)),
        )
        scored_df["tau_hat_10"] = tau_by_treatment[1]
        scored_df["tau_hat_20"] = tau_by_treatment[2]

        return scored_df

    return _inference

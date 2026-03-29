from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import auc

from .data_utils import CONTROL_VALUE, MODEL_KEY, TREATMENT_LABEL, TREATMENT_VALUE
from .meta_learners import estimate_ate_interval


def summarize_naive_vs_true(
    df: pd.DataFrame,
    treat_value: int = TREATMENT_VALUE,
    control_value: int = CONTROL_VALUE,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare naive observational ATE against oracle simulator ATE."""
    naive_conv_treat = df.loc[df["treatment"] == treat_value, "converted"].mean()
    naive_conv_control = df.loc[df["treatment"] == control_value, "converted"].mean()
    naive_ate = float(naive_conv_treat - naive_conv_control)

    true_conv_treat = float(df["true_conversion_20pct"].mean())
    true_conv_control = float(df["true_conversion_0pct"].mean())
    true_ate = float(df["true_cate_20pct"].mean())

    summary = pd.DataFrame(
        {
            "Metric": [
                "Naive conversion @ 20%",
                "Naive conversion @ 0%",
                "Naive ATE (20% - 0%)",
                "True conversion @ 20%",
                "True conversion @ 0%",
                "True ATE (physics oracle)",
                "Bias (Naive - True)",
            ],
            "Value": [
                naive_conv_treat,
                naive_conv_control,
                naive_ate,
                true_conv_treat,
                true_conv_control,
                true_ate,
                naive_ate - true_ate,
            ],
        }
    )

    metrics = {
        "naive_ate": naive_ate,
        "true_ate": true_ate,
    }
    return summary, metrics


def summarize_model_predictions(
    df: pd.DataFrame,
    learners: Mapping[str, Any],
    x: NDArray,
    treatment: NDArray,
    y: NDArray,
    model_key: Mapping[str, str] = MODEL_KEY,
    true_col: str = "true_cate_20pct",
    label: str = "dataset",
    treatment_label: int = TREATMENT_LABEL,
    bootstrap_ci: bool = False,
    n_bootstraps: int = 100,
    bootstrap_size: int = 5_000,
) -> pd.DataFrame:
    """Build per-model summary table with estimate/lower/upper and oracle mean CATE."""
    rows: list[dict[str, float | str]] = []
    for model_name in model_key:
        learner = learners[model_name]
        est, lb, ub = estimate_ate_interval(
            learner=learner,
            x=x,
            treatment=treatment,
            y=y,
            treatment_label=treatment_label,
            bootstrap_ci=bootstrap_ci,
            n_bootstraps=n_bootstraps,
            bootstrap_size=bootstrap_size,
        )

        rows.append(
            {
                "Model": model_name,
                f"Mean CATE Estimate ({label})": est,
                f"Mean CATE Lower Bound ({label})": lb,
                f"Mean CATE Upper Bound ({label})": ub,
                f"Mean True CATE ({label})": float(df[true_col].mean()),
            }
        )

    return pd.DataFrame(rows)


def decile_validation_table(
    df: pd.DataFrame,
    score_col: str,
    treatment_value: int = TREATMENT_VALUE,
    control_value: int = CONTROL_VALUE,
    n_deciles: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ranked users into deciles and compare predicted vs realized lift."""
    if df.empty:
        raise ValueError("Input dataframe is empty; decile validation cannot be computed.")

    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True).copy()
    n_bins = min(n_deciles, len(ranked))
    if n_bins < 2:
        raise ValueError("At least two rows are required for decile validation.")

    ranked["decile"] = pd.qcut(ranked.index, q=n_bins, labels=np.arange(1, n_bins + 1)) # type: ignore

    rows: list[dict[str, float | int]] = []
    for decile in range(1, n_bins + 1):
        sub = ranked[ranked["decile"] == decile]
        conv_treated = sub.loc[sub["treatment"] == treatment_value, "converted"].mean()
        conv_control = sub.loc[sub["treatment"] == control_value, "converted"].mean()

        rows.append(
            {
                "Decile": decile,
                "Predicted_Lift": float(sub[score_col].mean()),
                "Actual_Lift": float(conv_treated - conv_control),
                "n": int(len(sub)),
            }
        )

    return ranked, pd.DataFrame(rows)


def build_all_deciles(
    df: pd.DataFrame,
    model_score_cols: Mapping[str, str],
    n_deciles: int = 10,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Compute ranked frame and decile table for every model score column."""
    ranked_frames: dict[str, pd.DataFrame] = {}
    decile_tables: dict[str, pd.DataFrame] = {}

    for model_name, score_col in model_score_cols.items():
        ranked_df, decile_df = decile_validation_table(df, score_col=score_col, n_deciles=n_deciles)
        ranked_frames[model_name] = ranked_df
        decile_tables[model_name] = decile_df

    return ranked_frames, decile_tables


def summarize_decile_ranking(
    df: pd.DataFrame,
    decile_tables: Mapping[str, pd.DataFrame],
    model_score_cols: Mapping[str, str],
    true_col: str = "true_cate_20pct",
) -> pd.DataFrame:
    """Summarize top-vs-bottom decile spread and monotonicity against true CATE."""
    rows: list[dict[str, float | str]] = []

    for model_name, score_col in model_score_cols.items():
        decile_df = decile_tables[model_name]
        top_decile = int(decile_df["Decile"].min())
        bottom_decile = int(decile_df["Decile"].max())

        top_actual = float(decile_df.loc[decile_df["Decile"] == top_decile, "Actual_Lift"].iloc[0])
        bottom_actual = float(decile_df.loc[decile_df["Decile"] == bottom_decile, "Actual_Lift"].iloc[0])
        spread = top_actual - bottom_actual
        spearman_vs_true = float(pd.Series(df[score_col]).corr(df[true_col], method="spearman"))

        rows.append(
            {
                "Model": model_name,
                "Top Decile Actual Lift": top_actual,
                "Bottom Decile Actual Lift": bottom_actual,
                "Decile Lift Spread (Top-Bottom)": spread,
                "Spearman Corr(pred, true CATE)": spearman_vs_true,
            }
        )

    return pd.DataFrame(rows).sort_values("Decile Lift Spread (Top-Bottom)", ascending=False)


def calculate_qini_curve(
    df: pd.DataFrame,
    treatment_value: int = TREATMENT_VALUE,
) -> NDArray[np.float64]:
    """Compute cumulative incremental gain curve used in Qini analysis."""
    w = (df["treatment"] == treatment_value).astype(int).to_numpy()
    y = df["converted"].to_numpy(dtype=float)

    cum_treated = np.cumsum(w)
    cum_control = np.cumsum(1 - w)
    cum_y_treated = np.cumsum(y * w)
    cum_y_control = np.cumsum(y * (1 - w))

    eps = 1e-10
    incremental_gain = cum_y_treated - cum_y_control * (cum_treated / (cum_control + eps))
    return np.insert(incremental_gain, 0, 0.0)


def qini_analysis(
    df: pd.DataFrame,
    model_score_cols: Mapping[str, str],
    true_col: str = "true_cate_20pct",
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    dict[str, NDArray[np.float64]],
    pd.DataFrame,
]:
    """Compute random/perfect/model Qini curves and normalized Qini scores."""
    if df.empty:
        raise ValueError("Input dataframe is empty; Qini analysis cannot be computed.")

    x_axis = np.arange(len(df) + 1, dtype=float) / len(df)

    qini_endpoint = float(calculate_qini_curve(df)[-1])
    qini_random = x_axis * qini_endpoint
    area_random = float(auc(x_axis, qini_random))

    df_perfect = df.sort_values(true_col, ascending=False).reset_index(drop=True)
    qini_perfect = calculate_qini_curve(df_perfect)
    area_perfect = float(auc(x_axis, qini_perfect))

    qini_curves: dict[str, NDArray[np.float64]] = {}
    qini_rows: list[dict[str, float | str]] = []

    for model_name, score_col in model_score_cols.items():
        ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        qini_curve = calculate_qini_curve(ranked)
        qini_curves[model_name] = qini_curve

        area_model = float(auc(x_axis, qini_curve))
        denom = area_perfect - area_random
        qini_norm = np.nan if abs(denom) < 1e-12 else (area_model - area_random) / denom

        qini_rows.append(
            {
                "Model": model_name,
                "AUC_Model": area_model,
                "AUC_Random": area_random,
                "AUC_Perfect": area_perfect,
                "Qini_Normalized": qini_norm,
            }
        )

    qini_results = (
        pd.DataFrame(qini_rows)
        .sort_values("Qini_Normalized", ascending=False)
        .reset_index(drop=True)
    )

    return x_axis, qini_random, qini_perfect, qini_curves, qini_results

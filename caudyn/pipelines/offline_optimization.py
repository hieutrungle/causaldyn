from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from caudyn.value_optimization.lp_solver import ValueOptimizer
from caudyn.value_optimization.unit_selector import UnitSelector

from .common import (
    OPTIMIZATION_CANDIDATE_TREATMENTS,
    _format_int,
    _log_section,
    _prepare_optimization_dataframe,
)
from .contracts import CausalPipelineResult, ExperimentConfigLike, OptimizationPipelineResult


def _log_optimization_inputs(
    logger: logging.Logger,
    *,
    df_opt: pd.DataFrame,
    df_selected: pd.DataFrame,
    best_model: str,
    user_count: int,
) -> None:
    mu_0_by_user = df_opt.drop_duplicates(subset=["user_id"])["mu_0"]
    logger.info(
        "R-Learner mu(x) extracted for optimization (min=%.4f, mean=%.4f, max=%.4f).",
        float(mu_0_by_user.min()),
        float(mu_0_by_user.mean()),
        float(mu_0_by_user.max()),
    )
    logger.info(
        "Optimization CATE source: multi-treatment R-Learner with direct lift estimates vs control for treatments %s.",
        ", ".join(str(t) for t in OPTIMIZATION_CANDIDATE_TREATMENTS),
    )
    logger.info(
        "Optimization input rows prepared: %s users -> %s candidate rows (best eval model=%s).",
        _format_int(user_count),
        _format_int(len(df_opt)),
        best_model,
    )
    logger.info(
        "Unit selection pruning: %s -> %s rows (dropped %s).",
        _format_int(len(df_opt)),
        _format_int(len(df_selected)),
        _format_int(len(df_opt) - len(df_selected)),
    )


def _build_allocation_summary(df_allocated: pd.DataFrame) -> pd.DataFrame:
    base_summary = pd.DataFrame(
        {
            "treatment": [0, 1, 2],
            "allocated_fraction": [0.0, 0.0, 0.0],
        }
    )
    allocation_by_treatment = (
        df_allocated.groupby("treatment")["optimal_fraction"]
        .sum()
        .reset_index(name="allocated_fraction")
    )
    return (
        base_summary.drop(columns=["allocated_fraction"])
        .merge(allocation_by_treatment, on="treatment", how="left")
        .fillna({"allocated_fraction": 0.0})
    )


def _log_optimization_completion(
    logger: logging.Logger,
    *,
    budget: float,
    lambda_val: float,
    allocated_10: float,
    allocated_20: float,
    allocated_none: float,
) -> None:
    lambda_print = float(lambda_val) if not np.isnan(lambda_val) else float("nan")
    logger.info(
        "\n"
        "=========================================================\n"
        "       OFFLINE VALUE OPTIMIZATION COMPLETE\n"
        "=========================================================\n"
        "Total Budget: $%.2f\n"
        "Global Shadow Price (Lambda): %.4f\n"
        "-> This means right at the budget limit, spending $1 buys %.4f incremental rides!\n"
        "\n"
        "Allocation Summary:\n"
        "- Users receiving 10%% Discount: %.2f\n"
        "- Users receiving 20%% Discount: %.2f\n"
        "- Users receiving NO Discount: %.2f\n"
        "=========================================================",
        float(budget),
        lambda_print,
        lambda_print,
        allocated_10,
        allocated_20,
        allocated_none,
    )


def run_offline_optimization_pipeline(
    config: ExperimentConfigLike,
    causal_data: CausalPipelineResult,
    logger: logging.Logger | None = None,
) -> OptimizationPipelineResult:
    """Run Step 6: offline convex optimization with budget constraints."""
    log = logger or logging.getLogger(__name__)
    _log_section(log, "Step 6 - Offline Value Optimization")

    df_opt = _prepare_optimization_dataframe(
        df_rct=causal_data.df_rct,
        x_rct=causal_data.x_rct_full,
        r_learner=causal_data.r_learner_multi,
        mu_model=causal_data.mu_model_fitted,
        candidate_treatments=OPTIMIZATION_CANDIDATE_TREATMENTS,
    )

    selector = UnitSelector(
        user_id_col="user_id",
        treatment_col="treatment",
        base_prob_col="mu_0",
        cate_col="tau_hat",
        cost_col="face_value",
    )
    df_selected = selector.fit_transform(df_opt)

    lambda_val = np.nan
    df_allocated = df_selected.copy()
    allocation_summary = pd.DataFrame(
        {
            "treatment": [0, 1, 2],
            "allocated_fraction": [0.0, 0.0, 0.0],
        }
    )

    if df_selected.empty:
        log.warning("Unit selector returned no positive-lift candidates. Skipping optimization solve.")
    else:
        optimizer = ValueOptimizer(
            budget=config.simulation_budget,
            user_col="user_id",
            cate_col="tau_hat",
            cost_col="expected_cost",
        )
        df_allocated, lambda_val = optimizer.optimize(df_selected)
        allocation_summary = _build_allocation_summary(df_allocated)

    _log_optimization_inputs(
        log,
        df_opt=df_opt,
        df_selected=df_selected,
        best_model=causal_data.best_model,
        user_count=len(causal_data.df_rct),
    )

    allocated_10 = float(
        allocation_summary.loc[allocation_summary["treatment"] == 1, "allocated_fraction"].iloc[0]
    )
    allocated_20 = float(
        allocation_summary.loc[allocation_summary["treatment"] == 2, "allocated_fraction"].iloc[0]
    )
    no_discount = max(float(df_opt["user_id"].nunique()) - (allocated_10 + allocated_20), 0.0)

    _log_optimization_completion(
        log,
        budget=float(config.simulation_budget),
        lambda_val=float(lambda_val) if not np.isnan(lambda_val) else np.nan,
        allocated_10=allocated_10,
        allocated_20=allocated_20,
        allocated_none=no_discount,
    )

    metrics: dict[str, pd.DataFrame] = {
        "df_opt": df_opt,
        "df_selected": df_selected,
        "df_allocated": df_allocated,
        "allocation_summary": allocation_summary,
        "optimization_metrics": pd.DataFrame(
            [
                {
                    "budget": config.simulation_budget,
                    "lambda": float(lambda_val) if not np.isnan(lambda_val) else np.nan,
                    "allocated_10pct": allocated_10,
                    "allocated_20pct": allocated_20,
                    "allocated_none": no_discount,
                }
            ]
        ),
    }

    return OptimizationPipelineResult(
        lambda_val=float(lambda_val) if not np.isnan(lambda_val) else np.nan,
        metrics=metrics,
    )

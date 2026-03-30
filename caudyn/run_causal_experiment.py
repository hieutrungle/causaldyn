from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from caudyn.causal.data_utils import (
    DEFAULT_FEATURES,
    MODEL_KEY,
    add_oracle_counterfactuals,
    generate_randomized_holdout,
    prepare_binary_meta_dataset,
)
from caudyn.causal.meta_learners import (
    add_meta_predictions,
    build_base_learner,
    build_treatment_learner,
    fit_meta_learners,
    initialize_meta_learners,
)
from caudyn.causal.metrics import (
    build_all_deciles,
    qini_analysis,
    summarize_decile_ranking,
    summarize_model_predictions,
    summarize_naive_vs_true,
)
from caudyn.causal.plotting import plot_decile_comparison, plot_qini_comparison
from caudyn.environment import UberMarketplaceEnvironment
from caudyn.value_optimization.lp_solver import ValueOptimizer
from caudyn.value_optimization.unit_selector import UnitSelector

LOGGER = logging.getLogger(__name__)

AVERAGE_RIDE_VALUE = 25.0
FACE_VALUE_BY_TREATMENT = {
    0: 0.0,
    1: 0.10 * AVERAGE_RIDE_VALUE,
    2: 0.20 * AVERAGE_RIDE_VALUE,
}
OPTIMIZATION_BUDGET = 5_000.0
MU_CLIP_EPS = 1e-6
OPTIMIZATION_CANDIDATE_TREATMENTS: tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the end-to-end causalml experiment."""

    features: tuple[str, ...] = DEFAULT_FEATURES
    hist_rows: int = 200_000
    rct_rows: int = 20_000
    hist_seed: int = 42
    rct_seed: int = 999
    learner_seed: int = 42
    bootstrap_ci: bool = False
    n_bootstraps: int = 100
    bootstrap_size: int = 5_000
    show_plots: bool = True


def _format_percent(value: object) -> str:
    if pd.isna(value): # type: ignore
        return "nan"
    return f"{float(value):.2%}" # type: ignore


def _format_table(df: pd.DataFrame, percent_columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for column in percent_columns:
        if column in out.columns:
            out[column] = out[column].map(_format_percent)
    return out


def _log_section(title: str) -> None:
    LOGGER.info("\n%s\n%s", title, "-" * len(title))


def _format_int(value: int) -> str:
    return f"{value:,d}"


def _prepare_optimization_dataframe(
    *,
    df_rct: pd.DataFrame,
    x_rct: np.ndarray,
    r_learner: Any,
    x_train_mu: np.ndarray,
    y_train_mu: np.ndarray,
    candidate_treatments: Sequence[int] = OPTIMIZATION_CANDIDATE_TREATMENTS,
) -> pd.DataFrame:
    """Build optimization-ready data from holdout predictions.

    This function maps causal outputs into financial optimization inputs expected by
    UnitSelector and ValueOptimizer, while expanding each rider into multiple
    candidate promo tiers (e.g., 10% and 20%).

    Notes:
    - CATE values are predicted directly from a fitted multi-treatment R-Learner.
    - For each treatment t in `candidate_treatments`, `tau_hat` is the predicted
      lift of treatment t vs control treatment 0.
    - Baseline conversion `mu_0` is extracted from the R-Learner outcome model.
    """
    mu_0 = _extract_r_learner_mu0(
        r_learner=r_learner,
        x_train=x_train_mu,
        y_train=y_train_mu,
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


def _extract_r_learner_mu0(
    *,
    r_learner: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_score: np.ndarray,
) -> np.ndarray:
    """Extract baseline conversion mu(x) from the R-learner outcome model.

    CausalML's R-learner stores an outcome learner template (`model_mu`) and uses
    cross-validation internally for nuisance estimates. To recover a deployable
    baseline estimate, we fit a clone of that same learner on training data and
    score the holdout features.
    """
    model_mu_template = getattr(r_learner, "model_mu", None)
    if model_mu_template is None:
        raise ValueError("R-Learner does not expose an outcome learner template (`model_mu`).")

    model_mu_fitted = clone(model_mu_template)
    model_mu_fitted.fit(x_train, y_train)
    mu_0 = np.asarray(model_mu_fitted.predict(x_score), dtype=float).reshape(-1)
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


def _fit_multitreatment_r_learner(
    *,
    df_hist: pd.DataFrame,
    features: Sequence[str],
    random_state: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Fit a dedicated multi-treatment R-Learner on 0/10/20 historical data."""
    x_hist = df_hist[list(features)].to_numpy(dtype=float, copy=True)
    y_hist = df_hist["converted"].astype(float).to_numpy(copy=True)
    treatment_hist = df_hist["treatment"].to_numpy(copy=True)

    base_learner = build_base_learner(random_state=random_state)
    treatment_learner = build_treatment_learner(random_state=random_state)
    r_learner = initialize_meta_learners(base_learner, treatment_learner)["R-Learner"]
    r_learner.fit(X=x_hist, treatment=treatment_hist, y=y_hist)
    return r_learner, x_hist, y_hist


def run_experiment(config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """Execute the full causalml comparison pipeline from data generation to Qini evaluation."""
    # Step 1: Confounding diagnostics on biased logs.
    _log_section("Step 1 - Environment and Confounding Diagnostic")
    env_hist = UberMarketplaceEnvironment(seed=config.hist_seed)
    df_hist = env_hist.generate_biased_historical_data(num_samples=config.hist_rows)
    df_hist = add_oracle_counterfactuals(df_hist, env_hist)
    summary_step1, step1_metrics = summarize_naive_vs_true(df_hist)

    LOGGER.info("Rows generated: %s", _format_int(len(df_hist)))
    LOGGER.info("\n%s", _format_table(summary_step1, ["Value"]).to_string(index=False))

    policy_diagnosis = (
        df_hist.groupby("treatment")["frequency"].mean().rename("avg_frequency").to_frame()
    )
    LOGGER.info("\nAverage monthly frequency by treatment arm:\n%s", policy_diagnosis.to_string())

    # Step 2: Train meta-learners with dedicated control/treatment regressors.
    _log_section("Step 2 - Train T/X/R Learners")
    df_train_bin, x_train, y_train, t_train = prepare_binary_meta_dataset(df_hist, config.features)

    LOGGER.info("Training rows (0%% vs 20%% only): %s", _format_int(len(df_train_bin)))
    LOGGER.info("Dropped 10%% rows: %s", _format_int(len(df_hist) - len(df_train_bin)))

    base_learner = build_base_learner(random_state=config.learner_seed)
    treatment_learner = build_treatment_learner(random_state=config.learner_seed)
    learners = initialize_meta_learners(base_learner, treatment_learner)
    learners = fit_meta_learners(learners, x_train, t_train, y_train)

    LOGGER.info("Fitting dedicated multi-treatment R-Learner for optimization candidate scoring (0/10/20).")
    r_learner_multi, x_hist_multi, y_hist_multi = _fit_multitreatment_r_learner(
        df_hist=df_hist,
        features=config.features,
        random_state=config.learner_seed,
    )

    df_train_bin = add_meta_predictions(df_train_bin, learners, x_train)
    train_summary = summarize_model_predictions(
        df=df_train_bin,
        learners=learners,
        x=x_train,
        treatment=t_train,
        y=y_train,
        label="train",
        bootstrap_ci=config.bootstrap_ci,
        n_bootstraps=config.n_bootstraps,
        bootstrap_size=config.bootstrap_size,
    )
    train_percent_cols = [
        "Mean CATE Estimate (train)",
        "Mean CATE Lower Bound (train)",
        "Mean CATE Upper Bound (train)",
        "Mean True CATE (train)",
    ]
    LOGGER.info("\n%s", _format_table(train_summary, train_percent_cols).to_string(index=False))

    # Step 3: Randomized holdout and model scoring.
    _log_section("Step 3 - Randomized Holdout (RCT Sandbox)")
    env_rct, df_rct = generate_randomized_holdout(n_rows=config.rct_rows, seed=config.rct_seed)
    df_rct = add_oracle_counterfactuals(df_rct, env_rct)

    x_rct = df_rct[list(config.features)].to_numpy(dtype=float, copy=True)
    df_rct = add_meta_predictions(df_rct, learners, x_rct)

    holdout_summary = summarize_model_predictions(
        df=df_rct,
        learners=learners,
        x=x_rct,
        treatment=df_rct["treatment_label"].to_numpy(),
        y=df_rct["converted"].astype(float).to_numpy(),
        label="RCT",
        bootstrap_ci=config.bootstrap_ci,
        n_bootstraps=config.n_bootstraps,
        bootstrap_size=config.bootstrap_size,
    )
    holdout_percent_cols = [
        "Mean CATE Estimate (RCT)",
        "Mean CATE Lower Bound (RCT)",
        "Mean CATE Upper Bound (RCT)",
        "Mean True CATE (RCT)",
    ]

    treatment_share = df_rct["treatment"].value_counts(normalize=True).sort_index().rename("share")
    LOGGER.info("RCT rows generated: %s", _format_int(len(df_rct)))
    LOGGER.info("\nTreatment share:\n%s", treatment_share.to_string())
    LOGGER.info("\n%s", _format_table(holdout_summary, holdout_percent_cols).to_string(index=False))

    # Step 4: Decile ranking validation.
    _log_section("Step 4 - Decile Validation")
    model_score_cols = {model_name: f"pred_cate_{key}" for model_name, key in MODEL_KEY.items()}
    ranked_frames, decile_tables = build_all_deciles(df_rct, model_score_cols)

    for model_name, decile_df in decile_tables.items():
        LOGGER.info(
            "\n%s deciles:\n%s",
            model_name,
            _format_table(decile_df, ["Predicted_Lift", "Actual_Lift"]).to_string(index=False),
        )

    ranking_df = summarize_decile_ranking(
        df=df_rct,
        decile_tables=decile_tables,
        model_score_cols=model_score_cols,
        true_col="true_cate_20pct",
    )
    ranking_percent_cols = [
        "Top Decile Actual Lift",
        "Bottom Decile Actual Lift",
        "Decile Lift Spread (Top-Bottom)",
    ]
    LOGGER.info("\n%s", _format_table(ranking_df, ranking_percent_cols).to_string(index=False))

    if config.show_plots:
        plot_decile_comparison(decile_tables, show=True)

    # Step 5: Qini analysis and final comparison.
    _log_section("Step 5 - Qini Curves and Normalized Qini")
    x_axis, qini_random, qini_perfect, qini_curves, qini_results = qini_analysis(
        df=df_rct,
        model_score_cols=model_score_cols,
        true_col="true_cate_20pct",
    )

    LOGGER.info("\n%s", _format_table(qini_results, ["Qini_Normalized"]).to_string(index=False))

    best_model = str(qini_results.loc[0, "Model"])
    if config.show_plots:
        plot_qini_comparison(
            x_axis=x_axis,
            qini_random=qini_random,
            qini_perfect=qini_perfect,
            qini_curves=qini_curves,
            best_model=best_model,
            show=True,
        )

    final_comparison = ranking_df.merge(
        qini_results[["Model", "Qini_Normalized"]],
        on="Model",
        how="left",
    ).sort_values("Qini_Normalized", ascending=False)

    final_percent_cols = [
        "Top Decile Actual Lift",
        "Bottom Decile Actual Lift",
        "Decile Lift Spread (Top-Bottom)",
        "Qini_Normalized",
    ]
    LOGGER.info("\n%s", _format_table(final_comparison, final_percent_cols).to_string(index=False))
    LOGGER.info("Best model by normalized Qini: %s", best_model)

    # Step 6: Offline value optimization on top of causal predictions.
    _log_section("Step 6 - Offline Value Optimization")
    df_opt = _prepare_optimization_dataframe(
        df_rct=df_rct,
        x_rct=x_rct,
        r_learner=r_learner_multi,
        x_train_mu=x_hist_multi,
        y_train_mu=y_hist_multi,
        candidate_treatments=OPTIMIZATION_CANDIDATE_TREATMENTS,
    )

    mu_0_by_user = df_opt.drop_duplicates(subset=["user_id"])["mu_0"]
    LOGGER.info(
        "R-Learner mu(x) extracted for optimization (min=%.4f, mean=%.4f, max=%.4f).",
        float(mu_0_by_user.min()),
        float(mu_0_by_user.mean()),
        float(mu_0_by_user.max()),
    )
    LOGGER.info(
        "Optimization CATE source: multi-treatment R-Learner with direct lift estimates vs control for treatments %s.",
        ", ".join(str(t) for t in OPTIMIZATION_CANDIDATE_TREATMENTS),
    )

    LOGGER.info(
        "Optimization input rows prepared: %s users -> %s candidate rows (best eval model=%s).",
        _format_int(df_rct["treatment"].count()),
        _format_int(len(df_opt)),
        best_model,
    )

    selector = UnitSelector(
        user_id_col="user_id",
        treatment_col="treatment",
        base_prob_col="mu_0",
        cate_col="tau_hat",
        cost_col="face_value",
    )
    df_selected = selector.fit_transform(df_opt)
    LOGGER.info(
        "Unit selection pruning: %s -> %s rows (dropped %s).",
        _format_int(len(df_opt)),
        _format_int(len(df_selected)),
        _format_int(len(df_opt) - len(df_selected)),
    )

    lambda_val = np.nan
    df_allocated = df_selected.copy()
    allocation_summary = pd.DataFrame(
        {
            "treatment": [0, 1, 2],
            "allocated_fraction": [0.0, 0.0, 0.0],
        }
    )

    if df_selected.empty:
        LOGGER.warning("Unit selector returned no positive-lift candidates. Skipping optimization solve.")
    else:
        optimizer = ValueOptimizer(
            budget=OPTIMIZATION_BUDGET,
            user_col="user_id",
            cate_col="tau_hat",
            cost_col="expected_cost",
        )
        df_allocated, lambda_val = optimizer.optimize(df_selected)

        allocation_by_treatment = (
            df_allocated.groupby("treatment")["optimal_fraction"]
            .sum()
            .reset_index(name="allocated_fraction")
        )
        allocation_summary = (
            allocation_summary.drop(columns=["allocated_fraction"])
            .merge(
                allocation_by_treatment,
                on="treatment",
                how="left",
            )
            .fillna({"allocated_fraction": 0.0})
        )

    allocated_10 = float(
        allocation_summary.loc[allocation_summary["treatment"] == 1, "allocated_fraction"].iloc[0]
    )
    allocated_20 = float(
        allocation_summary.loc[allocation_summary["treatment"] == 2, "allocated_fraction"].iloc[0]
    )
    no_discount = max(float(df_opt["user_id"].nunique()) - (allocated_10 + allocated_20), 0.0)

    LOGGER.info(
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
        OPTIMIZATION_BUDGET,
        float(lambda_val) if not np.isnan(lambda_val) else float("nan"),
        float(lambda_val) if not np.isnan(lambda_val) else float("nan"),
        allocated_10,
        allocated_20,
        no_discount,
    )

    return {
        "summary_step1": summary_step1,
        "train_summary": train_summary,
        "holdout_summary": holdout_summary,
        "ranking_df": ranking_df,
        "qini_results": qini_results,
        "final_comparison": final_comparison,
        "df_hist": df_hist,
        "df_train_bin": df_train_bin,
        "df_rct": df_rct,
        "policy_diagnosis": policy_diagnosis,
        "step1_metrics": pd.DataFrame([step1_metrics]),
        "ranked_frames": pd.DataFrame(
            {
                "Model": list(ranked_frames.keys()),
                "Rows": [len(frame) for frame in ranked_frames.values()],
            }
        ),
        "df_opt": df_opt,
        "df_selected": df_selected,
        "df_allocated": df_allocated,
        "allocation_summary": allocation_summary,
        "optimization_metrics": pd.DataFrame(
            [
                {
                    "budget": OPTIMIZATION_BUDGET,
                    "lambda": float(lambda_val) if not np.isnan(lambda_val) else np.nan,
                    "allocated_10pct": allocated_10,
                    "allocated_20pct": allocated_20,
                    "allocated_none": no_discount,
                }
            ]
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the causalml T/X/R learner comparison pipeline on the Uber marketplace simulator."
    )
    parser.add_argument("--hist-rows", type=int, default=400_000, help="Number of biased historical rows.")
    parser.add_argument("--rct-rows", type=int, default=100_000, help="Number of randomized holdout rows.")
    parser.add_argument("--hist-seed", type=int, default=42, help="Random seed for historical data generation.")
    parser.add_argument("--rct-seed", type=int, default=999, help="Random seed for RCT holdout generation.")
    parser.add_argument("--learner-seed", type=int, default=42, help="Random seed for XGBoost learners.")
    parser.add_argument("--bootstrap-ci", action="store_true", help="Enable bootstrap confidence intervals.")
    parser.add_argument("--n-bootstraps", type=int, default=100, help="Bootstrap iterations for interval estimation.")
    parser.add_argument("--bootstrap-size", type=int, default=5_000, help="Sample size per bootstrap iteration.")
    parser.add_argument("--no-plots", action="store_true", help="Disable matplotlib rendering.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging.")
    return parser.parse_args()


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _configure_runtime_noise(verbose: bool) -> None:
    """Reduce third-party log and warning noise for CLI runs."""
    logging.getLogger("causalml").setLevel(logging.INFO if verbose else logging.WARNING)

    if verbose:
        return

    warnings.filterwarnings(
        "ignore",
        message=r"'penalty' was deprecated in version 1\.8 and will be removed in 1\.10\..*",
        category=FutureWarning,
        module=r"sklearn\.linear_model\._logistic",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The fitted attributes of LogisticRegressionCV will be simplified in scikit-learn 1\.10.*",
        category=FutureWarning,
        module=r"sklearn\.linear_model\._logistic",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The max_iter was reached which means the coef_ did not converge",
        category=ConvergenceWarning,
        module=r"sklearn\.linear_model\._sag",
    )


def main() -> None:
    args = _parse_args()
    _configure_logging(args.verbose)
    _configure_runtime_noise(args.verbose)

    config = ExperimentConfig(
        features=tuple(DEFAULT_FEATURES),
        hist_rows=args.hist_rows,
        rct_rows=args.rct_rows,
        hist_seed=args.hist_seed,
        rct_seed=args.rct_seed,
        learner_seed=args.learner_seed,
        bootstrap_ci=args.bootstrap_ci,
        n_bootstraps=args.n_bootstraps,
        bootstrap_size=args.bootstrap_size,
        show_plots=not args.no_plots,
    )

    run_experiment(config)


if __name__ == "__main__":
    main()

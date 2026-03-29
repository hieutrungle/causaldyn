from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
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

LOGGER = logging.getLogger(__name__)


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
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the causalml T/X/R learner comparison pipeline on the Uber marketplace simulator."
    )
    parser.add_argument("--hist-rows", type=int, default=200_000, help="Number of biased historical rows.")
    parser.add_argument("--rct-rows", type=int, default=20_000, help="Number of randomized holdout rows.")
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

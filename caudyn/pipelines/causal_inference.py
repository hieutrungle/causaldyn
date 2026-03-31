from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from caudyn.causal.data_utils import MODEL_KEY, add_oracle_counterfactuals, generate_randomized_holdout
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

from .common import _fit_r_learner_mu_model, _format_int, _format_table, _log_section
from .contracts import CausalPipelineResult, ExperimentConfigLike


def _log_step1_diagnostics(
    logger: logging.Logger,
    *,
    df_hist: pd.DataFrame,
    summary_step1: pd.DataFrame,
    policy_diagnosis: pd.DataFrame,
) -> None:
    logger.info("Rows generated: %s", _format_int(len(df_hist)))
    logger.info("\n%s", _format_table(summary_step1, ["Value"]).to_string(index=False))
    logger.info("\nAverage monthly frequency by treatment arm:\n%s", policy_diagnosis.to_string())


def _log_step2_training_snapshot(
    logger: logging.Logger,
    *,
    df_train: pd.DataFrame,
    train_summary: pd.DataFrame,
) -> None:
    logger.info("Training rows (0%%/10%%/20%%): %s", _format_int(len(df_train)))
    train_treatment_share = df_train["treatment"].value_counts(normalize=True).sort_index().rename("share")
    logger.info("\nTraining treatment share:\n%s", train_treatment_share.to_string())

    train_percent_cols = [
        "Mean CATE Estimate (train)",
        "Mean CATE Lower Bound (train)",
        "Mean CATE Upper Bound (train)",
        "Mean True CATE (train)",
    ]
    logger.info("\n%s", _format_table(train_summary, train_percent_cols).to_string(index=False))


def _log_holdout_snapshot(
    logger: logging.Logger,
    *,
    df_rct: pd.DataFrame,
    df_eval: pd.DataFrame,
    holdout_summary: pd.DataFrame,
) -> None:
    holdout_percent_cols = [
        "Mean CATE Estimate (RCT)",
        "Mean CATE Lower Bound (RCT)",
        "Mean CATE Upper Bound (RCT)",
        "Mean True CATE (RCT)",
    ]
    treatment_share_full = df_rct["treatment"].value_counts(normalize=True).sort_index().rename("share")
    treatment_share_binary = (
        df_eval["treatment"].value_counts(normalize=True).sort_index().rename("share")
    )

    logger.info("RCT rows generated (full 0/10/20): %s", _format_int(len(df_rct)))
    logger.info("\nFull treatment share:\n%s", treatment_share_full.to_string())
    logger.info("Binary validation rows (0/20): %s", _format_int(len(df_eval)))
    logger.info("\nBinary treatment share:\n%s", treatment_share_binary.to_string())
    logger.info("\n%s", _format_table(holdout_summary, holdout_percent_cols).to_string(index=False))


def _log_decile_validation(
    logger: logging.Logger,
    *,
    decile_tables: dict[str, pd.DataFrame],
    ranking_df: pd.DataFrame,
) -> None:
    for model_name, decile_df in decile_tables.items():
        logger.info(
            "\n%s deciles:\n%s",
            model_name,
            _format_table(decile_df, ["Predicted_Lift", "Actual_Lift"]).to_string(index=False),
        )

    ranking_percent_cols = [
        "Top Decile Actual Lift",
        "Bottom Decile Actual Lift",
        "Decile Lift Spread (Top-Bottom)",
    ]
    logger.info("\n%s", _format_table(ranking_df, ranking_percent_cols).to_string(index=False))


def _log_qini_summary(
    logger: logging.Logger,
    *,
    qini_results: pd.DataFrame,
    final_comparison: pd.DataFrame,
    best_model: str,
) -> None:
    logger.info("\n%s", _format_table(qini_results, ["Qini_Normalized"]).to_string(index=False))

    final_percent_cols = [
        "Top Decile Actual Lift",
        "Bottom Decile Actual Lift",
        "Decile Lift Spread (Top-Bottom)",
        "Qini_Normalized",
    ]
    logger.info("\n%s", _format_table(final_comparison, final_percent_cols).to_string(index=False))
    logger.info("Best model by normalized Qini: %s", best_model)


def run_causal_inference_pipeline(
    config: ExperimentConfigLike,
    logger: logging.Logger | None = None,
) -> CausalPipelineResult:
    """Run Steps 1-5: data generation, model training, and causal validation."""
    log = logger or logging.getLogger(__name__)

    _log_section(log, "Step 1 - Environment and Confounding Diagnostic")
    env_hist = UberMarketplaceEnvironment(seed=config.hist_seed)
    df_hist = env_hist.generate_biased_historical_data(num_samples=config.hist_rows)
    df_hist = add_oracle_counterfactuals(df_hist, env_hist)
    summary_step1, step1_metrics = summarize_naive_vs_true(df_hist)
    policy_diagnosis = (
        df_hist.groupby("treatment")["frequency"].mean().rename("avg_frequency").to_frame()
    )
    _log_step1_diagnostics(
        log,
        df_hist=df_hist,
        summary_step1=summary_step1,
        policy_diagnosis=policy_diagnosis,
    )

    _log_section(log, "Step 2 - Train T/X/R Learners")
    df_train = df_hist.copy()
    df_train["treatment_label"] = df_train["treatment"].astype(int)
    x_train = df_train[list(config.features)].to_numpy(dtype=np.float64, copy=True)
    y_train = df_train["converted"].to_numpy(dtype=np.float64, copy=True)
    t_train = df_train["treatment_label"].to_numpy(copy=True)

    base_learner = build_base_learner(random_state=config.learner_seed)
    treatment_learner = build_treatment_learner(random_state=config.learner_seed)
    learners = initialize_meta_learners(base_learner, treatment_learner)
    learners = fit_meta_learners(learners, x_train, t_train, y_train)

    r_learner_multi = learners["R-Learner"]
    mu_model_fitted = _fit_r_learner_mu_model(
        r_learner=r_learner_multi,
        x_train=x_train,
        y_train=y_train,
    )

    df_train = add_meta_predictions(df_train, learners, x_train)
    train_summary = summarize_model_predictions(
        df=df_train,
        learners=learners,
        x=x_train,
        treatment=t_train,
        y=y_train,
        label="train",
        bootstrap_ci=config.bootstrap_ci,
        n_bootstraps=config.n_bootstraps,
        bootstrap_size=config.bootstrap_size,
    )
    _log_step2_training_snapshot(log, df_train=df_train, train_summary=train_summary)

    _log_section(log, "Step 3 - Randomized Holdout (RCT Sandbox)")
    env_rct, df_rct = generate_randomized_holdout(
        n_rows=config.rct_rows,
        seed=config.rct_seed,
        treatments=(0, 1, 2),
        treatment_value=2,
    )
    df_rct = add_oracle_counterfactuals(df_rct, env_rct)
    x_rct_full = df_rct[list(config.features)].to_numpy(dtype=float, copy=True)

    df_rct_binary = df_rct[df_rct["treatment"].isin([0, 2])].copy()
    if df_rct_binary.empty:
        raise ValueError("Binary RCT slice is empty; cannot run decile or Qini evaluation.")
    if df_rct_binary["treatment"].nunique() < 2:
        raise ValueError("Binary RCT slice must contain both treatment 0 and treatment 2 rows.")

    x_rct_binary = df_rct_binary[list(config.features)].to_numpy(dtype=float, copy=True)
    df_eval = add_meta_predictions(df_rct_binary, learners, x_rct_binary)

    holdout_summary = summarize_model_predictions(
        df=df_eval,
        learners=learners,
        x=x_rct_binary,
        treatment=df_eval["treatment_label"].to_numpy(),
        y=df_eval["converted"].astype(float).to_numpy(),
        label="RCT",
        bootstrap_ci=config.bootstrap_ci,
        n_bootstraps=config.n_bootstraps,
        bootstrap_size=config.bootstrap_size,
    )
    _log_holdout_snapshot(log, df_rct=df_rct, df_eval=df_eval, holdout_summary=holdout_summary)

    _log_section(log, "Step 4 - Decile Validation")
    model_score_cols = {model_name: f"pred_cate_{key}" for model_name, key in MODEL_KEY.items()}
    ranked_frames, decile_tables = build_all_deciles(df_eval, model_score_cols)

    ranking_df = summarize_decile_ranking(
        df=df_eval,
        decile_tables=decile_tables,
        model_score_cols=model_score_cols,
        true_col="true_cate_20pct",
    )
    _log_decile_validation(log, decile_tables=decile_tables, ranking_df=ranking_df)

    if config.show_plots:
        plot_decile_comparison(decile_tables, show=True)

    _log_section(log, "Step 5 - Qini Curves and Normalized Qini")
    x_axis, qini_random, qini_perfect, qini_curves, qini_results = qini_analysis(
        df=df_eval,
        model_score_cols=model_score_cols,
        true_col="true_cate_20pct",
    )

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
    _log_qini_summary(
        log,
        qini_results=qini_results,
        final_comparison=final_comparison,
        best_model=best_model,
    )

    # metrics: dict[str, pd.DataFrame] = {
    #     "summary_step1": summary_step1,
    #     "train_summary": train_summary,
    #     "holdout_summary": holdout_summary,
    #     "ranking_df": ranking_df,
    #     "qini_results": qini_results,
    #     "final_comparison": final_comparison,
    #     "df_hist": df_hist,
    #     "df_train_bin": df_train,
    #     "df_rct": df_rct,
    #     "df_rct_binary": df_eval,
    #     "policy_diagnosis": policy_diagnosis,
    #     "step1_metrics": pd.DataFrame([step1_metrics]),
    #     "ranked_frames": pd.DataFrame(
    #         {
    #             "Model": list(ranked_frames.keys()),
    #             "Rows": [len(frame) for frame in ranked_frames.values()],
    #         }
    #     ),
    # }
    metrics = {}

    return CausalPipelineResult(
        r_learner_multi=r_learner_multi,
        mu_model_fitted=mu_model_fitted,
        df_rct=df_rct,
        x_rct_full=x_rct_full,
        best_model=best_model,
        metrics=metrics,
    )

"""Causal inference utilities for marketplace incentive experiments."""

from .data_utils import (
    CONTROL_LABEL,
    CONTROL_VALUE,
    DEFAULT_FEATURES,
    MODEL_KEY,
    TREATMENT_LABEL,
    TREATMENT_VALUE,
    add_oracle_counterfactuals,
    generate_randomized_holdout,
    prepare_binary_meta_dataset,
)
from .meta_learners import (
    add_meta_predictions,
    build_base_learner,
    build_treatment_learner,
    fit_meta_learners,
    initialize_meta_learners,
)
from .metrics import (
    build_all_deciles,
    calculate_qini_curve,
    decile_validation_table,
    qini_analysis,
    summarize_decile_ranking,
    summarize_model_predictions,
    summarize_naive_vs_true,
)
from .plotting import plot_decile_comparison, plot_qini_comparison

__all__ = [
    "CONTROL_LABEL",
    "CONTROL_VALUE",
    "DEFAULT_FEATURES",
    "MODEL_KEY",
    "TREATMENT_LABEL",
    "TREATMENT_VALUE",
    "add_meta_predictions",
    "add_oracle_counterfactuals",
    "build_all_deciles",
    "build_base_learner",
    "build_treatment_learner",
    "calculate_qini_curve",
    "decile_validation_table",
    "fit_meta_learners",
    "generate_randomized_holdout",
    "initialize_meta_learners",
    "plot_decile_comparison",
    "plot_qini_comparison",
    "prepare_binary_meta_dataset",
    "qini_analysis",
    "summarize_decile_ranking",
    "summarize_model_predictions",
    "summarize_naive_vs_true",
]

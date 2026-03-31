from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from caudyn.causal.data_utils import DEFAULT_FEATURES
from caudyn.pipelines import (
    DEFAULT_CAUSAL_ARTIFACT_FILENAME,
    load_causal_pipeline_result,
    OptimizationPipelineResult,
    run_causal_inference_pipeline,
    run_offline_optimization_pipeline,
    run_online_simulation_pipeline,
    save_causal_pipeline_result,
)

LOGGER = logging.getLogger(__name__)
OPTIMIZATION_BUDGET = 5_000.0


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
    run_simulation: bool = False
    simulation_budget: float = OPTIMIZATION_BUDGET
    simulation_hours: int = 24
    simulation_seed: int = 2026
    simulation_min_riders_per_hour: int = 800
    simulation_max_riders_per_hour: int = 1200
    simulation_base_lambda: float | None = None
    run_offline_optimization: bool = True
    save_causal_artifacts: bool = False
    load_causal_artifacts: bool = False
    causal_artifacts_dir: str = "tmp_models"
    causal_artifact_file: str = DEFAULT_CAUSAL_ARTIFACT_FILENAME


def _resolve_causal_results(config: ExperimentConfig):
    if config.load_causal_artifacts:
        artifact_dir = Path(config.causal_artifacts_dir)
        return load_causal_pipeline_result(
            artifact_dir=artifact_dir,
            file_name=config.causal_artifact_file,
            logger=LOGGER,
        )

    causal_results = run_causal_inference_pipeline(config, logger=LOGGER)
    if config.save_causal_artifacts:
        artifact_dir = Path(config.causal_artifacts_dir)
        save_causal_pipeline_result(
            causal_results,
            artifact_dir=artifact_dir,
            file_name=config.causal_artifact_file,
            logger=LOGGER,
        )
    return causal_results


def run_experiment(config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """Execute the full pipeline through modularized stage runners."""
    causal_results = _resolve_causal_results(config)
    optimization_results = OptimizationPipelineResult(lambda_val=float("nan"), metrics={})
    if config.run_offline_optimization:
        optimization_results = run_offline_optimization_pipeline(
            config,
            causal_results,
            logger=LOGGER,
        )
    else:
        LOGGER.info(
            "Skipping Step 6 - Offline Value Optimization (--skip-offline-optimization enabled)."
        )

    sim_results: dict[str, pd.DataFrame] = {
        "simulation_summary": pd.DataFrame(),
        "simulation_artifacts": pd.DataFrame(columns=["artifact_type", "path"]),
    }
    if config.run_simulation:
        sim_results = run_online_simulation_pipeline(
            config,
            causal_results,
            optimization_results,
            logger=LOGGER,
        )

    return {
        **causal_results.metrics,
        **optimization_results.metrics,
        **sim_results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the causalml T/X/R learner comparison pipeline on the Uber marketplace simulator."
    )
    parser.add_argument("--hist-rows", type=int, default=500_000, help="Number of biased historical rows.")
    parser.add_argument("--rct-rows", type=int, default=100_000, help="Number of randomized holdout rows.")
    parser.add_argument("--hist-seed", type=int, default=42, help="Random seed for historical data generation.")
    parser.add_argument("--rct-seed", type=int, default=999, help="Random seed for RCT holdout generation.")
    parser.add_argument("--learner-seed", type=int, default=42, help="Random seed for XGBoost learners.")
    parser.add_argument("--bootstrap-ci", action="store_true", help="Enable bootstrap confidence intervals.")
    parser.add_argument("--n-bootstraps", type=int, default=100, help="Bootstrap iterations for interval estimation.")
    parser.add_argument("--bootstrap-size", type=int, default=5_000, help="Sample size per bootstrap iteration.")
    parser.add_argument(
        "--skip-offline-optimization",
        action="store_true",
        help="Skip Step 6 (and therefore Step 7); useful for Step 1-5 train-and-save workflows.",
    )
    parser.add_argument("--run-simulation", action="store_true", help="Run Step 7 online event-loop simulation.")
    parser.add_argument(
        "--save-causal-artifacts",
        action="store_true",
        help="Save Step 1-5 causal outputs to disk for reuse in future Step 6/7 runs.",
    )
    parser.add_argument(
        "--load-causal-artifacts",
        action="store_true",
        help="Load Step 1-5 causal outputs from disk and skip causal retraining.",
    )
    parser.add_argument(
        "--causal-artifacts-dir",
        type=str,
        default="tmp_models",
        help="Directory for saved causal artifacts.",
    )
    parser.add_argument(
        "--causal-artifact-file",
        type=str,
        default=DEFAULT_CAUSAL_ARTIFACT_FILENAME,
        help="Causal artifact filename inside --causal-artifacts-dir.",
    )
    parser.add_argument("--simulation-budget", type=float, default=3_000.0, help="Simulation budget in dollars.")
    parser.add_argument("--simulation-hours", type=int, default=24, help="Number of simulation epochs (hours).")
    parser.add_argument("--simulation-seed", type=int, default=2026, help="Random seed for simulation.")
    parser.add_argument(
        "--simulation-min-riders",
        type=int,
        default=800,
        help="Minimum number of rider arrivals per simulation hour.",
    )
    parser.add_argument(
        "--simulation-max-riders",
        type=int,
        default=1200,
        help="Maximum number of rider arrivals per simulation hour.",
    )
    parser.add_argument(
        "--simulation-base-lambda",
        type=float,
        default=None,
        help=(
            "Optional base lambda override for simulation. "
            "If omitted, uses offline optimizer lambda when available."
        ),
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable matplotlib rendering.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging.")
    args = parser.parse_args()

    if args.skip_offline_optimization and args.run_simulation:
        parser.error("--run-simulation cannot be combined with --skip-offline-optimization.")

    return args


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
        run_offline_optimization=not args.skip_offline_optimization,
        run_simulation=args.run_simulation,
        save_causal_artifacts=args.save_causal_artifacts,
        load_causal_artifacts=args.load_causal_artifacts,
        causal_artifacts_dir=args.causal_artifacts_dir,
        causal_artifact_file=args.causal_artifact_file,
        simulation_budget=args.simulation_budget,
        simulation_hours=args.simulation_hours,
        simulation_seed=args.simulation_seed,
        simulation_min_riders_per_hour=args.simulation_min_riders,
        simulation_max_riders_per_hour=args.simulation_max_riders,
        simulation_base_lambda=args.simulation_base_lambda,
    )

    run_experiment(config)


if __name__ == "__main__":
    main()

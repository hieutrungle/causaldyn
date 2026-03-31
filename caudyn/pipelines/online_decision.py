from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from caudyn.simulation import MarketplaceSimulation

from .common import (
    DEFAULT_BASE_LAMBDA,
    OPTIMIZATION_CANDIDATE_TREATMENTS,
    _build_online_inference_fn,
    _log_section,
)
from .contracts import CausalPipelineResult, ExperimentConfigLike, OptimizationPipelineResult

STEP7_RESULTS_DIR = Path("tmp_results")
STEP7_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def _build_step7_artifact_paths(*, seed: int, hours: int) -> tuple[Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime(STEP7_TIMESTAMP_FORMAT)
    run_tag = f"{timestamp}_seed{seed}_h{hours}"
    csv_path = STEP7_RESULTS_DIR / f"step7_simulation_summary_{run_tag}.csv"
    png_path = STEP7_RESULTS_DIR / f"step7_simulation_plot_{run_tag}.png"
    return csv_path, png_path


def _export_step7_artifacts(
    *,
    logger: logging.Logger,
    simulation: MarketplaceSimulation,
    summary_df: pd.DataFrame,
    seed: int,
    hours: int,
    show_plot: bool,
) -> pd.DataFrame:
    if summary_df.empty:
        raise ValueError("Cannot export Step 7 artifacts because simulation_summary is empty.")

    STEP7_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path, png_path = _build_step7_artifact_paths(seed=seed, hours=hours)

    summary_df.to_csv(csv_path, index=False)
    simulation.plot_time_series(summary_df=summary_df, show=show_plot, save_path=png_path)

    logger.info("Step 7 artifacts exported under %s", STEP7_RESULTS_DIR.as_posix())
    logger.info("Step 7 CSV artifact: %s", csv_path.as_posix())
    logger.info("Step 7 PNG artifact: %s", png_path.as_posix())

    return pd.DataFrame(
        [
            {
                "artifact_type": "step7_simulation_summary",
                "path": csv_path.as_posix(),
            },
            {
                "artifact_type": "step7_simulation_plot",
                "path": png_path.as_posix(),
            },
        ]
    )


def _resolve_simulation_lambda(
    *,
    config: ExperimentConfigLike,
    optim_data: OptimizationPipelineResult,
) -> tuple[float, str]:
    if config.simulation_base_lambda is not None:
        return float(config.simulation_base_lambda), "cli_override"
    if not np.isnan(optim_data.lambda_val):
        return float(optim_data.lambda_val), "offline_optimizer"
    return DEFAULT_BASE_LAMBDA, "fallback_default"


def _log_simulation_config(
    logger: logging.Logger,
    *,
    simulation_lambda: float,
    lambda_source: str,
    config: ExperimentConfigLike,
) -> None:
    logger.info(
        "Running online simulation with model-backed inference | base_lambda=%.4f (%s) | "
        "budget=$%.2f | hours=%d | riders/hour=[%d, %d]",
        simulation_lambda,
        lambda_source,
        float(config.simulation_budget),
        int(config.simulation_hours),
        int(config.simulation_min_riders_per_hour),
        int(config.simulation_max_riders_per_hour),
    )


def run_online_simulation_pipeline(
    config: ExperimentConfigLike,
    causal_data: CausalPipelineResult,
    optim_data: OptimizationPipelineResult,
    logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame]:
    """Run Step 7: online decisioning with threshold dispatch + PID pacing."""
    log = logger or logging.getLogger(__name__)
    _log_section(log, "Step 7 - Online Event Loop Simulation")

    online_inference_fn = _build_online_inference_fn(
        features=config.features,
        r_learner=causal_data.r_learner_multi,
        mu_model=causal_data.mu_model_fitted,
        candidate_treatments=OPTIMIZATION_CANDIDATE_TREATMENTS,
    )

    simulation_lambda, lambda_source = _resolve_simulation_lambda(
        config=config,
        optim_data=optim_data,
    )
    _log_simulation_config(
        log,
        simulation_lambda=simulation_lambda,
        lambda_source=lambda_source,
        config=config,
    )

    simulation = MarketplaceSimulation(
        total_budget=float(config.simulation_budget),
        total_hours=int(config.simulation_hours),
        base_lambda=simulation_lambda,
        seed=int(config.simulation_seed),
        min_riders_per_hour=int(config.simulation_min_riders_per_hour),
        max_riders_per_hour=int(config.simulation_max_riders_per_hour),
        inference_fn=online_inference_fn,
    )

    simulation_summary = simulation.run_simulation()
    simulation_artifacts = _export_step7_artifacts(
        logger=log,
        simulation=simulation,
        summary_df=simulation_summary,
        seed=int(config.simulation_seed),
        hours=int(config.simulation_hours),
        show_plot=config.show_plots,
    )

    return {
        "simulation_summary": simulation_summary,
        "simulation_artifacts": simulation_artifacts,
    }

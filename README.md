# causal-ml-marketplace

Production-style causal marketplace optimization research stack.

This project combines causal effect estimation, constrained value optimization, and online pacing control in one pipeline. The core goal is to move from offline causal inference to a controllable online decision loop that can be stress-tested in simulation.

## Highlights

- End-to-end causal experiment runner with T-Learner, X-Learner, and R-Learner evaluation.
- Offline value optimization with budget constraints and shadow-price extraction.
- Online event-loop simulation with threshold dispatch and PID pacing control.
- Step 7 artifact tracking with automatic CSV and PNG export to tmp_results for reproducibility.

## Repository Layout

- [caudyn/environment.py](caudyn/environment.py): Marketplace simulator and counterfactual mechanics.
- [caudyn/run_causal_experiment.py](caudyn/run_causal_experiment.py): Unified entry point for offline pipeline and optional online simulation.
- [caudyn/value_optimization/](caudyn/value_optimization): Unit selection and LP optimization modules.
- [caudyn/decision_engine/](caudyn/decision_engine): Real-time dispatcher and PID pacing controller.
- [caudyn/simulation/system_orchestrator.py](caudyn/simulation/system_orchestrator.py): Step 7 event-loop orchestration and simulation time-series plotting.
- [tmp_results/](tmp_results): Generated Step 7 experiment artifacts (created at runtime).

## Requirements

- Python 3.10+
- Virtual environment recommended

## Installation

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For development dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Unified Experiment CLI

Most important workflows:

1. Full pipeline run (Step 1-6)

```bash
python -m caudyn.run_causal_experiment \
  --hist-rows 500000 \
  --rct-rows 100000 \
  --save-causal-artifacts \
  --causal-artifacts-dir tmp_models \
  --run-simulation \
  --simulation-hours 72 \
  --simulation-budget 10000 \
  --simulation-min-riders 1600 \
  --simulation-max-riders 2200
```

1. Train + save causal artifacts (cache Step 1-5 to tmp_models)

```bash
python -m caudyn.run_causal_experiment \
  --hist-rows 500000 \
  --rct-rows 100000 \
  --skip-offline-optimization \
  --save-causal-artifacts \
  --causal-artifacts-dir tmp_models
```

1. Load cached causal artifacts + run offline optimization + simulation (skip Step 1-5 retraining)

```bash
python -m caudyn.run_causal_experiment \
  --load-causal-artifacts \
  --causal-artifacts-dir tmp_models \
  --run-simulation \
  --simulation-hours 72 \
  --simulation-budget 10000 \
  --simulation-min-riders 1600 \
  --simulation-max-riders 2200
```

### Causal Artifact Flags

- --save-causal-artifacts: Save the output of Step 1-5 (run_causal_inference_pipeline) to a local artifact file.
- --load-causal-artifacts: Load a previously saved Step 1-5 artifact and skip causal retraining.
- --causal-artifacts-dir: Directory used for saved artifacts (default: tmp_models).
- --causal-artifact-file: Artifact filename inside the directory (default: causal_pipeline_result.pkl).

### Pipeline Control Flags

- --skip-offline-optimization: Stop after Step 1-5 (and optional save); Step 6 and Step 7 are not executed.
- --run-simulation: Execute Step 7 after Step 6. This cannot be combined with --skip-offline-optimization.

### Simulation Controls

- --simulation-budget: Total budget for Step 7.
- --simulation-hours: Number of event-loop epochs.
- --simulation-seed: Random seed for reproducibility.
- --simulation-min-riders: Lower bound for rider arrivals per hour.
- --simulation-max-riders: Upper bound for rider arrivals per hour.
- --simulation-base-lambda: Optional lambda override for online loop initialization.

## Step 7 Artifact Export

When --run-simulation is enabled, Step 7 artifacts are automatically exported under [tmp_results/](tmp_results):

- CSV: step7_simulation_summary_YYYYMMDDTHHMMSSZ_seedSEED_hHOURS.csv
- PNG: step7_simulation_plot_YYYYMMDDTHHMMSSZ_seedSEED_hHOURS.png

This export is intended for experiment tracking and allows easy comparisons across runs.

## Clean Code Principles

- Keep orchestration in [caudyn/run_causal_experiment.py](caudyn/run_causal_experiment.py) and domain logic in dedicated modules.
- Keep inference contracts explicit: simulation inference must produce mu_0, tau_hat_10, and tau_hat_20.
- Keep simulation strict: no mock inference fallback in the orchestrator; inject model-backed inference from the offline pipeline.
- Prefer small helper functions for data preparation, model extraction, and artifact export.
- Keep runtime behavior reproducible with explicit seeds and deterministic artifact naming.

## Reproducibility Notes

- Fix seeds using --hist-seed, --rct-seed, --learner-seed, and --simulation-seed.
- Keep artifact outputs for each run in tmp_results to support before/after analysis.
- Use --no-plots in headless environments while still exporting Step 7 PNG artifacts.

## Packaging

- Metadata source: [pyproject.toml](pyproject.toml)
- Runtime dependencies: [requirements.txt](requirements.txt)
- Console entry point: caudyn-run-causal-experiment -> caudyn.run_causal_experiment:main

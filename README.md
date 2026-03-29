# causal-ml-marketplace

Marketplace contextual bandit and causal learning experiments, including a modular causalml pipeline under the `caudyn` package.

## Prerequisites

- Python 3.10+
- A virtual environment (recommended)

## Installation

Install the project in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

This uses packaging metadata from `pyproject.toml` and loads runtime dependencies from `requirements.txt`.

## Development Installation

Install editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Current dev dependencies include:

- pytest
- black

## Run the Causal Experiment

After installation, run via console script:

```bash
caudyn-run-causal-experiment --hist-rows 1000 --rct-rows 500 --no-plots
```

Or run as a module:

```bash
python -m caudyn.run_causal_experiment --hist-rows 1000 --rct-rows 500 --no-plots
```

## Packaging Notes

- Core package metadata: `pyproject.toml`
- Runtime dependency source: `requirements.txt` (via dynamic dependencies)
- Optional development extras: `[project.optional-dependencies].dev`

This structure supports local editable installs now and can be used for future publishing with minimal changes.

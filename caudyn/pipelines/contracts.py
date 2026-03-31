from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd


class ExperimentConfigLike(Protocol):
    """Minimal config contract consumed by pipeline modules."""

    @property
    def features(self) -> Sequence[str]: ...

    @property
    def hist_rows(self) -> int: ...

    @property
    def rct_rows(self) -> int: ...

    @property
    def hist_seed(self) -> int: ...

    @property
    def rct_seed(self) -> int: ...

    @property
    def learner_seed(self) -> int: ...

    @property
    def bootstrap_ci(self) -> bool: ...

    @property
    def n_bootstraps(self) -> int: ...

    @property
    def bootstrap_size(self) -> int: ...

    @property
    def show_plots(self) -> bool: ...

    @property
    def simulation_budget(self) -> float: ...

    @property
    def simulation_hours(self) -> int: ...

    @property
    def simulation_seed(self) -> int: ...

    @property
    def simulation_min_riders_per_hour(self) -> int: ...

    @property
    def simulation_max_riders_per_hour(self) -> int: ...

    @property
    def simulation_base_lambda(self) -> float | None: ...


@dataclass
class CausalPipelineResult:
    """Typed output contract for Steps 1-5 (causal inference and validation)."""

    r_learner_multi: Any
    mu_model_fitted: Any
    df_rct: pd.DataFrame
    x_rct_full: np.ndarray
    best_model: str
    metrics: dict[str, pd.DataFrame]


@dataclass
class OptimizationPipelineResult:
    """Typed output contract for Step 6 (offline value optimization)."""

    lambda_val: float
    metrics: dict[str, pd.DataFrame]

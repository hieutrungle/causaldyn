from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from caudyn.environment import UberMarketplaceEnvironment

TREATMENT_VALUE = 2
CONTROL_VALUE = 0
TREATMENT_LABEL = 2
CONTROL_LABEL = 0

MODEL_KEY = {
    "T-Learner": "t",
    "X-Learner": "x",
    "R-Learner": "r",
}

DEFAULT_FEATURES: tuple[str, ...] = (
    "recency",
    "frequency",
    "weather_active",
    "surge_multiplier",
)


def context_from_row(row) -> dict[str, float]:
    """Build an environment context dict from a data row."""
    return {
        "recency": float(row["recency"]),
        "frequency": float(row["frequency"]),
        "weather_active": float(row["weather_active"]),
        "surge_multiplier": float(row["surge_multiplier"]),
    }


def add_oracle_counterfactuals(
    df: pd.DataFrame,
    env: UberMarketplaceEnvironment,
    treat_value: int = TREATMENT_VALUE,
    control_value: int = CONTROL_VALUE,
) -> pd.DataFrame:
    """Add oracle conversion probabilities and true CATE columns from simulator physics."""
    out = df.copy()
    contexts = out[list(DEFAULT_FEATURES)].to_dict(orient="records")

    treat_probs = [env._calculate_true_conversion(context_from_row(ctx), treat_value) for ctx in contexts]
    control_probs = [env._calculate_true_conversion(context_from_row(ctx), control_value) for ctx in contexts]

    out["true_conversion_20pct"] = np.asarray(treat_probs, dtype=float)
    out["true_conversion_0pct"] = np.asarray(control_probs, dtype=float)
    out["true_cate_20pct"] = out["true_conversion_20pct"] - out["true_conversion_0pct"]
    return out


def prepare_binary_meta_dataset(
    df: pd.DataFrame,
    features: Sequence[str],
    treat_value: int = TREATMENT_VALUE,
    control_value: int = CONTROL_VALUE,
    treatment_label: int = TREATMENT_LABEL,
    control_label: int = CONTROL_LABEL,
) -> tuple[pd.DataFrame, NDArray, NDArray, NDArray]:
    """Prepare binary-treatment arrays used by causalml meta-learners."""
    out = df[df["treatment"].isin([control_value, treat_value])].copy()
    out["treatment_label"] = np.where(out["treatment"] == treat_value, treatment_label, control_label)

    x = out[list(features)].to_numpy(dtype=float, copy=True)
    y = out["converted"].astype(float).to_numpy(copy=True)
    treatment = out["treatment_label"].to_numpy(copy=True)
    return out, x, y, treatment


def generate_randomized_holdout(
    n_rows: int = 20_000,
    seed: int = 999,
    treatments: Sequence[int] = (CONTROL_VALUE, TREATMENT_VALUE),
    treatment_value: int = TREATMENT_VALUE,
    treatment_label: int = TREATMENT_LABEL,
    control_label: int = CONTROL_LABEL,
) -> tuple[UberMarketplaceEnvironment, pd.DataFrame]:
    """Generate an RCT-style holdout where treatment assignment is randomized."""
    env = UberMarketplaceEnvironment(seed=seed)
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    for _ in range(n_rows):
        context = env.reset()
        action = int(rng.choice(treatments))
        _, reward, true_prob = env.step(action)

        row = dict(context)
        row["treatment"] = action
        row["treatment_label"] = treatment_label if action == treatment_value else control_label
        row["discount_value"] = env.discount_levels[action]
        row["converted"] = reward
        row["true_prob_observed"] = true_prob
        rows.append(row)

    return env, pd.DataFrame(rows)

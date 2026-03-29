from __future__ import annotations

import logging
from typing import Mapping

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def _supports_interactive_show() -> bool:
    backend = plt.get_backend().lower()

    try:
        from matplotlib.backends.registry import BackendFilter, backend_registry

        interactive_backends = {
            name.lower() for name in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
        }
        return backend in interactive_backends
    except Exception:
        # Fallback for older matplotlib versions.
        non_interactive_backends = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
        return backend not in non_interactive_backends


def plot_decile_comparison(
    decile_tables: Mapping[str, pd.DataFrame],
    show: bool = True,
    title: str = "Decile Validation: Predicted vs Actual Lift (RCT)",
) -> tuple:
    """Plot predicted lift vs actual lift by decile for each learner."""
    if not decile_tables:
        raise ValueError("decile_tables is empty; there is nothing to plot.")

    n_models = len(decile_tables)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = np.asarray([axes])

    for ax, (model_name, decile_df) in zip(axes, decile_tables.items()):
        x = np.arange(len(decile_df))
        width = 0.38
        ax.bar(x - width / 2, decile_df["Predicted_Lift"], width, label="Predicted", color="#4C78A8")
        ax.bar(x + width / 2, decile_df["Actual_Lift"], width, label="Actual", color="#F58518")
        ax.set_title(model_name)
        ax.set_xlabel("Decile (1 = highest predicted uplift)")
        ax.set_xticks(x)
        ax.set_xticklabels(decile_df["Decile"])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    axes[0].set_ylabel("Lift")
    axes[0].legend(loc="best")
    fig.suptitle(title, y=1.03)
    plt.tight_layout()

    if show:
        if _supports_interactive_show():
            plt.show()
        else:
            LOGGER.debug("Skipping plot display for non-interactive backend '%s'.", plt.get_backend())

    return fig, axes


def plot_qini_comparison(
    x_axis: np.ndarray,
    qini_random: np.ndarray,
    qini_perfect: np.ndarray,
    qini_curves: Mapping[str, np.ndarray],
    best_model: str,
    show: bool = True,
    title: str = "Qini Curves: T vs X vs R Learners",
) -> tuple:
    """Plot random, perfect, and model Qini curves on a single chart."""
    color_map = {
        "T-Learner": "#4C78A8",
        "X-Learner": "#E45756",
        "R-Learner": "#72B7B2",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_axis, qini_random, label="Random Baseline", color="#9E9E9E", linestyle="--", linewidth=2)
    ax.plot(x_axis, qini_perfect, label="Perfect (Oracle)", color="#54A24B", linewidth=2.5)

    for model_name, curve in qini_curves.items():
        ax.plot(
            x_axis,
            curve,
            label=model_name,
            color=color_map.get(model_name, "#4C78A8"),
            linewidth=2.2,
        )

    if best_model in qini_curves:
        best_color = color_map.get(best_model, "#4C78A8")
        ax.fill_between(
            x_axis,
            qini_random,
            qini_curves[best_model],
            where=(qini_curves[best_model] >= qini_random), # pyright: ignore[reportArgumentType]
            color=best_color,
            alpha=0.15,
            interpolate=True,
        )

    ax.set_title(title)
    ax.set_xlabel("Fraction of users targeted")
    ax.set_ylabel("Cumulative incremental conversions")
    ax.legend(loc="best")
    plt.tight_layout()

    if show:
        if _supports_interactive_show():
            plt.show()
        else:
            LOGGER.debug("Skipping plot display for non-interactive backend '%s'.", plt.get_backend())

    return fig, ax

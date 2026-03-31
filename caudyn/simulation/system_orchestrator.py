from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from caudyn.decision_engine import PIDPacingController, ThresholdDispatcher
from caudyn.environment import UberMarketplaceEnvironment


InferenceFn = Callable[[pd.DataFrame], pd.DataFrame]
REQUIRED_INFERENCE_COLUMNS: tuple[str, ...] = ("mu_0", "tau_hat_10", "tau_hat_20")


class MarketplaceSimulation:
    """Event-loop orchestrator for online marketplace decisioning."""

    RIDE_REVENUE = 25.0

    def __init__(
        self,
        total_budget: float = 5000.0,
        total_hours: int = 24,
        base_lambda: float = 0.1936,
        seed: int = 42,
        min_riders_per_hour: int = 800,
        max_riders_per_hour: int = 1200,
        *,
        inference_fn: InferenceFn,
    ):
        if total_budget <= 0:
            raise ValueError("total_budget must be positive")
        if total_hours <= 0:
            raise ValueError("total_hours must be positive")
        if min_riders_per_hour <= 0:
            raise ValueError("min_riders_per_hour must be positive")
        if max_riders_per_hour < min_riders_per_hour:
            raise ValueError("max_riders_per_hour must be >= min_riders_per_hour")

        self.total_budget = float(total_budget)
        self.total_hours = int(total_hours)
        self.base_lambda = float(base_lambda)
        self.min_riders_per_hour = int(min_riders_per_hour)
        self.max_riders_per_hour = int(max_riders_per_hour)

        self.env = UberMarketplaceEnvironment(seed=seed)
        self.dispatcher = ThresholdDispatcher(lambda_val=base_lambda)
        self.pacing_controller = PIDPacingController(
            lambda_base=base_lambda,
            # kp=0.005,
            # ki=0.0001,
            # kd=0.0002,
            kp=0.0006,  # Reduced by 10x
            ki=0.00005,  # Reduced to prevent integral windup on small errors
            kd=0.00025,  # Keep derivative slightly high to dampen sudden traffic spikes
        )

        self.target_spend_curve = [
            self.total_budget * (hour / self.total_hours)
            for hour in range(1, self.total_hours + 1)
        ]

        self._rng = np.random.default_rng(seed + 101)
        self._inference_fn = inference_fn

        self._treatment_face_values = {1: 2.50, 2: 5.00}
        self._hourly_records: list[dict[str, float | int]] = []
        self._cumulative_actual_spend = 0.0
        self._cumulative_control_money = 0.0
        self._cumulative_engine_money = 0.0
        self.logger = logging.getLogger(__name__)

    def _generate_hourly_arrivals(self, rider_count: int) -> pd.DataFrame:
        contexts = [self.env._generate_user_context() for _ in range(rider_count)]
        return pd.DataFrame(contexts)

    @staticmethod
    def _validate_inference_output(scored_df: pd.DataFrame) -> None:
        missing_cols = [
            col for col in REQUIRED_INFERENCE_COLUMNS if col not in scored_df.columns
        ]
        if missing_cols:
            raise ValueError(
                "Inference output is missing required columns: "
                + ", ".join(missing_cols)
            )

    def _run_inference(self, riders_df: pd.DataFrame) -> pd.DataFrame:
        scored_df = self._inference_fn(riders_df.copy())
        self._validate_inference_output(scored_df)
        return scored_df

    def _dispatch_and_resolve(
        self,
        scored_df: pd.DataFrame,
    ) -> tuple[float, int, dict[int, int], float, float]:
        hourly_spend = 0.0
        hourly_conversions = 0
        hourly_control_money = 0.0
        hourly_engine_money = 0.0
        treatment_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}

        model_cols = [
            "recency",
            "frequency",
            "weather_active",
            "surge_multiplier",
            "mu_0",
            "tau_hat_10",
            "tau_hat_20",
        ]
        for rider in scored_df[model_cols].to_dict(orient="records"):
            rider_payload = {
                "user_id": f"h{len(self._hourly_records) + 1}_r{self._rng.integers(1_000_000_000)}",
                "mu_0": float(rider["mu_0"]),
                "treatments": [
                    {
                        "treatment_id": 1,
                        "tau_hat": float(rider["tau_hat_10"]),
                        "face_value": self._treatment_face_values[1],
                    },
                    {
                        "treatment_id": 2,
                        "tau_hat": float(rider["tau_hat_20"]),
                        "face_value": self._treatment_face_values[2],
                    },
                ],
            }

            decision = self.dispatcher.dispatch(rider_payload)
            assigned_treatment = int(decision["assigned_treatment"])

            if assigned_treatment not in treatment_counts:
                treatment_counts[assigned_treatment] = 0
            treatment_counts[assigned_treatment] += 1

            context = {
                "recency": int(rider["recency"]),
                "frequency": int(rider["frequency"]),
                "weather_active": int(rider["weather_active"]),
                "surge_multiplier": float(rider["surge_multiplier"]),
            }

            true_prob = self.env._calculate_true_conversion(context, assigned_treatment)
            converted = int(self.env.np_random.binomial(1, true_prob))

            # Counterfactual baseline: expected money if this same rider always received control.
            control_prob = self.env._calculate_true_conversion(context, 0)
            hourly_control_money += self.RIDE_REVENUE * control_prob

            if converted:
                hourly_conversions += 1
                hourly_engine_money += self.RIDE_REVENUE
                if assigned_treatment in self._treatment_face_values:
                    hourly_spend += self._treatment_face_values[assigned_treatment]

        return (
            hourly_spend,
            hourly_conversions,
            treatment_counts,
            hourly_control_money,
            hourly_engine_money,
        )

    def _print_hourly_header(self) -> None:
        self.logger.info(
            "hour | riders | target_spend | actual_spend | conversions | lambda | t0 | t1 | t2"
        )
        self.logger.info(
            "-----+--------+--------------+--------------+-------------+--------+----+----+----"
        )

    def _print_hourly_row(self, hourly_record: dict[str, float | int]) -> None:
        self.logger.info(
            f"{int(hourly_record['hour']):>4d} | "
            f"{int(hourly_record['riders']):>6d} | "
            f"{float(hourly_record['target_spend']):>12.2f} | "
            f"{float(hourly_record['actual_spend']):>12.2f} | "
            f"{int(hourly_record['conversions']):>11d} | "
            f"{float(hourly_record['lambda']):>6.4f} | "
            f"{int(hourly_record['t0']):>2d} | "
            f"{int(hourly_record['t1']):>2d} | "
            f"{int(hourly_record['t2']):>2d}"
        )

    @staticmethod
    def _supports_interactive_show() -> bool:
        backend = plt.get_backend().lower()

        try:
            from matplotlib.backends.registry import BackendFilter, backend_registry

            interactive_backends = {
                name.lower()
                for name in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
            }
            return backend in interactive_backends
        except Exception:
            non_interactive_backends = {
                "agg",
                "cairo",
                "pdf",
                "pgf",
                "ps",
                "svg",
                "template",
            }
            return backend not in non_interactive_backends

    def plot_time_series(
        self,
        summary_df: pd.DataFrame | None = None,
        show: bool = True,
        save_path: str | Path | None = None,
    ) -> tuple:
        """Plot spend, lambda, and treatment dynamics across simulation time."""
        if summary_df is None:
            summary_df = pd.DataFrame(self._hourly_records)

        if summary_df.empty:
            raise ValueError("No simulation history is available to plot.")

        hours = summary_df["hour"].to_numpy(dtype=int)
        target_spend = summary_df["target_spend"].to_numpy(dtype=float)
        actual_spend = summary_df["actual_spend"].to_numpy(dtype=float)
        lambda_series = summary_df["lambda"].to_numpy(dtype=float)
        conversions = summary_df["conversions"].to_numpy(dtype=float)
        t0 = summary_df["t0"].to_numpy(dtype=float)
        t1 = summary_df["t1"].to_numpy(dtype=float)
        t2 = summary_df["t2"].to_numpy(dtype=float)

        fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

        axes[0].plot(
            hours,
            target_spend,
            label="Target cumulative spend",
            color="#4C78A8",
            linewidth=2.2,
        )
        axes[0].plot(
            hours,
            actual_spend,
            label="Actual cumulative spend",
            color="#F58518",
            linewidth=2.2,
        )
        axes[0].axhline(
            self.total_budget,
            color="#54A24B",
            linestyle=":",
            linewidth=2,
            label="Total budget",
        )
        axes[0].set_ylabel("Spend")
        axes[0].yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        axes[0].set_title("Budget Pacing Over Time")
        axes[0].legend(loc="best")
        axes[0].grid(alpha=0.25)

        axes[1].plot(hours, lambda_series, color="#E45756", linewidth=2.2)
        axes[1].axhline(
            self.base_lambda,
            color="#9E9E9E",
            linestyle="--",
            linewidth=1.5,
            label="Base lambda",
        )
        axes[1].set_ylabel("Lambda")
        axes[1].set_title("Controller Output (Shadow Price)")
        axes[1].legend(loc="best")
        axes[1].grid(alpha=0.25)

        axes[2].stackplot(
            hours,
            t0,
            t1,
            t2,
            labels=["Treatment 0", "Treatment 1", "Treatment 2"],
            colors=["#9E9E9E", "#72B7B2", "#F58518"],
            alpha=0.85,
        )
        conv_axis = axes[2].twinx()
        conv_axis.plot(
            hours,
            conversions,
            color="#4C78A8",
            linewidth=2.0,
            marker="o",
            label="Conversions",
        )

        axes[2].set_ylabel("Assignments")
        conv_axis.set_ylabel("Conversions")
        axes[2].set_xlabel("Hour")
        axes[2].set_title("Treatment Mix and Conversions")
        axes[2].grid(alpha=0.25)

        handles_left, labels_left = axes[2].get_legend_handles_labels()
        handles_right, labels_right = conv_axis.get_legend_handles_labels()
        axes[2].legend(
            handles_left + handles_right, labels_left + labels_right, loc="upper left"
        )

        axes[2].set_xticks(hours)

        fig.suptitle("Marketplace Simulation Time Series", y=0.995)
        plt.tight_layout()

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            self.logger.info(
                "Saved simulation time-series plot to %s", output_path.as_posix()
            )

        if show:
            if self._supports_interactive_show():
                plt.show()
            else:
                self.logger.debug(
                    "Skipping simulation plot display for non-interactive backend '%s'.",
                    plt.get_backend(),
                )

        return fig, axes

    def run_simulation(self) -> pd.DataFrame:
        self._hourly_records = []
        self._cumulative_actual_spend = 0.0
        self._cumulative_control_money = 0.0
        self._cumulative_engine_money = 0.0
        self.dispatcher.lambda_val = self.base_lambda
        self.pacing_controller.integral_error = 0.0
        self.pacing_controller.prev_error = 0.0

        self._print_hourly_header()

        for hour in range(1, self.total_hours + 1):
            rider_count = int(
                self._rng.integers(
                    low=self.min_riders_per_hour,
                    high=self.max_riders_per_hour + 1,
                )
            )

            riders_df = self._generate_hourly_arrivals(rider_count)
            scored_df = self._run_inference(riders_df)

            (
                hourly_spend,
                conversions,
                treatment_counts,
                hourly_control_money,
                hourly_engine_money,
            ) = self._dispatch_and_resolve(scored_df)
            self._cumulative_actual_spend += hourly_spend
            self._cumulative_control_money += hourly_control_money
            self._cumulative_engine_money += hourly_engine_money

            target_spend = self.target_spend_curve[hour - 1]
            new_lambda = self.pacing_controller.update(
                actual_spend=self._cumulative_actual_spend,
                target_spend=target_spend,
            )
            self.dispatcher.lambda_val = new_lambda

            hourly_record: dict[str, float | int] = {
                "hour": hour,
                "riders": rider_count,
                "target_spend": target_spend,
                "actual_spend": self._cumulative_actual_spend,
                "control_money": self._cumulative_control_money,
                "engine_money": self._cumulative_engine_money,
                "engine_net_money": self._cumulative_engine_money
                - self._cumulative_actual_spend,
                "conversions": conversions,
                "lambda": new_lambda,
                "t0": treatment_counts.get(0, 0),
                "t1": treatment_counts.get(1, 0),
                "t2": treatment_counts.get(2, 0),
            }
            self._hourly_records.append(hourly_record)
            self._print_hourly_row(hourly_record)

        summary_df = pd.DataFrame(self._hourly_records)

        total_conversions = int(summary_df["conversions"].sum())
        final_spend = float(summary_df["actual_spend"].iloc[-1])
        control_money = float(summary_df["control_money"].iloc[-1])
        engine_money = float(summary_df["engine_money"].iloc[-1])
        engine_net_money = float(summary_df["engine_net_money"].iloc[-1])
        net_delta_vs_control = engine_net_money - control_money
        final_lambda = float(summary_df["lambda"].iloc[-1])
        print("\nSimulation complete")
        print(
            f"Final spend: ${final_spend:,.2f} / ${self.total_budget:,.2f} | "
            f"Total conversions: {total_conversions:,d} | Final lambda: {final_lambda:.4f}"
        )
        print(
            "Financial summary | "
            f"Control-only money (no treatment): ${control_money:,.2f} | "
            f"Decision-engine money (gross): ${engine_money:,.2f} | "
            f"Decision-engine money (net): ${engine_net_money:,.2f} | "
            f"Net delta vs control: ${net_delta_vs_control:,.2f}"
        )

        return summary_df


if __name__ == "__main__":
    raise SystemExit(
        "MarketplaceSimulation requires a model-backed inference function. "
        "Run `python -m caudyn.run_causal_experiment --run-simulation` instead."
    )

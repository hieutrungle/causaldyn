from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class UnitSelector:
    """Deterministically clean and score treatment rows for downstream optimization.

    The unit selector converts raw causal model outputs into financially meaningful
    features while pruning rows that cannot improve an allocation decision.

    Attributes:
        user_id_col: Column containing a unique user identifier.
        treatment_col: Column containing treatment identifiers or tiers.
        base_prob_col: Baseline conversion probability under control (mu_0).
        cate_col: Predicted CATE / incremental lift (tau_hat).
        cost_col: Face-value treatment cost column.
    """

    _PROB_EPS = 1e-6
    _COST_EPS = 1e-6
    _TREATMENT_PROB_COL = "mu_1"
    _EXPECTED_COST_COL = "expected_cost"
    _ROI_COL = "roi"
    _ROW_ORDER_COL = "__unit_selector_row_order__"

    def __init__(
        self,
        user_id_col: str = "user_id",
        treatment_col: str = "treatment",
        base_prob_col: str = "mu_0",
        cate_col: str = "tau_hat",
        cost_col: str = "face_value",
    ):
        self.user_id_col = user_id_col
        self.treatment_col = treatment_col
        self.base_prob_col = base_prob_col
        self.cate_col = cate_col
        self.cost_col = cost_col

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all unit-selector transformations in a deterministic sequence.

        Args:
            df: Input dataframe containing user-treatment rows and model outputs.

        Returns:
            A cleaned dataframe with positive-lift units and additional columns:
            `mu_1`, `expected_cost`, and `roi`.
        """
        self._validate_required_columns(df)

        out = df.copy()
        out = self._clamp_probabilities(out)
        out = self._prune_unprofitable_units(out)
        out = self._calculate_expected_cost(out)
        out = self._calculate_roi(out)
        out = self._prune_dominated_treatments(out)
        out = out.reset_index(drop=True)

        LOGGER.info("Unit selector completed with %s remaining rows.", self._fmt_int(len(out)))
        return out

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        required_columns = {
            self.user_id_col,
            self.treatment_col,
            self.base_prob_col,
            self.cate_col,
            self.cost_col,
        }
        missing = sorted(required_columns.difference(df.columns))
        if missing:
            missing_fmt = ", ".join(missing)
            raise ValueError(f"Missing required columns: {missing_fmt}")

    def _clamp_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clamp model probabilities to (0, 1), then re-align CATE with bounds."""
        out = df.copy()

        mu_0_raw = out[self.base_prob_col].to_numpy(dtype=float, copy=True)
        tau_raw = out[self.cate_col].to_numpy(dtype=float, copy=True)
        mu_1_raw = mu_0_raw + tau_raw

        mu_0 = np.clip(mu_0_raw, self._PROB_EPS, 1.0 - self._PROB_EPS)
        mu_1 = np.clip(mu_1_raw, self._PROB_EPS, 1.0 - self._PROB_EPS)
        tau_hat = mu_1 - mu_0

        out[self.base_prob_col] = mu_0
        out[self._TREATMENT_PROB_COL] = mu_1
        out[self.cate_col] = tau_hat

        clamped_rows = int(np.count_nonzero((mu_0 != mu_0_raw) | (mu_1 != mu_1_raw)))
        LOGGER.info(
            "Probability clamping adjusted %s rows.",
            self._fmt_int(clamped_rows),
        )
        return out

    def _prune_unprofitable_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows that have non-positive incremental lift (tau_hat <= 0)."""
        before_rows = len(df)
        out = df[df[self.cate_col] > 0.0].copy()
        dropped = before_rows - len(out)

        LOGGER.info(
            "Dropped %s rows due to %s <= 0.",
            self._fmt_int(dropped),
            self.cate_col,
        )
        return out

    def _calculate_expected_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute expected cost as face value times treatment conversion probability."""
        out = df.copy()
        mu_1 = out[self._TREATMENT_PROB_COL].to_numpy(dtype=float, copy=True)
        face_value = out[self.cost_col].to_numpy(dtype=float, copy=True)

        expected_cost = face_value * mu_1
        zero_mask = expected_cost == 0.0
        if np.any(zero_mask):
            expected_cost = np.where(zero_mask, self._COST_EPS, expected_cost)

        out[self._EXPECTED_COST_COL] = expected_cost
        LOGGER.info(
            "Expected-cost guardrail applied to %s zero-cost rows.",
            self._fmt_int(int(np.count_nonzero(zero_mask))),
        )
        return out

    def _calculate_roi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute normalized ROI as incremental lift per expected dollar."""
        out = df.copy()
        tau_hat = out[self.cate_col].to_numpy(dtype=float, copy=True)
        expected_cost = out[self._EXPECTED_COST_COL].to_numpy(dtype=float, copy=True)
        out[self._ROI_COL] = tau_hat / expected_cost
        return out

    def _prune_dominated_treatments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop higher-cost rows dominated by lower-cost rows for the same user.

        A treatment is dominated when there exists another treatment with a lower
        cost and an equal-or-higher CATE for that same user.
        """
        if df.empty:
            LOGGER.info("Dropped 0 dominated treatment rows.")
            return df

        out = df.copy()
        out[self._ROW_ORDER_COL] = np.arange(len(out), dtype=int)
        out = out.sort_values(
            by=[self.user_id_col, self.cost_col, self._ROW_ORDER_COL],
            ascending=[True, True, True],
            kind="mergesort",
        )

        by_cost = out.groupby([self.user_id_col, self.cost_col], as_index=False)[self.cate_col].max()
        cumulative_best = by_cost.groupby(self.user_id_col)[self.cate_col].cummax()
        by_cost["best_lower_tau"] = cumulative_best.groupby(by_cost[self.user_id_col]).shift(1)

        out = out.merge(
            by_cost[[self.user_id_col, self.cost_col, "best_lower_tau"]],
            on=[self.user_id_col, self.cost_col],
            how="left",
        )

        dominated_mask = out["best_lower_tau"].notna() & (out[self.cate_col] <= out["best_lower_tau"])
        dropped = int(np.count_nonzero(dominated_mask))

        out = out.loc[~dominated_mask].copy()
        out = out.sort_values(self._ROW_ORDER_COL, kind="mergesort")
        out = out.drop(columns=["best_lower_tau", self._ROW_ORDER_COL])

        LOGGER.info("Dropped %s dominated treatment rows.", self._fmt_int(dropped))
        return out

    @staticmethod
    def _fmt_int(value: int) -> str:
        return f"{value:,d}"


def _build_demo_frame() -> pd.DataFrame:
    """Create a compact demo frame with clamping and pruning edge cases."""
    return pd.DataFrame(
        [
            {"user_id": "u1", "treatment": "promo_5", "mu_0": 0.10, "tau_hat": 0.05, "face_value": 5.0},
            {"user_id": "u1", "treatment": "promo_10", "mu_0": 0.10, "tau_hat": 0.04, "face_value": 10.0},
            {"user_id": "u2", "treatment": "promo_5", "mu_0": 1.20, "tau_hat": 0.10, "face_value": 5.0},
            {"user_id": "u3", "treatment": "promo_5", "mu_0": -0.20, "tau_hat": 0.30, "face_value": 5.0},
            {"user_id": "u4", "treatment": "promo_5", "mu_0": 0.60, "tau_hat": -0.10, "face_value": 5.0},
            {"user_id": "u5", "treatment": "promo_0", "mu_0": 0.20, "tau_hat": 0.90, "face_value": 0.0},
            {"user_id": "u5", "treatment": "promo_10", "mu_0": 0.20, "tau_hat": 0.85, "face_value": 10.0},
        ]
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    selector = UnitSelector()
    demo_df = _build_demo_frame()

    LOGGER.info("Demo input rows: %s", selector._fmt_int(len(demo_df)))
    selected_df = selector.fit_transform(demo_df)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print("\nFinal Unit Selector Output:\n")
    print(selected_df.to_string(index=False))

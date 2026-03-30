from __future__ import annotations

import logging
from typing import cast

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sp

LOGGER = logging.getLogger(__name__)


class ValueOptimizer:
    """Solve fractional treatment allocation under budget and user exclusivity.

    This optimizer solves a linear program:

    maximize    sum_i tau_i * x_i
    subject to  sum_i cost_i * x_i <= B
                sum_{i in user u} x_i <= 1, for each user u
                0 <= x_i <= 1

    where x_i is the selected fraction for candidate row i.

    Attributes:
        budget: Global expected-cost budget.
        user_col: User identifier column.
        cate_col: Predicted incremental lift column.
        cost_col: Expected spend column.
    """

    def __init__(
        self,
        budget: float,
        user_col: str = "user_id",
        cate_col: str = "tau_hat",
        cost_col: str = "expected_cost",
    ):
        if budget < 0:
            raise ValueError("budget must be non-negative")

        self.budget = float(budget)
        self.user_col = user_col
        self.cate_col = cate_col
        self.cost_col = cost_col

    def optimize(self, df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Optimize fractional allocations and return shadow price.

        Args:
            df: Cleaned dataframe containing candidate user-treatment rows.

        Returns:
            A tuple with:
            1. Dataframe augmented with `optimal_fraction`.
            2. Positive budget shadow price lambda from the dual value.

        Raises:
            ValueError: If required columns are missing or data is invalid.
            RuntimeError: If the optimization fails to solve optimally.
        """
        self._validate_input(df)

        n_rows = len(df)
        tau_hat = df[self.cate_col].to_numpy(dtype=float, copy=True)
        expected_cost = df[self.cost_col].to_numpy(dtype=float, copy=True)

        x = cp.Variable(n_rows)

        objective = cp.Maximize(tau_hat @ x)
        budget_constraint = expected_cost @ x <= self.budget

        a_users = self._build_user_incidence_matrix(df[self.user_col])
        a_users_cvx = cp.Constant(a_users)
        user_constraint = a_users_cvx @ x <= 1.0

        problem = cp.Problem(
            objective,
            [budget_constraint, user_constraint, x >= 0.0, x <= 1.0],
        )

        try:
            problem.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            LOGGER.warning("CLARABEL unavailable, falling back to cvxpy default solver.")
            problem.solve()

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"Optimization did not converge to an optimal solution. Status: {problem.status}")

        if x.value is None:
            raise RuntimeError("Optimization succeeded but no primal solution was returned.")

        if budget_constraint.dual_value is None:
            raise RuntimeError("Optimization succeeded but budget dual value is unavailable.")

        optimal_fraction = np.asarray(x.value, dtype=float).reshape(-1)
        optimal_fraction = np.clip(optimal_fraction, 0.0, 1.0)
        optimal_fraction = np.where(np.abs(optimal_fraction) < 1e-8, 0.0, optimal_fraction)
        optimal_fraction = np.where(np.abs(optimal_fraction - 1.0) < 1e-8, 1.0, optimal_fraction)
        shadow_price = float(abs(np.asarray(budget_constraint.dual_value, dtype=float).reshape(-1)[0]))

        out = df.copy()
        out["optimal_fraction"] = optimal_fraction

        if problem.value is None:
            objective_value = float("nan")
        else:
            objective_value = float(cast(float, problem.value))
        LOGGER.info("Optimization status: %s", problem.status)
        LOGGER.info("Objective value: %.6f", objective_value)
        LOGGER.info("Extracted budget shadow price lambda: %.6f", shadow_price)
        return out, shadow_price

    def _build_user_incidence_matrix(self, user_series: pd.Series) -> sp.csr_matrix:
        """Build sparse user-incidence matrix for exclusivity constraints.

        Args:
            user_series: Series aligned with dataframe rows containing user ids.

        Returns:
            CSR matrix A where A[u, i] = 1 if row i belongs to user u, else 0.
        """
        user_codes, unique_users = pd.factorize(user_series, sort=False)
        n_rows = len(user_series)
        n_users = len(unique_users)

        data = np.ones(n_rows, dtype=float)
        row_indices = user_codes.astype(np.int64, copy=False)
        col_indices = np.arange(n_rows, dtype=np.int64)

        return sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_users, n_rows))

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate schema and numerical columns before optimization."""
        required = {self.user_col, self.cate_col, self.cost_col}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        if df.empty:
            raise ValueError("Input dataframe is empty. Optimization requires at least one candidate row.")

        tau_hat = df[self.cate_col].to_numpy(dtype=float, copy=True)
        expected_cost = df[self.cost_col].to_numpy(dtype=float, copy=True)

        if not np.isfinite(tau_hat).all():
            raise ValueError(f"Column {self.cate_col} must contain only finite numeric values.")

        if not np.isfinite(expected_cost).all():
            raise ValueError(f"Column {self.cost_col} must contain only finite numeric values.")

        if np.any(expected_cost < 0.0):
            raise ValueError(f"Column {self.cost_col} must be non-negative.")


def _build_demo_frame() -> pd.DataFrame:
    """Create a compact demo with multi-treatment users and a tight budget."""
    return pd.DataFrame(
        [
            {"user_id": "A", "treatment": "promo_t1", "tau_hat": 0.060, "expected_cost": 1.00},
            {"user_id": "A", "treatment": "promo_t2", "tau_hat": 0.090, "expected_cost": 1.80},
            {"user_id": "B", "treatment": "promo_t1", "tau_hat": 0.050, "expected_cost": 0.90},
            {"user_id": "B", "treatment": "promo_t2", "tau_hat": 0.110, "expected_cost": 2.20},
            {"user_id": "C", "treatment": "promo_t1", "tau_hat": 0.040, "expected_cost": 0.80},
        ]
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    demo_df = _build_demo_frame()
    optimizer = ValueOptimizer(budget=2.50)

    solved_df, lambda_shadow = optimizer.optimize(demo_df)
    solved_df = solved_df.sort_values(["user_id", "treatment"]).reset_index(drop=True)

    print("\nOptimal Fractional Allocation:\n")
    print(solved_df.to_string(index=False))

    per_user_fraction = solved_df.groupby("user_id", as_index=False)["optimal_fraction"].sum()
    print("\nPer-user allocation sums (must be <= 1):\n")
    print(per_user_fraction.to_string(index=False))

    tolerance = 1e-8
    if (per_user_fraction["optimal_fraction"] > 1.0 + tolerance).any():
        raise RuntimeError("Mutually exclusive treatment constraint was violated.")

    print(f"\nBudget shadow price lambda: {lambda_shadow:.6f}")

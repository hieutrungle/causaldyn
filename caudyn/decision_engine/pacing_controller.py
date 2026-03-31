from __future__ import annotations

import logging
import time


class PIDPacingController:
    """PID controller for online budget pacing via dynamic lambda updates.

    The controller adjusts the dispatch hurdle `lambda` based on cumulative spend
    error:

        error = actual_spend - target_spend

    Positive error means overspending, so lambda should increase to make promo
    assignment stricter.

    Integral anti-windup is enforced by clamping the integral state such that the
    integral contribution `ki * integral_error` always remains within
    `[-integral_limit, integral_limit]`.
    """

    def __init__(
        self,
        lambda_base: float,
        kp: float,
        ki: float,
        kd: float,
        lambda_min: float = 0.01,
        lambda_max: float = 1.0,
        integral_limit: float = 0.5,
    ):
        """Initialize controller gains, bounds, and internal state.

        Args:
            lambda_base: Baseline shadow price from offline optimization.
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
            lambda_min: Lower saturation bound for lambda.
            lambda_max: Upper saturation bound for lambda.
            integral_limit: Max absolute magnitude allowed for the integral term.

        Raises:
            ValueError: If configuration values are invalid.
        """
        if lambda_base < 0:
            raise ValueError("lambda_base must be non-negative")
        if lambda_min <= 0:
            raise ValueError("lambda_min must be strictly positive")
        if lambda_max <= lambda_min:
            raise ValueError("lambda_max must be greater than lambda_min")
        if integral_limit < 0:
            raise ValueError("integral_limit must be non-negative")

        self.lambda_base = float(lambda_base)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.integral_limit = float(integral_limit)

        self.integral_error = 0.0
        self.prev_error = 0.0
        self.logger = logging.getLogger(__name__)

    def update(self, actual_spend: float, target_spend: float) -> float:
        """Compute the next lambda value from spend tracking error.

        Args:
            actual_spend: Observed cumulative spend so far.
            target_spend: Desired cumulative spend at this time point.

        Returns:
            The clamped lambda value to use in the decision threshold.
        """
        error = float(actual_spend) - float(target_spend)

        p_term = self.kp * error

        self.integral_error += error
        if self.ki != 0.0:
            max_abs_integral_error = self.integral_limit / abs(self.ki)
            self.integral_error = max(
                -max_abs_integral_error,
                min(max_abs_integral_error, self.integral_error),
            )

        i_term = self.ki * self.integral_error

        d_term = self.kd * (error - self.prev_error)
        self.prev_error = error

        new_lambda = self.lambda_base + p_term + i_term + d_term
        clamped_lambda = max(self.lambda_min, min(self.lambda_max, new_lambda))

        self.logger.info(
            "PID update | error=%.2f | P=%.4f I=%.4f D=%.4f | lambda=%.4f",
            error,
            p_term,
            i_term,
            d_term,
            clamped_lambda,
        )
        return clamped_lambda


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    controller = PIDPacingController(
        lambda_base=0.1936,
        kp=0.00045,
        ki=0.0001,
        kd=0.0009,
        lambda_min=0.01,
        lambda_max=1.0,
        integral_limit=0.05,
    )

    total_hours = 24
    hourly_target_increment = 100.0
    demand_shock_hour = 10
    demand_shock_overspend = 500.0

    target_spend = 0.0
    actual_spend = 0.0
    lambda_now = controller.lambda_base

    print("Hourly pacing simulation (24 steps):")
    print(
        "hour | timestamp    | target  | actual  | error   | lambda | note"
    )
    print("-----+--------------+---------+---------+---------+--------+-----------------------------")

    for hour in range(1, total_hours + 1):
        target_spend = hourly_target_increment * hour
        note = ""

        if hour == demand_shock_hour:
            actual_spend = target_spend + demand_shock_overspend
            note = "DEMAND SHOCK (+$500 overspend)"
        else:
            # Simple marketplace response model:
            # when lambda rises above baseline, promo approvals tighten and spend drops.
            lambda_gap = max(0.0, lambda_now - controller.lambda_base)
            suppression = min(0.95, 7.0 * lambda_gap)
            hourly_spend = hourly_target_increment * (1.0 - suppression)
            actual_spend += hourly_spend

        lambda_now = controller.update(actual_spend=actual_spend, target_spend=target_spend)
        error = actual_spend - target_spend

        print(
            f"{hour:>4d} | {time.time():>12.3f} | "
            f"{target_spend:>7.2f} | {actual_spend:>7.2f} | {error:>7.2f} | "
            f"{lambda_now:>6.4f} | {note}"
        )

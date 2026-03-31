from __future__ import annotations

import logging
import time
from typing import Any, Dict


class ThresholdDispatcher:
    """Serve real-time promo decisions using a shadow-price hurdle.

    This class evaluates each available treatment for a rider using the net causal
    value equation:

        V = bounded_tau - (lambda_val * expected_cost)

    where:
    - bounded_tau = mu_1 - mu_0
    - mu_1 = clip(mu_0 + tau_hat, 0, 1)
    - expected_cost = face_value * mu_1

    The dispatcher returns treatment `0` (control / no promo) unless at least one
    treatment has positive net causal value and exceeds the current maximum.
    """

    def __init__(self, lambda_val: float):
        """Initialize dispatcher.

        Args:
            lambda_val: Global shadow price (hurdle rate) from offline optimization.

        Raises:
            ValueError: If `lambda_val` is negative.
        """
        if lambda_val < 0:
            raise ValueError("lambda_val must be non-negative")

        self.lambda_val = float(lambda_val)
        self.logger = logging.getLogger(__name__)

    def dispatch(self, rider_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Assign an online treatment for one rider.

        Input schema:
            {
                "user_id": "user_123",
                "mu_0": 0.15,
                "treatments": [
                    {"treatment_id": 1, "tau_hat": 0.05, "face_value": 2.50},
                    {"treatment_id": 2, "tau_hat": 0.12, "face_value": 5.00}
                ]
            }

        Output schema:
            {
                "user_id": "user_123",
                "timestamp": 1678886400.123,
                "lambda_used": 0.1936,
                "assigned_treatment": 2,
                "net_causal_value": 0.0232,
            }

        Resilience behavior:
        - Any exception in parsing or arithmetic returns immediate control treatment
          with zero net value and logs the error.

        Args:
            rider_predictions: Real-time inference payload for one rider.

        Returns:
            Decision telemetry dictionary.
        """
        timestamp = time.time()
        user_id = str(rider_predictions.get("user_id", "unknown"))

        fallback = {
            "user_id": user_id,
            "timestamp": timestamp,
            "lambda_used": self.lambda_val,
            "assigned_treatment": 0,
            "net_causal_value": 0.0,
        }

        try:
            mu_0 = float(rider_predictions["mu_0"])
            mu_0 = max(0.0, min(1.0, mu_0))

            best_treatment = 0
            max_v = 0.0

            treatments = rider_predictions["treatments"]
            for treatment in treatments:
                treatment_id = int(treatment["treatment_id"])
                tau_hat = float(treatment["tau_hat"])
                face_value = float(treatment["face_value"])

                mu_1 = max(0.0, min(1.0, mu_0 + tau_hat))
                bounded_tau = mu_1 - mu_0
                expected_cost = face_value * mu_1
                value = bounded_tau - (self.lambda_val * expected_cost)

                if value > max_v:
                    max_v = value
                    best_treatment = treatment_id

            return {
                "user_id": user_id,
                "timestamp": timestamp,
                "lambda_used": self.lambda_val,
                "assigned_treatment": best_treatment,
                "net_causal_value": max_v,
            }
        except Exception as exc:
            self.logger.error(
                "Dispatch failed for user_id=%s; defaulting to control. Error: %s",
                user_id,
                exc,
            )
            return fallback


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    dispatcher = ThresholdDispatcher(lambda_val=0.1936)

    # Rider A: high baseline propensity with small incremental lift.
    # Expectation: no promo should pass the hurdle.
    rider_a = {
        "user_id": "rider_a_sure_thing",
        "mu_0": 0.85,
        "treatments": [
            {"treatment_id": 1, "tau_hat": 0.03, "face_value": 2.50},
            {"treatment_id": 2, "tau_hat": 0.05, "face_value": 5.00},
        ],
    }

    # Rider B: low baseline propensity with strong treatment-2 lift.
    # Expectation: treatment 2 should pass the hurdle and win.
    rider_b = {
        "user_id": "rider_b_persuadable",
        "mu_0": 0.10,
        "treatments": [
            {"treatment_id": 1, "tau_hat": 0.05, "face_value": 2.50},
            {"treatment_id": 2, "tau_hat": 0.90, "face_value": 4.50},
        ],
    }

    result_a = dispatcher.dispatch(rider_a)
    result_b = dispatcher.dispatch(rider_b)

    print("Rider A decision:")
    print(result_a)
    print("\nRider B decision:")
    print(result_b)

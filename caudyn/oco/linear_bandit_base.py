"""Shared base class for linear contextual bandit agents."""

import numpy as np


class LinearBanditAgentBase:
    """Shared state/update logic for linear contextual bandit agents."""

    def __init__(self, n_actions, n_features, prior_weight=1.0):
        self.n_actions = n_actions
        self.n_features = n_features

        # A higher prior_weight means the bandit is less likely to overwrite
        # offline intelligence during early exploration.
        self.A = [np.identity(n_features) * prior_weight for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]

    def inject_linear_priors(self, prior_thetas, prior_weight=1.0):
        """Injects per-arm linear priors directly into agent state.

        Args:
            prior_thetas: Dict/list mapping arm index -> theta vector.
            prior_weight: Pseudo-count confidence for initializing A matrices.
        """
        prior_weight = float(prior_weight)
        if prior_weight <= 0:
            raise ValueError("prior_weight must be positive")

        for a in range(self.n_actions):
            theta = np.asarray(prior_thetas[a], dtype=float)
            if theta.shape != (self.n_features,):
                raise ValueError(
                    f"Prior theta for arm {a} must have shape ({self.n_features},), got {theta.shape}"
                )

            self.A[a] = np.identity(self.n_features) * prior_weight
            self.b[a] = self.A[a].dot(theta)

    def _get_context_vector(self, state_dict):
        """Converts state dictionary into normalized feature vector."""
        return np.array(
            [
                1.0,
                state_dict["recency"] / 30.0,
                state_dict["frequency"] / 20.0,
                state_dict["weather_active"],
                state_dict["surge_multiplier"] / 3.0,
            ]
        )

    def choose_action(self, state_dict):
        raise NotImplementedError("Subclasses must implement choose_action")

    def update(self, action, state_dict, reward):
        """Updates covariance matrix and reward vector."""
        x = self._get_context_vector(state_dict)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

    def get_learned_weights(self, action):
        """Returns learned linear weights for one action."""
        A_inv = np.linalg.inv(self.A[action])
        return A_inv.dot(self.b[action])

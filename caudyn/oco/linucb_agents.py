"""LinUCB agent definitions.

This module contains only algorithm/agent state and update logic.
"""

import numpy as np
from .linear_bandit_base import LinearBanditAgentBase


class LinUCBAgent(LinearBanditAgentBase):
    """Original LinUCB agent with explicit covariance matrices."""

    def __init__(self, n_actions, n_features, alpha=1.0, prior_weight=1.0):
        super().__init__(
            n_actions=n_actions, n_features=n_features, prior_weight=prior_weight
        )
        self.alpha = alpha

    def choose_action(self, state_dict):
        x = self._get_context_vector(state_dict)
        ucb_scores = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv.dot(self.b[a])
            expected_reward = theta_a.dot(x)
            exploration_bonus = self.alpha * np.sqrt(x.dot(A_inv).dot(x))
            ucb_scores[a] = expected_reward + exploration_bonus

        max_ucb = np.max(ucb_scores)
        best_actions = np.where(np.isclose(ucb_scores, max_ucb))[0]
        return np.random.choice(best_actions)


class FastLinUCBAgent(LinearBanditAgentBase):
    """Discounted LinUCB using Sherman-Morrison inverse updates."""

    def __init__(self, n_actions, n_features, alpha=1.0, gamma=0.995, prior_weight=1.0):
        super().__init__(
            n_actions=n_actions, n_features=n_features, prior_weight=prior_weight
        )
        self.alpha = alpha
        self.gamma = gamma
        self.A_inv = [np.linalg.inv(a_matrix) for a_matrix in self.A]

    def inject_linear_priors(self, prior_thetas, prior_weight=1.0):
        """Injects direct linear priors and refreshes inverse matrices."""
        super().inject_linear_priors(prior_thetas, prior_weight=prior_weight)
        self.A_inv = [np.linalg.inv(a_matrix) for a_matrix in self.A]

    def choose_action(self, state_dict):
        x = self._get_context_vector(state_dict)
        ucb_scores = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            A_inv_a = self.A_inv[a]
            theta_a = A_inv_a.dot(self.b[a])
            expected_reward = theta_a.dot(x)
            exploration_bonus = self.alpha * np.sqrt(x.dot(A_inv_a).dot(x))
            ucb_scores[a] = expected_reward + exploration_bonus

        max_ucb = np.max(ucb_scores)
        best_actions = np.where(np.isclose(ucb_scores, max_ucb))[0]
        return np.random.choice(best_actions)

    def update(self, action, state_dict, reward):
        """Updates inverse covariance matrix and reward vector directly."""
        x = self._get_context_vector(state_dict)

        self.b[action] = (self.gamma * self.b[action]) + (reward * x)

        M = self.A_inv[action]
        Mx = M.dot(x)
        denominator = self.gamma + x.dot(Mx)
        numerator = np.outer(Mx, Mx)
        self.A_inv[action] = (1.0 / self.gamma) * (M - (numerator / denominator))

        # Keep A synchronized for compatibility with shared/base logic.
        self.A[action] = (self.gamma * self.A[action]) + np.outer(x, x)

    def get_learned_weights(self, action):
        """Returns learned linear weights for one action."""
        return self.A_inv[action].dot(self.b[action])

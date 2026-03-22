import numpy as np
from .linear_bandit_base import LinearBanditAgentBase


class LinTSAgent(LinearBanditAgentBase):
    """Vanilla Linear Thompson Sampling Agent.

    Inherits the A and b tracking matrices from the shared linear-bandit base class, but explores by
    sampling from a Multivariate Gaussian posterior rather than calculating
    a deterministic upper confidence bound.
    """

    def __init__(
        self, n_actions, n_features, v_squared=1.0, prior_weight=1.0, random_seed=42
    ):
        # We drop 'alpha' and replace it with 'v_squared'
        super().__init__(
            n_actions=n_actions, n_features=n_features, prior_weight=prior_weight
        )

        # v_squared is the variance of our prior exploration noise
        self.v_squared = v_squared
        self.rng = np.random.RandomState(random_seed)

    def choose_action(self, state_dict):
        """Samples weights from the posterior and selects the maximizing action."""
        x = self._get_context_vector(state_dict)
        sampled_rewards = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            # 1. Invert the A matrix (Vanilla approach, O(d^3) complexity)
            A_inv = np.linalg.inv(self.A[a])

            # 2. Calculate the expected mean weights: theta_hat = A_inv * b
            theta_hat = A_inv.dot(self.b[a])

            # 3. Calculate the covariance matrix: v^2 * A_inv
            covariance = self.v_squared * A_inv

            # PRACTICAL ENGINEERING TRICK:
            # Floating-point arithmetic over thousands of steps can cause the
            # covariance matrix to lose strict positive semi-definiteness.
            # We add a microscopic value to the diagonal to ensure the Cholesky
            # decomposition inside the sampler doesn't crash.
            covariance += np.eye(self.n_features) * 1e-8

            # 4. Sample a random weight vector from the Multivariate Normal distribution
            sampled_theta = self.rng.multivariate_normal(theta_hat, covariance)

            # 5. Calculate the expected reward for this sampled reality
            sampled_rewards[a] = sampled_theta.dot(x)

        # Act optimally according to the randomly sampled weights
        max_reward = np.max(sampled_rewards)
        best_actions = np.where(np.isclose(sampled_rewards, max_reward))[0]

        # We use the local random state to break ties, ensuring reproducibility
        return self.rng.choice(best_actions)

    # We do NOT need to write update() or inject_linear_priors().
    # The shared base class handles updating self.A and self.b.


class FastLinTSAgent(LinearBanditAgentBase):
    """Fast Discounted Linear Thompson Sampling Agent.

    Optimizes performance by tracking the inverse covariance matrix directly using
    Sherman-Morrison rank-1 updates. Incorporates a discount factor (gamma) to
    gracefully 'forget' old data, making it robust to non-stationary environments.
    """

    def __init__(
        self,
        n_actions,
        n_features,
        v_squared=1.0,
        gamma=0.995,
        prior_weight=1.0,
        random_seed=42,
    ):
        super().__init__(
            n_actions=n_actions, n_features=n_features, prior_weight=prior_weight
        )

        self.v_squared = v_squared
        self.gamma = gamma
        self.rng = np.random.RandomState(random_seed)

        # Pre-compute the inverse matrices for tracking
        self.A_inv = [np.linalg.inv(a_matrix) for a_matrix in self.A]

    def inject_linear_priors(self, prior_thetas, prior_weight=1.0):
        """Injects offline priors and refreshes the inverse matrices."""
        super().inject_linear_priors(prior_thetas, prior_weight=prior_weight)
        self.A_inv = [np.linalg.inv(a_matrix) for a_matrix in self.A]

    def choose_action(self, state_dict):
        """Samples weights from the posterior using pre-computed inverse matrices."""
        x = self._get_context_vector(state_dict)
        sampled_rewards = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            A_inv_a = self.A_inv[a]

            # 1. Calculate expected mean directly using our tracked inverse
            theta_hat = A_inv_a.dot(self.b[a])

            # 2. Covariance matrix with numerical stability padding
            covariance = self.v_squared * A_inv_a
            covariance += np.eye(self.n_features) * 1e-8

            # 3. Sample a random weight vector from the posterior
            sampled_theta = self.rng.multivariate_normal(theta_hat, covariance)

            # 4. Calculate the expected reward
            sampled_rewards[a] = sampled_theta.dot(x)

        # Act optimally according to the randomly sampled weights
        max_reward = np.max(sampled_rewards)
        best_actions = np.where(np.isclose(sampled_rewards, max_reward))[0]
        return self.rng.choice(best_actions)

    def update(self, action, state_dict, reward):
        """Updates inverse covariance matrix and reward vector using Sherman-Morrison."""
        x = self._get_context_vector(state_dict)

        # Apply exponential decay (discounting) to the reward vector
        self.b[action] = (self.gamma * self.b[action]) + (reward * x)

        # Rank-1 Sherman-Morrison update for A_inv with discounting
        M = self.A_inv[action]
        Mx = M.dot(x)
        denominator = self.gamma + x.dot(Mx)
        numerator = np.outer(Mx, Mx)
        self.A_inv[action] = (1.0 / self.gamma) * (M - (numerator / denominator))

        # Keep base A synchronized for debugging and mathematical consistency
        self.A[action] = (self.gamma * self.A[action]) + np.outer(x, x)

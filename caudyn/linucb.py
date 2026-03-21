'''
Algorithm: LinUCB (Linear Upper Confidence Bound)
Inputs:
  α : Exploration parameter (a strictly positive scalar)
  d : Dimension of the context feature vector

Initialization:
  FOR each arm a ∈ A:
      A_a = Identity Matrix of size d × d, Covariance Matrix for arm $a$
      b_a = Zero Vector of size d × 1, The Reward Vector for arm $a$
  END FOR

Execution Loop:
  FOR each time step t = 1, 2, ..., T:
      
      1. Observe the user context vector x_t (size d × 1)
      
      2. FOR each available arm a ∈ A:
          // Calculate the learned weights (Ridge Regression)
          // The Causal Weights for arm $a$
          θ_a = Inverse(A_a) * b_a
          
          // Calculate the expected reward (Exploitation)
          expected_reward = Transpose(θ_a) * x_t
          
          // Calculate the confidence bound (Exploration)
          confidence_bound = α * sqrt( Transpose(x_t) * Inverse(A_a) * x_t )
          
          // Calculate the Upper Confidence Bound (UCB)
          p_a = expected_reward + confidence_bound
      END FOR
      
      3. Choose arm a_t = argmax(p_a)  // Break ties randomly
      
      4. Execute action a_t and observe the actual reward r_t
      
      5. Update the matrices for the chosen arm a_t:
          A_{a_t} = A_{a_t} + (x_t * Transpose(x_t))
          b_{a_t} = b_{a_t} + (r_t * x_t)
          
  END FOR
'''



import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import Ridge

try:
    from environment import UberMarketplaceEnvironment # Script execution mode
except ImportError:
    from .environment import UberMarketplaceEnvironment # Package import mode

class LinUCBAgent:
    """Original LinUCB agent with explicit covariance matrices."""

    def __init__(self, n_actions, n_features, alpha=1.0, prior_weight=1.0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha
        
        # A is now initialized with our prior_weight instead of just 1.0
        # A higher prior_weight means the Bandit is less likely to overwrite
        # the offline intelligence during early exploration.
        self.A = [np.identity(n_features) * prior_weight for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]

    def inject_offline_prior(self, historical_df, model_y, cate_model, discount_levels):
        """
        Distills the Phase 2 Random Forests into linear weights and injects them.
        """
        print("-> Distilling Phase 2 R-Learner into LinUCB Priors...")
        
        features = ['recency', 'frequency', 'weather_active', 'surge_multiplier']
        X_raw = historical_df[features]
        
        # 1. Ask the offline models for the baseline and the CATE
        baseline_preds = model_y.predict(X_raw)
        cate_preds = cate_model.predict(X_raw)
        
        # We need to format the data exactly how the Bandit sees it (with the Intercept)
        X_bandit = np.array([
            np.ones(len(historical_df)),
            historical_df['recency'] / 30.0,
            historical_df['frequency'] / 20.0,
            historical_df['weather_active'],
            historical_df['surge_multiplier'] / 3.0
        ]).T
        
        # 2. For each arm, calculate the expected rewards and run a Ridge Regression
        for a in range(self.n_actions):
            discount = discount_levels[a]
            
            # The R-Learner's prediction of reality for this specific arm
            expected_rewards = baseline_preds + (cate_preds * discount)
            
            # Fit a linear model to distill the Random Forest logic into linear weights
            # We don't fit an intercept because we already included a column of 1s in X_bandit
            distillation_model = Ridge(fit_intercept=False, alpha=1.0)
            distillation_model.fit(X_bandit, expected_rewards)
            
            theta_prior = distillation_model.coef_
            
            # 3. Inject the prior into the Bandit's brain
            # b = A * theta_prior
            self.b[a] = self.A[a].dot(theta_prior)
            
            print(f"   Arm {a} Prior Injected successfully.")
        
    def _get_context_vector(self, state_dict):
        """
        Converts the state dictionary into a normalized numpy vector.
        Normalization is crucial for LinUCB so the exploration parameter (alpha)
        scales evenly across all features.
        """
        return np.array([
            1.0, # Intercept bias term
            state_dict['recency'] / 30.0,
            state_dict['frequency'] / 20.0,
            state_dict['weather_active'],
            state_dict['surge_multiplier'] / 3.0
        ])

    def choose_action(self, state_dict):
        x = self._get_context_vector(state_dict)
        ucb_scores = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            # Calculate A inverse
            A_inv = np.linalg.inv(self.A[a])
            
            # Calculate theta (the learned causal weights for this specific arm)
            theta_a = A_inv.dot(self.b[a])
            
            # Calculate the expected reward (Exploitation)
            expected_reward = theta_a.dot(x)
            
            # Calculate the confidence bound (Exploration)
            exploration_bonus = self.alpha * np.sqrt(x.dot(A_inv).dot(x))
            
            # Upper Confidence Bound
            ucb_scores[a] = expected_reward + exploration_bonus
            
        # Break ties randomly, otherwise pick the max UCB
        # np.isclose is used to handle floating point precision ties
        max_ucb = np.max(ucb_scores)
        best_actions = np.where(np.isclose(ucb_scores, max_ucb))[0]
        return np.random.choice(best_actions)

    def update(self, action, state_dict, reward):
        """Updates the covariance matrix and reward vector after observing reality."""
        x = self._get_context_vector(state_dict)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

    def get_learned_weights(self, action):
        """Helper to peek into the Bandit's brain."""
        A_inv = np.linalg.inv(self.A[action])
        return A_inv.dot(self.b[action])

class FastLinUCBAgent(LinUCBAgent):
    """Discounted LinUCB using Sherman-Morrison inverse updates."""

    def __init__(self, n_actions, n_features, alpha=1.0, gamma=0.995, prior_weight=1.0):
        super().__init__(n_actions=n_actions, n_features=n_features, alpha=alpha, prior_weight=prior_weight)
        self.gamma = gamma # The Discount/Forgetting factor (0 < gamma <= 1)

        # Keep direct inverse matrices for fast O(d^2) updates.
        self.A_inv = [np.linalg.inv(a_matrix) for a_matrix in self.A]

    def inject_offline_prior(self, historical_df, model_y, cate_model, discount_levels):
        """Injects priors and keeps inverse matrices synchronized."""
        super().inject_offline_prior(historical_df, model_y, cate_model, discount_levels)
        self.A_inv = [np.linalg.inv(a_matrix) for a_matrix in self.A]

    def choose_action(self, state_dict):
        x = self._get_context_vector(state_dict)
        ucb_scores = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            # We no longer compute the inverse from scratch.
            A_inv_a = self.A_inv[a]
            
            # Calculate theta (the learned causal weights for this specific arm)
            theta_a = A_inv_a.dot(self.b[a])
            
            # Calculate the expected reward (Exploitation)
            expected_reward = theta_a.dot(x)
            
            # Calculate the confidence bound (Exploration)
            exploration_bonus = self.alpha * np.sqrt(x.dot(A_inv_a).dot(x))
            
            # Upper Confidence Bound
            ucb_scores[a] = expected_reward + exploration_bonus
            
        # Break ties randomly, otherwise pick the max UCB
        max_ucb = np.max(ucb_scores)
        best_actions = np.where(np.isclose(ucb_scores, max_ucb))[0]
        return np.random.choice(best_actions)

    def update(self, action, state_dict, reward):
        """Updates the inverse covariance matrix and reward vector directly."""
        x = self._get_context_vector(state_dict)
        
        # 1. Decay and update the reward vector
        self.b[action] = (self.gamma * self.b[action]) + (reward * x)
        
        # 2. Sherman-Morrison Update for the Inverse Matrix with Discounting
        # The math: M_new = (1/gamma) * [ M - (M * x * x^T * M) / (gamma + x^T * M * x) ]
        M = self.A_inv[action]
        
        # Step A: Calculate M * x  (This is an O(d^2) operation)
        Mx = M.dot(x) 
        
        # Step B: Calculate the denominator (scalar)
        denominator = self.gamma + x.dot(Mx)
        
        # Step C: Calculate the numerator (outer product of Mx with itself)
        numerator = np.outer(Mx, Mx)
        
        # Step D: Apply the update rule
        self.A_inv[action] = (1.0 / self.gamma) * (M - (numerator / denominator))

        # Keep A synchronized for compatibility with shared/base logic.
        self.A[action] = (self.gamma * self.A[action]) + np.outer(x, x)

    def get_learned_weights(self, action):
        """Helper to peek into the Bandit's brain."""
        # No inversion needed here either
        return self.A_inv[action].dot(self.b[action])


def run_agent_simulation(agent_class, env_seed=100, n_steps=15000, progress_every=5000, **agent_kwargs):
    """Runs one full simulation and returns metrics for the given agent class."""
    env = UberMarketplaceEnvironment(seed=env_seed)
    agent = agent_class(n_actions=3, n_features=5, **agent_kwargs)

    cumulative_rewards = 0.0
    reward_history = []
    cumulative_true_conversion = 0.0
    true_conversion_history = []
    cumulative_oracle_conversion = 0.0
    oracle_conversion_history = []
    cumulative_regret = 0.0
    regret_history = []
    mse_history = []

    print(f"\n1. Launching {agent_class.__name__} online simulation for {n_steps} riders...")
    start_time = time.perf_counter()

    state = env.reset()
    for step in range(n_steps):
        action = agent.choose_action(state)

        # Oracle grading against all possible actions.
        true_probs = [env._calculate_true_conversion(state, a) for a in range(agent.n_actions)]
        optimal_prob = np.max(true_probs)
        chosen_prob = true_probs[action]

        cumulative_true_conversion += chosen_prob
        true_conversion_history.append(cumulative_true_conversion / (step + 1))
        cumulative_oracle_conversion += optimal_prob
        oracle_conversion_history.append(cumulative_oracle_conversion / (step + 1))

        step_regret = optimal_prob - chosen_prob
        cumulative_regret += step_regret
        regret_history.append(cumulative_regret)

        x = agent._get_context_vector(state)
        bandit_prediction = agent.get_learned_weights(action).dot(x)
        squared_error = (bandit_prediction - chosen_prob) ** 2
        mse_history.append(squared_error)

        next_state, reward, _ = env.step(action)
        agent.update(action, state, reward)

        cumulative_rewards += reward
        reward_history.append(cumulative_rewards / (step + 1))
        state = next_state

        if (step + 1) % progress_every == 0:
            current_rmse = np.sqrt(np.mean(mse_history[-progress_every:]))
            print(f"   -> Step {step + 1}: Average Conversion Rate = {reward_history[-1]:.1%}")
            print(f"      Cumulative Regret: {cumulative_regret:.2f}")
            print(f"      Recent Prediction RMSE: {current_rmse:.4f}")

    runtime_seconds = time.perf_counter() - start_time

    return {
        "agent_name": agent_class.__name__,
        "agent": agent,
        "env": env,
        "reward_history": reward_history,
        "true_conversion_history": true_conversion_history,
        "oracle_conversion_history": oracle_conversion_history,
        "regret_history": regret_history,
        "mse_history": mse_history,
        "avg_conversion": reward_history[-1],
        "avg_true_conversion": true_conversion_history[-1],
        "avg_oracle_conversion": oracle_conversion_history[-1],
        "cumulative_regret": cumulative_regret,
        "final_rmse": float(np.sqrt(np.mean(mse_history[-progress_every:]))),
        "runtime_seconds": runtime_seconds,
    }


def print_agent_weights(simulation_result):
    """Prints learned linear weights per action for one simulation result."""
    agent = simulation_result["agent"]
    env = simulation_result["env"]
    print(f"\n2. Peeking into {simulation_result['agent_name']} learned weights...")
    feature_names = ["Intercept", "Recency", "Frequency", "Weather", "Surge"]

    for a in range(agent.n_actions):
        discount = env.discount_levels[a] * 100
        weights = agent.get_learned_weights(a)
        print(f"\n--- Arm {a} ({discount}% Discount) Learned Weights ---")
        for name, weight in zip(feature_names, weights):
            print(f"  {name:10}: {weight:+.3f}")


def print_comparison(original_result, fast_result):
    """Prints a concise side-by-side comparison of key metrics."""
    print("\n3. Comparison: Original vs Fast")
    print("   Metric                        Original          Fast")
    print(f"   Avg Conversion                {original_result['avg_conversion']:.4f}          {fast_result['avg_conversion']:.4f}")
    print(f"   Cumulative Regret             {original_result['cumulative_regret']:.2f}          {fast_result['cumulative_regret']:.2f}")
    print(f"   Final RMSE                    {original_result['final_rmse']:.4f}          {fast_result['final_rmse']:.4f}")
    print(f"   Runtime (seconds)             {original_result['runtime_seconds']:.2f}          {fast_result['runtime_seconds']:.2f}")


def plot_comparison(original_result, fast_result):
    """Plots side-by-side learning curves for Original and Fast LinUCB."""
    original_color = "tab:blue"
    fast_color = "tab:orange"
    oracle_color = "black"

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(original_result["reward_history"], label="Original LinUCB", color=original_color)
    plt.plot(fast_result["reward_history"], label="Fast LinUCB", color=fast_color, alpha=0.9)
    plt.plot(
        original_result["oracle_conversion_history"],
        label="Ground Truth (Oracle)",
        color=oracle_color,
        linestyle="--",
        linewidth=1.5,
    )
    plt.xlabel("Steps")
    plt.ylabel("Average Conversion")
    plt.title("Average Conversion vs Ground Truth")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(original_result["regret_history"], label="Original LinUCB", color=original_color)
    plt.plot(fast_result["regret_history"], label="Fast LinUCB", color=fast_color, alpha=0.9)
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret Over Time")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(original_result["mse_history"], label="Original LinUCB", color=original_color, alpha=0.5)
    plt.plot(fast_result["mse_history"], label="Fast LinUCB", color=fast_color, alpha=0.5)
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.title("Prediction MSE Over Time")
    plt.legend()

    plt.tight_layout()
    plt.show()

# =====================================================================
# THE ONLINE SIMULATION RUNNER
# =====================================================================
if __name__ == "__main__":
    n_steps = 15000
    original_result = run_agent_simulation(
        LinUCBAgent,
        env_seed=100,
        n_steps=n_steps,
        progress_every=5000,
        alpha=0.5,
    )

    fast_result = run_agent_simulation(
        FastLinUCBAgent,
        env_seed=100,
        n_steps=n_steps,
        progress_every=5000,
        alpha=0.5,
        gamma=0.995,
    )

    print_agent_weights(original_result)
    print_agent_weights(fast_result)
    print_comparison(original_result, fast_result)

    print("\n4. Analysis:")
    print("Look at the 'Frequency' weight across the three arms in both methods.")
    print("Both agents should learn that 20% discount (Arm 2) for high-frequency users")
    print("is often weak or negative compared with low-frequency users.")

    plot_comparison(original_result, fast_result)
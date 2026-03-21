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
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import Ridge

try:
    from environment import UberMarketplaceEnvironment, UberMarketplaceEnvironmentWithShock # Script execution mode
except ImportError:
    from .environment import UberMarketplaceEnvironment, UberMarketplaceEnvironmentWithShock # Package import mode

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


def run_agent_simulation(agent_class, env_seed=100, n_steps=15000, progress_every=5000, env_class=None, env_kwargs=None, **agent_kwargs):
    """Runs one full simulation and returns metrics for the given agent class.
    
    Args:
        agent_class: The bandit agent class to instantiate
        env_seed: RNG seed for environment reproducibility
        n_steps: Number of simulation steps
        progress_every: Print progress every N steps
        env_class: Optional custom environment class (defaults to UberMarketplaceEnvironment)
        env_kwargs: Optional dict of kwargs to pass to environment constructor
        **agent_kwargs: Additional kwargs passed to agent_class constructor
    """
    if env_class is None:
        env_class = UberMarketplaceEnvironment
    if env_kwargs is None:
        env_kwargs = {}
    env = env_class(seed=env_seed, **env_kwargs)
    agent = agent_class(n_actions=3, n_features=5, **agent_kwargs)

    reward_history = []
    true_conversion_history = []
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

        # Store raw per-step probabilities (no cumulative averaging).
        true_conversion_history.append(chosen_prob)
        oracle_conversion_history.append(optimal_prob)

        step_regret = optimal_prob - chosen_prob
        cumulative_regret += step_regret
        regret_history.append(cumulative_regret)

        x = agent._get_context_vector(state)
        bandit_prediction = agent.get_learned_weights(action).dot(x)
        squared_error = (bandit_prediction - chosen_prob) ** 2
        mse_history.append(squared_error)

        next_state, reward, _ = env.step(action)
        agent.update(action, state, reward)

        # Store raw binary reward for this step (0 or 1).
        reward_history.append(reward)
        state = next_state

        if (step + 1) % progress_every == 0:
            current_rmse = np.sqrt(np.mean(mse_history[-progress_every:]))
            recent_avg_conversion = np.mean(reward_history[-progress_every:])
            print(f"   -> Step {step + 1}: Recent Avg Conversion Rate = {recent_avg_conversion:.1%}")
            print(f"      Cumulative Regret: {cumulative_regret:.2f}")
            print(f"      Recent Prediction RMSE: {current_rmse:.4f}")

    runtime_seconds = time.perf_counter() - start_time

    period_summaries = None
    if hasattr(env, "shock_step"):
        split_idx = int(min(max(env.shock_step, 0), len(reward_history)))

        def _build_period_summary(start_idx, end_idx, label):
            if end_idx <= start_idx:
                return None
            return {
                "label": label,
                "start_step": start_idx + 1,
                "end_step": end_idx,
                "avg_conversion": float(np.mean(reward_history[start_idx:end_idx])),
                "avg_true_conversion": float(np.mean(true_conversion_history[start_idx:end_idx])),
                "avg_oracle_conversion": float(np.mean(oracle_conversion_history[start_idx:end_idx])),
            }

        period_summaries = []
        pre_summary = _build_period_summary(0, split_idx, "Pre-shock")
        post_summary = _build_period_summary(split_idx, len(reward_history), "Post-shock")
        if pre_summary is not None:
            period_summaries.append(pre_summary)
        if post_summary is not None:
            period_summaries.append(post_summary)

    return {
        "agent_name": agent_class.__name__,
        "agent": agent,
        "env": env,
        "reward_history": reward_history,
        "true_conversion_history": true_conversion_history,
        "oracle_conversion_history": oracle_conversion_history,
        "regret_history": regret_history,
        "mse_history": mse_history,
        "avg_conversion": float(np.mean(reward_history)),
        "avg_true_conversion": float(np.mean(true_conversion_history)),
        "avg_oracle_conversion": float(np.mean(oracle_conversion_history)),
        "cumulative_regret": cumulative_regret,
        "final_rmse": float(np.sqrt(np.mean(mse_history[-progress_every:]))),
        "runtime_seconds": runtime_seconds,
        "period_summaries": period_summaries,
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

    original_periods = original_result.get("period_summaries")
    fast_periods = fast_result.get("period_summaries")
    if original_periods and fast_periods and len(original_periods) == len(fast_periods):
        print("\n   Period Summary (environment change detected):")
        print("   Period               Original Conv    Fast Conv    Oracle (Orig)   Oracle (Fast)")
        for original_period, fast_period in zip(original_periods, fast_periods):
            label = f"{original_period['label']} [{original_period['start_step']}-{original_period['end_step']}]"
            print(
                f"   {label:<20} "
                f"{original_period['avg_conversion']:.4f}         "
                f"{fast_period['avg_conversion']:.4f}      "
                f"{original_period['avg_oracle_conversion']:.4f}         "
                f"{fast_period['avg_oracle_conversion']:.4f}"
            )


def plot_comparison(original_result, fast_result):
    """Plots side-by-side learning curves for Original and Fast LinUCB."""
    original_color = "tab:blue"
    fast_color = "tab:orange"
    oracle_color = "black"

    def cumulative_average(data):
        """Returns cumulative average so the plotted metric matches its label."""
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            return arr
        return np.cumsum(arr) / np.arange(1, arr.size + 1)

    plt.figure(figsize=(18, 5))

    original_avg_conversion = cumulative_average(original_result["reward_history"])
    fast_avg_conversion = cumulative_average(fast_result["reward_history"])
    original_oracle_avg = cumulative_average(original_result["oracle_conversion_history"])
    fast_oracle_avg = cumulative_average(fast_result["oracle_conversion_history"])

    plt.subplot(1, 3, 1)
    plt.plot(original_avg_conversion, label="Original LinUCB", color=original_color)
    plt.plot(fast_avg_conversion, label="Fast LinUCB", color=fast_color, alpha=0.9)
    plt.plot(
        original_oracle_avg,
        label="Oracle (Original Trajectory)",
        color=oracle_color,
        linestyle="--",
        linewidth=1.5,
    )
    if len(original_oracle_avg) == len(fast_oracle_avg) and not np.allclose(original_oracle_avg, fast_oracle_avg):
        plt.plot(
            fast_oracle_avg,
            label="Oracle (Fast Trajectory)",
            color="tab:gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Average Conversion")
    plt.title("Cumulative Average Conversion vs Oracle")
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


def split_rolling_average(data, shock_step, window=1000):
    """Calculates a rolling average that respects structural breaks.
    
    Prevents pre-shock data from smearing into post-shock visualizations.
    Calculates rolling average independently for Phase 1 and Phase 2, then stitches them together.
    """
    # Phase 1: Pre-shock
    # Using min_periods=1 ensures the graph starts immediately at step 0
    phase_1 = pd.Series(data[:shock_step]).rolling(window=window, min_periods=1).mean()
    
    # Phase 2: Post-shock
    # The rolling window completely resets here. No Phase 1 data is allowed.
    phase_2 = pd.Series(data[shock_step:]).rolling(window=window, min_periods=1).mean()
    
    # Stitch them back together into a single continuous array
    return pd.concat([phase_1, phase_2]).values


def plot_shock_comparison(original_result, fast_result, shock_step=17500):
    """Plots comparison with shock environment, highlighting the market crash event.
    
    Includes a vertical line marking the exact moment of the economic shock.
    This reveals how each algorithm adapts: Original becomes paralyzed, Fast recovers.
    """
    original_color = "tab:blue"
    fast_color = "tab:orange"
    oracle_color = "black"
    shock_color = "violet"

    plt.figure(figsize=(18, 5))

    # Rolling average for conversion metrics to reveal non-stationary regime shifts.
    rolling_window = 1000

    original_reward_rolling = split_rolling_average(original_result["reward_history"], shock_step, window=rolling_window)
    fast_reward_rolling = split_rolling_average(fast_result["reward_history"], shock_step, window=rolling_window)
    oracle_rolling = split_rolling_average(original_result["oracle_conversion_history"], shock_step, window=rolling_window)

    # Subplot 1: Conversion (1,000-step rolling window, split at shock boundary)
    plt.subplot(1, 3, 1)
    plt.axvline(x=shock_step, color=shock_color, linestyle=':', linewidth=2.5, label='Economic Shock', alpha=0.7)
    plt.plot(original_reward_rolling, label="Original LinUCB (rolling=1000)", color=original_color)
    plt.plot(fast_reward_rolling, label="Fast LinUCB (rolling=1000)", color=fast_color, alpha=0.9)
    plt.plot(
        oracle_rolling,
        label="Ground Truth (Oracle, rolling=1000)",
        color=oracle_color,
        linestyle="--",
        linewidth=1.5,
    )
    plt.xlabel("Steps")
    plt.ylabel("Average Conversion")
    plt.title("Conversion: Adaptability to Market Shock (Split Rolling Window)")
    plt.legend()

    # Subplot 2: Regret
    plt.subplot(1, 3, 2)
    plt.axvline(x=shock_step, color=shock_color, linestyle=':', linewidth=2.5, label='Economic Shock', alpha=0.7)
    plt.plot(original_result["regret_history"], label="Original LinUCB", color=original_color)
    plt.plot(fast_result["regret_history"], label="Fast LinUCB", color=fast_color, alpha=0.9)
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Regret")
    plt.title("Regret: Recovery Speed After Shock")
    plt.legend()

    # Subplot 3: MSE
    plt.subplot(1, 3, 3)
    plt.axvline(x=shock_step, color=shock_color, linestyle=':', linewidth=2.5, label='Economic Shock', alpha=0.7)
    plt.plot(original_result["mse_history"], label="Original LinUCB", color=original_color, alpha=0.5)
    plt.plot(fast_result["mse_history"], label="Fast LinUCB", color=fast_color, alpha=0.5)
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.title("Prediction MSE: Model Resilience")
    plt.legend()

    plt.tight_layout()
    plt.show()


def run_multi_seed_comparison(
    seeds=None,
    n_steps=15000,
    progress_every=5000,
    env_class=None,
    env_kwargs=None,
    original_kwargs=None,
    fast_kwargs=None,
):
    """Runs both algorithms across multiple seeds and returns aggregated results.

    Args:
        seeds: Iterable of integer seeds. Defaults to 5 seeds.
        n_steps: Number of simulation steps for each run.
        progress_every: Progress logging interval.
        env_class: Optional environment class.
        env_kwargs: Optional kwargs passed to environment constructor.
        original_kwargs: Optional kwargs for LinUCBAgent.
        fast_kwargs: Optional kwargs for FastLinUCBAgent.
    """
    if seeds is None:
        seeds = [100, 101, 102, 103, 104]
    if env_kwargs is None:
        env_kwargs = {}
    if original_kwargs is None:
        original_kwargs = {"alpha": 0.5}
    if fast_kwargs is None:
        fast_kwargs = {"alpha": 0.5, "gamma": 0.995}

    original_runs = []
    fast_runs = []

    print("\n" + "=" * 70)
    print(f"MULTI-SEED BENCHMARK: {len(seeds)} seeds")
    print("=" * 70)

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n[Seed {run_idx}/{len(seeds)}] env_seed={seed}")
        original_result = run_agent_simulation(
            LinUCBAgent,
            env_seed=seed,
            n_steps=n_steps,
            progress_every=progress_every,
            env_class=env_class,
            env_kwargs=env_kwargs,
            **original_kwargs,
        )
        fast_result = run_agent_simulation(
            FastLinUCBAgent,
            env_seed=seed,
            n_steps=n_steps,
            progress_every=progress_every,
            env_class=env_class,
            env_kwargs=env_kwargs,
            **fast_kwargs,
        )
        original_runs.append(original_result)
        fast_runs.append(fast_result)

    def _aggregate_scalar_metric(runs, metric_name):
        values = np.array([run[metric_name] for run in runs], dtype=float)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }

    summary = {
        "avg_conversion": {
            "original": _aggregate_scalar_metric(original_runs, "avg_conversion"),
            "fast": _aggregate_scalar_metric(fast_runs, "avg_conversion"),
        },
        "cumulative_regret": {
            "original": _aggregate_scalar_metric(original_runs, "cumulative_regret"),
            "fast": _aggregate_scalar_metric(fast_runs, "cumulative_regret"),
        },
        "final_rmse": {
            "original": _aggregate_scalar_metric(original_runs, "final_rmse"),
            "fast": _aggregate_scalar_metric(fast_runs, "final_rmse"),
        },
        "runtime_seconds": {
            "original": _aggregate_scalar_metric(original_runs, "runtime_seconds"),
            "fast": _aggregate_scalar_metric(fast_runs, "runtime_seconds"),
        },
    }

    return {
        "seeds": list(seeds),
        "original_runs": original_runs,
        "fast_runs": fast_runs,
        "summary": summary,
    }


def print_multi_seed_summary(multi_seed_result):
    """Prints scalar mean/std summary for multi-seed runs."""
    summary = multi_seed_result["summary"]

    def _row(metric_key, label):
        orig = summary[metric_key]["original"]
        fast = summary[metric_key]["fast"]
        print(
            f"   {label:<24} "
            f"{orig['mean']:.4f} ± {orig['std']:.4f}    "
            f"{fast['mean']:.4f} ± {fast['std']:.4f}"
        )

    print("\nMulti-Seed Summary (mean ± std)")
    print(f"   Seeds: {multi_seed_result['seeds']}")
    print("   Metric                    Original                 Fast")
    _row("avg_conversion", "Avg Conversion")
    _row("cumulative_regret", "Cumulative Regret")
    _row("final_rmse", "Final RMSE")
    _row("runtime_seconds", "Runtime (seconds)")


def plot_multi_seed_comparison(multi_seed_result):
    """Plots mean ± std trajectories across seeds for both algorithms."""
    original_color = "tab:blue"
    fast_color = "tab:orange"

    original_runs = multi_seed_result["original_runs"]
    fast_runs = multi_seed_result["fast_runs"]

    def cumulative_average(data):
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            return arr
        return np.cumsum(arr) / np.arange(1, arr.size + 1)

    def mean_std_band(series_list):
        matrix = np.vstack(series_list)
        return np.mean(matrix, axis=0), np.std(matrix, axis=0)

    original_conv_mean, original_conv_std = mean_std_band(
        [cumulative_average(run["reward_history"]) for run in original_runs]
    )
    fast_conv_mean, fast_conv_std = mean_std_band(
        [cumulative_average(run["reward_history"]) for run in fast_runs]
    )

    original_regret_mean, original_regret_std = mean_std_band(
        [run["regret_history"] for run in original_runs]
    )
    fast_regret_mean, fast_regret_std = mean_std_band(
        [run["regret_history"] for run in fast_runs]
    )

    original_mse_mean, original_mse_std = mean_std_band(
        [run["mse_history"] for run in original_runs]
    )
    fast_mse_mean, fast_mse_std = mean_std_band(
        [run["mse_history"] for run in fast_runs]
    )

    x_conv = np.arange(len(original_conv_mean))
    x_regret = np.arange(len(original_regret_mean))
    x_mse = np.arange(len(original_mse_mean))

    plt.figure(figsize=(18, 5))

    # Subplot 1: Cumulative average conversion
    plt.subplot(1, 3, 1)
    plt.plot(x_conv, original_conv_mean, label="Original LinUCB Mean", color=original_color)
    plt.fill_between(
        x_conv,
        original_conv_mean - original_conv_std,
        original_conv_mean + original_conv_std,
        color=original_color,
        alpha=0.2,
        label="Original ±1 std",
    )
    plt.plot(x_conv, fast_conv_mean, label="Fast LinUCB Mean", color=fast_color)
    plt.fill_between(
        x_conv,
        fast_conv_mean - fast_conv_std,
        fast_conv_mean + fast_conv_std,
        color=fast_color,
        alpha=0.2,
        label="Fast ±1 std",
    )
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Average Conversion")
    plt.title("Conversion Across Seeds")
    plt.legend()

    # Subplot 2: Cumulative regret
    plt.subplot(1, 3, 2)
    plt.plot(x_regret, original_regret_mean, label="Original LinUCB Mean", color=original_color)
    plt.fill_between(
        x_regret,
        original_regret_mean - original_regret_std,
        original_regret_mean + original_regret_std,
        color=original_color,
        alpha=0.2,
    )
    plt.plot(x_regret, fast_regret_mean, label="Fast LinUCB Mean", color=fast_color)
    plt.fill_between(
        x_regret,
        fast_regret_mean - fast_regret_std,
        fast_regret_mean + fast_regret_std,
        color=fast_color,
        alpha=0.2,
    )
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Regret")
    plt.title("Regret Across Seeds")
    plt.legend()

    # Subplot 3: Prediction MSE
    plt.subplot(1, 3, 3)
    plt.plot(x_mse, original_mse_mean, label="Original LinUCB Mean", color=original_color)
    plt.fill_between(
        x_mse,
        original_mse_mean - original_mse_std,
        original_mse_mean + original_mse_std,
        color=original_color,
        alpha=0.2,
    )
    plt.plot(x_mse, fast_mse_mean, label="Fast LinUCB Mean", color=fast_color)
    plt.fill_between(
        x_mse,
        fast_mse_mean - fast_mse_std,
        fast_mse_mean + fast_mse_std,
        color=fast_color,
        alpha=0.2,
    )
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.title("Prediction MSE Across Seeds")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_multi_seed_shock_comparison(multi_seed_result, shock_step, rolling_window=1000):
    """Plots shock scenario mean ± std across seeds for both algorithms.

    Uses split rolling averages for conversion so pre-shock data does not smear
    into post-shock estimates.
    """
    original_color = "tab:blue"
    fast_color = "tab:orange"
    shock_color = "violet"

    original_runs = multi_seed_result["original_runs"]
    fast_runs = multi_seed_result["fast_runs"]

    def mean_std_band(series_list):
        matrix = np.vstack(series_list)
        return np.mean(matrix, axis=0), np.std(matrix, axis=0)

    original_conv_mean, original_conv_std = mean_std_band(
        [split_rolling_average(run["reward_history"], shock_step, rolling_window) for run in original_runs]
    )
    fast_conv_mean, fast_conv_std = mean_std_band(
        [split_rolling_average(run["reward_history"], shock_step, rolling_window) for run in fast_runs]
    )
    oracle_conv_mean, oracle_conv_std = mean_std_band(
        [split_rolling_average(run["oracle_conversion_history"], shock_step, rolling_window) for run in original_runs]
    )

    original_regret_mean, original_regret_std = mean_std_band(
        [run["regret_history"] for run in original_runs]
    )
    fast_regret_mean, fast_regret_std = mean_std_band(
        [run["regret_history"] for run in fast_runs]
    )

    original_mse_mean, original_mse_std = mean_std_band(
        [run["mse_history"] for run in original_runs]
    )
    fast_mse_mean, fast_mse_std = mean_std_band(
        [run["mse_history"] for run in fast_runs]
    )

    x_conv = np.arange(len(original_conv_mean))
    x_regret = np.arange(len(original_regret_mean))
    x_mse = np.arange(len(original_mse_mean))

    plt.figure(figsize=(18, 5))

    # Subplot 1: Conversion with split rolling window
    plt.subplot(1, 3, 1)
    plt.axvline(x=shock_step, color=shock_color, linestyle=":", linewidth=2.5, label="Economic Shock", alpha=0.7)
    plt.plot(x_conv, original_conv_mean, label=f"Original Mean (rolling={rolling_window})", color=original_color)
    plt.fill_between(
        x_conv,
        original_conv_mean - original_conv_std,
        original_conv_mean + original_conv_std,
        color=original_color,
        alpha=0.2,
        label="Original ±1 std",
    )
    plt.plot(x_conv, fast_conv_mean, label=f"Fast Mean (rolling={rolling_window})", color=fast_color)
    plt.fill_between(
        x_conv,
        fast_conv_mean - fast_conv_std,
        fast_conv_mean + fast_conv_std,
        color=fast_color,
        alpha=0.2,
        label="Fast ±1 std",
    )
    plt.plot(
        x_conv,
        oracle_conv_mean,
        label=f"Oracle Mean (rolling={rolling_window})",
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    plt.fill_between(
        x_conv,
        oracle_conv_mean - oracle_conv_std,
        oracle_conv_mean + oracle_conv_std,
        color="black",
        alpha=0.1,
    )
    plt.xlabel("Steps")
    plt.ylabel("Average Conversion")
    plt.title("Shock Conversion Across Seeds")
    plt.legend()

    # Subplot 2: Regret
    plt.subplot(1, 3, 2)
    plt.axvline(x=shock_step, color=shock_color, linestyle=":", linewidth=2.5, label="Economic Shock", alpha=0.7)
    plt.plot(x_regret, original_regret_mean, label="Original Mean", color=original_color)
    plt.fill_between(
        x_regret,
        original_regret_mean - original_regret_std,
        original_regret_mean + original_regret_std,
        color=original_color,
        alpha=0.2,
    )
    plt.plot(x_regret, fast_regret_mean, label="Fast Mean", color=fast_color)
    plt.fill_between(
        x_regret,
        fast_regret_mean - fast_regret_std,
        fast_regret_mean + fast_regret_std,
        color=fast_color,
        alpha=0.2,
    )
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Regret")
    plt.title("Shock Regret Across Seeds")
    plt.legend()

    # Subplot 3: MSE
    plt.subplot(1, 3, 3)
    plt.axvline(x=shock_step, color=shock_color, linestyle=":", linewidth=2.5, label="Economic Shock", alpha=0.7)
    plt.plot(x_mse, original_mse_mean, label="Original Mean", color=original_color)
    plt.fill_between(
        x_mse,
        original_mse_mean - original_mse_std,
        original_mse_mean + original_mse_std,
        color=original_color,
        alpha=0.2,
    )
    plt.plot(x_mse, fast_mse_mean, label="Fast Mean", color=fast_color)
    plt.fill_between(
        x_mse,
        fast_mse_mean - fast_mse_std,
        fast_mse_mean + fast_mse_std,
        color=fast_color,
        alpha=0.2,
    )
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.title("Shock Prediction MSE Across Seeds")
    plt.legend()

    plt.tight_layout()
    plt.show()

# =====================================================================
# THE ONLINE SIMULATION RUNNER
# =====================================================================
if __name__ == "__main__":
    # =====================================================================
    # BASELINE SIMULATION: Static Environment
    # =====================================================================
    print("\n" + "="*70)
    print("BASELINE SIMULATION: Static Environment")
    print("="*70)
    
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

    # =====================================================================
    # SHOCK SIMULATION: Non-Stationary Environment Test
    # =====================================================================
    print("\n" + "="*70)
    print("SHOCK SIMULATION: Testing Algorithm Adaptability to Market Crisis")
    print("="*70)
    
    n_steps_shock = 35000
    shock_step = 17500
    
    shock_result_original = run_agent_simulation(
        LinUCBAgent,
        env_seed=100,
        n_steps=n_steps_shock,
        progress_every=5000,
        env_class=UberMarketplaceEnvironmentWithShock,
        env_kwargs={'shock_step': shock_step},
        alpha=0.5,
    )

    shock_result_fast = run_agent_simulation(
        FastLinUCBAgent,
        env_seed=100,
        n_steps=n_steps_shock,
        progress_every=5000,
        env_class=UberMarketplaceEnvironmentWithShock,
        env_kwargs={'shock_step': shock_step},
        alpha=0.5,
        gamma=0.995,
    )

    print("\n5. Shock Scenario Analysis:")
    print(f"   Economic crash occurs at Step {shock_step} (exactly halfway through)")
    print("   \n   Original LinUCB (Blue):")
    print("   - Has 17,500 steps of 'good economy' data memorized in A_a")
    print("   - When shock hits, confidence bounds are near zero")
    print("   - Algorithm PARALYZED: continues offering old weak discounts")
    print("   - Conversion rate CRASHES, Regret spikes sharply\n")
    print("   Fast LinUCB (Orange):")
    print("   - Discount factor (γ=0.995) keeps recent memory limited")
    print("   - Old 'good economy' data exponentially fades away")
    print("   - When shock hits, algorithm IMMEDIATELY begins exploring")
    print("   - Rapidly discovers new optimal pricing, conversion recovers smoothly")
    
    print_comparison(shock_result_original, shock_result_fast)
    plot_shock_comparison(shock_result_original, shock_result_fast, shock_step=shock_step)
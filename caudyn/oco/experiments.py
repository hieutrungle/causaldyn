"""LinUCB experiment runners and offline-prior extraction utilities."""

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .linucb_agents import FastLinUCBAgent, LinUCBAgent
from .lints_agents import FastLinTSAgent, LinTSAgent

try:
    from environment import (
        UberMarketplaceEnvironment,
        UberMarketplaceEnvironmentWithShock,
    )  # Script execution mode
except ImportError:
    try:
        from ..environment import (
            UberMarketplaceEnvironment,
            UberMarketplaceEnvironmentWithShock,
        )  # Package import mode
    except ImportError:
        from caudyn.environment import (
            UberMarketplaceEnvironment,
            UberMarketplaceEnvironmentWithShock,
        )  # Absolute fallback

try:
    from r_learner import RLearner  # Script execution mode
except ImportError:
    try:
        from ..r_learner import RLearner  # Package import mode
    except ImportError:
        from caudyn.r_learner import RLearner  # Absolute fallback


def _build_bandit_feature_matrix(dataframe):
    """Builds the normalized feature matrix expected by LinUCB."""
    return np.array(
        [
            np.ones(len(dataframe)),
            dataframe["recency"].to_numpy(dtype=float) / 30.0,
            dataframe["frequency"].to_numpy(dtype=float) / 20.0,
            dataframe["weather_active"].to_numpy(dtype=float),
            dataframe["surge_multiplier"].to_numpy(dtype=float) / 3.0,
        ]
    ).T


def extract_rlearner_linear_priors_from_csv(
    offline_csv_path,
    ridge_alpha=1.0,
    epsilon=1e-6,
    rlearner_cv=5,
    rlearner_random_state=42,
    discount_levels=None,
):
    """Extracts linear per-arm priors from offline logs using R-learner-style residualization."""
    if discount_levels is None:
        discount_levels = [0.0, 0.1, 0.2]

    required_columns = {
        "recency",
        "frequency",
        "weather_active",
        "surge_multiplier",
        "discount_value",
        "converted",
    }

    historical_df = pd.read_csv(offline_csv_path)
    missing = required_columns.difference(historical_df.columns)
    if missing:
        raise ValueError(
            f"Offline dataset is missing required columns: {sorted(missing)}"
        )

    X_raw = historical_df[
        ["recency", "frequency", "weather_active", "surge_multiplier"]
    ]
    X_bandit = _build_bandit_feature_matrix(historical_df)
    Y = historical_df["converted"].to_numpy(dtype=float)
    T = historical_df["discount_value"].to_numpy(dtype=float)

    rlearner = RLearner(
        random_state=rlearner_random_state,
        cv=rlearner_cv,
        epsilon=epsilon,
    ).fit(X_raw, T, Y)

    cate_target = rlearner.predict_cate(X_raw)
    cate_weights = np.ones_like(cate_target)
    cate_linear = Ridge(alpha=ridge_alpha, fit_intercept=False)
    cate_linear.fit(X_bandit, cate_target, sample_weight=cate_weights)
    theta_tau = cate_linear.coef_

    control_mask = np.isclose(T, 0.0)
    if not np.any(control_mask):
        raise ValueError(
            "Offline dataset has no control samples with discount_value=0.0"
        )

    baseline_linear = Ridge(alpha=ridge_alpha, fit_intercept=False)
    baseline_linear.fit(X_bandit[control_mask], Y[control_mask])
    theta_0 = baseline_linear.coef_

    priors = {
        arm_idx: theta_0 + (discount * theta_tau)
        for arm_idx, discount in enumerate(discount_levels)
    }

    return {
        "priors": priors,
        "theta_control": theta_0,
        "theta_tau": theta_tau,
        "ridge_alpha": ridge_alpha,
        "rlearner_cv": rlearner_cv,
        "rlearner_random_state": rlearner_random_state,
        "n_samples": int(len(historical_df)),
        "X_columns": ["recency", "frequency", "weather_active", "surge_multiplier"],
    }


def run_agent_simulation(
    agent_class,
    env_seed=100,
    n_steps=15000,
    progress_every=5000,
    env_class=None,
    env_kwargs=None,
    prior_thetas=None,
    prior_weight=1.0,
    **agent_kwargs,
):
    """Runs one full simulation and returns metrics for the given agent class."""
    if env_class is None:
        env_class = UberMarketplaceEnvironment
    if env_kwargs is None:
        env_kwargs = {}
    env = env_class(seed=env_seed, **env_kwargs)
    agent = agent_class(n_actions=3, n_features=5, **agent_kwargs)

    if prior_thetas is not None:
        agent.inject_linear_priors(prior_thetas, prior_weight=prior_weight)
        print(f"   -> Injected offline priors with prior_weight={prior_weight:.1f}")

    reward_history = []
    true_conversion_history = []
    oracle_conversion_history = []
    cumulative_regret = 0.0
    regret_history = []
    mse_history = []

    print(
        f"\n1. Launching {agent_class.__name__} online simulation for {n_steps} riders..."
    )
    start_time = time.perf_counter()

    state = env.reset()
    for step in range(n_steps):
        action = agent.choose_action(state)

        true_probs = [
            env._calculate_true_conversion(state, a) for a in range(agent.n_actions)
        ]
        optimal_prob = np.max(true_probs)
        chosen_prob = true_probs[action]

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

        reward_history.append(reward)
        state = next_state

        if (step + 1) % progress_every == 0:
            current_rmse = np.sqrt(np.mean(mse_history[-progress_every:]))
            recent_avg_conversion = np.mean(reward_history[-progress_every:])
            print(
                f"   -> Step {step + 1}: Recent Avg Conversion Rate = {recent_avg_conversion:.1%}"
            )
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
                "avg_true_conversion": float(
                    np.mean(true_conversion_history[start_idx:end_idx])
                ),
                "avg_oracle_conversion": float(
                    np.mean(oracle_conversion_history[start_idx:end_idx])
                ),
            }

        period_summaries = []
        pre_summary = _build_period_summary(0, split_idx, "Pre-shock")
        post_summary = _build_period_summary(
            split_idx, len(reward_history), "Post-shock"
        )
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


def run_multi_seed_comparison(
    agent_configs=None,
    seeds=None,
    n_steps=15000,
    progress_every=5000,
    env_class=None,
    env_kwargs=None,
    original_kwargs=None,
    fast_kwargs=None,
    prior_thetas=None,
    prior_weight=1.0,
):
    """Runs one or more algorithms across multiple seeds and returns aggregated results.

    Args:
        agent_configs: Optional dict of
            {"display_name": {"class": AgentClass, "kwargs": {...}}}.
            If omitted, defaults to the classic two-agent LinUCB benchmark.
        original_kwargs: Backward-compatible kwargs for LinUCBAgent default config.
        fast_kwargs: Backward-compatible kwargs for FastLinUCBAgent default config.
    """
    if seeds is None:
        seeds = [100, 101, 102, 103, 104]
    if env_kwargs is None:
        env_kwargs = {}

    if agent_configs is None:
        if original_kwargs is None:
            original_kwargs = {"alpha": 0.5}
        if fast_kwargs is None:
            fast_kwargs = {"alpha": 0.5, "gamma": 0.995}

        agent_configs = {
            "original": {"class": LinUCBAgent, "kwargs": original_kwargs},
            "fast": {"class": FastLinUCBAgent, "kwargs": fast_kwargs},
        }

    all_runs = {agent_name: [] for agent_name in agent_configs.keys()}

    print("\n" + "=" * 70)
    print(f"MULTI-SEED BENCHMARK: {len(seeds)} seeds, {len(agent_configs)} agents")
    print("=" * 70)

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n[Seed {run_idx}/{len(seeds)}] env_seed={seed}")
        for agent_name, config in agent_configs.items():
            if "class" not in config:
                raise ValueError(
                    f"agent_configs['{agent_name}'] must contain a 'class' entry"
                )

            result = run_agent_simulation(
                agent_class=config["class"],
                env_seed=seed,
                n_steps=n_steps,
                progress_every=progress_every,
                env_class=env_class,
                env_kwargs=env_kwargs,
                prior_thetas=prior_thetas,
                prior_weight=prior_weight,
                **config.get("kwargs", {}),
            )
            result["agent_name"] = agent_name
            all_runs[agent_name].append(result)

    def _aggregate_scalar_metric(runs, metric_name):
        values = np.array([run[metric_name] for run in runs], dtype=float)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }

    metrics_to_aggregate = [
        "avg_conversion",
        "cumulative_regret",
        "final_rmse",
        "runtime_seconds",
    ]
    summary = {}
    for metric in metrics_to_aggregate:
        summary[metric] = {
            agent_name: _aggregate_scalar_metric(runs, metric)
            for agent_name, runs in all_runs.items()
        }

    result_payload = {
        "seeds": list(seeds),
        "runs": all_runs,
        "summary": summary,
    }

    # Backward-compatibility for existing reporting helpers and notebooks.
    if "original" in all_runs:
        result_payload["original_runs"] = all_runs["original"]
    if "fast" in all_runs:
        result_payload["fast_runs"] = all_runs["fast"]

    return result_payload


def run_static_warm_start_multi_seed(
    offline_csv_path,
    agent_configs=None,
    seeds=None,
    n_steps=35000,
    progress_every=5000,
    prior_weight=1.0,
    ridge_alpha=1.0,
    rlearner_cv=5,
    rlearner_random_state=42,
    original_kwargs=None,
    fast_kwargs=None,
):
    """Runs multi-seed static-environment comparison with optional custom agents."""
    prior_bundle = extract_rlearner_linear_priors_from_csv(
        offline_csv_path=offline_csv_path,
        ridge_alpha=ridge_alpha,
        rlearner_cv=rlearner_cv,
        rlearner_random_state=rlearner_random_state,
    )
    priors = prior_bundle["priors"]

    print("\n--- Offline Prior Extraction Complete ---")
    print(f"   Offline samples: {prior_bundle['n_samples']}")
    print(f"   prior_weight: {prior_weight}")

    results = run_multi_seed_comparison(
        agent_configs=agent_configs,
        seeds=seeds,
        n_steps=n_steps,
        progress_every=progress_every,
        env_class=UberMarketplaceEnvironment,
        env_kwargs={},
        original_kwargs=original_kwargs,
        fast_kwargs=fast_kwargs,
        prior_thetas=priors,
        prior_weight=prior_weight,
    )
    results["prior_bundle"] = prior_bundle
    return results


def run_default_demo():
    """Runs default demos for LinUCB and a generic multi-agent shock benchmark."""
    try:
        from reporting import (
            print_agent_weights,
            print_comparison,
            plot_comparison,
            plot_shock_comparison,
        )
    except ImportError:
        from .reporting import (
            print_agent_weights,
            print_comparison,
            plot_comparison,
            plot_shock_comparison,
        )

    print("\n" + "=" * 70)
    print("BASELINE SIMULATION: Static Environment")
    print("=" * 70)

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

    print("\n" + "=" * 70)
    print("SHOCK SIMULATION: Testing Algorithm Adaptability to Market Crisis")
    print("=" * 70)

    n_steps_shock = 35000
    shock_step = 17500

    shock_result_original = run_agent_simulation(
        LinUCBAgent,
        env_seed=100,
        n_steps=n_steps_shock,
        progress_every=5000,
        env_class=UberMarketplaceEnvironmentWithShock,
        env_kwargs={"shock_step": shock_step},
        alpha=0.5,
    )

    shock_result_fast = run_agent_simulation(
        FastLinUCBAgent,
        env_seed=100,
        n_steps=n_steps_shock,
        progress_every=5000,
        env_class=UberMarketplaceEnvironmentWithShock,
        env_kwargs={"shock_step": shock_step},
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
    print("   - Discount factor (gamma=0.995) keeps recent memory limited")
    print("   - Old 'good economy' data exponentially fades away")
    print("   - When shock hits, algorithm IMMEDIATELY begins exploring")
    print("   - Rapidly discovers new optimal pricing, conversion recovers smoothly")

    print_comparison(shock_result_original, shock_result_fast)
    plot_shock_comparison(
        shock_result_original, shock_result_fast, shock_step=shock_step
    )

    print("\n" + "=" * 70)
    print("GENERIC BENCHMARK: Fast LinUCB vs Fast LinTS (single seed)")
    print("=" * 70)

    generic_results = run_multi_seed_comparison(
        agent_configs={
            "Fast LinUCB": {
                "class": FastLinUCBAgent,
                "kwargs": {"alpha": 0.5, "gamma": 0.995},
            },
            "Fast LinTS": {
                "class": FastLinTSAgent,
                "kwargs": {"v_squared": 0.1, "gamma": 0.995},
            },
            "LinTS": {
                "class": LinTSAgent,
                "kwargs": {"v_squared": 0.1},
            },
        },
        seeds=[100],
        n_steps=n_steps_shock,
        progress_every=5000,
        env_class=UberMarketplaceEnvironmentWithShock,
        env_kwargs={"shock_step": shock_step},
    )

    print("\n6. Generic Multi-Agent Summary (single-seed statistics):")
    for metric_name, metric_values in generic_results["summary"].items():
        print(f"   {metric_name}:")
        for agent_name, stats in metric_values.items():
            print(
                f"      {agent_name:<16} mean={stats['mean']:.4f}, std={stats['std']:.4f}"
            )


if __name__ == "__main__":
    run_default_demo()

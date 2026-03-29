import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _extract_runs_map(multi_seed_result):
    """Returns a normalized agent->runs mapping from new or legacy payloads."""
    runs = multi_seed_result.get("runs")
    if isinstance(runs, dict) and runs:
        return runs

    runs = {}
    if "original_runs" in multi_seed_result:
        runs["original"] = multi_seed_result["original_runs"]
    if "fast_runs" in multi_seed_result:
        runs["fast"] = multi_seed_result["fast_runs"]

    if not runs:
        raise ValueError("Expected multi_seed_result to contain 'runs' or legacy run keys")
    return runs


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
    print(
        f"   Avg Conversion                {original_result['avg_conversion']:.4f}          {fast_result['avg_conversion']:.4f}"
    )
    print(
        f"   Cumulative Regret             {original_result['cumulative_regret']:.2f}          {fast_result['cumulative_regret']:.2f}"
    )
    print(
        f"   Final RMSE                    {original_result['final_rmse']:.4f}          {fast_result['final_rmse']:.4f}"
    )
    print(
        f"   Runtime (seconds)             {original_result['runtime_seconds']:.2f}          {fast_result['runtime_seconds']:.2f}"
    )

    original_periods = original_result.get("period_summaries")
    fast_periods = fast_result.get("period_summaries")
    if original_periods and fast_periods and len(original_periods) == len(fast_periods):
        print("\n   Period Summary (environment change detected):")
        print(
            "   Period               Original Conv    Fast Conv    Oracle (Orig)   Oracle (Fast)"
        )
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
    original_oracle_avg = cumulative_average(
        original_result["oracle_conversion_history"]
    )
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
    if len(original_oracle_avg) == len(fast_oracle_avg) and not np.allclose(
        original_oracle_avg, fast_oracle_avg
    ):
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
    plt.plot(
        original_result["regret_history"], label="Original LinUCB", color=original_color
    )
    plt.plot(
        fast_result["regret_history"], label="Fast LinUCB", color=fast_color, alpha=0.9
    )
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret Over Time")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(
        original_result["mse_history"],
        label="Original LinUCB",
        color=original_color,
        alpha=0.5,
    )
    plt.plot(
        fast_result["mse_history"], label="Fast LinUCB", color=fast_color, alpha=0.5
    )
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.title("Prediction MSE Over Time")
    plt.legend()

    plt.tight_layout()
    plt.show()


def split_rolling_average(data, shock_step, window=1000):
    """Calculates a rolling average that respects structural breaks."""
    phase_1 = pd.Series(data[:shock_step]).rolling(window=window, min_periods=1).mean()
    phase_2 = pd.Series(data[shock_step:]).rolling(window=window, min_periods=1).mean()
    return pd.concat([phase_1, phase_2]).values


def plot_shock_comparison(original_result, fast_result, shock_step=17500):
    """Plots comparison with shock environment, highlighting the market crash event."""
    original_color = "tab:blue"
    fast_color = "tab:orange"
    oracle_color = "black"
    shock_color = "violet"

    plt.figure(figsize=(18, 5))

    rolling_window = 1000

    original_reward_rolling = split_rolling_average(
        original_result["reward_history"], shock_step, window=rolling_window
    )
    fast_reward_rolling = split_rolling_average(
        fast_result["reward_history"], shock_step, window=rolling_window
    )
    oracle_rolling = split_rolling_average(
        original_result["oracle_conversion_history"], shock_step, window=rolling_window
    )

    plt.subplot(1, 3, 1)
    plt.axvline(
        x=shock_step,
        color=shock_color,
        linestyle=":",
        linewidth=2.5,
        label="Economic Shock",
        alpha=0.7,
    )
    plt.plot(
        original_reward_rolling,
        label="Original LinUCB (rolling=1000)",
        color=original_color,
    )
    plt.plot(
        fast_reward_rolling,
        label="Fast LinUCB (rolling=1000)",
        color=fast_color,
        alpha=0.9,
    )
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

    plt.subplot(1, 3, 2)
    plt.axvline(
        x=shock_step,
        color=shock_color,
        linestyle=":",
        linewidth=2.5,
        label="Economic Shock",
        alpha=0.7,
    )
    plt.plot(
        original_result["regret_history"], label="Original LinUCB", color=original_color
    )
    plt.plot(
        fast_result["regret_history"], label="Fast LinUCB", color=fast_color, alpha=0.9
    )
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Regret")
    plt.title("Regret: Recovery Speed After Shock")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.axvline(
        x=shock_step,
        color=shock_color,
        linestyle=":",
        linewidth=2.5,
        label="Economic Shock",
        alpha=0.7,
    )
    plt.plot(
        original_result["mse_history"],
        label="Original LinUCB",
        color=original_color,
        alpha=0.5,
    )
    plt.plot(
        fast_result["mse_history"], label="Fast LinUCB", color=fast_color, alpha=0.5
    )
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.title("Prediction MSE: Model Resilience")
    plt.legend()

    plt.tight_layout()
    plt.show()


def print_multi_seed_summary(multi_seed_result):
    """Prints scalar mean/std summary for one or more multi-seed runs."""
    summary = multi_seed_result["summary"]
    runs_map = _extract_runs_map(multi_seed_result)
    agent_names = list(runs_map.keys())

    print("\nMulti-Seed Summary (mean +- std)")
    print(f"   Seeds: {multi_seed_result['seeds']}")
    for metric_key, label in [
        ("avg_conversion", "Avg Conversion"),
        ("cumulative_regret", "Cumulative Regret"),
        ("final_rmse", "Final RMSE"),
        ("runtime_seconds", "Runtime (seconds)"),
    ]:
        print(f"   {label}:")
        for agent_name in agent_names:
            stats = summary[metric_key][agent_name]
            print(
                f"      {agent_name:<20} {stats['mean']:.4f} +- {stats['std']:.4f}"
            )


def plot_multi_seed_comparison(multi_seed_result, ylim_dict=None):
    """Plots mean +- std trajectories across seeds for one or more algorithms.

    Args:
        multi_seed_result: Result payload from run_multi_seed_comparison.
        ylim_dict: Optional dict mapping metric name to (ymin, ymax) tuple.
            Valid keys: "conversion", "regret", "mse".
            Example: {"conversion": (0.1, 0.5), "mse": (0, 0.3)}
    """
    runs_map = _extract_runs_map(multi_seed_result)
    agent_names = list(runs_map.keys())
    cmap = plt.get_cmap("tab10")

    if ylim_dict is None:
        ylim_dict = {}

    def cumulative_average(data):
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            return arr
        return np.cumsum(arr) / np.arange(1, arr.size + 1)

    def mean_std_band(series_list):
        matrix = np.vstack(series_list)
        return np.mean(matrix, axis=0), np.std(matrix, axis=0)

    metric_specs = [
        (
            "Cumulative Average Conversion",
            "Conversion Across Seeds",
            lambda run: cumulative_average(run["reward_history"]),
            "conversion",
        ),
        (
            "Cumulative Regret",
            "Regret Across Seeds",
            lambda run: run["regret_history"],
            "regret",
        ),
        (
            "MSE",
            "Prediction MSE Across Seeds",
            lambda run: run["mse_history"],
            "mse",
        ),
    ]

    plt.figure(figsize=(18, 5))

    for subplot_idx, (y_label, title, extractor, metric_key) in enumerate(metric_specs, start=1):
        plt.subplot(1, 3, subplot_idx)
        for idx, agent_name in enumerate(agent_names):
            series_list = [extractor(run) for run in runs_map[agent_name]]
            mean, std = mean_std_band(series_list)
            x = np.arange(len(mean))
            color = cmap(idx % 10)

            plt.plot(x, mean, label=f"{agent_name} Mean", color=color)
            plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

        plt.xlabel("Steps")
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()

        if metric_key in ylim_dict:
            plt.ylim(ylim_dict[metric_key])

    plt.tight_layout()
    plt.show()


def plot_multi_seed_shock_comparison(
    multi_seed_result, shock_step, rolling_window=1000, ylim_dict=None
):
    """Plots shock scenario mean +- std across seeds for one or more algorithms.
    
    Args:
        multi_seed_result: Result payload from run_multi_seed_comparison.
        shock_step: Step where shock occurs in the environment.
        rolling_window: Window size for rolling average.
        ylim_dict: Optional dict mapping subplot names to (ymin, ymax) tuples.
            Valid keys: "conversion", "regret", "mse".
            Example: {"conversion": (0.1, 0.5)}
    """
    shock_color = "violet"
    runs_map = _extract_runs_map(multi_seed_result)
    agent_names = list(runs_map.keys())
    cmap = plt.get_cmap("tab10")

    if ylim_dict is None:
        ylim_dict = {}

    def mean_std_band(series_list):
        matrix = np.vstack(series_list)
        return np.mean(matrix, axis=0), np.std(matrix, axis=0)

    plt.figure(figsize=(18, 5))

    # Shock conversion subplot
    plt.subplot(1, 3, 1)
    plt.axvline(
        x=shock_step,
        color=shock_color,
        linestyle=":",
        linewidth=2.5,
        label="Economic Shock",
        alpha=0.7,
    )
    for idx, agent_name in enumerate(agent_names):
        series_list = [
            split_rolling_average(run["reward_history"], shock_step, rolling_window)
            for run in runs_map[agent_name]
        ]
        mean, std = mean_std_band(series_list)
        x = np.arange(len(mean))
        color = cmap(idx % 10)
        plt.plot(x, mean, label=f"{agent_name} Mean", color=color)
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    # Oracle reference from the first agent trajectory set.
    oracle_source_runs = runs_map[agent_names[0]]
    oracle_mean, oracle_std = mean_std_band(
        [
            split_rolling_average(
                run["oracle_conversion_history"], shock_step, rolling_window
            )
            for run in oracle_source_runs
        ]
    )
    x_oracle = np.arange(len(oracle_mean))
    plt.plot(
        x_oracle,
        oracle_mean,
        label=f"Oracle Mean (rolling={rolling_window})",
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    plt.fill_between(
        x_oracle,
        oracle_mean - oracle_std,
        oracle_mean + oracle_std,
        color="black",
        alpha=0.1,
    )
    plt.xlabel("Steps")
    plt.ylabel("Average Conversion")
    plt.title("Shock Conversion Across Seeds")
    plt.legend()
    
    if "conversion" in ylim_dict:
        plt.ylim(ylim_dict["conversion"])

    # Shock regret subplot
    plt.subplot(1, 3, 2)
    plt.axvline(
        x=shock_step,
        color=shock_color,
        linestyle=":",
        linewidth=2.5,
        label="Economic Shock",
        alpha=0.7,
    )
    for idx, agent_name in enumerate(agent_names):
        series_list = [run["regret_history"] for run in runs_map[agent_name]]
        mean, std = mean_std_band(series_list)
        x = np.arange(len(mean))
        color = cmap(idx % 10)
        plt.plot(x, mean, label=f"{agent_name} Mean", color=color)
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Regret")
    plt.title("Shock Regret Across Seeds")
    plt.legend()
    
    if "regret" in ylim_dict:
        plt.ylim(ylim_dict["regret"])

    # Shock MSE subplot
    plt.subplot(1, 3, 3)
    plt.axvline(
        x=shock_step,
        color=shock_color,
        linestyle=":",
        linewidth=2.5,
        label="Economic Shock",
        alpha=0.7,
    )
    for idx, agent_name in enumerate(agent_names):
        series_list = [run["mse_history"] for run in runs_map[agent_name]]
        mean, std = mean_std_band(series_list)
        x = np.arange(len(mean))
        color = cmap(idx % 10)
        plt.plot(x, mean, label=f"{agent_name} Mean", color=color)
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.title("Shock Prediction MSE Across Seeds")
    plt.legend()
    
    if "mse" in ylim_dict:
        plt.ylim(ylim_dict["mse"])

    plt.tight_layout()
    plt.show()


def plot_cold_vs_warm_start_comparison(cold_start_result, warm_start_result):
    """Compares cold-start vs warm-start for each shared algorithm key.

    Supports both the new generic payload format:
        {"runs": {"AgentName": [run1, run2, ...]}, ...}
    and legacy payloads containing original_runs/fast_runs.
    """

    def cumulative_average(data):
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            return arr
        return np.cumsum(arr) / np.arange(1, arr.size + 1)

    def mean_std_band(series_list):
        matrix = np.vstack(series_list)
        return np.mean(matrix, axis=0), np.std(matrix, axis=0)

    def _extract_algorithm_curves(runs):
        if not runs:
            raise ValueError("Expected at least one run to plot cold vs warm comparison")

        conv_mean, conv_std = mean_std_band(
            [cumulative_average(run["reward_history"]) for run in runs]
        )
        regret_mean, regret_std = mean_std_band([run["regret_history"] for run in runs])
        mse_mean, mse_std = mean_std_band([run["mse_history"] for run in runs])

        return {
            "conv": (conv_mean, conv_std),
            "regret": (regret_mean, regret_std),
            "mse": (mse_mean, mse_std),
        }

    cold_runs_map = _extract_runs_map(cold_start_result)
    warm_runs_map = _extract_runs_map(warm_start_result)

    shared_agent_names = [
        agent_name for agent_name in cold_runs_map.keys() if agent_name in warm_runs_map
    ]
    if not shared_agent_names:
        raise ValueError(
            "No shared agent keys found between cold_start_result and warm_start_result"
        )

    n_agents = len(shared_agent_names)
    fig_height = max(4 * n_agents, 6)
    plt.figure(figsize=(18, fig_height))

    def _draw_metric_subplot(position, metric_key, ylabel, title, cold_data, warm_data):
        cold_mean, cold_std = cold_data[metric_key]
        warm_mean, warm_std = warm_data[metric_key]

        x_cold = np.arange(len(cold_mean))
        x_warm = np.arange(len(warm_mean))

        plt.subplot(2, 3, position)
        plt.plot(x_cold, cold_mean, color="tab:blue", label="Cold-start Mean")
        plt.fill_between(
            x_cold,
            cold_mean - cold_std,
            cold_mean + cold_std,
            color="tab:blue",
            alpha=0.2,
            label="Cold-start +-1 std",
        )
        plt.plot(x_warm, warm_mean, color="tab:green", label="Warm-start Mean")
        plt.fill_between(
            x_warm,
            warm_mean - warm_std,
            warm_mean + warm_std,
            color="tab:green",
            alpha=0.2,
            label="Warm-start +-1 std",
        )
        plt.xlabel("Steps")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()

    for row_idx, agent_name in enumerate(shared_agent_names):
        cold_curves = _extract_algorithm_curves(cold_runs_map[agent_name])
        warm_curves = _extract_algorithm_curves(warm_runs_map[agent_name])
        base_pos = (row_idx * 3) + 1

        _draw_metric_subplot(
            base_pos,
            "conv",
            "Cumulative Average Conversion",
            f"{agent_name}: Conversion (Cold vs Warm)",
            cold_curves,
            warm_curves,
        )
        _draw_metric_subplot(
            base_pos + 1,
            "regret",
            "Cumulative Regret",
            f"{agent_name}: Regret (Cold vs Warm)",
            cold_curves,
            warm_curves,
        )
        _draw_metric_subplot(
            base_pos + 2,
            "mse",
            "MSE",
            f"{agent_name}: MSE (Cold vs Warm)",
            cold_curves,
            warm_curves,
        )

    plt.tight_layout()
    plt.show()

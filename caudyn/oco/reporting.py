import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    """Prints scalar mean/std summary for multi-seed runs."""
    summary = multi_seed_result["summary"]

    def _row(metric_key, label):
        orig = summary[metric_key]["original"]
        fast = summary[metric_key]["fast"]
        print(
            f"   {label:<24} "
            f"{orig['mean']:.4f} +- {orig['std']:.4f}    "
            f"{fast['mean']:.4f} +- {fast['std']:.4f}"
        )

    print("\nMulti-Seed Summary (mean +- std)")
    print(f"   Seeds: {multi_seed_result['seeds']}")
    print("   Metric                    Original                 Fast")
    _row("avg_conversion", "Avg Conversion")
    _row("cumulative_regret", "Cumulative Regret")
    _row("final_rmse", "Final RMSE")
    _row("runtime_seconds", "Runtime (seconds)")


def plot_multi_seed_comparison(multi_seed_result):
    """Plots mean +- std trajectories across seeds for both algorithms."""
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

    plt.subplot(1, 3, 1)
    plt.plot(
        x_conv, original_conv_mean, label="Original LinUCB Mean", color=original_color
    )
    plt.fill_between(
        x_conv,
        original_conv_mean - original_conv_std,
        original_conv_mean + original_conv_std,
        color=original_color,
        alpha=0.2,
        label="Original +-1 std",
    )
    plt.plot(x_conv, fast_conv_mean, label="Fast LinUCB Mean", color=fast_color)
    plt.fill_between(
        x_conv,
        fast_conv_mean - fast_conv_std,
        fast_conv_mean + fast_conv_std,
        color=fast_color,
        alpha=0.2,
        label="Fast +-1 std",
    )
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Average Conversion")
    plt.title("Conversion Across Seeds")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(
        x_regret,
        original_regret_mean,
        label="Original LinUCB Mean",
        color=original_color,
    )
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

    plt.subplot(1, 3, 3)
    plt.plot(
        x_mse, original_mse_mean, label="Original LinUCB Mean", color=original_color
    )
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


def plot_multi_seed_shock_comparison(
    multi_seed_result, shock_step, rolling_window=1000
):
    """Plots shock scenario mean +- std across seeds for both algorithms."""
    original_color = "tab:blue"
    fast_color = "tab:orange"
    shock_color = "violet"

    original_runs = multi_seed_result["original_runs"]
    fast_runs = multi_seed_result["fast_runs"]

    def mean_std_band(series_list):
        matrix = np.vstack(series_list)
        return np.mean(matrix, axis=0), np.std(matrix, axis=0)

    original_conv_mean, original_conv_std = mean_std_band(
        [
            split_rolling_average(run["reward_history"], shock_step, rolling_window)
            for run in original_runs
        ]
    )
    fast_conv_mean, fast_conv_std = mean_std_band(
        [
            split_rolling_average(run["reward_history"], shock_step, rolling_window)
            for run in fast_runs
        ]
    )
    oracle_conv_mean, oracle_conv_std = mean_std_band(
        [
            split_rolling_average(
                run["oracle_conversion_history"], shock_step, rolling_window
            )
            for run in original_runs
        ]
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
        x_conv,
        original_conv_mean,
        label=f"Original Mean (rolling={rolling_window})",
        color=original_color,
    )
    plt.fill_between(
        x_conv,
        original_conv_mean - original_conv_std,
        original_conv_mean + original_conv_std,
        color=original_color,
        alpha=0.2,
        label="Original +-1 std",
    )
    plt.plot(
        x_conv,
        fast_conv_mean,
        label=f"Fast Mean (rolling={rolling_window})",
        color=fast_color,
    )
    plt.fill_between(
        x_conv,
        fast_conv_mean - fast_conv_std,
        fast_conv_mean + fast_conv_std,
        color=fast_color,
        alpha=0.2,
        label="Fast +-1 std",
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
        x_regret, original_regret_mean, label="Original Mean", color=original_color
    )
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

    plt.subplot(1, 3, 3)
    plt.axvline(
        x=shock_step,
        color=shock_color,
        linestyle=":",
        linewidth=2.5,
        label="Economic Shock",
        alpha=0.7,
    )
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


def plot_cold_vs_warm_start_comparison(cold_start_result, warm_start_result):
    """Compares cold-start vs warm-start for each algorithm separately."""

    def cumulative_average(data):
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            return arr
        return np.cumsum(arr) / np.arange(1, arr.size + 1)

    def mean_std_band(series_list):
        matrix = np.vstack(series_list)
        return np.mean(matrix, axis=0), np.std(matrix, axis=0)

    def _extract_algorithm_curves(multi_seed_result, algo_key):
        runs = (
            multi_seed_result["original_runs"]
            if algo_key == "original"
            else multi_seed_result["fast_runs"]
        )

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

    cold_original = _extract_algorithm_curves(cold_start_result, "original")
    warm_original = _extract_algorithm_curves(warm_start_result, "original")
    cold_fast = _extract_algorithm_curves(cold_start_result, "fast")
    warm_fast = _extract_algorithm_curves(warm_start_result, "fast")

    plt.figure(figsize=(18, 10))

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

    _draw_metric_subplot(
        1,
        "conv",
        "Cumulative Average Conversion",
        "Original LinUCB: Conversion (Cold vs Warm)",
        cold_original,
        warm_original,
    )
    _draw_metric_subplot(
        2,
        "regret",
        "Cumulative Regret",
        "Original LinUCB: Regret (Cold vs Warm)",
        cold_original,
        warm_original,
    )
    _draw_metric_subplot(
        3,
        "mse",
        "MSE",
        "Original LinUCB: MSE (Cold vs Warm)",
        cold_original,
        warm_original,
    )
    _draw_metric_subplot(
        4,
        "conv",
        "Cumulative Average Conversion",
        "Fast LinUCB: Conversion (Cold vs Warm)",
        cold_fast,
        warm_fast,
    )
    _draw_metric_subplot(
        5,
        "regret",
        "Cumulative Regret",
        "Fast LinUCB: Regret (Cold vs Warm)",
        cold_fast,
        warm_fast,
    )
    _draw_metric_subplot(
        6, "mse", "MSE", "Fast LinUCB: MSE (Cold vs Warm)", cold_fast, warm_fast
    )

    plt.tight_layout()
    plt.show()

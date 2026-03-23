#---------------------------------------------------------*/
# Benchmarking Utilities for Model Evaluation
#---------------------------------------------------------*/

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from stable_baselines3 import PPO  # For loading models

from src.env import RobustBatchRegulatorEnv
from src.testing import (
    evaluate_random_agent,
    evaluate_static_agent,
    evaluate_rule_agent,
    evaluate_action_sequence,
    evaluate_policy_object
)
from utils.cma_optimizer import optimize_action_trajectory

# ============================================================
# --- Helper 1: Evaluate Fast Agents (Simple + PPO) ---
# ============================================================


def _run_single_seed_fast_agents(
    seed,
    max_steps,
    scaling_factor,
    total_size,
    min_batch_size,
    max_batch_size,
    sensor_noise_std,
    input_generator_type,
    ppo_vanilla_best_path,
    ppo_vanilla_final_path,
    ppo_bc_best_path,
    ppo_bc_final_path
):
    """Evaluates all fast agents (simple baselines + pre-trained PPOs) for one seed."""
    print(f"[Test Seed {seed}] Evaluating fast agents...")

    test_env = RobustBatchRegulatorEnv(
        max_steps=max_steps,
        seed=seed,
        sensor_noise_std=sensor_noise_std,
        input_generator_type=input_generator_type,
        scaling_factor=scaling_factor,
        total_size=total_size,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
    )

    results = {}

    # --- 1. Baselines ---
    results["Random"] = evaluate_random_agent(
        test_env, max_steps=max_steps, seed=seed, verbose=False, render=False
    )
    for act in [0.1, 0.5, 1.0]:
        results[f"Static {act:.1f}"] = evaluate_static_agent(
            test_env, static_action=act, max_steps=max_steps, seed=seed, verbose=False, render=False
        )
    results["Rule-Based"] = evaluate_rule_agent(
        test_env, max_steps=max_steps, seed=seed, verbose=False, render=False
    )

    # --- 2. PPO Vanilla ---
    if ppo_vanilla_best_path and os.path.exists(ppo_vanilla_best_path):
        model = PPO.load(ppo_vanilla_best_path, device="cpu")
        results["PPO (Best)"] = evaluate_policy_object(
            model, test_env, max_steps=max_steps, verbose=False, render=False
        )
    if ppo_vanilla_final_path and os.path.exists(ppo_vanilla_final_path):
        model = PPO.load(ppo_vanilla_final_path, device="cpu")
        results["PPO (Final)"] = evaluate_policy_object(
            model, test_env, max_steps=max_steps, verbose=False, render=False
        )

    # --- 3. PPO+BC ---
    if ppo_bc_best_path and os.path.exists(ppo_bc_best_path):
        model = PPO.load(ppo_bc_best_path, device="cpu")
        results["PPO+BC (Best)"] = evaluate_policy_object(
            model, test_env, max_steps=max_steps, verbose=False, render=False
        )
    if ppo_bc_final_path and os.path.exists(ppo_bc_final_path):
        model = PPO.load(ppo_bc_final_path, device="cpu")
        results["PPO+BC (Final)"] = evaluate_policy_object(
            model, test_env, max_steps=max_steps, verbose=False, render=False
        )

    print(f"[Test Seed {seed}] Finished evaluating fast agents.")
    return seed, results


# ============================================================
# --- Helper 2: Evaluate Slow Agent (CMA-ES Oracle, Sequential) ---
# [KEINE ÄNDERUNGEN HIER]
# ============================================================


def _run_single_seed_cma_es(
    seed,
    max_steps,
    scaling_factor,
    total_size,
    min_batch_size,
    max_batch_size,
    sensor_noise_std,
    input_generator_type,
    sigma_init,
    popsize,
    max_generations,
    num_workers=None,
    total_seeds=None,
    seed_idx=None,
):
    """Optimizes and evaluates CMA-ES sequentially for one test seed (tracks best result)."""
    desc_str = f"[{seed_idx}/{total_seeds}] CMA-ES (Seed {seed})"
    print(f"\n{desc_str} → Starting optimization...")

    test_env = RobustBatchRegulatorEnv(
        max_steps=max_steps,
        seed=seed,
        sensor_noise_std=sensor_noise_std,
        input_generator_type=input_generator_type,
        scaling_factor=scaling_factor,
        total_size=total_size,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
    )

    results = {}
    t_start_cma = time.perf_counter()

    # Run CMA-ES (with internal tqdm)
    optimal_actions = optimize_action_trajectory(
        env=test_env,
        num_steps=max_steps,
        seed=seed,
        scaling_factor=scaling_factor,
        sigma_init=sigma_init,
        popsize=popsize,
        max_generations=max_generations,
        num_workers=num_workers,
        verbose=True,
        seed_idx=seed_idx,
        total_seeds=total_seeds,
        plot_progress=False,
    )

    cumulative = evaluate_action_sequence(
        optimal_actions, test_env, seed=seed, verbose=False, render=False
    )
    t_end_cma = time.perf_counter()

    results["CMA-ES (Best)"] = cumulative
    results["time_cma"] = t_end_cma - t_start_cma

    cma_min, cma_sec = divmod(int(results["time_cma"]), 60)
    print(
        f"✅ Finished CMA-ES (Seed {seed}) | Time: {cma_min}m {cma_sec}s | Reward: {cumulative['cumulative_reward']:.2f}"
    )
    return seed, results


# ============================================================
# --- Helper 3: Data Loading ---
# ============================================================

def load_benchmark_data(data_path):
    """Loads benchmark results from a .npz file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    # NPZ stores items as keyword arguments
    data = np.load(data_path, allow_pickle=True)
    all_results = {k: data[k].tolist() for k in data.files}
    print(f"✅ Loaded benchmark data from {data_path}")
    return all_results


# ============================================================
# --- Helper 4: Plotting Function (Refactored) ---
# ============================================================

def plot_and_save_benchmark_results(
    all_results,
    agent_keys,
    all_times=None,
    save_path=None,
    num_seeds=None,
    data_path=None,  # New parameter for loading data
    skip_plot=False
):
    """
    Generates benchmark plots, saves raw data, and optionally loads data from file.
    """

    if data_path and os.path.exists(data_path):
        all_results = load_benchmark_data(data_path)
        # Re-derive agent keys based on loaded data
        agent_keys = [k for k in agent_keys if k in all_results]
        num_seeds = len(next(iter(all_results.values()))) if all_results else 0
    elif save_path and not all_results:
        print("⚠️ Cannot plot: No results provided and no data path specified.")
        return

    # --- 1. Filter and prepare data ---
    filtered_keys = [a for a in agent_keys if len(all_results.get(a, [])) > 0]
    if not filtered_keys:
        print("⚠️ No valid agents to plot.")
        return

    # Calculate mean and std for each agent
    agent_data = []
    for key in filtered_keys:
        rewards = np.array(all_results[key])
        if len(rewards) > 0:
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            sem_reward = std_reward / np.sqrt(len(rewards))  # Standard error of the mean
            agent_data.append((mean_reward, std_reward, sem_reward, key, rewards))

    # Keep agent order as in agent_keys (no sorting)
    means = [d[0] for d in agent_data]
    stds = [d[1] for d in agent_data]
    sems = [d[2] for d in agent_data]
    keys = [d[3] for d in agent_data]
    raw_results = [d[4] for d in agent_data]

    # Print raw data and stats
    print("\n--- Raw Data / Statistics ---")
    for key, rewards in zip(keys, raw_results):
        print(f"[{key}] N={len(rewards)}, Mean={np.mean(rewards):.2f}, Std={np.std(rewards):.2f}")
        print(f"    Raw Rewards (First 5): {rewards[:5].tolist()}...")
    print("-----------------------------")

    # --- Save Raw Data (only if a save_path is provided) ---
    if save_path:
        # Save raw data to NPZ file
        data_save_path = save_path.replace(".png", "_raw_data.npz")
        # Ensure all_results dict has all values as NumPy arrays for saving
        save_dict = {k: np.array(v) for k, v in all_results.items() if len(v) > 0}
        np.savez_compressed(data_save_path, **save_dict)
        print(f"✅ Raw benchmark data saved to {data_save_path}")

    if skip_plot:
        return

    # Formatting for labels (optional, for line breaks)
    def _format_label(key):
        if "(Best)" in key:
            return key.replace("(Best)", "\n(Best)")
        if "(Final)" in key:
            return key.replace("(Final)", "\n(Final)")
        if key.startswith("Static "):
            prefix, val = key.split(" ", 1)
            return f"{prefix}\n({val})"
        return key

    formatted_labels = [_format_label(k) for k in keys]

    # --- Plot initialization ---
    # REDUCED Y DIMENSION: figsize=(10, 6) instead of (10, 8)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 1]})
    plt.subplots_adjust(hspace=0.1)  # Reduced hspace for smaller figure size

    # --- 2. Upper plot: Vertical scatter with error bars ---
    ax1 = axes[0]
    x_pos = np.arange(len(keys))

    # Plot means with error bars (vertical)
    ax1.errorbar(
        x_pos,
        means,
        yerr=sems,
        fmt='o',
        color='steelblue',
        ecolor='slategray',
        capsize=5,
        markersize=8,
        label='Mean Reward ± SEM',
        zorder=3
    )

    # Optionally connect points with a line
    ax1.plot(x_pos, means, linestyle='--', color='lightgray', alpha=0.7, zorder=2)

    # Horizontal line at 0 for reference
    ax1.axhline(0, color='gray', linestyle=':', linewidth=0.8, zorder=1)

    # Text labels next to points, with minimum offset
    fixed_offset = 3.0
    for i, (mean_val, sem_val) in enumerate(zip(means, sems)):
        # Always place label fixed_offset below the bottom of the error bar
        text_y = mean_val - sem_val - fixed_offset
        ax1.text(
            x_pos[i],
            text_y,
            f"{mean_val:.2f}",
            ha='center',
            va='top',
            fontsize=10,
            color='darkslategray',
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
        )

    # Axis and title labels
    ax1.set_xticks(x_pos)
    # REMOVED X LABELS FROM AX1
    ax1.set_xticklabels([])
    ax1.set_ylabel("Mean Cumulative Reward", fontsize=12)

    # Grid lines for readability
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Adjust y-axis limits to show all error bars
    min_y = min(m - s for m, s in zip(means, sems))
    max_y = max(m + s for m, s in zip(means, sems))
    ax1.set_ylim(min_y - abs(min_y * 0.1), max_y + abs(max_y * 0.1))

    # --- 3. Lower plot: Boxplot of reward distributions (vertical) ---
    ax2 = axes[1]

    if raw_results:
        ax2.boxplot(
            raw_results,
            labels=formatted_labels,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
            showmeans=True,
            vert=True
        )
        # Connect means with a line (boxplot positions are 1-based)
        ax2.plot(x_pos + 1, means, linestyle='--', color='lightgray', alpha=0.7, zorder=2)

    ax2.set_ylabel("Cumulative Rewards per Seed", fontsize=12)
    ax2.set_xticklabels(formatted_labels, fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # --- Save and show ---
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        svg_path = save_path.replace(".png", ".svg")
        eps_path = save_path.replace(".png", ".eps")
        plt.savefig(save_path, dpi=300)
        plt.savefig(svg_path)
        plt.savefig(eps_path)
        print(f"\n✅ Benchmark plot saved to {save_path} (and .svg, .eps)")
    plt.show()


# ============================================================
# --- Sequential Benchmark Wrapper (Updated to save raw data) ---
# ============================================================


def benchmark_multiple_seeds(
    seeds,
    max_steps=100,
    scaling_factor=1000,
    total_size=1000,
    min_batch_size=15,
    max_batch_size=25,
    sensor_noise_std=0.0,
    input_generator_type="proportional",
    sigma_init=0.1,
    popsize=12,
    max_generations=50,
    save_dir="./img/benchmark",
    num_workers=4,
    # Pre-trained model paths
    ppo_vanilla_best_path=None,
    ppo_vanilla_final_path=None,
    ppo_bc_best_path=None,
    ppo_bc_final_path=None,
):
    """
    Two-stage sequential benchmark:
      1. Evaluate fast agents (baselines, PPOs)
      2. Sequentially optimize and evaluate CMA-ES (Best)
    """
    os.makedirs(save_dir, exist_ok=True)

    agent_keys_all = [
        "Random",
        "Static 0.1",
        "Static 0.5",
        "Static 1.0",
        "Rule-Based",
        "PPO (Best)",
        "PPO (Final)",
        "PPO+BC (Best)",
        "PPO+BC (Final)",
        "CMA-ES (Best)",
    ]
    agent_keys_preview = [k for k in agent_keys_all if "CMA-ES" not in k]

    all_results = {k: [] for k in agent_keys_all}
    all_times = {"CMA-ES (Best)": []}

    # === STAGE 1: Fast Agents ===
    print(f"🚀 Stage 1: Evaluating fast agents on {len(seeds)} seeds...")
    for seed in tqdm(seeds, desc="Fast Agent Evaluation", ncols=100):
        seed, result = _run_single_seed_fast_agents(
            seed,
            max_steps,
            scaling_factor,
            total_size,
            min_batch_size,
            max_batch_size,
            sensor_noise_std,
            input_generator_type,
            ppo_vanilla_best_path,
            ppo_vanilla_final_path,
            ppo_bc_best_path,
            ppo_bc_final_path,
        )
        for k, v in result.items():
            if k in all_results and isinstance(v, dict) and "cumulative_reward" in v:
                all_results[k].append(v["cumulative_reward"])

    # === Preview plot (fast agents only) ===
    preview_save_path = os.path.join(save_dir, f"benchmark_preview_{len(seeds)}seeds.png")
    plot_and_save_benchmark_results(
        all_results, agent_keys_preview, all_times, preview_save_path, len(seeds)
    )

    # === STAGE 2: CMA-ES Sequential ===
    print(f"\n🚀 Stage 2: Running sequential CMA-ES on {len(seeds)} seeds...")
    for i, seed in enumerate(seeds, start=1):
        seed, result_cma = _run_single_seed_cma_es(
            seed=seed,
            max_steps=max_steps,
            scaling_factor=scaling_factor,
            total_size=total_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            sensor_noise_std=sensor_noise_std,
            input_generator_type=input_generator_type,
            sigma_init=sigma_init,
            popsize=popsize,
            max_generations=max_generations,
            num_workers=num_workers,
            total_seeds=len(seeds),
            seed_idx=i,
        )

        for k, v in result_cma.items():
            if k in all_results and isinstance(v, dict) and "cumulative_reward" in v:
                all_results[k].append(v["cumulative_reward"])
            elif k in all_times:
                all_times[k].append(v)

    # === Final Plot ===
    final_save_path = os.path.join(save_dir, f"benchmark_final_{len(seeds)}seeds.png")
    plot_and_save_benchmark_results(
        all_results, agent_keys_all, all_times, final_save_path, len(seeds)
    )

    # === Stats (Now based on all_results, which holds the raw data) ===
    means = {k: np.mean(v) for k, v in all_results.items() if len(v) > 0}
    stds = {k: np.std(v) for k, v in all_results.items() if len(v) > 0}

    print("\n--- Benchmark Finished ---")
    for k, v in means.items():
        print(f"  {k}: Mean Reward = {v:.2f} ± {stds[k]:.2f}")

    return all_results, {"mean": means, "std": stds}


# ============================================================
# --- Execution Block for Plotting from Saved Data ---
# ============================================================

if __name__ == "__main__":

    # --- Configuration for loading data and defining output location ---

    # Define the local directory for the output image (e.g., "./img/")
    OUTPUT_DIR = "./img"

    # Define the base filename for the benchmark results
    FILENAME = "benchmark_final_20seeds.png"

    # The full path where the image will be saved
    final_plot_path = os.path.join(OUTPUT_DIR, FILENAME)

    # The path to the stored raw data file (this is an example path and should be adjusted)
    # NOTE: The data path needs to point to the actual saved .npz file.
    data_load_path = "/Users/tom/GitHub/2026_ES_RL/img/5_Final_Run_1764067437_proportional_scale_1000_ep_100/benchmark_final_20seeds_raw_data.npz"

    # List of all agents, as defined in benchmark_multiple_seeds
    agent_keys_order = [
        "Random",
        "Static 0.1",
        "Static 0.5",
        "Static 1.0",
        "Rule-Based",
        "PPO (Best)",
        "PPO (Final)",
        "PPO+BC (Best)",
        "PPO+BC (Final)",
        "CMA-ES (Best)",
    ]

    print("Starting plotting from saved data...")

    try:
        # 1. Plot the data using the loaded file
        # The save_path now includes the full directory and filename.
        plot_and_save_benchmark_results(
            all_results={},  # Ignored because data_path is set
            agent_keys=agent_keys_order,
            data_path=data_load_path,
            save_path=final_plot_path,  # Provide the full path for saving
            all_times=None,
            skip_plot=False
        )

    except FileNotFoundError as e:
        # English translation of the error message
        print(f"FATAL ERROR: {e}")
        print("Please ensure you have executed 'benchmark_multiple_seeds' first ")
        print("to generate the raw data (*_raw_data.npz).")
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")

# ---------------------------------------------------------*/
#
# ---------------------------------------------------------*/

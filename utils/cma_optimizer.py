# ---------------------------------------------------------*
# Title: CMA-ES Optimization for Robust Batch Regulator
# ---------------------------------------------------------*/

import numpy as np
import cma
from tqdm import tqdm
import os
import concurrent.futures
import matplotlib.pyplot as plt
import time

# ---------------------------------------------------------*/
# Single Trajectory Optimization
# ---------------------------------------------------------*/


def optimize_action_trajectory(
    env,
    num_steps=100,
    seed=None,
    scaling_factor=10_00,
    sigma_init=0.1,
    popsize=12,
    max_generations=None,
    num_workers=None,
    verbose=True,
    seed_idx=None,
    total_seeds=None,
    plot_progress=False,  # optional plotting,
    save_path="./img"
):
    """
    Optimize a length-num_steps action trajectory using CMA-ES.

    - Thread-parallelized evaluation of population members per generation.
    - tqdm progress bar per seed (if seed_idx/total_seeds provided).
    - Tracks the *best overall* solution across all generations.
    - Optional plotting of best-fitness evolution.
    """
    np.random.seed(seed)
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    if num_workers is None:
        num_workers = min(popsize, os.cpu_count())

    # Progress bar label
    desc_str = f"[{seed_idx}/{total_seeds}] CMA-ES (Seed {seed})" if seed_idx is not None else f"CMA-ES (Seed {seed})"
    pbar = tqdm(total=max_generations, desc=desc_str, ncols=100, leave=True)

    # CMA setup
    dim = num_steps
    x0 = np.random.uniform(0.3, 0.7, size=dim)
    es_opts = {
        "seed": seed,
        "popsize": popsize,
        "bounds": [0.1, 1.0],
        "verb_disp": 0,
        "verbose": -9,
    }
    if max_generations is not None:
        es_opts["maxiter"] = int(max_generations)
    es = cma.CMAEvolutionStrategy(x0, sigma_init, es_opts)

    # Helper functions
    def quantize01(v):
        v = np.clip(v, 0.1, 1.0)
        if scaling_factor and scaling_factor > 0:
            v = np.round(v * scaling_factor) / scaling_factor
        return np.clip(v, 0.1, 1.0)

    def map_to_env_range(a):
        return action_low + (action_high - action_low) * a

    def evaluate_solution(s):
        """Thread-safe deterministic evaluation of a single CMA-ES candidate."""
        # Create an isolated environment copy for this candidate
        local_env = env.__class__(
            max_steps=env.max_steps,
            seed=seed,
            sensor_noise_std=env.sensor_noise_std,
            input_generator_type=env.input_generator_type,
            scaling_factor=env.scaling_factor,
            total_size=env.total_size,
            min_batch_size=env.min_batch_size,
            max_batch_size=env.max_batch_size,
        )

        # Local action bounds
        action_low = float(local_env.action_space.low[0])
        action_high = float(local_env.action_space.high[0])

        # Quantize and map actions
        a01 = quantize01(s)
        obs, _ = local_env.reset(seed=seed)
        total_reward = 0.0

        for a in a01:
            a_env = map01_to_env(a, action_low, action_high)
            action = np.array([a_env], dtype=np.float32)
            obs, r, done, _, _ = local_env.step(action)
            total_reward += r
            if done:
                break

        return -total_reward  # CMA minimizes

    # --- Optimization loop ---
    best_fitness_per_gen = []
    best_overall_reward = -np.inf
    best_overall_solution = None

    while not es.stop():
        solutions = es.ask()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            fitness = list(executor.map(evaluate_solution, solutions))

        es.tell(solutions, fitness)

        # Extract generation's best
        gen_rewards = [-f for f in fitness]
        best_gen_reward = float(np.max(gen_rewards))
        best_fitness_per_gen.append(best_gen_reward)

        # Check global best so far
        if best_gen_reward > best_overall_reward:
            best_overall_reward = best_gen_reward
            best_overall_solution = quantize01(solutions[int(np.argmin(fitness))])

        pbar.update(1)
        pbar.set_postfix({"BestGenReward": f"{best_gen_reward:.2f}", "GlobalBest": f"{best_overall_reward:.2f}"})

    pbar.close()

    if verbose:
        print(f"✅ Seed {seed} finished | Best overall reward: {best_overall_reward:.2f}")

    # --- Optional plot ---
    if plot_progress and len(best_fitness_per_gen) > 1:
        generations = np.arange(1, len(best_fitness_per_gen) + 1)
        best_idx = int(np.argmax(best_fitness_per_gen))
        best_val = best_fitness_per_gen[best_idx]
        final_val = best_fitness_per_gen[-1]

        plt.figure(figsize=(8, 5))
        plt.plot(generations, best_fitness_per_gen, label="Best reward per generation")
        plt.scatter(best_idx + 1, best_val, color="red", marker="x", s=100,
                    label=f"Overall best: {best_val:.2f}")
        plt.scatter(generations[-1], final_val, color="green", marker="o", s=80,
                    label=f"Final best: {final_val:.2f}")
        plt.xlabel("Generation")
        plt.ylabel("Best cumulative reward")
        plt.title(f"CMA-ES Progress (Seed {seed})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        save_dir = save_path
        os.makedirs(save_dir, exist_ok=True)
        png_path = os.path.join(save_dir, f"cma_es_progress_seed_{seed}.png")
        svg_path = os.path.join(save_dir, f"cma_es_progress_seed_{seed}.svg")
        plt.savefig(png_path, dpi=300)
        plt.savefig(svg_path)
        plt.show()

    # Return the *global best* action sequence across all generations
    best_actions = best_overall_solution if best_overall_solution is not None else quantize01(es.result.xbest)
    return best_actions


def map01_to_env(a01: float, action_low: float, action_high: float) -> float:
    a01 = float(np.clip(a01, 0.1, 1.0))
    return action_low + (action_high - action_low) * a01

# ---------------------------------------------------------*/
# Compare CMA-ES Population Sizes
# ---------------------------------------------------------*/


def run_cma_tuning(
    env,
    popsize_list,
    max_generations=25,
    num_steps=100,
    seed=None,
    scaling_factor=1000,
    sigma_init=0.1,
    verbose=True,
    dir=None
):
    """
    Runs CMA-ES for a fixed number of generations across different population sizes
    and generates a single comparison plot.
    Fully standalone version (no external optimize_action_trajectory required).
    """

    if not popsize_list:
        if verbose:
            print("No population sizes provided.")
        return {}

    if verbose:
        print(f"--- Comparing CMA-ES for popsizes {popsize_list} (max_gen={max_generations}) ---")

    results = {}
    colors = plt.cm.jet(np.linspace(0, 1, len(popsize_list)))
    all_rewards = []

    # Loop over each population size
    for i, popsize in enumerate(popsize_list):
        if verbose:
            print(f"\nRunning CMA-ES optimization for popsize = {popsize}...")

        np.random.seed(seed)
        dim = num_steps
        x0 = np.random.uniform(0.3, 0.7, size=dim)

        action_low = float(env.action_space.low[0])
        action_high = float(env.action_space.high[0])

        es_opts = {
            "seed": seed,
            "popsize": popsize,
            "bounds": [0.1, 1.0],
            "verb_disp": 0,
            "verbose": -9,
        }
        if max_generations is not None:
            es_opts["maxiter"] = int(max_generations)

        es = cma.CMAEvolutionStrategy(x0, sigma_init, es_opts)

        def quantize01(v):
            v = np.clip(v, 0.1, 1.0)
            if scaling_factor and scaling_factor > 0:
                v = np.round(v * scaling_factor) / scaling_factor
            return np.clip(v, 0.1, 1.0)

        def map_to_env_range(a):
            return action_low + (action_high - action_low) * a

        pbar = (
            tqdm(total=max_generations, desc=f"CMA-ES (pop={popsize})", ncols=80, leave=False)
            if verbose and (max_generations is not None)
            else None
        )

        best_fitness_per_gen = []

        while not es.stop():
            solutions = es.ask()
            fitness = []

            for s in solutions:
                a01 = quantize01(s)
                obs, _ = env.reset(seed=seed)
                total_reward = 0.0

                for a in a01:
                    action = np.array([map_to_env_range(a)], dtype=np.float32)
                    obs, r, terminated, truncated, _ = env.step(action)
                    total_reward += r
                    if terminated or truncated:
                        break

                fitness.append(-total_reward)

            es.tell(solutions, fitness)

            best_gen_reward = -float(np.min(fitness))
            best_fitness_per_gen.append(best_gen_reward)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"best_gen_reward": f"{best_gen_reward:.2f}"})
            elif verbose:
                print(f"gen {es.countiter:3d} | best_gen_reward = {best_gen_reward:.2f}")

        if pbar is not None:
            pbar.close()

        if verbose:
            print(f"✅ Finished popsize={popsize} | Best reward: {-es.result.fbest:.2f}")

        # Store fitness history
        results[popsize] = best_fitness_per_gen
        all_rewards.extend(best_fitness_per_gen)

    # ----- Plot results -----
    plt.figure(figsize=(10, 6))
    for i, popsize in enumerate(popsize_list):
        fitness_history = results[popsize]
        gens = np.arange(1, len(fitness_history) + 1)
        final_value = fitness_history[-1]
        best_idx = np.argmax(fitness_history)
        best_value = fitness_history[best_idx]
        color = colors[i]

        label = f"Popsize={popsize:<3} (Best: {best_value:7.2f}, Final: {final_value:7.2f})"
        plt.plot(gens, fitness_history, label=label, color=color, alpha=0.8)
        plt.scatter(best_idx + 1, best_value, color=color, marker="x", s=100, zorder=5)
        plt.scatter(gens[-1], final_value, color=color, marker="o", s=80, zorder=5)

    if verbose:
        print("\n--- Generating comparison plot ---")

    plt.xlabel("Generation")
    plt.ylabel("Best cumulative reward")
    plt.title(f"CMA-ES Progress vs. Population Size (Generations={max_generations}, Steps={num_steps})")
    if all_rewards:
        plot_min_y = np.min(all_rewards)
        plot_max_y = max(1, np.max(all_rewards))
        plt.ylim(plot_min_y * 0.95, plot_max_y * 1.05)

    plt.legend(loc="best", fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    max_gen = max([len(results[p]) for p in popsize_list])

    save_dir = dir
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, "cma_tuning_comparison.png")
    svg_path = os.path.join(save_dir, "cma_tuning_comparison.svg")
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)

    plt.xticks(np.arange(1, max_gen + 1, 5))
    plt.show()

    return results


# ----------------------------------------------------------------*
# --- Single CMA Worker (used by batch function)
# ----------------------------------------------------------------*/

def _run_cma_worker(
    seed,
    env_class,
    num_steps,
    scaling_factor,
    sigma_init,
    max_generations,
    popsize,
    input_generator_type,
    total_size,
    min_batch_size,
    max_batch_size,
    sensor_noise_std,
    num_workers,
    seed_idx,
    total_seeds
):
    """Single CMA-ES run for one environment (with its own progress bar)."""
    env = env_class(
        max_steps=num_steps,
        seed=seed,
        scaling_factor=scaling_factor,
        sensor_noise_std=sensor_noise_std,
        input_generator_type=input_generator_type,
        total_size=total_size,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size
    )

    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    best_actions = optimize_action_trajectory(
        env=env,
        num_steps=num_steps,
        seed=seed,
        scaling_factor=scaling_factor,
        sigma_init=sigma_init,
        max_generations=max_generations,
        popsize=popsize,
        verbose=False,
        num_workers=num_workers,
        seed_idx=seed_idx,
        total_seeds=total_seeds
    )

    obs_list, act_list, next_obs_list, rew_list, done_list = [], [], [], [], []
    obs, _ = env.reset(seed=seed)

    for a_cma in best_actions:
        action_env = (a_cma * 2.0) - 1.0
        action_env_clipped = np.clip(action_env, action_low, action_high)
        act_step = np.array([action_env_clipped], dtype=np.float32)

        obs_list.append(obs.copy())
        obs, reward, done, _, _ = env.step(act_step)
        next_obs_list.append(obs.copy())
        act_list.append(act_step)
        rew_list.append(reward)
        done_list.append(done)
        if done:
            break

    if len(done_list) < num_steps:
        done_list.extend([True])

    return {
        "seed": seed,
        "observations": np.array(obs_list, dtype=np.float32),
        "actions": np.array(act_list, dtype=np.float32),
        "rewards": np.array(rew_list, dtype=np.float32)
    }

# ---------------------------------------------------------*/
# Batched Trajectory Optimization (Sequential)
# ---------------------------------------------------------*/


def batch_optimize_cma_trajectories(
    env_class,
    num_envs=4,
    num_steps=100,
    seeds=None,
    scaling_factor=10_00,
    sigma_init=0.1,
    save_path="./models/demo_data_bc.npy",
    verbose=True,
    max_generations=50,
    popsize=12,
    input_generator_type="proportional",
    total_size=1000,
    min_batch_size=15,
    max_batch_size=25,
    sensor_noise_std=0.0,
    num_workers=None
):
    """
    Sequential CMA-ES optimization for multiple seeds.
    Each seed shows its own progress bar with the format:
      [i/total_n] CMA-ES (Seed x)
    """
    if seeds is None:
        seeds = list(range(num_envs))

    total_seeds = len(seeds)

    if verbose:
        print(f"--- Launching sequential CMA-ES batch optimization for {total_seeds} environments ---")
        print(f"Popsize={popsize}, Generations={max_generations}, Workers per run={num_workers or os.cpu_count()}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = []

    for idx, s in enumerate(seeds, start=1):
        result = _run_cma_worker(
            seed=s,
            env_class=env_class,
            num_steps=num_steps,
            scaling_factor=scaling_factor,
            sigma_init=sigma_init,
            max_generations=max_generations,
            popsize=popsize,
            input_generator_type=input_generator_type,
            total_size=total_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            sensor_noise_std=sensor_noise_std,
            num_workers=num_workers,
            seed_idx=idx,
            total_seeds=total_seeds
        )
        results.append(result)

    obs_all = np.stack([r["observations"] for r in results])
    act_all = np.stack([r["actions"] for r in results])
    rew_all = np.stack([r["rewards"] for r in results])

    demo_data = {
        "observations": obs_all,
        "actions": act_all,
        "rewards": rew_all
    }

    np.save(save_path, demo_data, allow_pickle=True)

    if verbose:
        print(f"\n✅ Saved demonstration data to: {save_path}")
        print(f"   → Observations: {obs_all.shape}")
        print(f"   → Actions:      {act_all.shape}")
        print(f"   → Rewards:      {rew_all.shape}")

    return demo_data


# -------------------------Notes-----------------------------------------------*\
# - Each seed now displays its own tqdm progress bar.
# - No overall global progress bar anymore.
# - Format: "[i/total_n] CMA-ES (Seed x)" for clarity.
# - Thread-parallel population evaluation (deterministic).
# -----------------------------------------------------------------------------*\

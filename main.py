# ---------------------------------------------------------*
# Title: Main execution script for the ES-RL Benchmark
# ---------------------------------------------------------*/
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Core Environment
from src.env import RobustBatchRegulatorEnv

# Evaluation utilities
from src.testing import (
    evaluate_random_agent,
    evaluate_static_agent,
    evaluate_rule_agent,
    evaluate_action_sequence,
    evaluate_policy_object
)

# CMA-ES Experiments
from utils.cma_optimizer import optimize_action_trajectory, batch_optimize_cma_trajectories, run_cma_tuning

# Training and Benchmarking
from src.training import train_agent
from utils.benchmark_models import benchmark_multiple_seeds

# Helper Utilities
from utils.input_generator import visualize_batch_system
from utils.plotting import plot_evaluation_results
from imitation.data.types import Transitions  # For BC data conversion
from stable_baselines3 import PPO  # To load final models

# ---------------------------------------------------------*\
# --- CONFIGURATION PROFILE ---
# ---------------------------------------------------------*
#
from configs.full_config import *
# from configs.small_config import *

# ---------------------------------------------------------*/
# Helper Function
# ---------------------------------------------------------*/


def convert_demo_dict_to_transitions(demo_data_dict, env):
    """Convert the dictionary produced by `batch_optimize_cma_trajectories` (or loaded
    from disk) into an `imitation.data.types.Transitions` object.

    This ensures `obs`, `acts`, `next_obs`, `dones`, and `infos` are present and
    shaped correctly for Behaviour Cloning training.
    """
    obs_flat = demo_data_dict["observations"].reshape(-1, env.observation_space.shape[0])
    act_flat = demo_data_dict["actions"].reshape(-1, env.action_space.shape[0])

    # next_obs: shift observations by one and repeat final observation for padding
    next_obs_flat = np.vstack([obs_flat[1:], obs_flat[-1:]])

    # Build dones: mark episode ends according to rewards array shape
    num_episodes, num_steps = demo_data_dict["rewards"].shape
    dones_flat = np.zeros(len(obs_flat), dtype=bool)
    for i in range(num_episodes):
        dones_flat[(i + 1) * num_steps - 1] = True

    infos = np.array([{} for _ in range(len(obs_flat))])

    return Transitions(obs=obs_flat, acts=act_flat, next_obs=next_obs_flat, dones=dones_flat, infos=infos)

# ---------------------------------------------------------*\
# Main Execution Logic
# ---------------------------------------------------------*/


def run_experiment():
    """Main function to run the selected experiment."""

    # ---------------------------------------------------------*
    # Create dedicated run directory inside ./img/
    # ---------------------------------------------------------*/
    run_dir = os.path.join("img", TAG)
    os.makedirs(run_dir, exist_ok=True)
    print(f"📁 Output directory: {run_dir}\n")

    # ---------------------------------------------------------*/
    # Experiment Header
    # ---------------------------------------------------------*/
    print("\n-------------------------------------------")
    print(f"Starting Experiment run with tag: {TAG}")
    print(f"→ Input Generator: {INPUT_GENERATOR_TYPE}")
    print(f"→ Scaling Factor:  {SCALING_FACTOR}")
    print("-------------------------------------------\n")

    print("--- Seed configuration ---")
    print(f"Test Seeds:     {TEST_SEEDS}")
    print(f"Training Seeds: {TRAINING_SEEDS}")
    print(f"PPO Eval Seeds: {PPO_EVAL_SEEDS}")
    print(f"CMA Demo Seeds: {CMA_DEMO_SEEDS}")
    print("---------------------------\n")

    # ---------------------------------------------------------*/
    # --- Create Master Test Environment ---
    # ---------------------------------------------------------*/
    # This env is used for all single-run evaluations
    test_env = RobustBatchRegulatorEnv(
        max_steps=MAX_STEPS,
        seed=SINGLE_TEST_SEED,  # <-- Use Test-Seed
        sensor_noise_std=SENSOR_NOISE_STD,
        input_generator_type=INPUT_GENERATOR_TYPE,
        scaling_factor=SCALING_FACTOR,
        total_size=TOTAL_SIZE,
        min_batch_size=MIN_BATCH_SIZE,
        max_batch_size=MAX_BATCH_SIZE
    )

    # This dictionary will collect all single-run results
    results = {}

    # Paths to trained models, to be passed to the benchmark
    ppo_vanilla_best_path = None
    ppo_vanilla_final_path = None
    ppo_bc_best_path = None
    ppo_bc_final_path = None

    # BC demonstration data
    demo_transitions = None

    # ---------------------------------------------------------*/
    # --- Unified Environment Test Block ---
    # ---------------------------------------------------------*/
    if TEST_ENV:
        print(f"\n--- Running Environment Evaluation Suite (Seed: {SINGLE_TEST_SEED}) ---")
        visualize_batch_system(seed=SINGLE_TEST_SEED, scaling_factor=SCALING_FACTOR, save_dir=run_dir)
        results["Random"] = evaluate_random_agent(
            test_env, max_steps=MAX_STEPS, seed=SINGLE_TEST_SEED, verbose=True, render=True, save_dir=run_dir)
        for act in [0.1, 0.5, 1.0]:
            results[f"Static {act:.1f}"] = evaluate_static_agent(
                test_env, static_action=act, max_steps=MAX_STEPS, seed=SINGLE_TEST_SEED, save_dir=run_dir, verbose=True, render=True, custom_title_part=f"Static Action {act:.1f}")
        results["Rule-Based"] = evaluate_rule_agent(test_env, max_steps=MAX_STEPS,
                                                    seed=SINGLE_TEST_SEED, verbose=True, render=True, save_dir=run_dir)

        plot_evaluation_results(results, dir=run_dir, env=test_env)

    # ---------------------------------------------------------*/
    # --- PPO Training (Vanilla) ---
    # ---------------------------------------------------------*/
    if TRAIN_PPO_AGENT:
        print(f"\n--- (FACTORY) Training Standard PPO agent (Seed: {SINGLE_TRAIN_SEED}) ---")

        train_env_ppo = RobustBatchRegulatorEnv(
            max_steps=MAX_STEPS, seed=SINGLE_TRAIN_SEED,
            sensor_noise_std=SENSOR_NOISE_STD, input_generator_type=INPUT_GENERATOR_TYPE,
            scaling_factor=SCALING_FACTOR, total_size=TOTAL_SIZE,
            min_batch_size=MIN_BATCH_SIZE, max_batch_size=MAX_BATCH_SIZE
        )

        # train_agent now returns (loaded_best_model, best_path, final_path)
        loaded_best_model, ppo_vanilla_best_path, ppo_vanilla_final_path = train_agent(
            env=train_env_ppo, run_dir=run_dir, total_timesteps=PPO_TIMESTEPS,
            tag=f"{TAG}_PPO_Vanilla", seed=SINGLE_TRAIN_SEED, use_bc=False,
            eval_episodes=5, progress_bar=True, device="cpu",
            eval_freq=PPO_EVAL_FREQ, verbose=True, eval_seeds=PPO_EVAL_SEEDS
        )

        # --- Evaluate immediately for the single-run dashboard ---
        print(f"--- Evaluating PPO Vanilla on Test Seed {SINGLE_TEST_SEED} ---")
        # 1. Evaluate the BEST model (already loaded)
        results["PPO (Best)"] = evaluate_policy_object(
            loaded_best_model, test_env, max_steps=MAX_STEPS, save_dir=run_dir,
            verbose=False, tag= "PPO (Best)", render=True  # Keep it quiet
        )
        # 2. Load and evaluate the FINAL model
        if ppo_vanilla_final_path and os.path.exists(ppo_vanilla_final_path):
            final_model = PPO.load(ppo_vanilla_final_path, device='cpu')
            results["PPO (Final)"] = evaluate_policy_object(
                final_model, test_env, max_steps=MAX_STEPS, save_dir=run_dir,
                verbose=False, tag="PPO (Final)", render=True
            )
        print(f"→ PPO (Best) Reward: {results.get('PPO (Best)', {}).get('cumulative_reward', 'N/A'):.2f}")
        print(f"→ PPO (Final) Reward: {results.get('PPO (Final)', {}).get('cumulative_reward', 'N/A'):.2f}")

        plot_evaluation_results(results, dir=run_dir, env=test_env)

    # ---------------------------------------------------------*/
    # --- CMA-ES Hyperparameter Tuning ---
    # ---------------------------------------------------------*/
    if RUN_CMA_TUNING:
        print(f"\n--- Running CMA-ES Hyperparameter Tuning (Seed: {SINGLE_TEST_SEED}) ---")
        test_env.reset(seed=SINGLE_TEST_SEED)
        run_cma_tuning(
            env=test_env, popsize_list=POP_SIZES, max_generations=MAX_GEN,
            num_steps=MAX_STEPS, seed=SINGLE_TEST_SEED,
            sigma_init=SIGMA_INIT, verbose=True, dir=run_dir
        )

    # ---------------------------------------------------------*/
    # --- CMA-ES Single Trajectory Optimization ---
    # ---------------------------------------------------------*/
    if RUN_CMA_SINGLE:
        print(f"\n🚀 Running CMA-ES (Oracle) (Seed: {SINGLE_TEST_SEED})...")
        test_env.reset(seed=SINGLE_TEST_SEED)
        test_env.sensor_noise_std = 0.0  # Oracle run is deterministic

        optimal_actions = optimize_action_trajectory(
            env=test_env, num_steps=MAX_STEPS, seed=SINGLE_TEST_SEED,
            scaling_factor=SCALING_FACTOR, sigma_init=SIGMA_INIT,
            popsize=POP_SIZE, max_generations=MAX_GEN, verbose=True, plot_progress=True, save_path=run_dir)

        results["CMA-ES"] = evaluate_action_sequence(optimal_actions, test_env,
                                                     seed=SINGLE_TEST_SEED, render=True, verbose=True, save_dir=run_dir)

        print(f"   → Best cumulative reward: {results['CMA-ES']['cumulative_reward']:.2f}")

        plot_evaluation_results(results, dir=run_dir, env=test_env)

    # ---------------------------------------------------------*/
    # --- CMA-ES Batch Trajectory Optimization ---
    # ---------------------------------------------------------*/
    if RUN_CMA_BATCH:

        print("\n🚀 (FACTORY) Running Batched CMA-ES Demonstration Generation...")

        demo_data_dict = batch_optimize_cma_trajectories(
            env_class=RobustBatchRegulatorEnv,
            num_envs=CMA_BATCH_NUM_ENVS,
            num_steps=MAX_STEPS,
            seeds=CMA_DEMO_SEEDS,
            scaling_factor=SCALING_FACTOR,
            sigma_init=SIGMA_INIT,
            save_path="./models/demo_data_bc.npy",
            verbose=True,
            max_generations=MAX_GEN,
            popsize=POP_SIZE,
            input_generator_type=INPUT_GENERATOR_TYPE,
            total_size=TOTAL_SIZE,
            min_batch_size=MIN_BATCH_SIZE,
            max_batch_size=MAX_BATCH_SIZE,
            sensor_noise_std=SENSOR_NOISE_STD,
            num_workers=NUM_WORKERS
        )
        try:
            demo_transitions = convert_demo_dict_to_transitions(demo_data_dict, test_env)
            print(f"✅ Demonstrations successfully converted (Total: {len(demo_transitions)} steps).")
        except Exception as e:
            print(f"❌ Error converting demonstration data: {e}")

    # ---------------------------------------------------------*/
    # --- PPO Training after CMA-ES Pretraining ---
    # ---------------------------------------------------------*/
    if TRAIN_PPO_CMA:
        # If we didn't just generate the data, try to load it from disk
        if demo_transitions is None:
            print("No demo data in memory, trying to load from disk...")
            demo_path = "./models/demo_data_bc.npy"
            if os.path.exists(demo_path):
                try:
                    demo_data_dict = np.load(demo_path, allow_pickle=True).item()
                    demo_transitions = convert_demo_dict_to_transitions(demo_data_dict, test_env)
                    print("✅ Loaded and converted data from disk.")
                except Exception as e:
                    print(f"Failed to load/convert from disk: {e}")
            else:
                print(f"File not found: {demo_path}")

        if demo_transitions is not None:
            print(f"\n--- (FACTORY) Training PPO+CMA agent (Seed: {SINGLE_TRAIN_SEED}) ---")

            cma_train_env = RobustBatchRegulatorEnv(
                max_steps=MAX_STEPS, seed=SINGLE_TRAIN_SEED,
                sensor_noise_std=SENSOR_NOISE_STD, input_generator_type=INPUT_GENERATOR_TYPE,
                scaling_factor=SCALING_FACTOR, total_size=TOTAL_SIZE,
                min_batch_size=MIN_BATCH_SIZE, max_batch_size=MAX_BATCH_SIZE
            )

            loaded_best_bc_model, ppo_bc_best_path, ppo_bc_final_path = train_agent(
                env=cma_train_env, run_dir=run_dir, total_timesteps=PPO_TIMESTEPS,
                tag=f"{TAG}_PPO_CMA", use_bc=True, bc_data=demo_transitions,
                bc_epochs=10, seed=SINGLE_TRAIN_SEED, eval_episodes=5,
                progress_bar=True, device="cpu", eval_freq=PPO_EVAL_FREQ,
                verbose=True, eval_seeds=PPO_EVAL_SEEDS
            )

            # --- Evaluate immediately for the single-run dashboard ---
            print(f"--- Evaluating PPO+CMA on Test Seed {SINGLE_TEST_SEED} ---")
            # 1. Evaluate the BEST model (already loaded)
            results["PPO+BC (Best)"] = evaluate_policy_object(
                loaded_best_bc_model, test_env, max_steps=MAX_STEPS, save_dir=run_dir,
                verbose=False, render=True, tag= "PPO+BC (Best)"
            )
            # 2. Load and evaluate the FINAL model
            if ppo_bc_final_path and os.path.exists(ppo_bc_final_path):
                final_bc_model = PPO.load(ppo_bc_final_path, device='cpu')
                results["PPO+BC (Final)"] = evaluate_policy_object(
                    final_bc_model, test_env, max_steps=MAX_STEPS, save_dir=run_dir,
                    verbose=False, render=True, tag="PPO+BC (Final)"
                )
            print(f"→ PPO+BC (Best) Reward: {results.get('PPO+BC (Best)', {}).get('cumulative_reward', 'N/A'):.2f}")
            print(f"→ PPO+BC (Final) Reward: {results.get('PPO+BC (Final)', {}).get('cumulative_reward', 'N/A'):.2f}")
        else:
            print("❌ Cannot train PPO+CMA: No demonstration data available.")

        plot_evaluation_results(results, dir=run_dir, env=test_env)

    # ---------------------------------------------------------*/
    # --- Multi-Seed Benchmark Experiment ---
    # ---------------------------------------------------------*/
    if RUN_BENCHMARK:
        print("\n--- Starting Final Multi-Seed Benchmark Evaluation ---")

        all_results, stats = benchmark_multiple_seeds(
            seeds=TEST_SEEDS,  # <-- Benchmark now runs on TEST_SEEDS
            max_steps=MAX_STEPS,
            scaling_factor=SCALING_FACTOR,
            total_size=TOTAL_SIZE,
            min_batch_size=MIN_BATCH_SIZE,
            max_batch_size=MAX_BATCH_SIZE,
            sensor_noise_std=SENSOR_NOISE_STD,
            input_generator_type=INPUT_GENERATOR_TYPE,
            save_dir=run_dir,
            num_workers=NUM_WORKERS,

            # --- Pass the pre-trained models ---
            ppo_vanilla_best_path=ppo_vanilla_best_path,
            ppo_vanilla_final_path=ppo_vanilla_final_path,
            ppo_bc_best_path=ppo_bc_best_path,
            ppo_bc_final_path=ppo_bc_final_path,

            # --- Pass CMA-ES params ---
            sigma_init=SIGMA_INIT,
            popsize=POP_SIZE,
            max_generations=MAX_GEN
        )

    print("\n-------------------------------------------")
    print("Main script finished successfully.")
    print("-------------------------------------------")


# ---------------------------------------------------------*\
# Main Entry Point
# ---------------------------------------------------------*/
if __name__ == "__main__":
    run_experiment()

# ---------------------------------------------------------*/
#
# ---------------------------------------------------------*/

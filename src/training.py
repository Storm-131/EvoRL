# ---------------------------------------------------------*\
# PPO Agent Training for the Robust Batch Regulator
# ---------------------------------------------------------*/

import os
import shutil
import numpy as np
from typing import List
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from imitation.algorithms import bc
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import rollout
from imitation.data.types import Transitions

from src.env import RobustBatchRegulatorEnv

# Imports for plotting helper
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import warnings
warnings.filterwarnings("ignore", message=".*symmetric and normalized Box action space.*")


# ---------------------------------------------------------*
# Helper Functions
# ---------------------------------------------------------*/

def save_ppo_model(model, prefix: str, timesteps: int):
    """Saves a PPO model with a given prefix and timestep count."""
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)

    # Clean up previous models with the same prefix
    prev_dir = os.path.join(models_dir, "prev")
    for f in os.listdir(models_dir):
        if f.startswith(prefix) and f.endswith(".zip"):
            os.makedirs(prev_dir, exist_ok=True)
            shutil.move(os.path.join(models_dir, f), os.path.join(prev_dir, f))

    # Save the new model
    model_path = os.path.join(models_dir, f"{prefix}_{timesteps}.zip")
    model.save(model_path)
    print(f"✅ Saved new {prefix} model to: {model_path}")


def _create_eval_env(base_env: RobustBatchRegulatorEnv, seed_val: int):
    """
    Helper function to create a new, monitored instance of the environment
    for evaluation, using the same params as the base_env but a new seed.
    """
    def _init():
        eval_env = RobustBatchRegulatorEnv(
            max_steps=base_env.max_steps,
            seed=seed_val,  # Set the specific eval seed
            sensor_noise_std=base_env.sensor_noise_std,
            input_generator_type=base_env.input_generator_type,
            scaling_factor=base_env.scaling_factor,
            total_size=base_env.total_size,
            min_batch_size=base_env.min_batch_size,
            max_batch_size=base_env.max_batch_size,
        )
        return Monitor(eval_env)  # Wrap in Monitor
    return _init


def plot_training_progress_from_tb(log_dir, tag_name, save_path, eval_npz_path=None):
    """
    Reads a TensorBoard log file and (optionally) an evaluations.npz file
    to plot training and evaluation reward curves.
    """
    print(f"Attempting to generate plot from TensorBoard log: {log_dir}")
    train_steps, train_values = None, None
    eval_steps, eval_values = None, None

    try:
        # --- 1. Load Training Data (from TensorBoard) ---
        ea = EventAccumulator(log_dir)
        ea.Reload()

        if tag_name not in ea.Tags()['scalars']:
            print(f"Error: Tag '{tag_name}' not found in logs. Available tags: {ea.Tags()['scalars']}")
            return

        scalar_events = ea.Scalars(tag_name)
        train_steps = [event.step for event in scalar_events]
        train_values = [event.value for event in scalar_events]

        if not train_steps:
            print(f"Error: No data found for tag '{tag_name}'.")

        # --- 2. Load Evaluation Data (from .npz) ---
        if eval_npz_path and os.path.exists(eval_npz_path):
            try:
                eval_data = np.load(eval_npz_path)
                eval_steps = eval_data['timesteps']
                # 'results' is (n_evals, n_episodes). We need the mean across episodes.
                if 'results' in eval_data and eval_data['results'].ndim == 2:
                    eval_values = eval_data['results'].mean(axis=1)
                elif 'results' in eval_data:
                    eval_values = eval_data['results']  # Fallback if already 1D
                print(f"Successfully loaded evaluation data from {eval_npz_path}")
            except Exception as e:
                print(f"Warning: Could not load eval data from {eval_npz_path}: {e}")

        # --- 3. Create the Plot ---
        plt.figure(figsize=(12, 6))

        # Plot Training
        if train_steps:
            plt.plot(train_steps, train_values, label="Training Reward (rollout/ep_rew_mean)", alpha=0.8)

        # Plot Evaluation
        if eval_steps is not None and eval_values is not None:
            plt.plot(eval_steps, eval_values, label="Evaluation Reward (Mean)", marker='o', linestyle='--')

        plt.xlabel("Timesteps")
        plt.ylabel("Reward (Mean)")
        plt.title(f"Training & Evaluation Progress ({os.path.basename(log_dir)})")
        plt.grid(True)
        plt.legend()

        # Save and close
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        svg_path = save_path.replace(".png", ".svg")
        plt.savefig(save_path, dpi=300)
        plt.savefig(svg_path)
        plt.show()  # Use close() instead of show() to not block the script
        print(f"✅ Training progress plot saved to {save_path} (and .svg)")

    except Exception as e:
        print(f"❌ Error generating plot from TensorBoard logs: {e}")

# ---------------------------------------------------------*
# Behavioral Cloning (BC) for Warm-Starting
# ---------------------------------------------------------*/


def load_demonstrations_for_bc(demo_folder: str, env: RobustBatchRegulatorEnv) -> Transitions:
    """
    Loads expert demonstrations from JSON files and formats them for the imitation library.
    (This is a placeholder, you must adapt it to load your .npy file)
    """
    raise NotImplementedError("This function is a placeholder. Data is loaded in main.py.")


def train_behavioral_cloning(env: RobustBatchRegulatorEnv, demo_folder: str, n_epochs: int = 100) -> bc.BC:
    """
    Trains a policy using Behavioral Cloning (BC) from demonstration data.
    """
    transitions = load_demonstrations_for_bc(demo_folder, env)

    print("\n----------------------------------------")
    print(f"🏋🏽‍♂️ Training BC policy from {len(transitions.obs)} transitions...")
    print("----------------------------------------\n")

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=np.random.RandomState(42),
    )

    bc_trainer.train(n_epochs=n_epochs)
    print("\n✅ Behavioral Cloning training completed.\n")

    return bc_trainer

# ---------------------------------------------------------*
# Main PPO Training Function
# ---------------------------------------------------------*/


def train_agent(
    env: RobustBatchRegulatorEnv,
    run_dir: str,
    total_timesteps: int = 100_000,
    tag: str = "ppo_run",
    use_bc: bool = False,
    bc_data=None,  # Expects a Transitions object or path
    bc_epochs: int = 50,
    seed: int = 42,
    eval_episodes: int = 10,
    progress_bar=False,
    device="cpu",
    eval_freq: int = 5000,
    verbose: bool = False,
    eval_seeds: List[int] = [100, 101, 102, 103, 104]
):
    """
    Trains a PPO agent, optionally warm-started by BC.
    Saves the BEST and FINAL models.
    Returns the loaded best model, and paths to both best and final models.
    """
    # Wrap train env for monitoring and checks
    env_mon = Monitor(env)
    check_env(env_mon)

    # --- Create vectorized evaluation environments ---
    if verbose:
        print(f"Creating {len(eval_seeds)} evaluation environments with seeds: {eval_seeds}")

    eval_vec_env = DummyVecEnv(
        [_create_eval_env(env, seed) for seed in eval_seeds]
    )

    # ---------------------------------------------------------*/
    # --- Define EvalCallback ---
    # ---------------------------------------------------------*/
    best_model_save_path = f"./log/models/best_{tag}/"
    eval_log_path = os.path.join(best_model_save_path, "eval_logs")
    os.makedirs(best_model_save_path, exist_ok=True)
    os.makedirs(eval_log_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=best_model_save_path,
        log_path=eval_log_path,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
        verbose=0
    )

    POLICY_KWARGS = dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],  # Policy + Value Layer
        activation_fn=torch.nn.Tanh                  # gleiche Aktivierungsfunktion wie SB3-Default
    )

    # ---------------------------------------------------------*/
    # --- Behavioral Cloning Setup ---
    # ---------------------------------------------------------*/
    pretrained_policy = None
    if use_bc:
        if isinstance(bc_data, str):
            # Assumes a function `load_demonstrations_for_bc` exists
            print("Loading demonstrations from path...")
            transitions = load_demonstrations_for_bc(bc_data, env)
        elif isinstance(bc_data, Transitions):
            print("Using provided Transitions object for BC.")
            transitions = bc_data
        else:
            raise ValueError("bc_data must be a path (str) or a Transitions object.")

        # ---------------------------------------------------------*/
        # Train BC policy
        # ---------------------------------------------------------*/

        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            rng=np.random.RandomState(seed),
        )

        bc_trainer.train(n_epochs=bc_epochs)
        pretrained_policy = bc_trainer.policy
        if verbose:
            print("🧠 BC pre-training complete.")

    # ---------------------------------------------------------*/
    # --- Init PPO Model ---
    # ---------------------------------------------------------*/
    policy_kwargs = dict(
        net_arch=dict(pi=[32, 32], vf=[32, 32]),  # <-- Dictionary
        activation_fn=torch.nn.Tanh,
    )

    model = PPO(
        "MlpPolicy",
        env_mon,
        verbose=0,
        tensorboard_log="./log/tensorboard/",
        seed=seed,
        device=device,
        policy_kwargs=policy_kwargs,
    )

    # Load BC weights into PPO policy if available
    if use_bc and pretrained_policy is not None:
        try:
            model.policy.load_state_dict(pretrained_policy.state_dict(), strict=False)
            if verbose:
                print("🧠 PPO policy initialized from BC policy weights.")
        except Exception as e:
            print(f"⚠️ Could not load BC weights into PPO policy: {e}")

    # ---------------------------------------------------------*/
    # --- Train PPO ---
    # ---------------------------------------------------------*/
    if verbose:
        print(f"Starting PPO training for {total_timesteps} timesteps...")

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=progress_bar,
        callback=eval_callback,
        tb_log_name=tag
    )

    if verbose:
        print("PPO training finished.")

    # --- Get log paths ---
    actual_log_dir = model.logger.get_dir()
    eval_results_path = os.path.join(eval_log_path, "evaluations.npz")

    # --- Plot training curve ---
    if verbose:
        plot_save_path = os.path.join(run_dir, f"ppo_training_curve_{tag}.png")
        plot_training_progress_from_tb(
            log_dir=actual_log_dir,
            tag_name="rollout/ep_rew_mean",
            save_path=plot_save_path,
            eval_npz_path=eval_results_path
        )

    # --- Save FINAL model ---
    # 'model' is the last-step model
    final_model_path = os.path.join(best_model_save_path, "final_model.zip")
    model.save(final_model_path)
    if verbose:
        print(f"✅ Saved FINAL model to: {final_model_path}")

    # --- Load BEST model ---
    best_model_path = os.path.join(best_model_save_path, "best_model.zip")
    loaded_model = model  # Fallback: return the final model

    if eval_freq <= total_timesteps:
        if os.path.exists(best_model_path) and os.path.exists(eval_results_path):
            try:
                eval_data = np.load(eval_results_path)

                timesteps = eval_data["timesteps"]
                results = eval_data["results"]

                # --- Robust aggregation across episodes ---
                if results.ndim == 2:
                    scores = results.mean(axis=1)
                else:
                    scores = results

                assert len(timesteps) == len(scores), (
                    f"Mismatch: timesteps={len(timesteps)} vs scores={len(scores)}"
                )

                best_idx = int(np.nanargmax(scores))
                best_timestep = int(timesteps[best_idx])

                if verbose:
                    print(f"✅ Loading BEST model (from timestep {best_timestep})")

                loaded_model = PPO.load(best_model_path, env=env)

            except Exception as e:
                if verbose:
                    print(f"⚠️ Error loading best model: {e}. Returning the *last* trained model.")
        else:
            if verbose:
                print("⚠️ Could not find 'best_model.zip'. Returning the *last* trained model.")
    else:
        if verbose:
            print("Callback was disabled. Returning the *last* trained model.")

    # Return the loaded model (best or final) and paths to both
    return loaded_model, best_model_path, final_model_path

# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\

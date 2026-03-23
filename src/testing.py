# ---------------------------------------------------------*\
# Evaluation Utilities for the Robust Batch Regulator
# ---------------------------------------------------------*/

import numpy as np
import torch


# ---------------------------------------------------------*/
# Baseline Checks
# ---------------------------------------------------------*/

def evaluate_random_agent(env, max_steps=100, seed=None, verbose=True, render=True, save_dir="./img"):
    """
    Evaluates a random agent for a single episode with a specified number of steps.
    Returns the cumulative reward and step-wise reward list.
    """
    if verbose:
        print(f"--- Evaluating Random Agent for {max_steps} steps ---")

    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    cumulative_reward = 0.0
    step_rewards = []

    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        cumulative_reward += reward
        step_rewards.append(reward)
        if done:
            break

    result = {
        "cumulative_reward": cumulative_reward,
        "step_rewards": step_rewards,
        "cumulative_input": np.cumsum(getattr(env, "quantity_history", [0])),
        "container_contents": getattr(env, "container_contents", {}),
    }
    
    if render:
        env.render(
            show=True,
            save=True,
            log_dir=save_dir,
            filename="random_agent",
            title_seed=f"Seed {seed}" if seed is not None else "",
            custom_title_part="Random Agent Evaluation"
        )

    if verbose:
        print(f"Final Cumulative Reward: {cumulative_reward:.2f}\n")
    
    return result


def evaluate_static_agent(env, static_action, max_steps=100,
                          seed=None, custom_title_part="", verbose=True, render=True, save_dir="./img"):
    """
    Evaluates a static agent that always performs the same action for a fixed number of steps.
    Returns the cumulative reward and step-wise reward list.
    """
    if verbose:
        print(f"\n--- Evaluating Static Agent with action={static_action:.2f} for {max_steps} steps ---")

    # Map [0, 1] → [-1, 1] for the environment action space
    action_to_perform = np.array([2 * np.clip(static_action, 0.0, 1.0) - 1], dtype=np.float32)

    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed)
    cumulative_reward = 0.0
    step_rewards = []

    for _ in range(max_steps):
        obs, reward, done, truncated, info = env.step(action_to_perform)
        cumulative_reward += reward
        step_rewards.append(reward)
        if done:
            break

    if render:
        env.render(
            show=True,
            save=True,
            log_dir=save_dir,
            filename=f"static_agent_a{static_action:.2f}",
            title_seed=f"Seed {seed}" if seed is not None else "",
            custom_title_part=custom_title_part
        )

    if verbose:
        print(f"✅ Evaluation finished.")
        print(f"   - Static action: {static_action:.2f}")
        print(f"   - Steps: {len(step_rewards)}")
        print(f"   - Cumulative Reward: {cumulative_reward:.2f}\n")

    result = {
        "cumulative_reward": cumulative_reward,
        "step_rewards": step_rewards,
        "cumulative_input": np.cumsum(getattr(env, "quantity_history", [0])),
        "container_contents": getattr(env, "container_contents", {}),
    }
    return result


def evaluate_rule_agent(env, max_steps=100,
                        start_action=0.5,
                        step_up=0.005,
                        step_down=0.005,
                        seed=None,
                        verbose=True,
                        render=True,
                        save_dir="./img"):
    """
    Evaluates a simple rule-based agent for a single episode.
    - Increase input only if BOTH purities are at or above their thresholds.
    - Otherwise decrease input.
    Action is clamped to [0.1, 1.0].
    Returns the cumulative reward and step-wise reward list.
    """
    if verbose:
        print(f"--- Evaluating Rule-Based Agent for {max_steps} steps ---")

    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed)
    cumulative_reward = 0.0
    step_rewards = []
    action_value = float(np.clip(start_action, 0.1, 1.0))

    min_action, max_action = 0.1, 1.0
    thr_A = env.purity_thresholds["A"]
    thr_B = env.purity_thresholds["B"]

    for _ in range(max_steps):
        # map [0, 1] → [-1, 1] for environment
        action = np.array([2 * action_value - 1], dtype=np.float32)
        obs, reward, done, _, info = env.step(action)

        cumulative_reward += reward
        step_rewards.append(reward)

        # get current purities from env info
        pA = info.get("purity_A", getattr(env, "last_purity_A", 1.0))
        pB = info.get("purity_B", getattr(env, "last_purity_B", 1.0))

        # rule: increase only if BOTH >= thresholds, else decrease
        if (pA >= thr_A) and (pB >= thr_B):
            action_value = min(max_action, action_value + step_up)
        else:
            action_value = max(min_action, action_value - step_down)

        if done:
            break

    if render:
        env.render(
            show=True,
            save=True,
            log_dir=save_dir,
            filename="rule_based_agent",
            title_seed=f"Seed {seed}" if seed is not None else "",
            custom_title_part="Rule-Based Evaluation"
        )

    if verbose:
        print(f"Final Cumulative Reward: {cumulative_reward:.2f} | Final Action: {action_value:.2f}\n")

    result = {
        "cumulative_reward": cumulative_reward,
        "step_rewards": step_rewards,
        "cumulative_input": np.cumsum(getattr(env, "quantity_history", [0])),
        "container_contents": getattr(env, "container_contents", {}),
    }
    return result


# ---------------------------------------------------------*/
# CMA-ES and BC Policy Evaluations
# ---------------------------------------------------------*/

def map01_to_env(a01: float, action_low: float, action_high: float) -> float:
    a01 = float(np.clip(a01, 0.1, 1.0))
    return action_low + (action_high - action_low) * a01

def evaluate_action_sequence(sequence, env, seed=None, verbose=True, render=True, save_dir="./img",
                             sequence_space: str = "01"):
    """
    sequence_space: "01"  -> values in [0.1, 1.0] (CMA output), map with map01_to_env
                     "env" -> values already in env action space, only clip to Box
    """
    if verbose:
        print(f"--- Evaluating fixed action sequence with {len(sequence)} steps ---")

    # Debugging: print RNG state checksum
    # print(f"Thread eval start: RNG state checksum = {hash(str(env.np_random.bit_generator.state)[:100])}")

    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    cumulative_reward = 0.0
    step_rewards = []

    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    for a in sequence:
        if sequence_space == "01":
            a_env = map01_to_env(a, action_low, action_high)
            act = np.array([a_env], dtype=np.float32)
        else:  # "env"
            # Assume already in env space; clip to ensure validity
            a_env = float(np.clip(a, action_low, action_high))
            act = np.array([a_env], dtype=np.float32)

        obs, reward, done, _, _ = env.step(act)
        cumulative_reward += reward
        step_rewards.append(reward)
        if done:
            break

    if render:
        env.render(
            show=True,
            save=True,
            log_dir=save_dir,
            filename="cma_es_evaluation",
            title_seed=f"Seed {seed}" if seed is not None else "",
            custom_title_part="CMA-ES Evaluation"
        )

    if verbose:
        print(f"-> Final Sequence Reward: {cumulative_reward:.2f}\n")

    result = {
        "cumulative_reward": cumulative_reward,
        "step_rewards": step_rewards,
        "cumulative_input": np.cumsum(getattr(env, "quantity_history", [0])),
        "container_contents": getattr(env, "container_contents", {}),
    }
    return result



def evaluate_policy_object(policy, env, max_steps=100, num_episodes=None, seed=None,
                           verbose=True, render=True, tag="RL", save_dir="./img"):
    """
    Evaluates a given policy object (e.g., from BC, CMA-ES, or PPO) for one or several episodes.
    Returns the mean cumulative reward and step-wise reward list of the last episode.
    """
    if verbose:
        print(f"--- Evaluating Policy Object ---")

    # Default: single episode if num_episodes not set
    if num_episodes is None:
        num_episodes = 1

    total_rewards = []
    is_sb3_model = hasattr(policy, 'predict')

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed)
        cumulative_reward = 0.0
        step_rewards = []

        for _ in range(max_steps):
            if is_sb3_model:
                action, _ = policy.predict(obs, deterministic=True)
            elif isinstance(policy, torch.nn.Module):  # PyTorch model
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).float()
                    action = policy(obs_tensor).numpy()
            else:
                raise TypeError("Unsupported policy type. Must have .predict() or be a PyTorch Module.")

            obs, reward, done, _, _ = env.step(action)
            cumulative_reward += reward
            step_rewards.append(reward)
            if done:
                break

        total_rewards.append(cumulative_reward)

    mean_reward = np.mean(total_rewards)
    title = f"Policy Evaluation (PPO - {tag})" if tag else "Policy Evaluation"
             
    if render:
        env.render(
            show=True,
            save=True,
            log_dir=save_dir,
            filename=tag,
            title_seed=f"Seed {seed}" if seed is not None else "",
            custom_title_part=title
        )

    if verbose:
        print(f"→ Mean Reward over {num_episodes} episodes: {mean_reward:.2f}")

    result = {
        "cumulative_reward": cumulative_reward,
        "step_rewards": step_rewards,
        "cumulative_input": np.cumsum(getattr(env, "quantity_history", [0])),
        "container_contents": getattr(env, "container_contents", {}),
    }
    return result


# -------------------------Notes-----------------------------------------------*\
# - Plots are saved under the current run directory (img/<TAG>/...).
# - Each function supports optional seed and consistent rendering.
# -----------------------------------------------------------------------------*\

#---------------------------------------------------------*/
# Robust Batch Regulator Environment
#---------------------------------------------------------*/ 

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque
from utils.input_generator import BatchInputGenerator
from utils.plotting import plot_env_state


class RobustBatchRegulatorEnv(gym.Env):
    """
    A Gymnasium environment for the "Robust Batch Regulator" problem.
    The agent controls the normalized input quantity (0–1 range) to maximize throughput
    while maintaining the purity of sortable products A and B.
    Material X acts as an uncontrollable contaminant.
    """

    def __init__(self,
                 max_steps=100,
                 seed=None,
                 sensor_noise_std=0.00,
                 purity_thresholds=None,
                 input_generator_type="random",
                 scaling_factor=10_000,
                 total_size=10_000,
                 min_batch_size=500,
                 max_batch_size=1500,
                 ):

        super().__init__()

        # --- Basic configuration ---
        self.max_steps = max_steps
        self.sensor_noise_std = sensor_noise_std
        self.scaling_factor = scaling_factor
        self.total_size = total_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.input_generator_type = input_generator_type

        if purity_thresholds is None:
            self.purity_thresholds = {"A": 0.75, "B": 0.70}
        else:
            self.purity_thresholds = purity_thresholds

        self.material_names = ["A", "B", "X"]

        # --- Random number generator ---
        master_rng = np.random.default_rng(seed)
        self.np_random = master_rng
        subseed = master_rng.integers(1e9)

        # --- Input generator setup ---
        self.input_generator = BatchInputGenerator(
            materials=self.material_names,
            seed=int(subseed),
            sampling_mode=input_generator_type,
            total_size=self.total_size,
            min_batch_size=self.min_batch_size,
            max_batch_size=self.max_batch_size,
            scaling_factor=self.scaling_factor
        )

        # --- Action and observation spaces ---
        min_action = 2 * 0.1 - 1.0   # = -0.8, so that minimum action is 0.1 after mapping
        self.action_space = spaces.Box(low=min_action, high=1.0, shape=(1,), dtype=np.float32)

        # Observation = last 5 ratios (A/B mixture) + purities of A and B
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

        # --- Internal state ---
        self.current_step = 0
        self.ratio_history = None
        self.last_purity_A = 1.0
        self.last_purity_B = 1.0

        self.container_A = None
        self.container_B = None
        self.container_X = None
        self._history = {}

        self.quantity_history = []
        
        self.container_contents = {
            "A": {"A": 0, "B": 0, "X": 0},
            "B": {"A": 0, "B": 0, "X": 0},
            "X": {"A": 0, "B": 0, "X": 0},
        }
        
        self.reset(seed=seed)

    # ---------------------------------------------------------*/
    # Reset
    # ---------------------------------------------------------*/
    def reset(self, seed=None):
        """Resets the environment and the input generator."""
        super().reset(seed=seed)

        if seed is not None:
            master_rng = np.random.default_rng(seed)
            self.np_random = master_rng
            subseed = master_rng.integers(1e9)
            self.input_generator.reset(seed=int(subseed))
        else:
            self.input_generator.reset(seed=None)
            
        self.current_step = 0
        self.ratio_history = deque([0.5] * 5, maxlen=5)
        self.last_purity_A = 1.0
        self.last_purity_B = 1.0

        self.container_A = {"A": 0, "B": 0, "X": 0}
        self.container_B = {"A": 0, "B": 0, "X": 0}
        self.container_X = {"A": 0, "B": 0, "X": 0}

        self._history = {
            "rewards": [],
            "reward_quantity": [],
            "reward_quality": [],
            "quantities": [],
            "accuracies_A": [],
            "accuracies_B": [],
            "purities_A": [],
            "purities_B": [],
            "composition_A": [],
            "composition_B": [],
            "composition_X": [],
            "ratio_A": [],
            "ratio_B": [],
            "ratio_X": [],
            "actions_raw": [],
            "actions_scaled": []
        }

        self.quantity_history = []

        self.container_contents = {"A": {"A": 0, "B": 0, "X": 0},
                                   "B": {"A": 0, "B": 0, "X": 0},
                                   "X": {"A": 0, "B": 0, "X": 0}}

        observation = self._get_obs()
        return observation, {}

    # ---------------------------------------------------------*/
    # Step Function
    # ---------------------------------------------------------*/
    def step(self, action):
        """Executes one environment step.

        Action: float in [-1, 1].
        The InputGenerator scales it internally to integer material units.
        """
        # 1. Normalize and forward to generator
        # Map [-1, 1] to [0, 1] for input generator
        q_input = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
        composition = self.input_generator.draw_samples(quantity=q_input)

        # 2. Simulate sorting process
        accuracies = self._update_containers(composition)

        # 3. Compute purities and reward
        current_purity_A, current_purity_B = self._calculate_purities()
        total_quantity = sum(composition.values())
        reward = self._calculate_total_reward(total_quantity, current_purity_A, current_purity_B)

        # 4. Update internal state
        self.last_purity_A = current_purity_A
        self.last_purity_B = current_purity_B

        total_ab = composition["A"] + composition["B"]
        true_ratio_ab = composition["A"] / total_ab if total_ab > 0 else 0.5
        noisy_ratio = np.clip(true_ratio_ab + self.np_random.normal(0, self.sensor_noise_std), 0, 1)
        self.ratio_history.append(noisy_ratio)

        # Log actions
        self._last_action_raw = q_input
        self._last_action_scaled = q_input * self.scaling_factor

        # 5. Log data
        self._log_history(reward, composition, accuracies, current_purity_A, current_purity_B)
        self.quantity_history.append(sum(composition.values()))
        self.container_contents = {
            "A": dict(self.container_A),
            "B": dict(self.container_B),
            "X": dict(self.container_X)
        }

        # 6. Termination
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False

        observation = self._get_obs()
        info = {
            "purity_A": current_purity_A,
            "purity_B": current_purity_B,
            "composition": composition,
            "action_raw": self._last_action_raw,
            "action_scaled": self._last_action_scaled
        }
        return observation, reward, done, truncated, info

    # ---------------------------------------------------------*/
    # Observation Construction
    # ---------------------------------------------------------*/
    def _get_obs(self):
        """Constructs the observation vector."""
        return np.concatenate([
            np.array(self.ratio_history, dtype=np.float32),
            np.array([self.last_purity_A, self.last_purity_B], dtype=np.float32)
        ])

    # ---------------------------------------------------------*/
    # Sorting Simulation and Container Updates
    # ---------------------------------------------------------*/
    def _get_sorting_accuracy(self, material_type: str, current_load: float) -> float:
        """
        Strong degradation version:
        - Material A: hard linear drop
        - Material B: strongly curved (exponent 2.5)
        """

        # Normalize to [0, 1]
        load_ratio = min(current_load / self.scaling_factor, 1.0)

        if material_type == "A":
            # Strong linear drop
            min_acc, max_acc = 0.70, 0.95
            delta = max_acc - min_acc
            base_accuracy = max_acc - delta * load_ratio    # linear

        elif material_type == "B":
            # Strong curved drop, heavily convex downward
            min_acc, max_acc = 0.60, 0.85
            delta = max_acc - min_acc
            base_accuracy = max_acc - delta * (load_ratio ** 2.5)

        else:
            return 0.0

        # Add noise component
        noise = self.np_random.uniform(0, self.sensor_noise_std)
        return float(np.clip(base_accuracy - noise, 0.0, 1.0))


    def _simulate_sequential_sorting(self, composition: dict):
        # Copy incoming material stream so we can mutate safely
        stream = composition.copy()

        # Per-step outputs of stage A and stage B
        step_container_A = {"A": 0, "B": 0, "X": 0}
        step_container_B = {"A": 0, "B": 0, "X": 0}
        accuracy_A, accuracy_B = 0.0, 0.0

        # -------- Stage A: target = "A" --------
        total_quantity = sum(stream.values())
        if total_quantity > 0:
            # Load-dependent accuracy for stage A
            accuracy_A = self._get_sorting_accuracy("A", total_quantity)

            # Correctly sorted A goes to container A
            correctly_sorted_A = stream["A"] * accuracy_A
            step_container_A["A"] += correctly_sorted_A
            stream["A"] -= correctly_sorted_A

            # Only non-A material can be (wrongly) sent to A
            nonA_before_to_A = stream["B"] + stream["X"]
            incorrect_material_to_A = nonA_before_to_A * (1 - accuracy_A)
            if nonA_before_to_A > 0:
                # Split misrouted material to A proportionally to its residual mix
                ratio_B = stream["B"] / nonA_before_to_A
                step_container_A["B"] += incorrect_material_to_A * ratio_B
                step_container_A["X"] += incorrect_material_to_A * (1 - ratio_B)

                # Remove the misrouted amounts from the residual stream
                stream["B"] -= incorrect_material_to_A * ratio_B
                stream["X"] -= incorrect_material_to_A * (1 - ratio_B)

        # -------- Stage B: target = "B" (on the residual) --------
        total_remaining = sum(stream.values())
        if total_remaining > 0:
            # Load-dependent accuracy for stage B
            accuracy_B = self._get_sorting_accuracy("B", total_remaining)

            # Correctly sorted B goes to container B
            correctly_sorted_B = stream["B"] * accuracy_B
            step_container_B["B"] += correctly_sorted_B
            stream["B"] -= correctly_sorted_B

            # Only non-B material can be (wrongly) sent to B
            nonB_before_to_B = stream["A"] + stream["X"]
            incorrect_material_to_B = nonB_before_to_B * (1 - accuracy_B)
            if nonB_before_to_B > 0:
                # Split misrouted material to B proportionally to its residual mix
                ratio_A = stream["A"] / nonB_before_to_B
                step_container_B["A"] += incorrect_material_to_B * ratio_A
                step_container_B["X"] += incorrect_material_to_B * (1 - ratio_A)

                # Remove the misrouted amounts from the residual stream
                stream["A"] -= incorrect_material_to_B * ratio_A
                stream["X"] -= incorrect_material_to_B * (1 - ratio_A)

        # Whatever remains after two stages is discarded as X
        step_container_X = stream

        # Return per-stage outputs and achieved accuracies
        return step_container_A, step_container_B, step_container_X, {"A": accuracy_A, "B": accuracy_B}

    def _update_containers(self, composition: dict):
        """Updates global material containers."""
        sorted_A, sorted_B, sorted_X, accuracies = self._simulate_sequential_sorting(composition)
        
        for mat in self.material_names:
            self.container_A[mat] += sorted_A[mat]
            self.container_B[mat] += sorted_B[mat]
            self.container_X[mat] += sorted_X[mat]
        return accuracies

    def _calculate_purities(self):
        """Calculates current purity in containers A and B."""
        total_in_A = sum(self.container_A.values())
        total_in_B = sum(self.container_B.values())
        purity_A = self.container_A["A"] / total_in_A if total_in_A > 0 else 1.0
        purity_B = self.container_B["B"] / total_in_B if total_in_B > 0 else 1.0
        return purity_A, purity_B

    # ---------------------------------------------------------*/
    # Logging and Rendering
    # ---------------------------------------------------------*/

    def _log_history(self, reward, composition, accuracies, purity_A, purity_B):
        """Logs data for analysis or rendering."""
        h = self._history

        h["rewards"].append(reward)
        h["reward_quantity"] = h.get("reward_quantity", [])
        h["reward_quality"] = h.get("reward_quality", [])
        h["reward_quantity"].append(self._last_reward_quantity)
        h["reward_quality"].append(self._last_reward_quality)

        h["quantities"].append(sum(composition.values()))
        h["accuracies_A"].append(accuracies["A"])
        h["accuracies_B"].append(accuracies["B"])
        h["purities_A"].append(purity_A)
        h["purities_B"].append(purity_B)
        h["composition_A"].append(composition["A"])
        h["composition_B"].append(composition["B"])
        h["composition_X"].append(composition["X"])
        h["actions_raw"].append(self._last_action_raw)
        h["actions_scaled"].append(self._last_action_scaled)

        total_comp = sum(composition.values())
        if total_comp > 0:
            h["ratio_A"].append(composition["A"] / total_comp)
            h["ratio_B"].append(composition["B"] / total_comp)
            h["ratio_X"].append(composition["X"] / total_comp)
        else:
            h["ratio_A"].append(1 / 3)
            h["ratio_B"].append(1 / 3)
            h["ratio_X"].append(1 / 3)

    def render(self, save=False, show=True, log_dir='./img/', filename='env_render', title_seed="", custom_title_part=""):
        """Plots the environment state."""
        last_composition = {
            'A': self._history["composition_A"][-1] if self._history["composition_A"] else 0,
            'B': self._history["composition_B"][-1] if self._history["composition_B"] else 0,
            'X': self._history["composition_X"][-1] if self._history["composition_X"] else 0
        }

        plot_env_state(
            current_step=self.current_step,
            composition=last_composition,
            accuracies={"A": self._history["accuracies_A"][-1] if self._history["accuracies_A"] else 0,
                        "B": self._history["accuracies_B"][-1] if self._history["accuracies_B"] else 0},
            container_contents={'A': self.container_A, 'B': self.container_B, 'X': self.container_X},
            reward_history=self._history["rewards"],
            ratio_history_A=self._history["ratio_A"],
            ratio_history_B=self._history["ratio_B"],
            ratio_history_X=self._history["ratio_X"],
            quantity_history=self._history["quantities"],
            accuracy_history_A=self._history["accuracies_A"],
            accuracy_history_B=self._history["accuracies_B"],
            purity_history_A=self._history["purities_A"],
            purity_history_B=self._history["purities_B"],
            reward_quantity_history=self._history.get("reward_quantity", []),
            reward_quality_history=self._history.get("reward_quality", []),
            max_steps=self.max_steps,
            purity_thresholds=self.purity_thresholds,
            save=save, show=show, log_dir=log_dir, filename=filename,
            title_seed=title_seed, custom_title_part=custom_title_part
        )

    # ---------------------------------------------------------*/
    # Reward Components
    # ---------------------------------------------------------*/

    def _calculate_total_reward(self, total_quantity, purity_A, purity_B):
        """
        Strong-penalty purity version:

        Quantity reward:        [-0.25, +0.25]
        Purity reward per bin:  [-10.00, +0.25]
        Threshold = purity_thresholds[X]

        Below threshold:
            purity = 0    →  -10.0
            purity = T    →  +0.25
            (linear rise)

        Above threshold:
            linear decay:
            purity = T    → +0.25
            purity = 1.0  → 0.0
        """

        # -----------------------------------------------------------
        # 1) Quantity reward (unchanged)
        # -----------------------------------------------------------
        q_norm = np.clip(total_quantity / self.scaling_factor, 0.0, 1.0)
        r_quantity = 0.25 * (2.0 * q_norm - 1.0)   # [-0.25 … +0.25]


        # -----------------------------------------------------------
        # 2) Purity reward per container with very strong penalty
        # -----------------------------------------------------------
        def purity_reward(purity, threshold):
            purity = np.clip(purity, 0.0, 1.0)

            # ---- BELOW threshold: massive penalty  ----
            if purity < threshold:
                if threshold == 0.0:
                    return -10.0
                # linear from -10 → +0.25
                progress = purity / threshold
                return -10.0 + (10.25 * progress)

            # ---- ABOVE threshold: +0.25 → 0.0 ----
            if threshold == 1.0:
                return 0.25 if purity == 1.0 else -10.0

            overshoot = (purity - threshold) / (1.0 - threshold)
            return 0.25 * (1.0 - overshoot)   # +0.25 → 0.0

        r_A = purity_reward(purity_A, self.purity_thresholds["A"])
        r_B = purity_reward(purity_B, self.purity_thresholds["B"])

        r_quality = r_A + r_B     # Range ≈ [-20.0 … +0.50]


        # -----------------------------------------------------------
        # 3) Total reward
        # -----------------------------------------------------------
        total_reward = r_quantity + r_quality

        # -----------------------------------------------------------
        # 4) Logging
        # -----------------------------------------------------------
        self._last_reward_quantity = r_quantity
        self._last_reward_quality = r_quality
        self._last_reward_total = total_reward

        return total_reward


    # def _calculate_total_reward(self, total_quantity: float, purity_A: float, purity_B: float):
    #     """
    #     Peak-reward per container based on current purity.
    #     The peak is exactly at the respective threshold.
    #     Final total reward = (rA + rB) * q, where q = quantity in [0,1].
    #     """

    #     # Normalize quantity
    #     q = np.clip(total_quantity / self.scaling_factor, 0.0, 1.0)

    #     # Thresholds
    #     tA = self.purity_thresholds["A"]
    #     tB = self.purity_thresholds["B"]

    #     # --- Helper function for curve shape (generalized peak function) ---
    #     def peak_curve(p, threshold, 
    #                   w=0.05,     # Plateau width (in x-space)
    #                   k=40.0,     # Steepness of the rise
    #                   pexp=2.0,   # Exponent for left sigmoid
    #                   rate=6.0,   # Strength of right exponential drop
    #                   y_end=-0.5  # Minimum value far above threshold
    #                   ):
    #         """
    #         Peak at p = threshold.
    #         mapping: x = p/thresh  (x = 1 => peak)
    #         Left: sigmoid (sharp rise)
    #         Plateau: [1, 1+w]
    #         Right: exponential drop to y_end
    #         """
    #         # mapping purity → normalized x
    #         x = p / threshold

    #         # Plateau region
    #         x0 = 1.0          # Peak / threshold
    #         x_start = x0 + w  # End of plateau

    #         # Left sigmoid
    #         def sigma(z):
    #             return 1.0 / (1.0 + np.exp(-z))

    #         def clip01(a):
    #             return np.clip(a, 0.0, 1.0)

    #         # Case 1: Left (below threshold)
    #         if x < x0:
    #             g_left = (2.0 * sigma(k * (x - x0))) ** pexp
    #             y = 2.0 * clip01(g_left) - 1.0

    #         # Case 2: Plateau
    #         elif x0 <= x <= x_start:
    #             y = 1.0

    #         # Case 3: Right (above threshold)
    #         else:
    #             # Normalized distance
    #             T = max(1.0 - x_start, 1e-12)
    #             dx = x - x_start
    #             g_end = (y_end + 1.0) / 2.0

    #             num = np.exp(-rate * dx) - np.exp(-rate * T)
    #             den = 1.0 - np.exp(-rate * T)
    #             g_right = g_end + (1.0 - g_end) * (num / den)
    #             y = 2.0 * clip01(g_right) - 1.0

    #         return float(np.clip(y, -1.0, 1.0))

    #     # --- Partial rewards ---
    #     rA = peak_curve(purity_A, tA)
    #     rB = peak_curve(purity_B, tB)

    #     # --- Quantities ---
    #     qa = self.container_A["A"] / (self.container_A["A"] + self.container_A["B"] + self.container_A["X"])
    #     qb = self.container_B["B"] / (self.container_B["A"] + self.container_B["B"] + self.container_B["X"])

    #     # --- Final reward ---
    #     current_reward = qa*rA + qb*rB

    #     # Difference to last reward
    #     last_reward = getattr(self, "_last_reward_total_calc", 0)
    #     reward = current_reward - last_reward

    #     # Logging
    #     self._last_reward_quantity = 0
    #     self._last_reward_quality = 0
    #     self._last_reward_total = reward
    #     self._last_reward_total_calc = current_reward

    #     return reward


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\

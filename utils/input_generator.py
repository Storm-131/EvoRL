#---------------------------------------------------------*/
# Batch Input Generator for Material Compositions
#---------------------------------------------------------*/

import os
import copy
from collections import deque

import numpy as np
import matplotlib.pyplot as plt


class BatchInputGenerator:
    """
    Generates a master batch of material composed of sequential, smaller sub-batches
    with varying compositions. Supports both random and proportional sampling.

    Scaling factor controls *resolution only*: same seed → identical material patterns,
    just with replicated units for higher scaling factors.
    """

    def __init__(self, materials, seed=None,
                 total_size=100, min_batch_size=15, max_batch_size=25,
                 sampling_mode="random", scaling_factor=10_00):
        # --- Validation ---
        if len(materials) != 3:
            raise ValueError("Expected exactly three materials (A, B, X).")
        if sampling_mode not in ("random", "proportional"):
            raise ValueError("sampling_mode must be 'random' or 'proportional'.")
        if scaling_factor <= 0:
            raise ValueError("scaling_factor must be positive.")

        self.materials = materials
        self.sampling_mode = sampling_mode
        self.scaling_factor = int(scaling_factor)
        self.rng = np.random.default_rng(seed)

        # --- Base unscaled parameters ---
        self.total_size_unscaled = total_size
        self.min_batch_size_unscaled = min_batch_size
        self.max_batch_size_unscaled = max_batch_size

        # --- Derived scaled values (for compatibility only) ---
        self.total_size = int(total_size * self.scaling_factor)
        self.min_batch_size = int(min_batch_size * self.scaling_factor)
        self.max_batch_size = int(max_batch_size * self.scaling_factor)

        # --- Generate structure ---
        self._generate_base_master_batch()
        self._apply_scaling()

        # --- Runtime state ---
        self._remaining_batches = deque([dict(b, remaining=b["size"]) for b in self.batch_info])

    # ---------------------------------------------------------*/
    # Generate base master batch structure
    # ---------------------------------------------------------*/
    def _generate_base_master_batch(self):
        """Generate base batch structure without scaling."""
        self.base_batch_info = []
        generated_size = 0

        while generated_size < self.total_size_unscaled:
            batch_size = self.rng.integers(self.min_batch_size_unscaled,
                                           self.max_batch_size_unscaled + 1)
            if generated_size + batch_size > self.total_size_unscaled:
                batch_size = self.total_size_unscaled - generated_size
            if batch_size <= 0:
                break

            ratio_X = self.rng.uniform(0.10, 0.30)
            remaining_ratio = 1.0 - ratio_X
            ab_ratios = self.rng.dirichlet(alpha=[2.0, 2.0])
            ratio_A = ab_ratios[0] * remaining_ratio
            ratio_B = ab_ratios[1] * remaining_ratio

            composition = {m: r for m, r in zip(self.materials, [ratio_A, ratio_B, ratio_X])}

            self.base_batch_info.append({
                "size": batch_size,
                "comp": composition
            })
            generated_size += batch_size

    # ---------------------------------------------------------*/
    # Helper Functions
    # ---------------------------------------------------------*/
    def _apply_scaling(self):
        """Scale the metadata according to the scaling factor."""
        self.batch_info = [{
            "size": int(info["size"] * self.scaling_factor),
            "comp": dict(info["comp"])
        } for info in self.base_batch_info]

        self.total_size = sum(b["size"] for b in self.batch_info)
        self.min_batch_size = int(self.min_batch_size_unscaled * self.scaling_factor)
        self.max_batch_size = int(self.max_batch_size_unscaled * self.scaling_factor)

    # ---------------------------------------------------------*/
    def get_initial_state(self):
        """Returns metadata of all generated sub-batches."""
        return self.batch_info

    # ---------------------------------------------------------*/
    def reset(self, seed=None):
        """Reset the internal state for a new run."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._generate_base_master_batch()
            self._apply_scaling()
        self._remaining_batches = deque([dict(b, remaining=b["size"]) for b in self.batch_info])

    # ---------------------------------------------------------*/
    def is_empty(self):
        """Check if all batches are depleted."""
        return not self._remaining_batches

    # ---------------------------------------------------------*/
    # Draw samples from current batch structure
    # ---------------------------------------------------------*/
    def draw_samples(self, quantity: float):
        """Draw samples proportional to scaling factor (fractional input scaled)."""
        scaled_quantity = int(np.round(quantity * self.scaling_factor))
        if scaled_quantity <= 0:
            return {m: 0 for m in self.materials}

        if self.sampling_mode == "random":
            return self._draw_random(scaled_quantity)
        return self._draw_proportional(scaled_quantity)

    # ---------------------------------------------------------*/
    # Draw samples randomly from current batch structure
    # ---------------------------------------------------------*/
    def _draw_random(self, quantity: int):
        drawn_composition = {m: 0 for m in self.materials}
        to_draw = quantity

        while to_draw > 0 and self._remaining_batches:
            current_batch = self._remaining_batches[0]
            batch_size = current_batch["remaining"]
            comp = current_batch["comp"]
            draw_now = min(to_draw, batch_size)
            to_draw -= draw_now
            current_batch["remaining"] -= draw_now

            # stochastic sampling according to batch composition
            int_counts = self.rng.multinomial(draw_now, [comp[m] for m in self.materials])
            for m, c in zip(self.materials, int_counts):
                drawn_composition[m] += c

            if current_batch["remaining"] == 0:
                self._remaining_batches.popleft()

        return drawn_composition

    # ---------------------------------------------------------*/
    # Draw samples proportionally from current batch structure
    # ---------------------------------------------------------*/
    def _draw_proportional(self, quantity: int):
        drawn_composition = {m: 0 for m in self.materials}
        to_draw = quantity

        while to_draw > 0 and self._remaining_batches:
            current_batch = self._remaining_batches[0]
            batch_size = current_batch["remaining"]
            comp = current_batch["comp"]

            draw_now = min(to_draw, batch_size)
            to_draw -= draw_now
            current_batch["remaining"] -= draw_now

            ideal_counts = np.array([comp[m] * draw_now for m in self.materials])

            int_counts = np.floor(ideal_counts).astype(int)
            remainder = draw_now - int_counts.sum()

            if remainder > 0:
                fractional_parts = ideal_counts - int_counts
                top_indices = np.argsort(fractional_parts)[::-1][:remainder]
                for idx in top_indices:
                    int_counts[idx] += 1

            for m, c in zip(self.materials, int_counts):
                drawn_composition[m] += c

            if current_batch["remaining"] == 0:
                self._remaining_batches.popleft()

        return drawn_composition


# ---------------------------------------------------------*/
# Visualization
# ---------------------------------------------------------*/

def visualize_batch_system(seed=1, num_steps=100, scaling_factor=1.0, save_dir="./img"):
    """
    Visualizes both sampling modes ("random" and "proportional") over time.
    The resulting plot is saved into the specified directory (save_dir).
    """
    print(f"Generating plot for both sampling modes with seed={seed}, scaling_factor={scaling_factor}...")

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10),
                             gridspec_kw={"width_ratios": [1.5, 1, 1, 1]})
    colors = {"A": "lightblue", "B": "lightgreen", "X": "lightcoral"}

    draw_quantities = [0.1, 0.5, 1.0]
    sampling_modes = ["random", "proportional"]

    for row_idx, mode in enumerate(sampling_modes):
        ax_master = axes[row_idx, 0]
        generator = BatchInputGenerator(
            ["A", "B", "X"],
            seed=seed,
            sampling_mode=mode,
            scaling_factor=scaling_factor
        )
        batch_info = generator.get_initial_state()

        # Plot master batch structure
        current_pos = 0
        for info in batch_info:
            r = info["comp"]
            size = info["size"]
            ax_master.bar(current_pos + size / 2, r["A"], width=size, color=colors["A"])
            ax_master.bar(current_pos + size / 2, r["B"], bottom=r["A"], width=size, color=colors["B"])
            ax_master.bar(current_pos + size / 2, r["X"], bottom=r["A"] + r["B"],
                          width=size, color=colors["X"])
            current_pos += size

        ax_master.set_title(f"{mode.capitalize()} Sampling: Master Batch", fontweight="bold", fontsize=12)
        ax_master.set_ylabel("Material Ratio")
        ax_master.set_xlim(0, generator.total_size)
        ax_master.set_ylim(0, 1)
        ax_master.grid(axis="y", linestyle="--")
        if row_idx == 1:
            ax_master.set_xlabel("Cumulative Material Units")

        # Simulate draw evolution for each draw fraction
        for col_idx, q_fractional in enumerate(draw_quantities):
            ax_sim = axes[row_idx, col_idx + 1]
            generator.reset()
            ratio_history = []

            for _ in range(num_steps):
                if generator.is_empty():
                    break
                drawn = generator.draw_samples(q_fractional)
                total_drawn = sum(drawn.values())
                if total_drawn > 0:
                    ratio_history.append({m: drawn[m] / total_drawn for m in generator.materials})
                else:
                    ratio_history.append({m: 0 for m in generator.materials})

            mat_A = [r["A"] for r in ratio_history]
            mat_B = [r["B"] for r in ratio_history]
            mat_X = [r["X"] for r in ratio_history]

            draw_real_units = int(np.round(q_fractional * generator.scaling_factor))
            ax_sim.stackplot(range(len(ratio_history)), mat_A, mat_B, mat_X,
                             colors=list(colors.values()), alpha=0.8)
            ax_sim.set_title(f"Draw Quantity = {draw_real_units}", fontsize=10)
            ax_sim.set_xlim(0, num_steps)
            ax_sim.set_ylim(0, 1)
            ax_sim.grid(True, linestyle="--")
            if row_idx == 1:
                ax_sim.set_xlabel("Time Steps")

    # Legend and layout
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, colors.keys(), loc="upper center",
               bbox_to_anchor=(0.5, 0.96), ncol=3, fontsize=12)

    fig.suptitle(
        f"Comparison of Random and Proportional Sampling\n(Seed={seed}, Scaling={scaling_factor:g})",
        fontsize=18, fontweight="bold", y=1.02
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Save to given directory
    save_path_png = os.path.join(save_dir, f"batch_system_comparison_seed{seed}_scale{scaling_factor:g}.png")
    save_path_svg = os.path.join(save_dir, f"batch_system_comparison_seed{seed}_scale{scaling_factor:g}.svg")
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_svg, bbox_inches="tight")
    print(f"✅ Plot successfully saved to '{save_path_png}' (and .svg)")

    plt.show()


# ---------------------------------------------------------*/
# Example usage
# ---------------------------------------------------------*/
if __name__ == "__main__":
    visualize_batch_system(seed=1, scaling_factor=100)
    visualize_batch_system(seed=42, scaling_factor=100)
    visualize_batch_system(seed=42, scaling_factor=1000)


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\

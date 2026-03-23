# ---------------------------------------------------------*\
# Plotting Utilities
# ---------------------------------------------------------*/

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------*/
# Dashboard Plot for RobustBatchRegulatorEnv State
# ---------------------------------------------------------*/
def plot_env_state(
    # Current step data
    current_step: int,
    composition: dict,
    accuracies: dict,
    container_contents: dict,
    # Historical data (lists)
    reward_history: list,
    reward_quantity_history: list,
    reward_quality_history: list,
    ratio_history_A: list,
    ratio_history_B: list,
    ratio_history_X: list,
    quantity_history: list,
    accuracy_history_A: list,
    accuracy_history_B: list,
    purity_history_A: list,
    purity_history_B: list,
    # General info
    max_steps: int,
    purity_thresholds: dict,
    save: bool = False,
    show: bool = True,
    log_dir: str = './img/log/',
    filename: str = 'env_state',
    title_seed: str = "",
    custom_title_part: str = ""
):
    """
    Generates a comprehensive dashboard plot for the state of the RobustBatchRegulatorEnv with 3 materials.
    """
    # --- Global line thickness parameters ---
    thickness_quantity = 3
    thickness_mean_quantity = 2
    thickness_accuracy = 2
    thickness_deviation = 2
    thickness_reward = 2
    thickness_stackplot = 0.8  # alpha for stackplot (increased)

    fig = plt.figure(figsize=(20, 18))
    # Compose title: show seed and custom part, not step
    title = "Robust Batch Regulator"
    if title_seed:
        title += f" - {title_seed}"
    if custom_title_part:
        title += f" - {custom_title_part}"
    fig.suptitle(title, fontsize=20, fontweight='bold')

    # Define colors
    color_A = 'lightblue'
    color_B = 'lightgreen'
    color_X = 'lightcoral'
    color_A_dark = 'dodgerblue'
    color_B_dark = 'seagreen'
    color_quantity = 'darkorange'

    # ---------------------------------------------------------*/
    # Plot 1: Current Input Composition (Pie)
    # ---------------------------------------------------------*/
    ax1 = plt.subplot2grid((3, 4), (0, 0))
    ax1.set_title('Current Input Composition', fontweight='bold')
    comp_values = [composition.get('A', 0), composition.get('B', 0), composition.get('X', 0)]
    total_comp = sum(comp_values)
    labels = [f"A ({comp_values[0]})", f"B ({comp_values[1]})", f"X ({comp_values[2]})"]
    ax1.pie(comp_values, labels=labels, colors=[color_A, color_B, color_X],
            autopct='%1.0f%%', startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax1.text(0, -1.3, f"Total: {total_comp}", ha='center', va='center', fontweight='bold')

    # ---------------------------------------------------------*/
    # Plot 2: Current Input & Accuracies (Bar)
    # ---------------------------------------------------------*/
    ax2 = plt.subplot2grid((3, 4), (0, 1))
    ax2.set_title('Current Input & Accuracies', fontweight='bold')
    materials = ['A', 'B', 'X']
    input_quantities = [composition.get('A', 0), composition.get('B', 0), composition.get('X', 0)]
    bars = ax2.bar(materials, input_quantities, color=[color_A_dark, color_B_dark, color_X])
    ax2.set_ylabel('Absolute Quantity')

    # Text for quantities inside the bars
    for i, bar in enumerate(bars):
        quantity = input_quantities[i]
        if quantity > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, quantity / 2,
                     f'{quantity}', ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    # Text for Accuracies above the bars
    acc_A, acc_B = accuracies.get('A', 0), accuracies.get('B', 0)
    if input_quantities[0] > 0:
        ax2.text(bars[0].get_x() + bars[0].get_width()/2., input_quantities[0],
                 f'Acc: {acc_A:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    if input_quantities[1] > 0:
        ax2.text(bars[1].get_x() + bars[1].get_width()/2., input_quantities[1],
                 f'Acc: {acc_B:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    if ax2.get_ylim()[1] < 110:
        ax2.set_ylim(top=ax2.get_ylim()[1]*1.15)

    # ---------------------------------------------------------*/
    # Plot 3: Container Contents (Stacked Bars with Purity Text)
    # ---------------------------------------------------------*/
    ax3 = plt.subplot2grid((3, 4), (0, 2))
    ax3.set_title('Container Contents', fontweight='bold')

    # Loop over the three containers
    for name, container in [('A', container_contents['A']), ('B', container_contents['B']), ('X', container_contents['X'])]:
        bottom = 0
        ax3.bar(name, container['A'], color=color_A, label='A' if name == 'A' else None)
        bottom += container['A']
        ax3.bar(name, container['B'], bottom=bottom, color=color_B, label='B' if name == 'A' else None)
        bottom += container['B']
        ax3.bar(name, container['X'], bottom=bottom, color=color_X, label='X' if name == 'A' else None)

    # Purity text for A and B containers
    for name in ['A', 'B']:
        container = container_contents[name]
        total = sum(container.values())
        if total > 0:
            purity = (container[name] / total) * 100
            ax3.text(name, total / 2, f'{purity:.1f}%', ha='center',
                     va='center', color='black', fontweight='bold', fontsize=12)

    ax3.set_ylabel('Total Quantity')
    ax3.legend()

    # ---------------------------------------------------------*/
    # Plot 4: Cumulative Rewards
    # ---------------------------------------------------------*/
    ax4 = plt.subplot2grid((3, 4), (0, 3))
    ax4.set_title('Cumulative Rewards', fontweight='bold')

    # safe loading
    total_rewards = np.array(reward_history)
    quantity_rewards = np.array(reward_quantity_history)
    quality_rewards = np.array(reward_quality_history)
    if len(total_rewards) == 0:
        total_rewards = np.array([0])
    if len(quantity_rewards) == 0:
        quantity_rewards = np.array([0])
    if len(quality_rewards) == 0:
        quality_rewards = np.array([0])

    # cumulative sums
    cumulative_total = np.cumsum(total_rewards)
    cumulative_quantity = np.cumsum(quantity_rewards)
    cumulative_quality = np.cumsum(quality_rewards)

    ax4.plot(cumulative_total, label="Total", color='purple', linewidth=2.5)
    ax4.plot(cumulative_quantity, label="Quantity", color='orange', linestyle='--')
    ax4.plot(cumulative_quality, label="Quality", color='blue', linestyle='--')

    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Cumulative Reward')
    # Place legend in lower left to avoid overlap with ax5/ax6 legends
    ax4.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='black', framealpha=0.8)
    ax4.grid(True, linestyle='--')

    # Place text in upper left to avoid legend
    ax4.text(0.05, 0.95,
             f'Total: {cumulative_total[-1]:.2f}\nQty: {cumulative_quantity[-1]:.2f}\nQual: {cumulative_quality[-1]:.2f}',
             transform=ax4.transAxes, ha='left', va='top', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    # ---------------------------------------------------------*/
    # Plot 5: Timeline - Input Composition, Quantity & Resulting Accuracy
    # ---------------------------------------------------------*/
    ax5 = plt.subplot2grid((3, 1), (1, 0))
    ax5.set_title('Timeline: Input Composition, Quantity & Resulting Accuracy', fontweight='bold')
    timesteps = np.arange(len(quantity_history))
    ax5.stackplot(
        timesteps, ratio_history_A, ratio_history_B, ratio_history_X,
        labels=['A Ratio', 'B Ratio', 'X Ratio'],
        colors=[color_A, color_B, color_X], alpha=thickness_stackplot
    )
    ax5.set_ylabel('Material Ratio / Accuracy')
    ax5.set_ylim(0, 1)

    # Calculate mean accuracies for legend
    mean_acc_A = np.mean(accuracy_history_A) if len(accuracy_history_A) > 0 else 0.0
    mean_acc_B = np.mean(accuracy_history_B) if len(accuracy_history_B) > 0 else 0.0

    # Plot accuracies on the LEFT y-axis (thicker lines)
    accA_label = f'Accuracy A (mean={mean_acc_A:.2f})'
    accB_label = f'Accuracy B (mean={mean_acc_B:.2f})'
    if len(accuracy_history_A) > 0:
        ax5.plot(timesteps, accuracy_history_A, color=color_A_dark, linewidth=thickness_accuracy, label=accA_label)
    if len(accuracy_history_B) > 0:
        ax5.plot(timesteps, accuracy_history_B, color=color_B_dark, linewidth=thickness_accuracy, label=accB_label)

    # RIGHT axis for quantity with fixed range 0-105, thick orange solid line
    ax5_twin = ax5.twinx()
    mean_quantity = np.mean(quantity_history) if len(quantity_history) > 0 else 0.0
    ax5_twin.plot(
        timesteps, quantity_history, color=color_quantity, linestyle='-',  # solid line
        marker='.', markersize=4, label=f'Input Quantity (mean={mean_quantity:.1f})', linewidth=thickness_quantity
    )
    ax5_twin.set_ylabel('Quantity')
    ax5_twin.set_ylim(0,)

    # Show mean input quantity as horizontal dashed orange line
    if len(quantity_history) > 0:
        mean_quantity = np.mean(quantity_history)
        ax5_twin.axhline(
            mean_quantity, color=color_quantity, linestyle=':', linewidth=thickness_mean_quantity,
            label=f'Mean Input ({mean_quantity:.1f})'
        )

    # Combine legends and set a high zorder to ensure it's drawn on top.
    lines, labels = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    legend = ax5_twin.legend(
        lines + lines2, labels + labels2,
        loc='upper left',
        frameon=True,
        facecolor='white',
        edgecolor='black',
        framealpha=0.8
    )
    legend.set_zorder(100)  # This tells matplotlib to draw the legend on the top layer.

    ax5.set_xlim(0, max_steps)

    # ---------------------------------------------------------*/
    # Plot 6: Timeline - Purities & Cumulative Input
    # ---------------------------------------------------------*/
    ax6 = plt.subplot2grid((3, 1), (2, 0))
    ax6.set_title('Timeline: Purities & Cumulative Input', fontweight='bold')

    # --- Purity time series (A and B) ---
    ax6.plot(
        timesteps, purity_history_A, color=color_A_dark,
        label='Purity A', linewidth=thickness_deviation
    )
    ax6.plot(
        timesteps, purity_history_B, color=color_B_dark,
        label='Purity B', linewidth=thickness_deviation
    )

    # --- Fill area under the curves ---
    ax6.fill_between(
        timesteps, purity_history_A, color=color_A_dark, alpha=0.15
    )
    ax6.fill_between(
        timesteps, purity_history_B, color=color_B_dark, alpha=0.15
    )

    # --- Threshold reference lines (thicker) ---
    ax6.axhline(
        y=purity_thresholds['A'], color=color_A_dark, linestyle='--',
        linewidth=2.0, label=f'A threshold ({purity_thresholds["A"]:.0%})'
    )
    ax6.axhline(
        y=purity_thresholds['B'], color=color_B_dark, linestyle='--',
        linewidth=2.0, label=f'B threshold ({purity_thresholds["B"]:.0%})'
    )

    ax6.set_ylabel('Purity')
    ax6.set_ylim(0.0, 1.0)

    # --- Secondary axis: cumulative input ---
    ax6_twin = ax6.twinx()
    cumulative_input = np.cumsum(quantity_history) if quantity_history else np.array([0])
    ax6_twin.plot(
        timesteps, cumulative_input, color=color_quantity, linestyle='-',
        linewidth=thickness_quantity, label='Cumulative Input'
    )
    ax6_twin.set_ylabel('Total Processed Quantity')

    # annotate total input
    if getattr(cumulative_input, "any", lambda: False)():
        ax6_twin.text(
            0.98, 0.95, f'Total Input: {cumulative_input[-1]:.0f}',
            transform=ax6_twin.transAxes, ha='right', va='top',
            fontweight='bold', fontsize=12, bbox=dict(facecolor='white', alpha=0.8)
        )

    # --- Legend & layout ---
    lines, labels = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6_twin.legend(lines + lines2, labels + labels2, loc='upper left')

    ax6.set_xlabel('Timesteps')
    ax6.set_xlim(0, max_steps)

    # --- Finalize and Show/Save ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        os.makedirs(log_dir, exist_ok=True)
        png_path = os.path.join(log_dir, f"{filename}.png")
        svg_path = os.path.join(log_dir, f"{filename}.svg")
        plt.savefig(png_path, dpi=300)
        plt.savefig(svg_path)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------*/
# Plot Evaluation Results Comparison
# ---------------------------------------------------------*/

def plot_evaluation_results(results_dict, dir, env, group_ppo=True):
    """
    Plots a comparison of all evaluated agents.
    Automatically separates PPO, PPO+BC, and other PPO variants.
    Ensures unique labels for each PPO variant.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # --------------------------------------------------------
    # STEP 1 — Agent keys
    # --------------------------------------------------------
    orig_agent_names = list(results_dict.keys())

    # Detect all PPO-like agents
    ppo_variants = [n for n in orig_agent_names if "ppo" in n.lower()]
    other_agents = [n for n in orig_agent_names if "ppo" not in n.lower()]

    # Sort PPO variants (Best → Final → Others)
    def ppo_sort_key(n):
        name = n.lower()
        if "best" in name:
            return 0
        if "final" in name:
            return 1
        return 2

    ppo_variants = sorted(ppo_variants, key=ppo_sort_key)

    # --------------------------------------------------------
    # STEP 2 —  PPO-Labels 
    # --------------------------------------------------------
    ppo_group_labels = []
    for key in ppo_variants:
        base = key.replace("(", "").replace(")", "")
        parts = base.split()

        algo = parts[0]      # PPO or PPO+BC
        tag = parts[1]       # Best or Final

        pretty = f"{algo}\n({tag})"  
        ppo_group_labels.append(pretty)

    # Final order of all agents for plotting:
    agent_names = other_agents + ppo_group_labels

    # --------------------------------------------------------
    # STEP 3 — Mapping label → result-key
    # --------------------------------------------------------
    key_lookup = {}

    # Map PPO variants cleanly
    for orig, pretty in zip(ppo_variants, ppo_group_labels):
        key_lookup[pretty] = orig

    # Map non-PPO agents 1:1
    for name in other_agents:
        key_lookup[name] = name

    # --------------------------------------------------------
    # STEP 4 — Build arrays
    # --------------------------------------------------------
    cumulative_rewards = [results_dict[key_lookup[k]]["cumulative_reward"] for k in agent_names]
    step_reward_lists = [results_dict[key_lookup[k]]["step_rewards"] for k in agent_names]

    cumulative_inputs = []
    purity_A_final = []
    purity_B_final = []

    for k in agent_names:
        res = results_dict[key_lookup[k]]

        cumulative_inputs.append(res["cumulative_input"][-1])

        cont = res.get("container_contents", {})
        if "A" in cont and "B" in cont:
            totA = sum(cont["A"].values()); totB = sum(cont["B"].values())
            pA = (cont["A"]["A"] / totA * 100) if totA > 0 else 0
            pB = (cont["B"]["B"] / totB * 100) if totB > 0 else 0
        else:
            pA = pB = 0

        purity_A_final.append(pA)
        purity_B_final.append(pB)

    # --------------------------------------------------------
    # --- Plotting
    # --------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)

    # === 1) Cumulative Reward ===
    bars = axes[0].bar(agent_names, cumulative_rewards, color="steelblue", alpha=0.8)
    axes[0].set_title("Cumulative Reward per Agent", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Total Reward")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    for bar, value in zip(bars, cumulative_rewards):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                     f"{value:.2f}", ha="center", va="center", fontweight="bold")

    # === 2) Step Reward Boxplots ===
    axes[1].boxplot(step_reward_lists, labels=agent_names, patch_artist=True,
                    boxprops=dict(facecolor="lightgrey"), medianprops=dict(color="red"))
    axes[1].set_title("Step-wise Reward Distributions", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Reward per Step")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    # === 3) Total Input + Purities ===
    x = np.arange(len(agent_names))
    width = 0.25
    ax_left = axes[2]
    ax_right = ax_left.twinx()

    cumulative_inputs_k = np.asarray(cumulative_inputs) / 1000
    ax_left.bar(x - width, cumulative_inputs_k, width, color="darkorange", label="Total Input [K]")
    ax_left.set_ylabel("Total Input Quantity [K]")

    ax_right.bar(x, purity_A_final, width, color='lightblue', label="Purity A")
    ax_right.bar(x + width, purity_B_final, width, color='lightgreen', label="Purity B")
    ax_right.set_ylabel("Purity [%]")
    ax_right.set_ylim(0, 100)

    # Thresholds
    ax_right.axhline(env.purity_thresholds["A"] * 100, linestyle='--', color='lightblue')
    ax_right.axhline(env.purity_thresholds["B"] * 100, linestyle='--', color='lightgreen')

    # X-Labels
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(agent_names, rotation=0)

    # Save + Show
    png_path = os.path.join(dir, "evaluation_comparison.png")
    svg_path = os.path.join(dir, "evaluation_comparison.svg")
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.show()



# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\

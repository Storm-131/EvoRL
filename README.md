# Evolutionary Warm-Starts for Reinforcement Learning in Industrial Continuous Control ⚙️🤖

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Code Style: autopep8](https://img.shields.io/badge/code%20style-autopep8-lightgrey)](https://pypi.org/project/autopep8/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project investigates the use of the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to generate high-quality demonstration data for warm-starting Proximal Policy Optimization (PPO) agents in an industrial continuous-control task. The central focus is on leveraging offline evolutionary trajectory optimization to stabilize and accelerate reinforcement learning in stochastic, long-horizon control settings.

At the core of this research is a custom benchmark environment that adapts an industrial sorting process into a continuous input regulation problem. The agent controls the normalized material inflow to balance throughput and output purity under non-stationary and stochastic input conditions.

The primary objective of this project is to evaluate how CMA-ES–generated oracle trajectories can serve as expert demonstrations for Behavioral Cloning (BC), thereby improving the stability, robustness, and sample efficiency of PPO agents during online training.

---

## 🤖 Environment Design

The project is built around a custom [gymnasium](https://gymnasium.farama.org/index.html)-based environment representing a simplified industrial sorting and regulation process. The environment exposes a one-dimensional continuous control signal corresponding to the regulated input quantity. Load-dependent processing accuracy and cumulative purity dynamics induce a non-trivial trade-off between throughput and quality.

Stochasticity arises from a parameterized input generator that produces varying material compositions and batch sizes, making the task non-stationary and challenging for purely reactive controllers.

---

## 🧪 Experimental Focus

This benchmark focuses on the following core components:

1. **CMA-ES Oracle Trajectory Optimization**  
   CMA-ES is used to optimize full episode-long sequences of continuous control actions on fixed-seed environments. This yields seed-specific oracle trajectories that exploit full foresight of the stochastic input sequence and provide an empirical upper performance bound.

2. **Demonstration-Based Warm-Starting (PPO+BC)**  
   Optimized oracle trajectories are aggregated across multiple seeds and used as expert demonstrations to pretrain PPO policies via supervised Behavioral Cloning before standard online reinforcement learning.

3. **Baseline and Comparative Evaluation**  
   The performance of PPO trained from scratch is compared against PPO+BC, simple rule-based controllers, and the CMA-ES oracle to assess learning stability, robustness, and convergence behavior.

---

## 🏗 Folder Structure

```
📦ES-RL-Refactored
┣ 📂configs              --> Parameter configurations
┣ 📂img                  --> Figures and plots
┣ 📂log                  --> Logging output (e.g., TensorBoard)
┣ 📂models               --> Trained model checkpoints
┣ 📂src                  --> Environment, training, and evaluation code
┃ ┣ 📜env.py
┃ ┣ 📜testing.py
┃ ┗ 📜training.py
┣ 📂utils                --> Benchmark utilities and optimizers
┃ ┣ 📜benchmark_models.py
┃ ┣ 📜cma_optimizer.py
┃ ┣ 📜input_generator.py
┃ ┗ 📜plotting.py
┣ 📜environment.yaml
┣ 📜README.md
┣ 📜main.py              --> Entry point for experiments
```

---

## 📚 Setup

To set up the environment locally, follow these steps:

```bash
git clone <repository-url>

# Create and activate the conda environment
conda env create -f environment.yaml -n industrial_es_rl
conda activate industrial_es_rl
````

---

## 🚀 Quickstart

All experiments can be executed via the main entry script:

```bash
# Configure experiment parameters and run
python main.py
```

Configuration files allow switching between baseline controllers, PPO training from scratch, CMA-ES trajectory optimization, and PPO warm-started via Behavioral Cloning.

---

## 📄 Reproducibility and Code Availability

The code used for all experiments in this project is publicly available and accompanies the corresponding research publication. Configuration files and fixed random seeds are provided to facilitate reproducibility.

---

## Contact 📬

For questions related to the benchmark or experiments, please contact the corresponding [author](https://www.ini.rub.de/the_institute/people/tom-maus/).

---


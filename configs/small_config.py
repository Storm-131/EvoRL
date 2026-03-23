# ---------------------------------------------------------*/
# Small configuration for testing code
# ---------------------------------------------------------*/

import numpy as np
import time 

# 1. Experiment Selector
# ---------------------------------------------------------*/
TEST_ENV = 1                 # (1/0) Run unified test mode (evaluates static/rule-based agents)

TRAIN_PPO_AGENT = 0          # (1/0) Train PPO agent from scratch

RUN_CMA_TUNING = 0           # (1/0) Run CMA-ES hyperparameter tuning
RUN_CMA_SINGLE = 1           # (1/0) Demo CMA-ES optimization experiment
RUN_CMA_BATCH = 0            # (1/0) Run CMA-ES optimization experiment for BC data

TRAIN_PPO_CMA = 0            # (1/0) Train PPO after CMA-ES pretraining (BC warm-start)

RUN_BENCHMARK = 1            # (1/0) Run multi-seed benchmark experiment

# 2. Env Configuration
# ---------------------------------------------------------*/
SCALING_FACTOR = 10_00       # Scaling factor for material quantities
MAX_STEPS = 100              # Max steps per episode
SENSOR_NOISE_STD = 0.0       # Sensor noise standard deviation

# 3. Input Generator Configuration
# ---------------------------------------------------------*/
INPUT_GENERATOR_TYPE = "proportional"  # Input generator type ("random" vs. "proportional")
TOTAL_SIZE = 1000                      # Total size of the input material stream
MIN_BATCH_SIZE = 15                    # Minimum batch size (from input)
MAX_BATCH_SIZE = 25                    # Maximum batch size (from input)

# 4. Benchmark Parameters
# ---------------------------------------------------------*/
BENCHMARK_N_SEEDS = 10                  # Number of seeds for benchmarking

# 5. CMA-ES Parameters
# ---------------------------------------------------------*/
POP_SIZES = [4, 8, 16, 32, 64]    # Population sizes to test during CMA-ES tuning [4, 8, 16, 32, 64]

MAX_GEN = 30                      # CMA-ES: Max generations (75)
POP_SIZE = 16                      # CMA-ES: Population size (32)
SIGMA_INIT = 0.1                  # CMA-ES: Initial sigma (0.1)

CMA_BATCH_NUM_ENVS = 10            # Number of envs for batch CMA-ES

# 6. Seed Management
# ---------------------------------------------------------*/
GLOBAL_MASTER_SEED = 42
_rng = np.random.RandomState(GLOBAL_MASTER_SEED)  # Master Random Number Generator

# Define disjoint Seed-Lists
TEST_SEEDS = list(range(0, 0 + BENCHMARK_N_SEEDS))                 # Seeds for final evaluation (e.g., [0, 1, 2, 3])
TRAINING_SEEDS = list(range(1000, 1000 + BENCHMARK_N_SEEDS))       # Seeds for model training (e.g., [1000, 1001, ...])
PPO_EVAL_SEEDS = list(range(2000, 2000 + BENCHMARK_N_SEEDS + 1))   # Seeds for PPO EvalCallback (e.g., [2000, ...])
CMA_DEMO_SEEDS = list(range(3000, 3000 + CMA_BATCH_NUM_ENVS))      # Seeds for CMA-ES demo generation (e.g., [3000,])

# Seeds Single Runs (non-benchmark)
SINGLE_TEST_SEED = 42      # e.g., 0
SINGLE_TRAIN_SEED = TRAINING_SEEDS[0]      # e.g., 1000

# 7. PPO Parameters
# ---------------------------------------------------------*/
PPO_TIMESTEPS = 50_000    # PPO training timesteps
PPO_EVAL_FREQ = 5_000        # PPO evaluation frequency during training

# 8. Logging and Tag Configuration
# ---------------------------------------------------------*/
NAME = f"1_Peak_Reward_{int(time.time())}"
TAG = f"{NAME}_{INPUT_GENERATOR_TYPE}_scale_{SCALING_FACTOR}_ep_{MAX_STEPS}"

# 9. Computer Resources
# ---------------------------------------------------------*/
NUM_WORKERS = 8              # Number of parallel workers for benchmarking

# ---------------------------------------------------------*/
# For local Testing..
# ---------------------------------------------------------*/


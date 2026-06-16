# Robometer Policy Learning

A flexible reinforcement \ imitation learning framework supporting multiple algorithms (SAC, IQL, BC), reward models (Robometer, RoboReward), and distributed training. We will soon include detailed guides for DSRL (Diffusion-steering RL) with Pi0/0.5 on LIBERO and Real World tasks.

>[!WARNING]
>
> This repository is under active development, so some modules and features may change over time. You may encounter issues when using features that are not yet documented in this README. Please feel free to open an issue — we will do our best to help.

## Table of Contents

- [Setup](#setup)
- [Training](#training)
  - [Basic Training](#basic-training-ground-truth-rewards)
  - [Training with Reward Model](#training-with-robometer-reward-model)
  - [Example: Online RL in LIBERO](#example-online-rl-in-libero)
- [Real-World Online RL with DSRL + Remote Reward Labeling](#real-world-online-rl-with-dsrl--remote-reward-labeling)
- [Project Structure](#project-structure)
---

## Setup

### Prerequisites

- Git
- Python 3.10+
- NVIDIA Drivers (for GPU support)

### Installation

1. **Install `uv` (if not already installed):**
  ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
2. **Clone and setup submodules (required for DSRL/LIBERO):**
  ```bash
   git submodule init
   git submodule update --recursive
  ```
3. **Create and sync the virtual environment:**
  ```bash
   # Install dependencies from pyproject.toml
   uv sync

   # Optional: Install with development dependencies
   uv sync --extra dev
  ```

**Activate the environment:**

```bash
 source .venv/bin/activate
```



**Note that this repo assumes robometer is installed as a git submodule and located at `./robometer`. If you made any changes to robometer/have your own robometer fork, replace the submodule** 

---

## Training

### Basic Training (Ground Truth Rewards)

Train with ground truth rewards (default, no reward model - Online only):

```bash
uv run python scripts/train.py \
  --config-path=../robometer_policy_learning/configs \
  --config-name=config \
  algorithm@online_algorithm=sac \
  alg.online_alg_name=sac \
  env.use_gt_rewards=true
```

Offline pretraining with online fine-tuning:

```bash
uv run python scripts/train.py \
  --config-path=../robometer_policy_learning/configs \
  --config-name=config \
  algorithm@offline_algorithm=iql \
  algorithm@online_algorithm=sac \
  alg.offline_alg_name=iql \
  alg.online_alg_name=sac \
  env.use_gt_rewards=true
```

### Training with Robometer Reward Model

Train with Robometer reward model (Online only):

```bash
uv run python scripts/train.py \
  --config-path=../robometer_policy_learning/configs \
  --config-name=config \
  reward_model=robometer \
  algorithm@online_algorithm=sac \
  alg.online_alg_name=sac \
  env.use_gt_rewards=false \
  reward_model.model_path=robometer/Robometer-4B
```

Train with Robometer reward model (Offline-to-online):

```bash
uv run python scripts/train.py   \
  --config-path=../robometer_policy_learning/configs   \
  --config-name=config   \
  reward_model=robometer   \
  algorithm@online_algorithm=sac   \
  alg.online_alg_name=sac   \
  algorithm@offline_algorithm=iql   \
  alg.offline_alg_name=iql   \
  env.use_gt_rewards=false   \
  reward_model.model_path=robometer/Robometer-4B
```

### Example: Online RL in LIBERO

Train a SAC policy in LIBERO using ground-truth rewards:

```bash
uv run python scripts/train.py   \
  --config-path=../robometer_policy_learning/configs  \
  --config-name=libero_online_rl  \
  env.env_name=libero_90  \
  env.task_id=28  \
  env.use_gt_rewards=true  \
  algorithm@online_algorithm=sac   \
  alg.online_alg_name=sac   \
  training.num_rollouts=100000  \
  training.seed=100  \
  eval.eval_freq=5000  \
  eval.eval_num_episodes=20  \
  online_algorithm.num_critic_updates_per_actor_update=1  \
  online_algorithm.learning_starts=5000  \
  online_algorithm.critic_optimizer_lr=1e-5  \
  online_algorithm.actor_optimizer_lr=1e-5  \
  logging.wandb_name=libero_online_rl_gt_rewards  \
  logging.wandb_entity=YOUR_WANDB_ENTITY
```

Train a SAC policy in LIBERO using Robometer rewards:

```bash
uv run python scripts/train.py   \
  --config-path=../robometer_policy_learning/configs  \
  --config-name=libero_online_rl  \
  reward_model=robometer  \
  reward_model.model_path=robometer/Robometer-4B  \
  reward_model.add_estimated_reward=true  \
  reward_model.use_success_detection=false  \
  env.env_name=libero_90  \
  env.task_id=28  \
  env.use_gt_rewards=false  \
  algorithm@online_algorithm=sac   \
  alg.online_alg_name=sac   \
  training.num_rollouts=100000  \
  training.seed=100  \
  eval.eval_freq=5000  \
  eval.eval_num_episodes=20  \
  online_algorithm.num_critic_updates_per_actor_update=1  \
  online_algorithm.learning_starts=5000  \
  online_algorithm.critic_optimizer_lr=1e-5  \
  online_algorithm.actor_optimizer_lr=1e-5  \
  logging.wandb_name=libero_online_rl_robometer_rewards  \
  logging.wandb_entity=YOUR_WANDB_ENTITY
```

You should see evaluation curves similar to the example below:

<p align="center">
  <img src="docs/libero_rl.png" alt="LIBERO RL Experiments" width="600"/>
</p>

---

# Real-World Online RL with DSRL + Remote Reward Labeling

Train DSRL policies on a real Franka/DROID or WidowX robot with async reward relabeling via Robometer over gRPC.

**Full guide:** [docs/REAL_ROBOT_README.md](docs/REAL_ROBOT_README.md) — covers robot server setup, Pinggy tunneling, reward relabel server, training, and evaluation.

### Quick commands (DROID / Franka)

**Terminal 1 — Training machine (reward model server):**
```bash
uv run python scripts/start_reward_relabel_server.py \
    reward_model=robometer \
    reward_model.model_path="robometer/Robometer-4B" \
    server.port=50052 server.host="0.0.0.0" device=cuda \
    server.image_keys='["observation/exterior_image_1_left"]'
```

**Terminal 2 — Training machine (DSRL training):**
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python scripts/train_dsrl.py \
    --config-name dsrl_remote_robot_async_relabel_config \
    logging.wandb_entity=YOUR_ENTITY \
    remote_robot.host=YOUR_ROBOT_HOST \
    remote_robot.port=YOUR_ROBOT_PORT
```

**Terminal 3 — Robot machine (robot server):**
```bash
uv run python robometer_policy_learning/robots/droid_remote_server.py \
    --left-camera-id "..." --wrist-camera-id "..." \
    --external-camera left --server-port 6000 \
    --prompt "put the red block in the bowl"
```

**Terminal 4 — Robot machine (Pinggy tunnel):**
```bash
ssh -p 443 -R0:localhost:6000 qr+tcp@free.pinggy.io
```

See the [real robot guide](docs/REAL_ROBOT_README.md) for WidowX instructions, eval with `eval_pi0.py`, and troubleshooting. 

---

# Additional Resources

- gRPC service definitions:
  `robometer_policy_learning/distributed/protos/`

- Configuration files:
  `robometer_policy_learning/configs/`

- Algorithm-specific configs:
  `robometer_policy_learning/configs/algorithm/`

## Project Structure

```text
.
├── docs/                         # Documentation assets and figures
├── robometer_policy_learning/    # Main policy learning package
│   ├── algorithms/               # BC, IQL, SAC, and DSRL algorithm code
│   ├── buffers/                  # Replay and offline data buffers
│   ├── configs/                  # Hydra configs for algorithms, envs, and reward models
│   ├── distributed/              # Distributed training and reward relabeling services
│   ├── envs/                     # Environment wrappers and task interfaces
│   ├── loggers/                  # Logging integrations
│   ├── modules/                  # Policy, critic, and value network modules
│   ├── robots/                   # Real-robot interfaces
│   ├── rollouts/                 # Rollout collection utilities
│   ├── runners/                  # Training and evaluation runners
│   └── utils/                    # Shared helpers
├── scripts/                      # Training, evaluation, and server entrypoints
├── tests/                        # Test suite
```


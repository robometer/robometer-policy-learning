# Real Robot Training with DSRL + Async Reward Relabeling

**See also:** [robometer_policy_learning/robots/README.md](../robometer_policy_learning/robots/README.md) (canonical DROID + WidowX TCP servers, shared protocol, Pinggy workflow).

This guide explains how to train DSRL on a real robot using remote TCP servers, async reward relabeling over gRPC, and TCP tunneling (Pinggy/ngrok).

## Architecture Overview

```
┌─────────────────────────────┐     TCP/Pickle      ┌──────────────────────────────────┐
│       Robot Machine         │◄────────────────────│        Training Machine           │
│                             │                     │                                  │
│  ┌──────────────────────┐   │                     │  ┌────────────────────────────┐  │
│  │  Franka/DROID Robot  │   │                     │  │  train_dsrl.py             │  │
│  └──────────┬───────────┘   │                     │  │  (SAC + Pi0 + DINOv2)     │  │
│             │               │                     │  └─────────────┬──────────────┘  │
│  ┌──────────▼───────────┐   │    Tunnel (Pinggy)  │                │                 │
│  │ droid_remote_server  │───┼─────────────────────┼─► RemoteEnv    │                 │
│  │ (port 6000)          │   │                     │                │                 │
│  └──────────────────────┘   │                     │  ┌─────────────▼──────────────┐  │
│                             │                     │  │  AsyncRewardRelabelWrapper │  │
│  Keyboard: s=success,       │                     │  │  (accumulates trajectory)  │  │
│            f=failure, q=quit│                     │  └─────────────┬──────────────┘  │
└─────────────────────────────┘                     │                │ gRPC            │
                                                    │  ┌─────────────▼──────────────┐  │
                                                    │  │  Reward Relabel Server     │  │
                                                    │  │  (localhost:50052)         │  │
                                                    │  │  Robometer / RoboReward   │  │
                                                    │  └────────────────────────────┘  │
                                                    └──────────────────────────────────┘
```

The reward relabel server runs **locally** on the training machine and scores trajectories automatically. The robot server runs on the robot machine and accepts manual success/failure labels via keyboard.

## Quick Start (DROID / Franka Panda)

### Prerequisites

```bash
# On the robot machine: install droid and openpi-client
pip install droid openpi-client

# On the training machine: clone and install this repo
git clone --recurse-submodules <this-repo>
cd robometer-policy-learning
uv sync
```

### 1. Start the Robot Server (Robot Machine)

```bash
# From the robometer-policy-learning checkout on the robot machine:
uv run python robometer_policy_learning/robots/droid_remote_server.py \
    --left-camera-id "24259877" \
    --right-camera-id "24514023" \
    --wrist-camera-id "13062452" \
    --external-camera left \
    --server-port 6000 \
    --prompt "put the red block in the bowl" \
    --max-steps 600
```

**Arguments:**
- `--left-camera-id` / `--right-camera-id` / `--wrist-camera-id`: Camera serial numbers (**required**)
- `--external-camera`: Which external camera to use for the policy (`left` or `right`, default: `left`)
- `--server-port`: Port the TCP server listens on (default: `6000`)
- `--prompt`: Task instruction for the robot (**required**)
- `--max-steps`: Max steps per episode before timeout (default: `600`)
- `--resolution`: Image resolution (default: `224`)

### 2. Set Up Tunneling (Robot Machine)

Open a **second terminal** on the robot machine:

```bash
# Pinggy (free tier):
ssh -p 443 -R0:localhost:6000 qr+tcp@free.pinggy.io
```

Note the generated URL (e.g., `abc123.a.pinggy.io`) and port (e.g., `443`). You will use these as `remote_robot.host` and `remote_robot.port` in step 4.

### 3. Start the Reward Relabel Server (Training Machine)

Open a terminal on the training machine:

```bash
# Robometer (Qwen-based progress + success prediction):
uv run python scripts/start_reward_relabel_server.py \
    reward_model=robometer \
    reward_model.model_path="robometer/Robometer-4B" \
    server.port=50052 \
    server.host="0.0.0.0" \
    device=cuda \
    server.image_keys='["observation/exterior_image_1_left"]'
```

This loads the reward model and listens for gRPC requests on `localhost:50052`. Keep it running in the background.

### 4. Start Training (Training Machine)

In a **second terminal** on the training machine, download the Pi0 checkpoint first:

```bash
# Download Pi0 DROID checkpoint (one-time):
uv run hf download jesbu1/pi0_droid --local-dir checkpoints/pi0_droid
```

Then start training. **Only three values need to be changed** — your W&B entity, and the robot tunnel address from step 2:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python scripts/train_dsrl.py \
    --config-name dsrl_remote_robot_async_relabel_config \
    logging.wandb_entity=YOUR_WANDB_ENTITY \
    remote_robot.host=YOUR_ROBOT_HOST \
    remote_robot.port=YOUR_ROBOT_PORT
```

**Example** (with Pinggy tunnel):
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python scripts/train_dsrl.py \
    --config-name dsrl_remote_robot_async_relabel_config \
    logging.wandb_entity=myusername \
    remote_robot.host=abc123.a.pinggy.io \
    remote_robot.port=443
```

All other settings (async relabel, buffer, reward model server address, noise bound, eval config) are baked into the config file at `robometer_policy_learning/configs/dsrl_remote_robot_async_relabel_config.yaml`. To customize them, either edit the config or pass CLI overrides.

**Config reference** (see `dsrl_remote_robot_async_relabel_config.yaml`):
| Setting | Default | Description |
|---------|---------|-------------|
| `dsrl.pi0_checkpoint` | `checkpoints/pi0_droid` | Path to Pi0 DROID checkpoint |
| `dsrl.noise_action_bound` | `1.0` | Action noise bound |
| `dsrl.action_exec_len` | `8` | Actions to execute per chunk |
| `dsrl.noise_dim` | `32` | Dimension of noise vector |
| `reward_model.async_reward_relabel_server_address` | `localhost:50052` | gRPC reward server address |
| `env.reward_relabel_batch_size` | `1` | Trajectories per relabel batch |
| `buffer.capacity` | `10000` | Replay buffer capacity |
| `training.num_rollouts` | `10000` | Total env steps to train |
| `remote_robot.connect_timeout` | `300.0` | Connection timeout (seconds) |

### 5. Evaluate Pi0 (Training Machine)

You can evaluate the Pi0 policy (with random noise or a trained DSRL policy) against the remote robot without training:

```bash
# Random noise evaluation (baseline):
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python scripts/eval_pi0.py \
    server.host=YOUR_ROBOT_HOST \
    server.port=YOUR_ROBOT_PORT \
    logging.wandb_entity=YOUR_ENTITY

# Evaluate with a trained DSRL policy:
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python scripts/eval_pi0.py \
    use_random_noise=false \
    policy_checkpoint=./replay_buffers/policy.pt \
    server.host=YOUR_ROBOT_HOST \
    server.port=YOUR_ROBOT_PORT \
    logging.wandb_entity=YOUR_ENTITY
```

**Config reference** (see `robometer_policy_learning/configs/eval_pi0.yaml`):
| Setting | Default | Description |
|---------|---------|-------------|
| `use_random_noise` | `true` | Use random Gaussian noise (baseline) |
| `policy_checkpoint` | `null` | Path to trained DSRL policy checkpoint |
| `pi0.checkpoint` | `checkpoints/pi0_droid` | Path to Pi0 DROID checkpoint |
| `pi0.action_exec_len` | `8` | Actions to execute per chunk |
| `pi0.noise_dim` | `32` | Noise vector dimension |
| `noise_scale` | `1.0` | Scale factor for random noise |
| `eval.num_episodes` | `20` | Number of evaluation episodes |
| `eval.record_video` | `true` | Record evaluation videos |

**SIMPLER simulation** (no robot needed):
```bash
uv run python scripts/eval_pi0.py \
    env.env_type=simpler \
    server.host=localhost server.port=6000
```

## Quick Start (WidowX / BRIDGE stack)

For the WidowX manipulator with the BRIDGE data collection stack, see the **WidowX controller** section below first.

### WidowX controller (BRIDGE stack)

1. **Install** WidowX-related dependencies:
   ```bash
   uv pip install -e ".[widowx]"
   ```

2. **Robot hardware**: follow the [BRIDGE repository](https://github.com/rail-berkeley/bridge_data_robot) instructions.

3. **Environment service** (from the `bridge_data_robot` checkout):
   ```bash
   USB_CONNECTOR_CHART=$(pwd)/usb_connector_chart.yml docker compose up --build robonet
   # In a separate terminal:
   docker compose exec robonet bash -lic "widowx_env_service --server"
   ```

### 1. Start the Robot Server (Robot Machine)

```bash
uv run python robometer_policy_learning/robots/widowx_remote_server.py \
    --robot-ip localhost \
    --robot-port 5556 \
    --server-port 6000 \
    --prompt "pick up the red block and place it in the bowl" \
    --max-steps 60
```

**Arguments:**
- `--robot-ip`: IP of the WidowX controller (default: `localhost`)
- `--robot-port`: Port of the WidowX controller (default: `5556`)
- `--server-port`: Port for the remote server (default: `6000`)
- `--prompt`: Task instruction (**required**)
- `--max-steps`: Max steps per episode (default: `120`)
- `--resolution`: Image resolution (default: `224`)
- `--wait-for-enter` / `--no-wait-for-enter`: Wait for Enter before each episode (default: `True`)

### 2. Set Up Tunneling + Training

Same as DROID steps 2-4 above. Use `dsrl_bridge_config.yaml` as the config name for training.

## Episode Workflow

By default, the robot waits for you to press **Enter** before starting each episode:

```
============================================================
Resetting robot for new episode...
============================================================

Task: put the red block in the bowl
Max steps: 600

────────────────────────────────────────────────────────────
Robot is ready. Set up the scene if needed.
────────────────────────────────────────────────────────────
>>> Press ENTER to start episode...
```

This gives you time to:
- Position objects in the scene
- Move obstacles out of the way
- Ensure the robot's workspace is clear

**Important for DROID:** The arm physically resets to its home position before the prompt appears. Reset your scene *during* this motion — if you wait until the prompt is shown, the initial observation may be captured with the arm already back at home while the scene is still in a post-episode state. Press Enter only after both the arm and the scene are ready.

## Keyboard Controls

While the robot server is running:

| Key | Action |
|-----|--------|
| `ENTER` | Start episode after reset |
| `s` | Mark current episode as **SUCCESS** (reward = 1) |
| `f` | Mark current episode as **FAILURE** (reward = 0) |
| `q` | Quit server |

## Reward Structure

- **Operator marks success (presses 's'):** reward = 1.0 (server sends 1.0)
- **Operator marks failure (presses 'f'):** reward = 0.0 (server sends 0.0)
- **Ongoing step:** reward = 0.0
- **Timeout (max_steps reached):** episode truncated, operator prompted to label

With async reward relabeling enabled, the Robometer server additionally computes per-frame progress predictions and success probabilities, which are merged into the replay buffer retroactively.

## Connection Handling

The training client automatically:
- Retries connection for up to 5 minutes (configurable via `remote_robot.connect_timeout`)
- Reconnects if connection is lost during training
- Handles network interruptions gracefully

## WidowX observation layout

The WidowX remote server processes end-effector state before sending over TCP:
1. Quaternion rotation converted to Euler angles (top-down frame, BRIDGE conventions)
2. Proprio state layout: `[x, y, z, roll, pitch, yaw, gripper_openness]`
3. RGB images resized to requested resolution from the configured camera

## Protocol Reference

Length-prefixed **pickle** over TCP. See [robots/README.md](../robometer_policy_learning/robots/README.md) for full protocol details.

**RESET:**
```python
# Client sends:
{'type': 'RESET'}

# Server responds (DROID format):
{
    'observation/exterior_image_1_left': np.ndarray,   # (224, 224, 3) uint8
    'observation/wrist_image_left': np.ndarray,          # (224, 224, 3) uint8
    'observation/joint_position': np.ndarray,            # (7,) float32
    'observation/gripper_position': np.ndarray,          # (1,) float32
    'prompt': str                                        # Task instruction
}
```

**STEP:**
```python
# Client sends:
{'type': 'STEP', 'action': np.ndarray}  # (8,) action for DROID

# Server responds:
{
    'observation/...': ...,   # Same keys as RESET
    'reward': float,          # 0.0 or 1.0
    'done': bool,
    'truncated': bool,
    'info': dict
}
```

**CLOSE:**
```python
{'type': 'CLOSE'}  # Server closes connection
```

## Troubleshooting

### Connection Refused
```
Connection failed: [Errno 111] Connection refused
```
- Ensure the robot server is running on the robot machine
- Check that the Pinggy/ngrok tunnel is active
- Verify `remote_robot.host` and `remote_robot.port` match the tunnel

### Timeout During Training
```
Failed to connect after 300s
```
- Increase `remote_robot.connect_timeout` via CLI override
- Check network connectivity
- Restart the tunnel

### Robot Not Responding
- Check the robot controller is running (DROID hardware, or WidowX docker stack)
- Look for errors in the robot server terminal
- Try restarting the robot server

### Reward Relabel Server Errors
```
gRPC connection failed
```
- Ensure the reward relabel server is running on the training machine (step 3)
- Verify `reward_model.async_reward_relabel_server_address` matches (default: `localhost:50052`)
- Check that the model downloaded successfully

### Image or Camera Issues
- Verify camera serial numbers for DROID (`--left-camera-id`, `--wrist-camera-id`, etc.)
- Check camera permissions and USB connections
- Ensure `--resolution` matches what the policy expects (default `224`)

## Safety

- Supervise the robot whenever the server is running
- Keep an emergency stop within reach
- Start with small motions and clear the workspace before long runs
- Prefer a dry run in a constrained workspace before long training sessions

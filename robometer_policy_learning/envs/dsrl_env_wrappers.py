from robometer_policy_learning.envs.libero_pi0_wrapper import LiberoPI0Wrapper, VectorLiberoPromptWrapper
import copy
from typing import List, Dict, Optional, Any
import numpy as np
import gymnasium as gym
import sys
import os
from robometer_policy_learning.envs.remote_env import RemoteEnv
from robometer_policy_learning.utils.env_utils import GymToGymnasiumWrapper
from transformers import AutoModel, AutoImageProcessor
from loguru import logger
from robometer_policy_learning.envs.dino_wrapper import DinoEmbeddingWrapper, VectorDinoEmbeddingWrapper
import torch
from sentence_transformers import SentenceTransformer


class SimpleDSRLWrapper(gym.ObservationWrapper):
    """
    Lightweight wrapper to convert RemoteEnv observations to DSRL format.
    Adds language embeddings and handles key mapping.
    """

    def __init__(
        self,
        env: gym.Env,
        sentence_model: SentenceTransformer = None,
    ):
        super().__init__(env)
        self.sentence_model = sentence_model
        self.language_instruction = None
        self.language_encoding = None
        self.observation_space = env.observation_space
        assert hasattr(env, "dsrl_key_mapping"), f"env {env.__class__.__name__} must have dsrl_key_mapping attribute"
        self.dsrl_key_mapping = env.dsrl_key_mapping

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Extract prompt from info (RemoteEnv puts it there) or obs (fallback)
        self.language_instruction = obs.get("prompt")
        assert self.language_instruction is not None, "Prompt is None"

        # Compute language embedding
        if self.sentence_model is not None:
            from robometer.utils.embedding_utils import compute_text_embeddings

            enc = compute_text_embeddings(self.language_instruction, self.sentence_model)
            self.language_encoding = enc.cpu().numpy().astype(np.float32)
        else:
            self.language_encoding = np.zeros((384,), dtype=np.float32)

        return self._format_obs(obs), info

    def observation(self, obs):
        return self._format_obs(obs)

    def _format_obs(self, obs):
        # Handle missing keys gracefully (e.g., during errors)
        formatted_obs = obs
        if obs["prompt"] != self.language_instruction:
            # Compute language embedding
            if self.sentence_model is not None:
                from robometer.utils.embedding_utils import compute_text_embeddings

                enc = compute_text_embeddings(self.language_instruction, self.sentence_model)
                self.language_encoding = enc.cpu().numpy().astype(np.float32)
            else:
                self.language_encoding = np.zeros((384,), dtype=np.float32)
            self.language_instruction = obs["prompt"]
        formatted_obs["language"] = self.language_encoding
        formatted_obs["prompt"] = self.language_instruction
        return formatted_obs


class SimplerDenseRewardWrapper(gym.Wrapper):
    """
    Dense reward wrapper for SimplerEnv WidowX env to just test out DSRL with gt dense reward.

    Computes dense rewards based on the info dict returned by ManiSkill:
    - is_src_obj_grasped: Object is being grasped
    - consecutive_grasp: Grasp is maintained over steps
    - src_on_target: Object is on/near target location
    - moved_correct_obj: Correct object was moved
    - moved_wrong_obj: Wrong object was moved (penalty)
    - success: Task completed successfully

    """

    # Reward components (can be overridden)
    STEP_COST = 0
    GRASP_REWARD = 0.1
    CONSECUTIVE_GRASP_REWARD = 0.1
    ON_TARGET_REWARD = 0.1
    MOVED_CORRECT_REWARD = 0.1
    MOVED_WRONG_PENALTY = -0.1
    SUCCESS_BONUS = 100

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.dsrl_key_mapping = env.dsrl_key_mapping
        self._prev_info = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_info = {}
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Compute dense reward from info dict
        dense_reward = self._compute_dense_reward(info)

        # Store for potential delta-based rewards in future
        self._prev_info = info

        return obs, dense_reward, done, truncated, info

    def _compute_dense_reward(self, info: dict) -> float:
        """Compute dense reward from ManiSkill info dict."""
        reward = self.STEP_COST

        # Grasp rewards
        if info["is_src_obj_grasped"]:
            reward += self.GRASP_REWARD

        if info["consecutive_grasp"]:
            reward += self.CONSECUTIVE_GRASP_REWARD

        # Object placement rewards
        if info["src_on_target"]:
            reward += self.ON_TARGET_REWARD

        # Movement rewards/penalties
        if info["moved_correct_obj"]:
            reward += self.MOVED_CORRECT_REWARD

        if info["moved_wrong_obj"]:
            reward += self.MOVED_WRONG_PENALTY

        # Success bonus
        if info["success"]:
            reward += self.SUCCESS_BONUS

        return reward


def _hide_collision_geoms(env):
    """Hide LIBERO's collision geoms in the offscreen render.

    robosuite already tries to do this -- MujocoEnv._reset_internal sets
    vopt.geomgroup[0]=0 when render_collision_mesh is False -- but under
    mujoco 2.3.7 writing to MjvOption.geomgroup sets every entry to the same
    value, so the next line (geomgroup[1]=1 for render_visual_mesh) turns all
    groups back on, collision included:

        >>> o = mujoco.MjvOption()
        >>> o.geomgroup[0] = 0   # -> [0, 0, 0, 0, 0, 0]
        >>> o.geomgroup[1] = 1   # -> [1, 1, 1, 1, 1, 1]

    The result is that agentview frames show collision geometry drawn over the
    visual meshes (e.g. KITCHEN_SCENE6's microwave renders in its
    micro_collision_blue material, and the Panda base in the group-0 rgbas from
    robot.xml). Since these frames are what the reward model scores, it seemed
    worth fixing.

    Setting geomgroup from Python doesn't work on this version, so hide the
    collision geoms in the model instead by zeroing their alpha. Re-applied on
    every reset because hard_reset=True rebuilds the model. This is a no-op on
    setups where the flag already works.
    """
    reset = env.reset

    def reset_and_hide(*args, **kwargs):
        out = reset(*args, **kwargs)
        model = env.sim.model
        collision = np.asarray(model.geom_group) == 0
        model.geom_rgba[collision, 3] = 0.0
        return out

    env.reset = reset_and_hide
    return env


def setup_libero_env(
    task_suite_name: str,
    task_id: int,
    n_envs: int,
    dinov2_model: AutoModel = None,
    dinov2_processor: AutoImageProcessor = None,
    sentence_model: SentenceTransformer = None,
    device: torch.device = None,
    seed: int = None,
    max_episode_steps: int = 400,
    image_keys: List[str] = ["observation/image"],
    extra_keys_to_drop: List[str] = [],
    async_reward_relabel_kwargs: Optional[Dict] = None,
):
    """
    Setup LIBERO environment.

    Args:
        task_suite_name: Name of LIBERO task suite
        task_id: Task ID within suite
        n_envs: Number of parallel environments
        dinov2_model: DINOv2 model for embedding wrapper
        dinov2_processor: DINOv2 processor for preprocessing images
        sentence_model: Sentence transformer model for language embeddings
        device: Device to load DINOv2 model on
        seed: Random seed
        max_episode_steps: Maximum number of steps per episode
        image_keys: List of image keys for DINO embedding
        extra_keys_to_drop: Additional keys to drop from observations
        async_reward_relabel_kwargs: Optional dict with async reward relabeling config. If None, async relabeling is disabled.
            Expected keys: server_address, batch_size, max_queue_size, timeout, flush_interval,
            success_detection_duration, success_detection_threshold, use_relative_rewards, use_placeholder_rewards
    Returns:
        env: Vectorized LIBERO environment
        remove_obs_keys: Keys to remove from observations for replay buffer
    """
    try:
        from libero.libero.envs import OffScreenRenderEnv, DummyVectorEnv
        from libero.libero import benchmark, get_libero_path
    except ImportError:
        logger.error("LIBERO not found. Please install LIBERO.")
        sys.exit(1)

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Setting up LIBERO environment: {task_suite_name}, task {task_id}")

    # Get task info
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)

    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    # Process async reward relabeling config
    use_async_reward_relabel = async_reward_relabel_kwargs is not None
    if use_async_reward_relabel:
        if "server_address" not in async_reward_relabel_kwargs:
            raise ValueError("server_address is required in async_reward_relabel_kwargs")
        logger.info(
            f"Async reward relabeling enabled for {n_envs} environments (server: {async_reward_relabel_kwargs['server_address']})"
        )

    # Create environments
    env_fns = []
    for i in range(n_envs):

        def make_env():
            env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}
            base_env = OffScreenRenderEnv(**env_args)
            base_env.seed(seed + i)
            _hide_collision_geoms(base_env)
            base_env = GymToGymnasiumWrapper(base_env, time_limit=max_episode_steps)
            wrapped_env = LiberoPI0Wrapper(
                base_env,
            )

            # Wrap with async reward relabeling if enabled
            # This must be done before DinoEmbeddingWrapper so it has access to raw images
            if use_async_reward_relabel:
                from robometer_policy_learning.envs.async_reward_relabel_wrapper import AsyncRewardRelabelEnvWrapper
                from robometer_policy_learning.distributed.clients.reward_relabel_client import RewardRelabelClient

                # Extract client settings from kwargs
                client_kwargs = {
                    "address": async_reward_relabel_kwargs["server_address"],
                    "max_queue_size": async_reward_relabel_kwargs.get("max_queue_size", 100),
                    "timeout": async_reward_relabel_kwargs.get("timeout", 60.0),
                    "flush_interval": async_reward_relabel_kwargs.get("flush_interval", 0.1),
                }

                env_reward_relabel_client = RewardRelabelClient(**client_kwargs)

                # Extract wrapper kwargs (exclude server_address and client settings)
                wrapper_kwargs = {
                    "batch_size": async_reward_relabel_kwargs.get("batch_size", 32),
                    "success_detection_duration": async_reward_relabel_kwargs.get("success_detection_duration", 2),
                    "success_detection_threshold": async_reward_relabel_kwargs.get("success_detection_threshold", 0.65),
                    "use_relative_rewards": async_reward_relabel_kwargs.get("use_relative_rewards", False),
                    "sync_mode": async_reward_relabel_kwargs.get("sync_mode", False),  # Default to async
                    "action_exec_len": async_reward_relabel_kwargs.get(
                        "action_exec_len", None
                    ),  # For DSRL mode: None means no chunking
                }

                wrapped_env = AsyncRewardRelabelEnvWrapper(
                    env=wrapped_env,
                    reward_relabel_client=env_reward_relabel_client,
                    **wrapper_kwargs,
                )
                sync_mode_str = "sync" if wrapper_kwargs["sync_mode"] else "async"
                logger.debug(f"Wrapped env {i} with AsyncRewardRelabelEnvWrapper (mode={sync_mode_str})")

            if dinov2_model is not None:
                # make a regular dino embedding wrapper for the env so the step with action chunk works properly
                # as it uses the unwrapped env in dsrl_rollout_worker and dsrl_evaluation_worker
                wrapped_env = DinoEmbeddingWrapper(
                    wrapped_env, dinov2_model, dinov2_processor, device=device, image_keys=image_keys
                )

            # Set metadata
            wrapped_env.task_id = task_id
            wrapped_env.task_suite = task_suite

            return wrapped_env

        env_fns.append(make_env)

    env = gym.vector.SyncVectorEnv(env_fns)
    env = VectorLiberoPromptWrapper(env, sentence_model)
    # # Create vectorized environment
    # if dinov2_model is not None:
    #     single_space = getattr(env, "single_observation_space", env.observation_space)
    #     if isinstance(single_space, gym.spaces.Dict) and "observation/image" in single_space.spaces:
    #         # uses only observation/image no wrist imag
    #         env = VectorDinoEmbeddingWrapper(env, dinov2_model, dinov2_processor, device=device, image_keys=["observation/image"])

    logger.info(
        f"✓ Created {n_envs} LIBERO environments"
        + (" with async reward relabeling" if use_async_reward_relabel else "")
    )
    remove_obs_keys = ["observation/wrist_image", "language", "image", "wrist_image", "prompt"] + extra_keys_to_drop
    if dinov2_model:
        remove_obs_keys += image_keys
    return env, remove_obs_keys


# Create dummy DSRL environment to get observation and action spaces
class DummyDSRLEnv(gym.Env):
    def __init__(
        self,
        observation_space,
        action_space,
        pi0_wrapper,
        noise_dim,
        chunk_size=None,
        action_bound=1.0,
        use_vlm_features=True,
        combine_image_features=False,
    ):
        # Use provided observation_space and action_space to construct dsrl_observation_space

        # Take a sample obs to compute vlm_feature_dim
        example_obs = observation_space.sample()
        if "dino_embedding" in example_obs:
            example_obs.pop("dino_embedding")
        example_obs["prompt"] = "test"

        ## Remove 'observation/' prefix from observation space keys
        # obs_space_dict = {}
        # for k, v in observation_space.spaces.items():
        #    new_k = k
        #    if k.startswith("observation/"):
        #        new_k = k[len("observation/"):]
        #    obs_space_dict[new_k] = v
        obs_space_dict = copy.deepcopy(observation_space.spaces)
        # Remove any thing in obs_space_dict in remove_obs_keys
        # for k in observation_space.spaces.keys():
        # if k in remove_obs_keys:
        # obs_space_dict.pop(k)

        ## Concatenate new state dim
        # orig_state_dim = obs_space_dict['state'].shape[0]
        # new_state_dim = orig_state_dim + vlm_feature_dim
        # obs_space_dict['state'] = gym.spaces.Box(
        #    low=-np.inf, high=np.inf, shape=(orig_state_dim,), dtype=np.float32
        # )
        self.use_vlm_features = use_vlm_features
        if use_vlm_features:
            with torch.inference_mode():
                vlm_features = pi0_wrapper.get_features(example_obs).detach().cpu().numpy()
            vlm_feature_dim = vlm_features.shape[-1] if len(vlm_features.shape) > 1 else vlm_features.shape[0]

            obs_space_dict["vlm_features"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(vlm_feature_dim,), dtype=np.float32
            )

        self.observation_space = gym.spaces.Dict(obs_space_dict)
        self.action_space = gym.spaces.Box(low=-action_bound, high=action_bound, shape=(noise_dim,), dtype=float)
        if chunk_size is not None:
            self.chunk_action_space = gym.spaces.Box(
                low=-action_bound, high=action_bound, shape=(chunk_size, noise_dim), dtype=float
            )
        else:
            self.chunk_action_space = None

    def sample_action(self):
        if self.chunk_action_space is not None:
            return self.chunk_action_space.sample()[None, ...]
        else:
            return self.action_space.sample()[None, ...]

    def reset(self):
        return None, None

    def step(self, action):
        return None, 0.0, True, {}


def make_simpler_env(
    n_envs: int,
    extra_keys_to_drop: List[str] = [],
    dinov2_model: AutoModel = None,
    dinov2_processor: AutoImageProcessor = None,
    sentence_model: SentenceTransformer = None,
    device: torch.device = None,
    use_dense_reward: bool = False,
    host: str = "0.0.0.0",
    port: int = 6000,
    num_stages: int = 1,
):
    """
    Create a vectorized SimplerEnv using the RemoteSimplerEnv wrapper.
    Connects to a locally (or remotely) running SimplerEnv server.

    Args:
        n_envs: Number of parallel environments
        extra_keys_to_drop: Addtl keys to drop from observation beyond image_keys (default: [])
        dinov2_model: DINOv2 model for image embedding
        dinov2_processor: DINOv2 image processor
        sentence_model: Sentence transformer for language embedding
        device: Torch device for embedding models
        use_dense_reward: If True, use dense rewards based on ManiSkill info dict
                         If False, use sparse rewards (-1 per step, 0 on success)
        host: Server hostname (default: "0.0.0.0")
        port: Server port (default: 6000)
        num_stages: Number of stages for multi-stage tasks (default: 1)

    Returns:
        env: Vectorized environment
        remove_obs_keys: Keys to remove from observation for replay buffer
    """
    env_fns = []
    for i in range(n_envs):

        def make_env(
            host=host,
            port=port,
            sentence_model=sentence_model,
            use_dense_reward=use_dense_reward,
            dinov2_model=dinov2_model,
            dinov2_processor=dinov2_processor,
            device=device,
            num_stages=num_stages,
        ):
            # Use new RemoteEnv with URL-based connection
            server_url = f"tcp://{host}:{port}"
            env = RemoteEnv(
                server_url=server_url,
                obs_format="widowx",
                socket_timeout=30.0,  # 30s timeout for recv/send operations
                connect_timeout=300.0,  # 5 min timeout to wait for server to start
                retry_interval=5.0,  # Retry every 5 seconds
                num_stages=num_stages,
            )

            # Apply dense reward wrapper if requested
            if use_dense_reward:
                env = SimplerDenseRewardWrapper(env)

            env = SimpleDSRLWrapper(env, sentence_model)
            if dinov2_model is not None:
                env = DinoEmbeddingWrapper(
                    env, dinov2_model, dinov2_processor, device=device, image_keys=["observation.images.image_0"]
                )
            return env

        env_fns.append(make_env)

    env = gym.vector.SyncVectorEnv(env_fns)

    reward_type = "dense" if use_dense_reward else "sparse"
    logger.info(f"✓ Created {n_envs} SimplerEnv environment(s) with {reward_type} rewards")

    remove_obs_keys = ["language", "prompt"] + extra_keys_to_drop
    if dinov2_model is not None:
        remove_obs_keys.extend(["observation.images.image_0", "observation_images_image_0"])
    return env, remove_obs_keys


def make_remote_robot_env(
    n_envs: int,
    host: str = "localhost",
    port: int = 6000,
    dinov2_model: AutoModel = None,
    dinov2_processor: AutoImageProcessor = None,
    sentence_model: SentenceTransformer = None,
    device: torch.device = None,
    # Configurable observation keys
    image_keys: List[str] = ["observation.images.image_0"],
    extra_keys_to_drop: List[str] = [],
    obs_format: str = "widowx",
    num_stages: int = 1,
    async_reward_relabel_kwargs: Optional[Dict] = None,
):
    """
    Create a vectorized remote robot environment for DSRL training.
    
    Connects to a remote robot server (e.g., widowx_remote_server.py) over TCP.
    Works with tunneling services like ngrok and Pinggy.
    
    This function can also be used as a drop-in replacement for make_simpler_env
    by using RemoteEnv instead of RemoteSimplerEnv (with the correct protocol).
    
    Args:
        n_envs: Number of parallel environments (typically 1 for real robots)
        host: Robot server hostname. Examples:
              - "localhost" for local connection
              - "0.tcp.ngrok.io" for ngrok TCP tunnel
              - "your-subdomain.a.pinggy.io" for Pinggy
        port: Robot server port
        dinov2_model: DINOv2 model for image embedding
        dinov2_processor: DINOv2 image processor
        sentence_model: Sentence transformer for language embedding
        device: Torch device for embedding models
        image_keys: Keys for dino conversion in DSRL observation (default: ['observation.images.image_0'])
        extra_keys_to_drop: Addtl keys to drop from observation beyond image_keys (default: [])
        obs_format: Observation format - "droid" or "widowx"
        num_stages: Number of stages for multi-stage tasks (default: 1)
        async_reward_relabel_kwargs: Optional dict with async reward relabeling config. If None, async relabeling is disabled.
            Expected keys: server_address, batch_size, max_queue_size, timeout, flush_interval,
            success_detection_duration, success_detection_threshold, use_relative_rewards, action_exec_len, sync_mode
    
    Returns:
        Vectorized gymnasium environment
    
    Example usage with Pinggy:
        # Terminal 1 (robot machine): Start robot server
        python robots/widowx_remote_server.py --prompt "pick up the red block"
        
        # Terminal 2 (robot machine): Start Pinggy tunnel
        ssh -p 443 -R0:localhost:6000 a.pinggy.io
        # Note the assigned URL, e.g., "xyz123.a.pinggy.io:443"
        
        # Terminal 3 (training machine): Start training
        python scripts/train_dsrl.py \\
            env_name="REMOTE_ROBOT" \\
            remote_robot.host="xyz123.a.pinggy.io" \\
            remote_robot.port=443
    
    Example usage with ngrok:
        # Terminal 1 (robot machine): Start robot server
        python robots/widowx_remote_server.py --prompt "pick up the red block"
        
        # Terminal 2 (robot machine): Start ngrok tunnel
        ngrok tcp 6000
        # Note the assigned URL, e.g., "0.tcp.ngrok.io:12345"
        
        # Terminal 3 (training machine): Start training
        python scripts/train_dsrl.py \\
            env_name="REMOTE_ROBOT" \\
            remote_robot.host="0.tcp.ngrok.io" \\
            remote_robot.port=12345
    
    Example with custom keys:
        env = make_remote_robot_env(
            n_envs=1,
            host="localhost",
            port=6000,
            image_keys=["observation.images.image_0"],
        )
    """
    logger.info(f"Creating remote robot environment connecting to {host}:{port}")
    logger.info(f"  image keys: {image_keys}")

    # Process async reward relabeling config
    use_async_reward_relabel = async_reward_relabel_kwargs is not None
    if use_async_reward_relabel:
        if "server_address" not in async_reward_relabel_kwargs:
            raise ValueError("server_address is required in async_reward_relabel_kwargs")
        logger.info(
            f"Async reward relabeling enabled for {n_envs} environments (server: {async_reward_relabel_kwargs['server_address']})"
        )

    env_fns = []
    for i in range(n_envs):
        # Capture variables in closure properly
        def make_env(
            host=host,
            port=port,
            sentence_model=sentence_model,
            dinov2_model=dinov2_model,
            dinov2_processor=dinov2_processor,
            device=device,
            image_keys=image_keys,
            async_reward_relabel_kwargs=async_reward_relabel_kwargs,
            num_stages=num_stages,
        ):
            # Use new RemoteEnv with URL-based connection
            server_url = f"tcp://{host}:{port}"
            env = RemoteEnv(
                server_url=server_url,
                obs_format=obs_format,  # SimplerEnv format (state + image)
                socket_timeout=30.0,  # 30s timeout for recv/send operations
                connect_timeout=300.0,  # 5 min timeout to wait for server to start
                retry_interval=5.0,  # Retry every 5 seconds
                num_stages=num_stages,
            )
            # Wrap with DSRL format converter
            env = SimpleDSRLWrapper(
                env,
                sentence_model,
            )

            # Wrap with async reward relabeling if enabled
            # This must be done BEFORE DinoEmbeddingWrapper so it has access to raw images
            if use_async_reward_relabel:
                from robometer_policy_learning.envs.async_reward_relabel_wrapper import AsyncRewardRelabelEnvWrapper
                from robometer_policy_learning.distributed.clients.reward_relabel_client import RewardRelabelClient

                # Extract client settings from kwargs
                client_kwargs = {
                    "address": async_reward_relabel_kwargs["server_address"],
                    "max_queue_size": async_reward_relabel_kwargs.get("max_queue_size", 100),
                    "timeout": async_reward_relabel_kwargs.get("timeout", 60.0),
                    "flush_interval": async_reward_relabel_kwargs.get("flush_interval", 0.1),
                }

                env_reward_relabel_client = RewardRelabelClient(**client_kwargs)

                # Extract wrapper kwargs (exclude server_address and client settings)
                wrapper_kwargs = {
                    "batch_size": async_reward_relabel_kwargs.get("batch_size", 32),
                    "success_detection_duration": async_reward_relabel_kwargs.get("success_detection_duration", 2),
                    "success_detection_threshold": async_reward_relabel_kwargs.get("success_detection_threshold", 0.65),
                    "use_relative_rewards": async_reward_relabel_kwargs.get("use_relative_rewards", False),
                    "sync_mode": async_reward_relabel_kwargs.get("sync_mode", False),  # Default to async
                    "action_exec_len": async_reward_relabel_kwargs.get(
                        "action_exec_len", None
                    ),  # For DSRL mode: None means no chunking
                }

                env = AsyncRewardRelabelEnvWrapper(
                    env=env,
                    reward_relabel_client=env_reward_relabel_client,
                    **wrapper_kwargs,
                )
                sync_mode_str = "sync" if wrapper_kwargs["sync_mode"] else "async"
                logger.debug(f"Wrapped remote robot env {i} with AsyncRewardRelabelEnvWrapper (mode={sync_mode_str})")

            if dinov2_model is not None:
                # make a regular dino embedding wrapper for the env so the step with action chunk works properly
                # as it uses the unwrapped env in dsrl_rollout_worker and dsrl_evaluation_worker
                env = DinoEmbeddingWrapper(env, dinov2_model, dinov2_processor, device=device, image_keys=image_keys)
            return env

        env_fns.append(make_env)

    env = gym.vector.SyncVectorEnv(env_fns)
    logger.info(
        f"✓ Created remote robot environment with {n_envs} env(s)"
        + (" with async reward relabeling" if use_async_reward_relabel else "")
    )
    remove_obs_keys = extra_keys_to_drop + ["language", "prompt"]
    if dinov2_model:  # remove dino image keys
        remove_obs_keys += image_keys
    return env, remove_obs_keys
